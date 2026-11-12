"""Hardware layer: stage + piezo + photon counter, plus a combined positioner.

``PyscanHardware`` talks to the real instruments through pyscan (same calls as the
old notebook). ``SimulatedHardware`` fakes a sample with alignment markers and
emitters so the whole GUI can be exercised without the microscope.

``Positioner`` hides the two coordinate systems: callers ask for an absolute
position in um and it decides how much goes to the stage and how much to the piezo.
"""

import ctypes
import os
import sys
import threading
import time

import numpy as np

from .config import AXES


class MoveError(RuntimeError):
    pass


class HardwareBase:
    """Low-level interface. Stage in mm, piezo in um, counts in counts/s."""

    name = "base"
    speed = 1.0  # simulator only: >1 shortens waits

    def wait(self, seconds):
        if seconds > 0:
            time.sleep(seconds / self.speed)

    def get_stage(self):
        raise NotImplementedError

    def set_stage_axis(self, axis, value_mm):
        """Move one stage axis (0=x, 1=y, 2=z) only; the others are not commanded."""
        raise NotImplementedError

    def get_piezo(self):
        raise NotImplementedError

    def set_piezo(self, xyz_um):
        raise NotImplementedError

    def set_piezo_axis(self, axis, value_um):
        """Move one piezo axis only."""
        raise NotImplementedError

    def get_count_rates(self):
        """Return (channel0, channel1) count rates."""
        raise NotImplementedError

    def home_stage(self, progress=None):
        """Home the stage axes (z first, so it moves away from the objective).
        ``progress(xyz_mm)`` is called with the current position while homing."""
        raise NotImplementedError

    def close(self):
        pass


def _wait_until_reached(set_fn, get_fn, target, tol, accept, timeout, poll=0.02, still_time=0.3):
    """Command ``target`` and poll the readback until every axis is within ``tol``.

    Like the old notebook, the move is re-issued if the device stops short. If it
    has stopped within ``accept`` of the target, that position is used (the piezo
    makes up the difference, and the readback is what gets recorded).
    Returns the final readback.
    """
    target = np.asarray(target, float)
    t0 = time.monotonic()
    set_fn(target)
    last = None
    last_change = time.monotonic()
    while True:
        pos = np.asarray(get_fn(), float)
        err = np.max(np.abs(pos - target))
        if err <= tol:
            return pos
        now = time.monotonic()
        if last is None or np.max(np.abs(pos - last)) > tol / 4:
            last, last_change = pos, now
        elif now - last_change > still_time:       # device has stopped
            if err <= accept:
                return pos
            set_fn(target)                          # stopped short: re-issue
            last, last_change = None, now
        if now - t0 > timeout:
            raise MoveError(f"move to {np.round(target, 5)} timed out at {np.round(pos, 5)}")
        time.sleep(poll)


class PyscanHardware(HardwareBase):
    name = "pyscan"

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg["pyscan_path"] not in sys.path:
            sys.path.append(cfg["pyscan_path"])
        import pyscan as ps  # imported here so the simulator works without pyscan
        self.ps = ps
        self.stage = self.piezo = self.ph = None
        try:
            # PicoHarp first: if it can't be opened we stop before touching the motors
            self.ph = self._open_picoharp(ps, self._remembered_picoharp_index())
            self._remember_picoharp_index(self.ph.dev)
            self.ph.init_correlation_mode()
            self.stage = ps.ThorlabsBSC203(serial=cfg["stage_serial"])
            time.sleep(1)
            self.piezo = ps.ThorlabsBSC303(serial=cfg["piezo_serial"])
            self._seed_stage_cache()
        except Exception:
            self.close()   # release whatever was opened, so a retry can open it again
            raise

    def _picoharp_index_path(self):
        return self.cfg["picoharp_index_file"] or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".picoharp_index.txt")

    def _remembered_picoharp_index(self):
        """The device number that worked last time, else the config value."""
        try:
            with open(self._picoharp_index_path()) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return int(self.cfg["picoharp_index"])

    def _remember_picoharp_index(self, idx):
        try:
            with open(self._picoharp_index_path(), "w") as f:
                f.write(str(idx))
        except OSError:
            pass

    @staticmethod
    def _open_picoharp(ps, preferred):
        """pyscan's PicoHarp300 does not raise when the device can't be opened; it
        prints e.g. "0 no device" and leaves out ``.dev``, so later calls fail with
        "no attribute dev". Check for that and try the other number (0 <-> 1), then
        the rest."""
        order = [preferred, 1 - preferred if preferred in (0, 1) else 0]
        for idx in order + [i for i in range(8) if i not in order]:
            ph = ps.PicoHarp300(idx)
            if hasattr(ph, "dev"):
                return ph
        raise RuntimeError(
            "Could not open the PicoHarp 300 (tried device numbers 0-7).\n\n"
            "Usually another program still has it open: a Jupyter notebook that ran ps.PicoHarp300(...), "
            "the PicoHarp software, or another copy of this GUI (look for python.exe in Task Manager). "
            "Close it, or unplug and replug the PicoHarp USB cable, then click Retry.\n"
            "The PicoHarp's own error message is in the black console window.")

    # ---- direct Kinesis access (the thorlabs_kinesis module pyscan uses) ----
    # Status bits of a Thorlabs stepper channel (Kinesis SBC_GetStatusBits)
    MOVING_BITS = 0x00000010 | 0x00000020 | 0x00000040 | 0x00000080
    HOMING_BIT = 0x00000200
    HOMED_BIT = 0x00000400
    HOME_POLL_S = 0.2
    HOME_RESEND_S = 5.0        # re-send Home if an axis hasn't started moving by then
    HOME_ATTEMPTS = 3

    def _kinesis(self):
        """(bsm, serial) from pyscan's driver, or (None, None) if not available."""
        mod = sys.modules.get(type(self.stage).__module__)
        bsm = getattr(mod, "bsm", None)
        serial = getattr(self.stage, "serial", None)
        return (bsm, serial) if bsm is not None and serial is not None else (None, None)

    def _status_bits(self, channel):
        """Status bits of a channel, or None if they can't be read."""
        bsm, serial = self._kinesis()
        if bsm is None:
            return None
        try:
            bsm.SBC_RequestStatusBits(serial, ctypes.c_short(channel))
            time.sleep(0.02)
            bits = bsm.SBC_GetStatusBits(serial, ctypes.c_short(channel))
            return None if bits is None else int(bits)
        except Exception:
            return None

    def _fresh_axis(self, axis):
        """Current position (mm) of one stage axis, requested from the controller."""
        bsm, serial = self._kinesis()
        if bsm is not None:
            try:
                bsm.SBC_RequestPosition(serial, ctypes.c_short(axis + 1))
                time.sleep(0.02)
            except Exception:
                pass
        return float(self.stage.xyz[axis])

    def get_stage(self):
        """Fresh stage position (mm). pyscan reads the last *polled* value, which can be
        stale, so ask the controller for the current position first."""
        bsm, serial = self._kinesis()
        if bsm is not None:
            try:
                for ch in (1, 2, 3):
                    bsm.SBC_RequestPosition(serial, ctypes.c_short(ch))
                time.sleep(0.03)
            except Exception:
                pass
        return np.array(self.stage.xyz, float)

    def set_stage_axis(self, axis, value_mm):
        c = self.cfg
        return _wait_until_reached(lambda t: self._command_stage(axis, float(t[0])),
                                   lambda: [self._fresh_axis(axis)], [value_mm], c["stage_tolerance_um"] / 1000.0,
                                   c["stage_accept_um"] / 1000.0, c["stage_move_timeout_s"], poll=0.05)

    def _command_stage(self, axis, value_mm):
        """Start a move of one stage axis without waiting (we wait with our own
        timeout). pyscan's x/y/z setters wait for an *exact* position match with no
        timeout and can hang forever, and its xfast/yfast/zfast properties fail on
        some versions (missing _x), so call its channel move methods directly."""
        channel = axis + 1
        cache = "_" + "xyz"[axis]
        self._seed_stage_cache()
        try:
            if hasattr(self.stage, "move_channel_fast"):
                self.stage.move_channel_fast(channel, value_mm)
            elif hasattr(self.stage, "move_channel_to"):
                self.stage.move_channel_to(channel, value_mm, wait=False)
            else:
                setattr(self.stage, "xyz"[axis], value_mm)
        except AttributeError as e:
            # Some pyscan versions read their cached _x/_y/_z right after sending
            # the move. The move has already gone out, so this is harmless; the
            # wait loop checks the real position (and re-sends if needed).
            if cache not in str(e):
                raise
        self._set_stage_cache(cache, value_mm)

    def _set_stage_cache(self, name, value):
        try:
            setattr(self.stage, name, float(value))
        except Exception:
            pass

    def _seed_stage_cache(self):
        """Make sure pyscan's cached _x/_y/_z exist (its own x/y/z setters create them,
        but they are missing after start-up or homing on some versions)."""
        try:
            missing = [ax for ax in "xyz" if not hasattr(self.stage, "_" + ax)]
        except Exception:
            missing = list("xyz")
        if missing:
            pos = self.get_stage()
            for ax in missing:
                self._set_stage_cache("_" + ax, pos["xyz".index(ax)])

    def get_piezo(self):
        return np.array(self.piezo.xyz, float)

    def set_piezo(self, xyz_um):
        """Returns the piezo readback (or None if verify_piezo_moves is off)."""
        c = self.cfg
        values = [float(v) for v in xyz_um]
        if not c["verify_piezo_moves"]:
            self.piezo.xyz = values
            return None
        return _wait_until_reached(lambda t: setattr(self.piezo, "xyz", values), self.get_piezo,
                                   xyz_um, c["piezo_tolerance_um"], c["piezo_accept_um"],
                                   c["piezo_move_timeout_s"], still_time=0.1)

    def set_piezo_axis(self, axis, value_um):
        c = self.cfg
        name = "xyz"[axis]
        if not c["verify_piezo_moves"]:
            setattr(self.piezo, name, float(value_um))
            return None
        _wait_until_reached(lambda t: setattr(self.piezo, name, float(t[0])), lambda: [self.piezo.xyz[axis]],
                            [value_um], c["piezo_tolerance_um"], c["piezo_accept_um"],
                            c["piezo_move_timeout_s"], still_time=0.1)
        return self.get_piezo()

    def set_piezo_axis_raw(self, axis, value_um):
        """Set one piezo axis exactly like the notebook (pyscan's setter, no read-back loop)."""
        setattr(self.piezo, "xyz"[axis], float(value_um))

    def get_count_rates(self):
        return float(self.ph.get_count_rate_0()), float(self.ph.get_count_rate_1())

    def _send_home(self, channel):
        bsm, serial = self._kinesis()
        if bsm is None:                       # fall back to pyscan's helper
            self.stage.home_channel(channel)
            return
        ch = ctypes.c_short(channel)
        for fn in ("SBC_EnableChannel", "SBC_ClearMessageQueue"):
            try:
                getattr(bsm, fn)(serial, ch)
            except Exception:
                pass
        try:
            bsm.SBC_StartPolling(serial, ch, 100)
        except Exception:
            pass
        time.sleep(0.1)
        bsm.SBC_Home(serial, ch)

    def _home_axis(self, axis, progress=None):
        """Home one axis and wait until the controller says it is homed and stopped.
        Without status bits: until it reads 0 and stays there for 1 s. Re-sends the
        Home command if the axis never starts moving."""
        channel = axis + 1
        name = "xyz"[axis]
        timeout = self.cfg["stage_home_timeout_s"]
        deadline = getattr(self, "_home_deadline", None) or time.monotonic() + timeout
        self._send_home(channel)
        attempts, started = 1, False
        t_sent = time.monotonic()
        pos0 = last = float(self.get_stage()[axis])
        zero_since = None
        bits = None
        while True:
            time.sleep(self.HOME_POLL_S)
            now = time.monotonic()
            xyz = self.get_stage()
            if progress:
                progress(xyz)
            pos = float(xyz[axis])
            bits = self._status_bits(channel)
            if abs(pos - pos0) > 1e-3 or (bits is not None and bits & (self.HOMING_BIT | self.MOVING_BITS)):
                started = True
            at_zero = abs(pos) <= 1e-3
            moving = bits is not None and bits & (self.HOMING_BIT | self.MOVING_BITS)
            if bits is not None and bits & self.HOMED_BIT and not moving and (started or at_zero):
                return pos
            # fallback (no status bits, or a controller that never sets "homed"):
            # standing still at 0 for long enough
            zero_since = (zero_since or now) if at_zero and not moving and abs(pos - last) <= 1e-4 else None
            if zero_since is not None and now - zero_since >= (1.0 if bits is None else 2.0):
                return pos
            last = pos
            if not started and now - t_sent > self.HOME_RESEND_S:
                if attempts >= self.HOME_ATTEMPTS:
                    raise MoveError(f"the {name} axis did not start homing after {attempts} tries "
                                    f"(reads {pos:.4f} mm{self._bits_text(bits)})")
                self._send_home(channel)
                attempts += 1
                t_sent = now
            if now > deadline:
                raise MoveError(f"homing timed out after {timeout:.0f} s while homing {name} "
                                f"(reads {pos:.4f} mm{self._bits_text(bits)})")

    @staticmethod
    def _bits_text(bits):
        return "" if bits is None else f", controller status 0x{bits:08X}"

    def home_stage(self, progress=None):
        """Home z, then x, then y (z first: it goes away from the objective)."""
        if self._kinesis()[0] is None and not hasattr(self.stage, "home_channel"):
            raise MoveError("this pyscan version can't home the stage; home it in Kinesis")
        self._home_deadline = time.monotonic() + self.cfg["stage_home_timeout_s"]
        try:
            for axis in (2, 0, 1):
                pos = self._home_axis(axis, progress)
                self._set_stage_cache("_" + "xyz"[axis], pos)
        finally:
            self._home_deadline = None

    def close(self):
        for dev in (self.stage, self.piezo):
            if dev is None:
                continue
            try:
                dev.close()
            except Exception:
                pass
        self._close_picoharp()

    def _close_picoharp(self):
        """pyscan never closes the PicoHarp; do it so the next program can open it."""
        if self.ph is None or not hasattr(self.ph, "dev"):
            return
        lib = getattr(sys.modules.get(type(self.ph).__module__), "phlib", None)
        if lib is not None:
            try:
                lib.PH_CloseDevice(ctypes.c_int(self.ph.dev))
            except Exception:
                pass
        self.ph = None


class SimulatedHardware(HardwareBase):
    """Fake microscope for testing.

    The sample has four bright alignment markers on a rotated rectangle and an
    array of emitters starting ~55 um right/up of the bottom-left marker. The focal
    plane is tilted, emitters sit slightly below the surface, and stage moves land
    with a small random error (like a real stepper stage).
    """

    name = "simulator"

    def __init__(self, cfg, speed=1.0, seed=1, stage_error_um=1.0, rotation_deg=2.0, unhomed=False,
                 home_fails=False, gated=True, zeroed=False):
        self.cfg = cfg
        self.speed = speed
        # Like the PicoHarp, count rates are averages over fixed gates of a virtual
        # clock, so a reading can include counts from before / during a move.
        self.gated = gated
        self.gate = cfg["count_gate_s"]
        self.clock = 0.0
        self._history = []          # (time, absolute position) after each change
        self.rng = np.random.default_rng(seed)
        self.stage_error_um = stage_error_um
        self._lock = threading.Lock()
        self._stage = np.array([2.30, 1.10, 1.700])
        self.homed = not (unhomed or zeroed)
        self.home_fails = home_fails
        self.reads_zero = zeroed     # like a freshly powered controller: reads (0, 0, 0), not homed
        self._piezo = np.array(cfg["piezo_center_um"], float)
        self.laser_level = 1.0   # set by simulated lasers (fraction of full power)

        # Grid orientation matches the on-screen arrows so that jogging "right"
        # moves toward the right-hand markers, plus a small rotation.
        right = _screen_vector(cfg["screen_right"])
        up = _screen_vector(cfg["screen_up"])
        theta = np.deg2rad(rotation_deg)  # sample rotation relative to the stage axes
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.u_hat = rot @ right
        self.v_hat = rot @ up
        self.W, self.H = cfg["marker_width_um"], cfg["marker_height_um"]
        p = self._abs(self._stage, self._piezo)
        self.origin = p[:2] + 25.0 * right - 15.0 * up  # bottom-left marker, near start
        # focal plane: z_f = z0 + gu*u + gv*v (um), in absolute coordinates
        self.z0 = p[2] + 3.0
        self.gu, self.gv = 0.012, -0.008
        self.emitter_depth = 0.4   # emitters are this far below the surface focus
        # emitters: 6x6 array, pitch 3 um, starting at (55, 55) um in grid coords
        uu, vv = np.meshgrid(55 + 3.0 * np.arange(6), 55 + 3.0 * np.arange(6))
        self.emitters = np.c_[uu.ravel() + self.rng.normal(0, 0.2, uu.size),
                              vv.ravel() + self.rng.normal(0, 0.2, uu.size)]
        self.emitter_amp = self.rng.uniform(5e4, 3e5, len(self.emitters))
        self.emitter_amp[13] = 6e5  # one clearly brightest emitter

    def _abs(self, stage, piezo):
        return 1000.0 * stage + np.asarray(self.cfg["piezo_sign"]) * piezo

    def _sleep(self, t):
        if t > 0:
            self.clock += t
            if self.speed < 1e6:
                time.sleep(t / self.speed)

    def wait(self, seconds):
        self._sleep(seconds)

    def _record(self):
        """Remember the (new) position at the current time, for the gated counter."""
        self._history.append((self.clock, self._abs(self._stage, self._piezo)))
        cutoff = self.clock - 3 * self.gate
        while len(self._history) > 2 and self._history[1][0] <= cutoff:
            self._history.pop(0)

    def _gate_rate(self):
        """Mean signal over the last complete gate (piecewise-constant positions)."""
        if not self.gated or not self._history:
            return self._signal(self._abs(self._stage, self._piezo))
        end = np.floor(self.clock / self.gate + 1e-9) * self.gate
        start = end - self.gate
        total = 0.0
        for k, (t, p) in enumerate(self._history):
            t_next = self._history[k + 1][0] if k + 1 < len(self._history) else np.inf
            a, b = max(t, start), min(t_next, end)
            if k == 0:
                a = start                     # before the first record: that position
            if b > a:
                total += (b - a) * self._signal(p)
        return total / self.gate

    def grid_to_abs(self, u, v, dz=0.0):
        xy = self.origin + u * self.u_hat + v * self.v_hat
        return np.array([xy[0], xy[1], self.z0 + self.gu * u + self.gv * v + dz])

    def true_markers(self):
        return {"BL": self.grid_to_abs(0, 0), "BR": self.grid_to_abs(self.W, 0),
                "TL": self.grid_to_abs(0, self.H), "TR": self.grid_to_abs(self.W, self.H)}

    def true_brightest(self):
        u, v = self.emitters[np.argmax(self.emitter_amp)]
        return self.grid_to_abs(u, v, -self.emitter_depth)

    def _signal(self, p):
        d = p[:2] - self.origin
        u, v = d @ self.u_hat, d @ self.v_hat
        zf = self.z0 + self.gu * u + self.gv * v
        rate = 2.0e4  # background
        for mu, mv in ((0, 0), (self.W, 0), (0, self.H), (self.W, self.H)):
            if abs(u - mu) < 3 and abs(v - mv) < 3:   # 6 um bright marker squares
                rate += 4e5 * np.exp(-0.5 * ((p[2] - zf) / 2.0) ** 2)
        dz = p[2] - (zf - self.emitter_depth)
        zfac = np.exp(-0.5 * (dz / 0.7) ** 2)
        r2 = (self.emitters[:, 0] - u) ** 2 + (self.emitters[:, 1] - v) ** 2
        rate += zfac * np.sum(self.emitter_amp * np.exp(-0.5 * r2 / 0.25 ** 2))
        return rate

    def home_stage(self, progress=None):
        if self.home_fails:            # like a controller that ignores Home
            if progress:
                progress(self.get_stage())
            return
        self.homed = True
        start = self._stage.copy()
        for f in np.linspace(1, 0, 11):    # moves to 0 over ~0.5 s
            self._sleep(0.05)
            with self._lock:
                self._stage = start * f
                self._record()
            if progress:
                progress(self.get_stage())

    def get_stage(self):
        with self._lock:
            # an unhomed stepper reports a meaningless (here: negative) position
            if self.homed:
                return self._stage.copy()
            return np.zeros(3) if self.reads_zero else self._stage - 3.0

    def set_stage_axis(self, axis, value_mm):
        if not self.homed:
            raise MoveError("simulated stage is not homed")
        dist = abs(value_mm - self._stage[axis]) * 1000
        self._sleep(0.2 + dist / 1000.0)  # ~1 mm/s
        err = self.rng.normal(0, self.stage_error_um) / 1000.0 if dist > 0 else 0.0
        with self._lock:
            self._stage[axis] = value_mm + err
            self._record()

    def get_piezo(self):
        with self._lock:
            return self._piezo.copy()

    def set_piezo(self, xyz_um):
        self._sleep(0.01)                    # motion time: still at the old position
        with self._lock:
            self._piezo = np.asarray(xyz_um, float).copy()
            self._record()
            return self._piezo.copy()

    def set_piezo_axis(self, axis, value_um):
        self._sleep(0.01)
        with self._lock:
            self._piezo[axis] = float(value_um)
            self._record()
            return self._piezo.copy()

    def set_piezo_axis_raw(self, axis, value_um):
        self.set_piezo_axis(axis, value_um)

    def get_count_rates(self):
        with self._lock:
            sig = self._gate_rate()
        rate = 300.0 + self.laser_level * sig   # 300 cps dark counts
        c0 = self.rng.poisson(rate * 0.55 * 0.1) / 0.1
        c1 = self.rng.poisson(rate * 0.45 * 0.1) / 0.1
        return float(c0), float(c1)


def _screen_vector(spec):
    axis, sign = spec
    v = np.zeros(2)
    v[AXES[axis]] = sign
    return v


class Positioner:
    """Moves to absolute positions (um) using stage + piezo together.

    abs = 1000 * stage_mm + piezo_sign * piezo_um
    Small moves only use the piezo. When a target falls outside the usable piezo
    range, the stage is moved (on that axis only) and the piezo makes up the
    difference using the stage *readback*, so stage inaccuracy does not matter.
    """

    def __init__(self, hw, cfg):
        self.hw = hw
        self.cfg = cfg
        self.sign = np.asarray(cfg["piezo_sign"], float)
        self.stage_lim = np.asarray(cfg["stage_limits_mm"], float)
        self.piezo_lim = np.asarray(cfg["piezo_limits_um"], float)
        self.center = np.asarray(cfg["piezo_center_um"], float)
        self.margin = float(cfg["piezo_margin_um"])
        self.stage = hw.get_stage()
        self.piezo = hw.get_piezo()

    # -- reading ---------------------------------------------------------------
    def refresh(self):
        self.stage = self.hw.get_stage()
        self.piezo = self.hw.get_piezo()
        return self.stage.copy(), self.piezo.copy()

    def to_abs(self, stage, piezo):
        return 1000.0 * np.asarray(stage, float) + self.sign * np.asarray(piezo, float)

    @property
    def abs(self):
        return self.to_abs(self.stage, self.piezo)

    def piezo_for(self, target, stage=None):
        stage = self.stage if stage is None else stage
        return (np.asarray(target, float) - 1000.0 * stage) / self.sign

    def _piezo_ok(self, p, margin):
        lo = self.piezo_lim[:, 0] + margin
        hi = self.piezo_lim[:, 1] - margin
        return (p >= lo) & (p <= hi)

    # -- low level moves with limit checks ------------------------------------
    # A stepper can land a few um past 0 when it returns to home; a lost home
    # position reads far outside the range (hundreds of um or more).
    HOMED_SLACK_MM = 0.01

    def stage_homed(self):
        """False if the stage reads outside its range (e.g. negative), which means
        the stepper controller lost its home position."""
        slack = self.HOMED_SLACK_MM
        return bool(np.all(self.stage >= self.stage_lim[:, 0] - slack) and
                    np.all(self.stage <= self.stage_lim[:, 1] + slack))

    def home_stage(self, progress=None):
        """Home, then prove it worked: several fresh readings must be inside the stage
        range, and every axis must make a small test move (10 um out and back).
        Any failure raises MoveError telling the user to restart the controller."""
        try:
            self.hw.home_stage(progress)
            self.hw.wait(self.cfg["stage_settle_s"])
            for _ in range(3):
                self.stage = self.hw.get_stage()
                if progress:
                    progress(self.stage.copy())
                if not self.stage_homed():
                    raise MoveError(f"After homing the stage still reads {np.round(self.stage, 4)} mm.")
                self.hw.wait(0.3)
            saved = self.cfg["stage_move_timeout_s"]
            self.cfg["stage_move_timeout_s"] = min(saved, 10.0)
            try:
                lo, hi = self.stage_lim[:, 0], self.stage_lim[:, 1]
                home = np.clip(self.stage, lo, hi)
                for k in range(3):
                    out = home.copy()
                    out[k] = min(home[k] + 0.01, hi[k])
                    for target in (out, home):
                        try:
                            self.move_stage(target)
                        except MoveError as e:
                            raise MoveError(f"After homing the {'xyz'[k]} axis would not move ({e}).") from e
                        if progress:
                            progress(self.stage.copy())
            finally:
                self.cfg["stage_move_timeout_s"] = saved
        except Exception as e:
            try:
                self.stage = self.hw.get_stage()
            except Exception:
                pass
            raise MoveError(f"{e}\n\nRestart the stepper controller (switch it off and on again), then relaunch the GUI.") from e
        return self.stage.copy()

    def move_stage(self, xyz_mm):
        xyz_mm = np.asarray(xyz_mm, float)
        if not self.stage_homed():
            raise MoveError(f"The stage reads {np.round(self.stage, 4)} mm, outside its range: it has lost its "
                            "home position. Click 'Home stage' on the Move tab.")
        cur = self.stage
        # Only axes that change are commanded (and limit-checked), one at a time. When
        # z moves away from the objective it goes first; toward it, it goes last.
        changed = [k for k in range(3) if abs(xyz_mm[k] - cur[k]) > 1e-9]
        if not changed:
            return
        for k in changed:
            if not self.stage_lim[k, 0] <= xyz_mm[k] <= self.stage_lim[k, 1]:
                raise MoveError(f"stage target {np.round(xyz_mm, 4)} mm is outside limits")
        z_away = (xyz_mm[2] < cur[2]) == bool(self.cfg["z_up_is_toward_objective"])
        if 2 in changed and z_away:
            changed = [2] + [k for k in changed if k != 2]
        for k in changed:
            self.hw.set_stage_axis(k, xyz_mm[k])
        self.hw.wait(self.cfg["stage_settle_s"])
        self.stage = self.hw.get_stage()

    def move_piezo(self, xyz_um):
        xyz_um = np.asarray(xyz_um, float)
        if not np.all(self._piezo_ok(xyz_um, 0.0)):
            raise MoveError(f"piezo target {np.round(xyz_um, 3)} um is outside limits")
        actual = self.hw.set_piezo(xyz_um)
        self.piezo = xyz_um.copy() if actual is None else np.asarray(actual, float)
        if self.cfg["piezo_settle_s"] > 0:
            self.hw.wait(self.cfg["piezo_settle_s"])

    # -- combined moves --------------------------------------------------------
    def plan_stage(self, target, direction=None):
        """Stage position (mm) needed to reach ``target`` (abs um), or None if the
        piezo alone can do it. ``direction`` (per-axis sign of the motion) places
        the piezo near the start of its range so a scan line runs longer before
        the next stage move."""
        target = np.asarray(target, float)
        p = self.piezo_for(target)
        ok = self._piezo_ok(p, self.margin)
        if np.all(ok):
            return None
        stage = self.stage.copy()
        lo = self.piezo_lim[:, 0] + self.margin + 0.5
        hi = self.piezo_lim[:, 1] - self.margin - 0.5
        for k in np.where(~ok)[0]:
            want = self.center[k]
            if direction is not None and k < 2 and direction[k] != 0:
                # moving toward +abs: the piezo should start at its low-abs end
                want = lo[k] if direction[k] * self.sign[k] > 0 else hi[k]
            stage[k] = (target[k] - self.sign[k] * want) / 1000.0
        return stage

    def check_reachable(self, target):
        """Raise MoveError if ``target`` (abs um) cannot be reached within stage limits."""
        target = np.asarray(target, float)
        stage = (target - self.sign * self.center) / 1000.0
        if np.any(stage < self.stage_lim[:, 0]) or np.any(stage > self.stage_lim[:, 1]):
            raise MoveError(f"position {np.round(target, 2)} um is outside the stage range")

    def piezo_reaches(self, target):
        """True if ``target`` (abs um) is inside the piezo range with the stage where it is."""
        return bool(np.all(self._piezo_ok(self.piezo_for(target), 0.0)))

    def move_piezo_first(self, target, direction=None):
        """Reach ``target`` with the piezo alone if possible (stage stays put),
        otherwise with stage + piezo."""
        target = np.asarray(target, float)
        if self.piezo_reaches(target):
            self.move_piezo(self.piezo_for(target))
            return self.abs
        return self.move_abs(target, direction)

    def move_abs(self, target, direction=None):
        target = np.asarray(target, float)
        stage = self.plan_stage(target, direction)
        if stage is not None:
            self.move_stage(stage)
        p = self.piezo_for(target)
        if not np.all(self._piezo_ok(p, 0.0)):
            # stage landed far from where we asked; recenter once more
            self.move_stage((target - self.sign * self.center) / 1000.0)
            p = self.piezo_for(target)
            if not np.all(self._piezo_ok(p, 0.0)):
                raise MoveError(f"could not reach {np.round(target, 3)} (stage error too large)")
        self.move_piezo(p)
        return self.abs

    def jog_stage(self, axis, delta_abs_um):
        s = self.stage.copy()
        s[axis] += delta_abs_um / 1000.0
        self.move_stage(s)

    def move_piezo_axes(self, xyz_um):
        """Move the piezo to ``xyz_um``, commanding only the axes that change."""
        xyz_um = np.asarray(xyz_um, float)
        if not np.all(self._piezo_ok(xyz_um, 0.0)):
            raise MoveError(f"piezo target {np.round(xyz_um, 3)} um is outside limits")
        for k in range(3):
            if abs(xyz_um[k] - self.piezo[k]) > 1e-9:
                actual = self.hw.set_piezo_axis(k, xyz_um[k])
                p = self.piezo.copy()
                p[k] = xyz_um[k]
                self.piezo = p if actual is None else np.asarray(actual, float)

    def jog_piezo(self, axis, delta_abs_um):
        """Move one piezo axis so the *absolute* position changes by ``delta_abs_um``."""
        p = self.piezo.copy()
        p[axis] += delta_abs_um / self.sign[axis]
        if not np.all(self._piezo_ok(p, 0.0)):
            raise MoveError(f"piezo target {np.round(p, 3)} um is outside limits")
        actual = self.hw.set_piezo_axis(axis, p[axis])
        self.piezo = p if actual is None else np.asarray(actual, float)

    def recenter_piezo(self, attempts=3):
        """Bring the piezo as close to mid-range as the stage allows, keeping the
        absolute position. Only axes more than ``recenter_tolerance_um`` off-center
        are moved (the stage cannot land more precisely than that), so pressing it
        again once centered does nothing. Returns the number of stage moves made."""
        target = self.abs
        tol = self.cfg["recenter_tolerance_um"]
        moves = 0
        for _ in range(attempts):
            off = np.abs(self.piezo - self.center) > tol
            if not np.any(off):
                break
            stage = self.stage.copy()
            stage[off] = ((target - self.sign * self.center) / 1000.0)[off]
            self.move_stage(stage)
            self.move_piezo(self.piezo_for(target))
            moves += 1
        return moves

    def piezo_fraction(self):
        """Fraction (0-1) of the piezo range in use on each axis."""
        lo, hi = self.piezo_lim[:, 0], self.piezo_lim[:, 1]
        return np.clip((self.piezo - lo) / (hi - lo), 0, 1)


def counter_time(cfg, dwell):
    """Seconds one reading takes: one gate to settle plus the averaged gates."""
    gate = cfg["count_gate_s"]
    return gate * (1 + max(1, int(round(dwell / gate))))


def make_counter(hw, cfg):
    """Return read(dwell) -> (counts, c0, c1).

    The PicoHarp reports count rates measured over fixed gates (100 ms). A reading
    taken less than one full gate after a move includes counts from before / during
    the move (the old notebook and earlier versions of this program did this: about
    half of each pixel came from the previous position). So: wait one gate after the
    move, then average max(1, dwell / gate) consecutive gates, all measured after the
    move. A longer dwell therefore really averages longer."""
    channel = cfg["counts_channel"]
    gate = cfg["count_gate_s"]

    def read(dwell):
        hw.wait(gate * 1.02)                  # let the gate running during the move finish
        n = max(1, int(round(dwell / gate)))
        total = np.zeros(2)
        for _ in range(n):
            hw.wait(gate * 1.02)              # a complete gate measured after the move
            total += hw.get_count_rates()
        c0, c1 = total / n
        c = {"sum": c0 + c1, "ch0": c0, "ch1": c1}[channel]
        return float(c), float(c0), float(c1)

    return read
