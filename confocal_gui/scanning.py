"""Raster scans in the marker-aligned grid frame and z sweeps.

All scans run on the hardware worker thread. They check ``stop_event`` between
points and report every point through ``on_point`` so the GUI can update live.
"""

import time
from dataclasses import dataclass, field

import numpy as np


def n_steps(length, step):
    """Number of points every ``step`` that fit in ``length`` (never more than the range)."""
    if step <= 0 or length < 0:
        raise ValueError("step must be > 0 and range >= 0")
    return int(np.floor(length / step + 1e-9)) + 1


def axis_values(start, length, step):
    """Points from ``start`` every ``step``, staying within ``start + length``."""
    return start + step * np.arange(n_steps(length, step))


def centered_values(center, length, step):
    """Points every ``step`` centred on ``center``, spanning at most ``length``."""
    n = n_steps(length, step)
    return center - step * (n - 1) / 2.0 + step * np.arange(n)


def fit_window_in_piezo(pos, frame, u_vals, v_vals, z_offset):
    """Shift an XY scan window (in its own frame) so the piezo can reach all of it with
    the stage where it is: as little as possible, centred on the piezo range if it is
    too big to fit. Returns the shifted (u, v)."""
    u, v = np.asarray(u_vals, float), np.asarray(v_vals, float)
    lo, hi = pos.piezo_lim[:2, 0], pos.piezo_lim[:2, 1]
    for _ in range(2):
        p = np.array([pos.piezo_for(a)[:2] for a in raster_corners(frame, u, v, z_offset)])
        pmin, pmax = p.min(axis=0), p.max(axis=0)
        need = np.zeros(2)
        for k in range(2):
            if pmax[k] - pmin[k] > hi[k] - lo[k]:
                need[k] = 0.5 * (lo[k] + hi[k]) - 0.5 * (pmin[k] + pmax[k])
            elif pmin[k] < lo[k]:
                need[k] = lo[k] - pmin[k]
            elif pmax[k] > hi[k]:
                need[k] = hi[k] - pmax[k]
        if np.all(np.abs(need) < 1e-9):
            break
        # piezo x/y change per unit step in u and in v
        p0 = pos.piezo_for(frame.to_abs(u[0], v[0], z_offset))[:2]
        ju = pos.piezo_for(frame.to_abs(u[0] + 1, v[0], z_offset))[:2] - p0
        jv = pos.piezo_for(frame.to_abs(u[0], v[0] + 1, z_offset))[:2] - p0
        du, dv = np.linalg.solve(np.column_stack([ju, jv]), need)
        u, v = u + du, v + dv
    return u, v


def directed_values(start, length, step, direction):
    """Points from ``start`` every ``step`` toward +``direction`` (+1 / -1), within ``length``."""
    return start + direction * step * np.arange(n_steps(length, step))


def z_in_piezo(pos, z_vals):
    """True if the piezo z alone (stage where it is) covers all of ``z_vals``."""
    lo, hi = pos.piezo_lim[2]
    pz = (np.asarray(z_vals, float) - 1000.0 * pos.stage[2]) / pos.sign[2]
    return bool(pz.min() >= lo - 1e-9 and pz.max() <= hi + 1e-9)


def fit_z_in_piezo(pos, z_vals):
    """Shift absolute z values so the piezo z covers them with the stage where it is."""
    z = np.asarray(z_vals, float)
    lo, hi = pos.piezo_lim[2]
    sign = pos.sign[2]
    pz = (z - 1000.0 * pos.stage[2]) / sign
    if pz.max() - pz.min() > hi - lo:
        shift = 0.5 * (lo + hi) - 0.5 * (pz.min() + pz.max())
    elif pz.min() < lo:
        shift = lo - pz.min()
    elif pz.max() > hi:
        shift = hi - pz.max()
    else:
        return z
    return z + shift * sign


def estimate_time(n_points, dwell_s, overhead_s):
    """(dwell-only seconds, estimated seconds including motion overhead)."""
    return n_points * dwell_s, n_points * (dwell_s + overhead_s)


def format_duration(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


@dataclass
class RasterResult:
    kind: str
    u: np.ndarray
    v: np.ndarray
    z_offset: float
    dwell_s: float
    counts: np.ndarray = None
    counts0: np.ndarray = None
    counts1: np.ndarray = None
    abs_xyz: np.ndarray = None
    stage_mm: np.ndarray = None
    piezo_um: np.ndarray = None
    order: list = field(default_factory=list)   # (j, i) in acquisition order
    frame_kind: str = "marker"                   # "marker" (marker edges) or "piezo" (piezo x/y)
    tilt: bool = False                           # z followed the marker-plane tilt
    method: str = ""                             # "notebook" (the old notebook's piezo scan) or "raster"
    mode: str = ""                               # "piezo" (stage fixed) or "stage" (stage moves too)
    clamped: int = 0                             # points held at the piezo limit (stepper kept fixed)
    start_stage: np.ndarray = None               # stage (mm) and piezo (um) when the scan was started,
    start_piezo: np.ndarray = None               # i.e. the scan centre: go back here to repeat it
    scan_stage: np.ndarray = None                # stage during the scan (after any single pre-move)
    started: float = 0.0
    finished: float = 0.0
    completed: bool = False

    def __post_init__(self):
        shape = (len(self.v), len(self.u))
        self.counts = np.full(shape, np.nan)
        self.counts0 = np.full(shape, np.nan)
        self.counts1 = np.full(shape, np.nan)
        self.abs_xyz = np.full(shape + (3,), np.nan)
        self.stage_mm = np.full(shape + (3,), np.nan)
        self.piezo_um = np.full(shape + (3,), np.nan)

    @property
    def n_points(self):
        return self.counts.size

    def best(self):
        """(j, i, u, v, counts) of the highest-count pixel so far, or None."""
        if np.all(np.isnan(self.counts)):
            return None
        j, i = np.unravel_index(np.nanargmax(self.counts), self.counts.shape)
        return int(j), int(i), float(self.u[i]), float(self.v[j]), float(self.counts[j, i])


@dataclass
class ZSweepResult:
    z_abs: np.ndarray
    xy_abs: np.ndarray
    dwell_s: float
    counts: np.ndarray = None
    counts0: np.ndarray = None
    counts1: np.ndarray = None
    stage_mm: np.ndarray = None
    piezo_um: np.ndarray = None
    started: float = 0.0
    finished: float = 0.0
    completed: bool = False
    mode: str = ""
    clamped: int = 0
    start_stage: np.ndarray = None
    start_piezo: np.ndarray = None
    scan_stage: np.ndarray = None

    def __post_init__(self):
        n = len(self.z_abs)
        self.counts = np.full(n, np.nan)
        self.counts0 = np.full(n, np.nan)
        self.counts1 = np.full(n, np.nan)
        self.stage_mm = np.full((n, 3), np.nan)
        self.piezo_um = np.full((n, 3), np.nan)

    def best(self):
        if np.all(np.isnan(self.counts)):
            return None
        k = int(np.nanargmax(self.counts))
        return k, float(self.z_abs[k]), float(self.counts[k])


def piezo_plan(pos, abs_points, force=False):
    """How a set of points can be reached:
    ("piezo", None)      every point is inside the piezo range now: stage stays put
    ("recenter", stage)  they fit the piezo range after one stage move to ``stage``
                         (with ``force``: always centre on the piezo range, even if the
                         points overhang it slightly; those get held at the limit)
    ("stage", None)      larger than the piezo range: the stage must move during the scan
    """
    pts = np.atleast_2d(np.asarray(abs_points, float))
    lo, hi = pos.piezo_lim[:, 0], pos.piezo_lim[:, 1]
    p = np.array([pos.piezo_for(a) for a in pts])
    pmin, pmax = p.min(axis=0), p.max(axis=0)
    fits_now = (pmin >= lo) & (pmax <= hi)
    if np.all(fits_now):
        return "piezo", None
    span = pts.max(axis=0) - pts.min(axis=0)
    if force or np.all(span <= (hi - lo)):
        stage = pos.stage.copy()
        mid = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
        need = ~fits_now
        stage[need] = (mid[need] - pos.sign[need] * pos.center[need]) / 1000.0
        return "recenter", stage
    return "stage", None


def piezo_only_move(pos, target):
    """Move with the piezo only (the stage never moves). A target a hair outside the
    piezo range is held at the limit; returns True if that happened."""
    p = pos.piezo_for(target)
    lo, hi = pos.piezo_lim[:, 0], pos.piezo_lim[:, 1]
    clipped = np.clip(p, lo, hi)
    pos.move_piezo(clipped)
    return bool(np.any(np.abs(clipped - p) > 1e-6))


def raster_corners(frame, u_vals, v_vals, z_offset):
    return np.array([frame.to_abs(u, v, z_offset) for u in (u_vals[0], u_vals[-1])
                     for v in (v_vals[0], v_vals[-1])])


def check_raster_reachable(pos, frame, u_vals, v_vals, z_offset):
    for u in (u_vals[0], u_vals[-1]):
        for v in (v_vals[0], v_vals[-1]):
            pos.check_reachable(frame.to_abs(u, v, z_offset))


def run_raster(kind, pos, read, frame, u_vals, v_vals, z_offset, dwell, stop_event,
               on_point=None, piezo_only=None):
    """Snake raster in the grid frame: row 0 goes right (+u), then step up (+v)
    and come back, etc. z follows the marker focal plane + ``z_offset``.
    ``piezo_only``: True keeps the stage fixed during the scan (it may move once
    before); False lets it move; None decides from whether the area fits."""
    res = RasterResult(kind, np.asarray(u_vals, float), np.asarray(v_vals, float),
                       float(z_offset), float(dwell), frame_kind=getattr(frame, "kind", "marker"),
                       tilt=bool(getattr(frame, "tilt", False)), method="raster")
    res.started = time.time()
    res.start_stage, res.start_piezo = pos.stage.copy(), pos.piezo.copy()
    check_raster_reachable(pos, frame, res.u, res.v, z_offset)
    corners = raster_corners(frame, res.u, res.v, z_offset)
    if piezo_only is None:
        piezo_only = piezo_plan(pos, corners)[0] == "piezo"
    res.mode = "piezo" if piezo_only else "stage"      # piezo only: the stage never moves
    res.scan_stage = pos.stage.copy()
    nx = len(res.u)
    for j, v in enumerate(res.v):
        row_dir = 1 if j % 2 == 0 else -1
        direction = frame.xy_direction(row_dir, 0)
        cols = range(nx) if row_dir > 0 else range(nx - 1, -1, -1)
        for i in cols:
            if stop_event.is_set():
                res.finished = time.time()
                return res
            target = frame.to_abs(res.u[i], v, z_offset)
            if res.mode == "stage":
                pos.move_abs(target, direction)
            else:
                res.clamped += piezo_only_move(pos, target)
            c, c0, c1 = read(dwell)
            res.counts[j, i], res.counts0[j, i], res.counts1[j, i] = c, c0, c1
            res.abs_xyz[j, i] = pos.abs
            res.stage_mm[j, i] = pos.stage
            res.piezo_um[j, i] = pos.piezo
            res.order.append((j, i))
            if on_point:
                on_point(res, j, i)
    res.completed = True
    res.finished = time.time()
    return res


def run_xy_notebook(pos, hw, frame, u_vals, v_vals, z_offset, dt, stop_event, on_point=None,
                    channel="sum", start_settle_s=3.0):
    """The old notebook's piezo XY scan (pyscan Sweep with loop0 = piezo x, loop1 = piezo
    y, dt = ``dt`` on both), so it gives the same image:

    * hardware piezo x is the fast axis; every line starts again from its first x
      (no back-and-forth),
    * each point: set only the piezo axes that change (pyscan's setter, no read-back
      loop), wait ``dt`` (``2 dt`` at the start of a line: y and x both change), then
      read the PicoHarp count rates once,
    * the stage never moves; z stays where it is, or follows the marker-plane tilt
      when ``frame.tilt`` (only then is the piezo z set during the scan).
    It first moves to the first point and waits ``start_settle_s`` (the notebook's sleep(3)).
    """
    res = RasterResult("xy", np.asarray(u_vals, float), np.asarray(v_vals, float), float(z_offset), float(dt),
                       frame_kind=frame.kind, tilt=bool(getattr(frame, "tilt", False)), method="notebook")
    res.started = time.time()
    res.start_stage, res.start_piezo = pos.stage.copy(), pos.piezo.copy()
    res.mode = "piezo"
    res.scan_stage = pos.stage.copy()
    lo, hi = pos.piezo_lim[:, 0], pos.piezo_lim[:, 1]
    # image index (j over v, i over u); the fast axis is whichever of u / v is hardware x
    fast_u = frame.x_axis == 0
    outer = range(len(res.v)) if fast_u else range(len(res.u))
    inner = range(len(res.u)) if fast_u else range(len(res.v))
    slow_hw, fast_hw = 1, 0

    def target(j, i):
        p = pos.piezo_for(frame.to_abs(res.u[i], res.v[j], z_offset))
        c = np.clip(p, lo, hi)
        return c, bool(np.any(np.abs(c - p) > 1e-6))

    first = (0, 0)
    p0, _ = target(*first)
    pos.move_piezo(p0)
    hw.wait(start_settle_s)
    cur = pos.piezo.copy()
    for a in outer:
        for b_n, b in enumerate(inner):
            if stop_event.is_set():
                res.finished = time.time()
                return res
            j, i = (a, b) if fast_u else (b, a)
            p, clipped = target(j, i)
            res.clamped += clipped
            axes = [slow_hw, fast_hw] + ([2] if res.tilt else [])
            for k in axes:
                if abs(p[k] - cur[k]) > 1e-6:
                    hw.set_piezo_axis_raw(k, p[k])
                    cur[k] = p[k]
            pos.piezo = cur.copy()
            hw.wait(dt * (2 if b_n == 0 and (a, b) != (0, 0) else 1))
            c0, c1 = hw.get_count_rates()
            c = {"sum": c0 + c1, "ch0": c0, "ch1": c1}[channel]
            res.counts[j, i], res.counts0[j, i], res.counts1[j, i] = c, c0, c1
            res.abs_xyz[j, i] = pos.abs
            res.stage_mm[j, i] = pos.stage
            res.piezo_um[j, i] = pos.piezo
            res.order.append((j, i))
            if on_point:
                on_point(res, j, i)
    res.completed = True
    res.finished = time.time()
    return res


def run_z_sweep(pos, read, xy_abs, z_vals, dwell, stop_event, on_point=None, piezo_only=None):
    """Hold x, y (absolute) and step z through ``z_vals`` (absolute um)."""
    res = ZSweepResult(np.asarray(z_vals, float), np.asarray(xy_abs, float)[:2], float(dwell))
    res.started = time.time()
    res.start_stage, res.start_piezo = pos.stage.copy(), pos.piezo.copy()
    first = np.r_[res.xy_abs, res.z_abs[0]]
    last = np.r_[res.xy_abs, res.z_abs[-1]]
    pos.check_reachable(first)
    pos.check_reachable(last)
    if piezo_only is None:
        piezo_only = piezo_plan(pos, [first, last])[0] == "piezo"
    res.mode = "piezo" if piezo_only else "stage"
    res.scan_stage = pos.stage.copy()
    for k, z in enumerate(res.z_abs):
        if stop_event.is_set():
            res.finished = time.time()
            return res
        target = np.r_[res.xy_abs, z]
        if res.mode == "stage":
            pos.move_abs(target)
        else:
            res.clamped += piezo_only_move(pos, target)
        c, c0, c1 = read(dwell)
        res.counts[k], res.counts0[k], res.counts1[k] = c, c0, c1
        res.stage_mm[k] = pos.stage
        res.piezo_um[k] = pos.piezo
        if on_point:
            on_point(res, k)
    res.completed = True
    res.finished = time.time()
    return res


def average_counts(read, total_s, dwell):
    """Mean and std of the count rate over ``total_s`` seconds."""
    n = max(1, int(round(total_s / dwell)))
    vals = np.array([read(dwell)[0] for _ in range(n)])
    return float(vals.mean()), float(vals.std())

def sweep_values(start, stop, step):
    """Percent values from start to stop (either direction) every step, never past stop."""
    n = n_steps(abs(stop - start), step)
    return start + np.sign(stop - start or 1) * step * np.arange(n)


@dataclass
class PowerSweepResult:
    pct: np.ndarray                 # laser diode current, % of max
    time_s: float
    counts: np.ndarray = None
    counts0: np.ndarray = None
    counts1: np.ndarray = None
    stage_mm: np.ndarray = None     # where it was measured
    piezo_um: np.ndarray = None
    started: float = 0.0
    finished: float = 0.0
    completed: bool = False

    def __post_init__(self):
        n = len(self.pct)
        self.counts = np.full(n, np.nan)
        self.counts0 = np.full(n, np.nan)
        self.counts1 = np.full(n, np.nan)

    def best(self):
        if np.all(np.isnan(self.counts)):
            return None
        k = int(np.nanargmax(self.counts))
        return k, float(self.pct[k]), float(self.counts[k])


def run_power_sweep(laser, read, pct_vals, time_s, settle_s, wait, stop_event, pos=None, on_point=None):
    """Set each laser current (% of max), wait ``settle_s``, count for ``time_s``."""
    pct_vals = np.asarray(pct_vals, float)
    res = PowerSweepResult(pct_vals, float(time_s))
    res.started = time.time()
    if pos is not None:
        res.stage_mm, res.piezo_um = pos.stage.copy(), pos.piezo.copy()
    for k, pct in enumerate(pct_vals):
        if stop_event.is_set():
            res.finished = time.time()
            return res
        laser.set_power(float(pct))
        wait(settle_s)
        c, c0, c1 = read(time_s)
        res.counts[k], res.counts0[k], res.counts1[k] = c, c0, c1
        if on_point:
            on_point(res, k)
    res.completed = True
    res.finished = time.time()
    return res
