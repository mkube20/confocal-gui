"""Laser control, taken from the laser functions in the original setup notebook.

Each laser takes a power in % (0-100). 0 % turns the output off.
* "visa_current": Thorlabs current drivers (532 nm ITC, 637 nm CLD) over pyvisa.
  Current = % / 100 * max_current_A (notebook: 532 -> 0.4 A, 637 -> 0.14 A at 100 %).
* "toptica_ibeam": Toptica iBeam over a COM port (pylablib), channel 1.
  Power = % / 100 * max_power_mW (the notebook passed 0-100 straight through as mW).
Devices are only connected the first time they are used.
"""


class Laser:
    def __init__(self, spec):
        self.spec = spec
        self.name = spec["name"]
        self.dev = None

    def ensure(self):
        if self.dev is None:
            self.dev = self._connect()

    def set_power(self, pct):
        if not 0 <= pct <= 100:
            raise ValueError("laser power must be 0-100 %")
        self.ensure()
        if pct == 0:
            self.off()
        else:
            self._set(pct)

    def off(self):
        """Turn the output off (connects first, so this also works if the laser
        was already on when the program started)."""
        self.ensure()
        self._off()

    def read_state(self):
        """(on, pct) as reported by the device, or None if it can't be read."""
        self.ensure()
        return self._state()

    def _state(self):
        return None

    def close(self):
        pass


class VisaCurrentLaser(Laser):
    def _connect(self):
        import pyvisa
        dev = pyvisa.ResourceManager().open_resource(self.spec["resource"])
        dev.query("*IDN?")
        return dev

    def _set(self, pct):
        amps = pct / 100.0 * self.spec["max_current_A"]
        self.dev.write(f"source:current:level:amplitude {amps:.5f}")
        self.dev.write(":OUTP1:STAT ON")

    def _off(self):
        self.dev.write(":OUTP1:STAT OFF")

    def _state(self):
        on = self.dev.query(":OUTP1:STAT?").strip().upper() in ("1", "ON")
        try:
            amps = float(self.dev.query("source:current:level:amplitude?"))
            pct = 100.0 * amps / self.spec["max_current_A"]
        except Exception:
            pct = None
        return on, pct

    def close(self):
        if self.dev is not None:
            self.dev.close()


class TopticaIBeamLaser(Laser):
    def _connect(self):
        from pylablib.devices import Toptica
        return Toptica.TopticaIBeam(self.spec["port"])

    def _set(self, pct):
        watts = pct / 100.0 * self.spec["max_power_mW"] / 1000.0
        self.dev.enable_channel(1, enabled=True)
        self.dev.enable_channel(2, enabled=False)
        self.dev.set_channel_power(1, watts)
        self.dev.enable()

    def _off(self):
        self.dev.set_channel_power(1, 0)
        self.dev.enable(enabled=False)

    def close(self):
        if self.dev is not None:
            self.dev.close()


class SimLaser(Laser):
    """Simulated laser: scales the simulator's count rate."""

    def __init__(self, spec, hw):
        super().__init__(spec)
        self.hw = hw

    def _connect(self):
        return object()

    def _set(self, pct):
        self.hw.laser_level = pct / 100.0

    def _off(self):
        self.hw.laser_level = 0.0

    def _state(self):
        level = self.hw.laser_level
        return level > 0, 100.0 * level


def make_lasers(cfg, hw=None, simulated=False):
    """Return {name: Laser} in the order given in cfg["lasers"]."""
    kinds = {"visa_current": VisaCurrentLaser, "toptica_ibeam": TopticaIBeamLaser}
    out = {}
    for spec in cfg["lasers"]:
        out[spec["name"]] = SimLaser(spec, hw) if simulated else kinds[spec["type"]](spec)
    return out