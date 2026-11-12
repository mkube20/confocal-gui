"""Configuration for the confocal GUI.

All values here can be overridden by a JSON file (default: confocal_config.json next
to run_confocal_gui.py). Only the keys you want to change need to be in that file.

Coordinate conventions
----------------------
* Stage (Thorlabs BSC203) positions are in **mm**, range 0-4 mm.
* Piezo (Thorlabs BSC303) positions are in **um**, range ~0.1-19.9 um.
* Internally everything is converted to one "absolute" frame in um:

      abs = 1000 * stage_mm + piezo_sign * piezo_um

  so a target position can be reached by any combination of stage + piezo.
  ``piezo_sign`` MUST match your hardware (see README: "Checking piezo_sign").
"""

import copy
import json
import os

DEFAULTS = {
    # ---- hardware connection -------------------------------------------------
    # Replace these placeholders with your own values (or override them in confocal_config.json).
    "pyscan_path": r"C:\path\to\pyscan",
    "stage_serial": "YOUR_BSC203_SERIAL",
    "piezo_serial": "YOUR_BSC303_SERIAL",
    "picoharp_index": 0,             # first device number to try; 0 <-> 1 is tried automatically
    "picoharp_index_file": "",       # "" = .picoharp_index.txt next to run_confocal_gui.py (remembers what worked)
    # Laser controlled from the Laser tab (see lasers.py). Set in % of the diode
    # controller's max current (100 % = max_current_A), not % of optical power.
    "lasers": [
        {"name": "637 nm", "type": "visa_current",
         "resource": "USB0::0x1313::0x804F::YOUR_LASER_SERIAL::0::INSTR", "max_current_A": 0.14},
    ],

    # ---- limits --------------------------------------------------------------
    # Stage soft limits [min, max] in mm for x, y, z. Above ~4 mm in z the sample
    # hits the objective, so z max is kept slightly below that.
    "stage_limits_mm": [[0.0, 4.0], [0.0, 4.0], [0.0, 3.95]],
    "piezo_limits_um": [[0.1, 19.9], [0.1, 19.9], [0.1, 19.9]],
    "piezo_center_um": [10.0, 10.0, 10.0],
    # Piezo targets closer than this to a limit trigger a stage move instead.
    "piezo_margin_um": 0.3,
    # "Recenter piezo" leaves an axis alone if it is already this close to center.
    "recenter_tolerance_um": 2.0,
    # True if increasing stage z moves the sample toward the objective. Used to
    # order moves so z retracts first when moving away.
    "z_up_is_toward_objective": True,

    # ---- coordinate mapping --------------------------------------------------
    # abs = 1000*stage + piezo_sign*piezo (per axis x, y, z)
    "piezo_sign": [1, 1, 1],
    # Which stage/piezo axis (and sign) the arrow buttons move. The buttons are
    # always labelled right = +x, up = +y; this chooses the hardware axis behind them.
    "screen_right": ["y", 1],
    "screen_up": ["x", 1],

    # ---- motion --------------------------------------------------------------
    "stage_tolerance_um": 0.5,       # readback must be this close to target
    "piezo_tolerance_um": 0.02,
    # If a device stops short of the target but within this distance, accept it
    # (the actual readback is used/recorded; for the stage the piezo compensates).
    "stage_accept_um": 5.0,
    "piezo_accept_um": 0.05,
    "stage_move_timeout_s": 60.0,
    "stage_home_timeout_s": 60.0,    # for the whole homing (all three axes)
    # Last good stage position, offered after homing ("" = next to run_confocal_gui.py).
    "last_stage_file": "",
    # Last completed alignment (marker positions + layout), "" = .alignment.json next to run_confocal_gui.py
    "alignment_file": "",
    "piezo_move_timeout_s": 3.0,
    "stage_settle_s": 0.2,           # wait after a stage move
    "piezo_settle_s": 0.0,           # extra wait after a piezo move (on top of dwell)
    "verify_piezo_moves": True,      # read back the piezo after every move

    # ---- counting ------------------------------------------------------------
    "dwell_s": 0.1,                  # counting time per point (averaged over whole gates)
    # PicoHarp count-rate gate. Each reading first waits one gate after a move, so
    # every averaged gate is measured entirely at the new position.
    "count_gate_s": 0.1,
    "counts_channel": "sum",         # "sum", "ch0" or "ch1"
    # Rough extra time per point for motion + readback, used in time estimates.
    # Starting guess (measured on the lab setup: 20x20 um, 2 um steps took ~1 min).
    # After each finished scan the measured value is saved in .scan_timing.json and
    # used for the next estimate of that scan type.
    "overhead_per_point_s": 0.4,
    "xy_overhead_per_point_s": 0.05,  # XY scans (notebook method): time per point on top of dwell
    "xy_start_settle_s": 3.0,        # wait at the first point before an XY scan (the notebook's sleep(3))
    "scan_timing_file": "",          # "" = .scan_timing.json next to run_confocal_gui.py
    "final_count_average_s": 1.0,    # averaging time for the final max count rate

    # ---- alignment -----------------------------------------------------------
    # Marker rectangle size used by the simulator only (any rectangle works for
    # the real alignment, which uses the measured positions).
    "marker_width_um": 150.0,
    "marker_height_um": 170.0,
    # A marker counts as "accepted" if the user ends up within this distance of
    # where the system put them.
    "accept_tolerance_um": 0.05,

    # ---- scan defaults -------------------------------------------------------
    "scan_start_offset_um": [75.0, 85.0],   # right, up from the bottom-left marker
    # XY scan presets. The piezo covers 19.8 um, so scans up to ~18 x 18 um run on the
    # piezo alone (the stepper stays put); larger ones move the stepper during the scan.
    "coarse_range_um": [18.0, 18.0],
    "coarse_step_um": 2.0,
    "fine_range_um": [1.0, 1.0],
    "fine_step_um": 0.1,
    "z_range_um": 15.0,
    "z_step_um": 0.1,
    # laser current sweep defaults (Laser tab), in % of max current
    "sweep_start_pct": 30.0,
    "sweep_stop_pct": 100.0,
    "sweep_step_pct": 5.0,
    "sweep_time_s": 0.5,             # counting time per current setting
    "sweep_settle_s": 0.3,           # wait after changing the current
    "min_scan_step_um": 0.1,

    # ---- jog defaults (um) ---------------------------------------------------
    "stage_xy_step_um": 10.0,
    "stage_xy_min_step_um": 10.0,
    "stage_z_step_um": 1.0,
    "stage_z_min_step_um": 1.0,
    "piezo_xy_step_um": 0.1,
    "piezo_z_step_um": 0.1,
    "piezo_min_step_um": 0.1,

    # ---- GUI -----------------------------------------------------------------
    "default_save_folder": "",
    "poll_interval_ms": 250,
}

AXES = {"x": 0, "y": 1, "z": 2}


def default_config_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "confocal_config.json")


def load_config(path=None):
    """Return DEFAULTS updated with the contents of the JSON file at ``path``."""
    cfg = copy.deepcopy(DEFAULTS)
    path = path or default_config_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            print(f"Warning: unknown config keys ignored: {sorted(unknown)}")
        cfg.update({k: v for k, v in user.items() if k in DEFAULTS})
    return cfg
