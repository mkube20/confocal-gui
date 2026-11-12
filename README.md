# Confocal Microscope Control GUI

A PyQt desktop application for aligning, focusing, and scanning a confocal microscope used to find
and characterize single-photon emitters. It replaces a Jupyter-notebook workflow with one guided
window, cutting alignment, focus, and scan time from hours to minutes.

It drives a Thorlabs BSC203 stepper stage, a Thorlabs BSC303 piezo scanner, a Thorlabs laser current
driver, and a PicoHarp 300 photon counter with two single-photon detectors. A built-in microscope
simulator lets you develop and practice without any hardware.

Author: Michael J. P. Kuban

## Features

- **Hardware abstraction layer**: the same GUI runs on real instruments or a simulated microscope (`--sim`).
- **Four-marker alignment** with sample-tilt correction, remembered between sessions.
- **XY and Z scans** with live images, adjustable contrast, and one-click *Go to optimum*;
  scans larger than the piezo range are stitched automatically using the stepper stage.
- **Live count rate**, **laser control**, and **laser current sweeps**.
- **Photon-correlation (g2) analysis** (`g2_analysis.py`) and a reader for raw PicoHarp T2 timestamp
  streams (`ptu_timestamps.py`).

## Quick start

Try it without hardware:

```
pip install -r requirements.txt
python run_confocal_gui.py --sim
```

`--sim-speed 20` makes the simulator run 20x faster.

On Windows you can instead double-click `Start Confocal GUI.bat`, which finds a suitable Python
(Anaconda / Miniconda / PATH) and starts the GUI; add `--sim` to use the simulator.

## Using it with real hardware

1. Install the hardware packages (see `requirements.txt`), including the pyscan instrument-control package.
2. In `confocal_gui/config.py`, replace the placeholders with your own values:
   - `pyscan_path`: folder containing pyscan (also set `PYSCAN_PATH` in `start_confocal.py`)
   - `stage_serial`, `piezo_serial`: serial numbers of your Thorlabs controllers
   - `lasers[...]["resource"]`: VISA resource string of your laser driver

   Instead of editing the file, you can put only the keys you want to change in a
   `confocal_config.json` next to `run_confocal_gui.py`.
3. If the launcher picks the wrong Python, put the full path to `python.exe` as the only line of a
   `python_path.txt` file next to `Start Confocal GUI.bat`.
4. Before trusting it on a new microscope, check `piezo_sign`, the arrow directions, the z direction, and
   the laser's 100 % current. See "Check these on the microscope" in
   [`confocal_gui/README.md`](confocal_gui/README.md).

## Documentation

Detailed documentation of every tab, the saved file formats, photon counting, and the coordinate
system is in [`confocal_gui/README.md`](confocal_gui/README.md).

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
