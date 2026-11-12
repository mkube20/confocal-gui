"""Launch the confocal alignment / scan GUI.

    python run_confocal_gui.py            # real hardware (pyscan)
    python run_confocal_gui.py --sim      # simulated microscope, no hardware needed
"""

import argparse
import os
import sys

from confocal_gui.config import load_config


def _add_conda_dll_path():
    """Conda puts Qt's DLLs in <env>/Library/bin, which is only on PATH when the env
    is activated. Add it so the GUI also starts when python.exe is run directly."""
    lib = os.path.join(sys.prefix, "Library", "bin")
    if os.name == "nt" and os.path.isdir(lib):
        os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(lib)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="use the simulated microscope")
    ap.add_argument("--sim-speed", type=float, default=1.0,
                    help="simulator time speed-up (e.g. 10 makes scans 10x faster)")
    ap.add_argument("--config", default=None, help="JSON file overriding confocal_gui/config.py defaults")
    args = ap.parse_args()
    cfg = load_config(args.config)

    _add_conda_dll_path()
    try:
        from confocal_gui.gui import MainWindow, QtWidgets
    except ImportError as e:
        if "Qt binding" not in str(e):
            raise
        print(f"No working Qt library found for this Python ({sys.executable}).")
        print("Install one, e.g.:  conda install pyqt   or   pip install PyQt5")
        return 1
    from confocal_gui.hardware import PyscanHardware, SimulatedHardware
    from confocal_gui.lasers import make_lasers

    app = QtWidgets.QApplication(sys.argv)
    if args.sim:
        hw = SimulatedHardware(cfg, speed=args.sim_speed)
    else:
        hw = None
        while hw is None:
            try:
                hw = PyscanHardware(cfg)
            except Exception as e:
                box = QtWidgets.QMessageBox()
                box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
                box.setWindowTitle("Hardware connection failed")
                box.setText(f"Could not connect to the instruments:\n\n{e}")
                retry = box.addButton("Retry", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
                sim = box.addButton("Use simulator", QtWidgets.QMessageBox.ButtonRole.ActionRole)
                box.addButton("Quit", QtWidgets.QMessageBox.ButtonRole.RejectRole)
                box.exec()
                if box.clickedButton() is sim:
                    hw = SimulatedHardware(cfg, speed=args.sim_speed)
                elif box.clickedButton() is not retry:
                    return 1
    win = MainWindow(hw, cfg, make_lasers(cfg, hw, simulated=hw.name == "simulator"))
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
