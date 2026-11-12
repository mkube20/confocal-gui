"""Find a Python on this computer that can run the confocal GUI, then start it.

Called by "Start Confocal GUI.bat". Only uses the standard library, so it runs on any
Python. It looks through the usual Anaconda / Miniconda locations (including their
environments) and Python on PATH, and prefers the one that can import pyscan
(i.e. the one the lab notebooks use). The choice is remembered in
.confocal_python.txt. To force a particular Python, put its full path to python.exe
in python_path.txt next to this file.

    python start_confocal.py            find Python and start the GUI
    python start_confocal.py --check    only report what was found
    python start_confocal.py --sim      start the GUI with the simulator
"""

import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".confocal_python.txt")
PINNED = os.path.join(HERE, "python_path.txt")
WHEELS = os.path.join(HERE, "offline_wheels")
PYSCAN_PATH = r"C:\path\to\pyscan"  # placeholder: set to the same folder as "pyscan_path" in confocal_gui/config.py

CHECK_BASE = "import numpy, matplotlib"
CHECK_PYSCAN = "import sys; sys.path.append(r'%s'); import pyscan" % PYSCAN_PATH
CHECK_QT = ("import matplotlib\n"
            "from matplotlib.backends.qt_compat import QtCore\n")


def candidates():
    seen, out = set(), []

    def add(exe):
        exe = os.path.normcase(os.path.abspath(exe))
        if exe not in seen and os.path.isfile(exe) and "windowsapps" not in exe:
            seen.add(exe)
            out.append(exe)

    home = os.path.expanduser("~")
    roots = []
    for base in (home, os.environ.get("LOCALAPPDATA", ""), os.environ.get("ProgramData", r"C:\ProgramData"), "C:\\"):
        for name in ("anaconda3", "Anaconda3", "miniconda3", "Miniconda3", "anaconda", "miniforge3", "mambaforge"):
            roots.append(os.path.join(base, name))
    for r in roots:
        add(os.path.join(r, "python.exe"))
        for env in sorted(glob.glob(os.path.join(r, "envs", "*"))):
            add(os.path.join(env, "python.exe"))
    for d in os.environ.get("PATH", "").split(os.pathsep):
        add(os.path.join(d, "python.exe"))
    for pattern in (r"C:\Python3*\python.exe", os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", "Python3*", "python.exe"),
                    r"C:\Program Files\Python3*\python.exe"):
        for exe in sorted(glob.glob(pattern)):
            add(exe)
    return out


def env_for(exe):
    """Environment as if a conda env were activated (its DLL folders on PATH)."""
    root = os.path.dirname(exe)
    extra = [root, os.path.join(root, "Library", "mingw-w64", "bin"), os.path.join(root, "Library", "usr", "bin"),
             os.path.join(root, "Library", "bin"), os.path.join(root, "Scripts")]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([p for p in extra if os.path.isdir(p)] + [env.get("PATH", "")])
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def works(exe, code, timeout=90):
    try:
        r = subprocess.run([exe, "-c", code], env=env_for(exe), cwd=HERE, timeout=timeout,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except Exception:
        return False


def choose(verbose):
    if os.path.isfile(PINNED):
        exe = open(PINNED).read().strip().strip('"')
        if verbose:
            print("Using Python from python_path.txt:", exe)
        return exe, works(exe, CHECK_PYSCAN)
    if os.path.isfile(CACHE):
        exe = open(CACHE).read().strip()
        if os.path.isfile(exe) and works(exe, CHECK_BASE):
            return exe, works(exe, CHECK_PYSCAN)
    fallback = None
    for exe in candidates():
        if not works(exe, CHECK_BASE):
            if verbose:
                print("  skip (no numpy/matplotlib):", exe)
            continue
        if works(exe, CHECK_PYSCAN):
            if verbose:
                print("  OK, has pyscan:", exe)
            return exe, True
        if verbose:
            print("  usable, but no pyscan:", exe)
        fallback = fallback or exe
    return fallback, False


def ensure_qt(exe):
    if works(exe, CHECK_QT):
        return True
    print("\nThis Python has no working Qt library (needed for the window):\n  " + exe)
    if not glob.glob(os.path.join(WHEELS, "PyQt5-*.whl")):
        print("Install PyQt5 into it (e.g. 'conda install pyqt'), then try again.")
        return False
    ans = input("Install PyQt5 from the offline_wheels folder into this Python? [y/N] ").strip().lower()
    if ans != "y":
        return False
    r = subprocess.run([exe, "-m", "pip", "install", "--no-index", "--find-links", WHEELS, "PyQt5"],
                       env=env_for(exe), cwd=HERE)
    return r.returncode == 0 and works(exe, CHECK_QT)


def main():
    args = sys.argv[1:]
    check = "--check" in args
    args = [a for a in args if a != "--check"]
    print("Looking for a Python that can run the confocal GUI...")
    exe, has_pyscan = choose(verbose=True)
    if not exe:
        print("\nNo Python with numpy and matplotlib was found on this computer.\n"
              "Put the full path of the python.exe your notebooks use into python_path.txt "
              "(next to this file) and try again.")
        return 1
    print("\nUsing:", exe)
    if not has_pyscan and "--sim" not in args:
        print("Note: this Python cannot import pyscan from %s - only the simulator will work." % PYSCAN_PATH)
    if not ensure_qt(exe):
        return 1
    with open(CACHE, "w") as f:
        f.write(exe)
    if check:
        print("Check finished: everything needed is available.")
        return 0
    print("Starting the GUI (keep this window open)...")
    return subprocess.call([exe, os.path.join(HERE, "run_confocal_gui.py")] + args, env=env_for(exe), cwd=HERE)


if __name__ == "__main__":
    code = main()
    if code:
        input("\nPress Enter to close.")
    sys.exit(code)