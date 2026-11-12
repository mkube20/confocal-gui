"""Read photon timestamps from the g2 runs.

Each g2 run in the notebook's loop writes two files into the run folder
(e.g. C:\\path\\to\\g2\\run_folder):

* ``0``, ``1``, ``2`` ... (no extension): written by ``devices.ph.run_correlation``.
  They hold the timestamped photon data (every detection: when it happened and on
  which detector): either PicoQuant PTU files, or (pyscan, as in the lab) the bare
  PicoHarp T2 record stream with no header (4 ps time tags); both are read.
* ``*.hdf5``: the small pyscan background XY scan taken before that run. Only
  count rates on a few points, no timestamps (``show_hdf5`` lists what is inside).

Usage (Python / notebook):

    from ptu_timestamps import read_ptu, show_hdf5
    d = read_ptu(r"C:\\path\\to\\g2\\run_folder\\0")
    d["times_s"]      # photon arrival times in seconds from the start of the run
    d["channel"]      # detector: 0 = detector 1 (PicoHarp input 0), 1 = detector 2
    d["header"]       # all header tags (resolution, acquisition time, ...)

Command line (writes 0_timestamps.csv next to each file, or .npz with --npz):

    python ptu_timestamps.py C:\\path\\to\\g2\\run_folder
"""

import csv
import os
import struct
import sys

import numpy as np

# tag types (PicoQuant PTU format)
_TY = {
    0xFFFF0008: "empty", 0x00000008: "bool", 0x10000008: "int", 0x11000008: "bitset",
    0x12000008: "color", 0x20000008: "float", 0x21000008: "datetime", 0x2001FFFF: "floatarray",
    0x4001FFFF: "ansistring", 0x4002FFFF: "widestring", 0xFFFFFFFF: "blob",
}
# record types (TTResultFormat_TTTRRecType)
PICOHARP_T2 = 0x00010203
PICOHARP_T3 = 0x00010303
T2_WRAP = 210698240          # PicoHarp T2 time-tag overflow period (ticks)
T3_WRAP = 65536              # PicoHarp T3 sync-counter overflow period
RAW_T2_RESOLUTION = 4e-12    # PicoHarp 300 T2 time-tag resolution


def read_header(f):
    """Header tags as a dict; leaves ``f`` at the first record."""
    magic = f.read(8).rstrip(b"\0").decode("ascii", "replace")
    if magic != "PQTTTR":
        raise ValueError(f"not a PTU (TTTR) file (magic {magic!r})")
    f.read(8)                                      # version
    tags = {}
    while True:
        ident = f.read(32).rstrip(b"\0").decode("ascii", "replace")
        idx, typ = struct.unpack("<iI", f.read(8))
        raw = f.read(8)
        name = ident if idx < 0 else f"{ident}({idx})"
        kind = _TY.get(typ, "unknown")
        if kind == "float":
            value = struct.unpack("<d", raw)[0]
        elif kind in ("ansistring", "widestring", "floatarray", "blob"):
            n = struct.unpack("<q", raw)[0]
            data = f.read(n)
            if kind == "ansistring":
                value = data.rstrip(b"\0").decode("latin-1")
            elif kind == "widestring":
                value = data.decode("utf-16-le", "replace").rstrip("\0")
            elif kind == "floatarray":
                value = np.frombuffer(data, "<f8")
            else:
                value = data
        else:
            value = struct.unpack("<q", raw)[0]
        tags[name] = value
        if ident == "Header_End":
            return tags


def read_ptu(path):
    """Photon timestamps of one PicoHarp PTU file.

    Returns a dict with
      times_s   arrival time of every photon (s, from the start of the measurement)
      channel   detector of every photon (0 or 1)
      markers_s, marker   marker events, if any (usually none)
      header    all header tags
    For T3 files there is also ``dtime_s`` (time after the last sync pulse)."""
    with open(path, "rb") as f:
        if f.read(6) == b"PQTTTR":
            f.seek(0)
            hdr = read_header(f)
            n = int(hdr["TTResult_NumberOfRecords"])
            rec = np.fromfile(f, dtype="<u4", count=n)
        else:
            # pyscan's run_correlation writes the bare PicoHarp T2 record stream
            # (no PTU header): 32-bit records, 4 ps time tags
            f.seek(0)
            rec = np.fromfile(f, dtype="<u4")
            hdr = {"raw_picoharp_t2": True, "TTResultFormat_TTTRRecType": PICOHARP_T2,
                   "MeasDesc_GlobalResolution": RAW_T2_RESOLUTION, "TTResult_NumberOfRecords": len(rec)}
    rtype = int(hdr["TTResultFormat_TTTRRecType"])
    res = float(hdr["MeasDesc_GlobalResolution"])          # s per tick (T2: 4 ps)
    ch = (rec >> 28).astype(np.int64)
    out = {"header": hdr}
    if rtype == PICOHARP_T2:
        t = (rec & 0x0FFFFFFF).astype(np.int64)
        special = ch == 0xF
        overflow = special & ((t & 0xF) == 0)
        truetime = t + np.cumsum(overflow) * T2_WRAP
        photon = ~special
        marker = special & ~overflow
        out["times_s"] = truetime[photon] * res
        out["channel"] = ch[photon]
        out["markers_s"] = truetime[marker] * res
        out["marker"] = (t[marker] & 0xF)
    elif rtype == PICOHARP_T3:
        dt = ((rec >> 16) & 0x0FFF).astype(np.int64)
        nsync = (rec & 0xFFFF).astype(np.int64)
        special = ch == 0xF
        overflow = special & ((dt & 0xF) == 0)
        sync = nsync + np.cumsum(overflow) * T3_WRAP
        photon = ~special
        marker = special & ~overflow
        dres = float(hdr.get("MeasDesc_Resolution", 0.0))    # s per dtime bin
        out["times_s"] = sync[photon] * res + dt[photon] * dres
        out["dtime_s"] = dt[photon] * dres
        out["channel"] = ch[photon] - 1                         # T3 detector channels are 1-based
        out["markers_s"] = sync[marker] * res
        out["marker"] = dt[marker] & 0xF
    else:
        raise ValueError(f"record type 0x{rtype:08X} is not PicoHarp T2/T3")
    return out


def show_hdf5(path):
    """Print every group / dataset in an HDF5 file (e.g. the background scans)."""
    import h5py

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: dataset {obj.shape} {obj.dtype}")
        else:
            print(f"  {name}/")
        for k, v in obj.attrs.items():
            print(f"      @{k} = {v!r}"[:200])

    with h5py.File(path, "r") as f:
        print(path)
        for k, v in f.attrs.items():
            print(f"  @{k} = {v!r}"[:200])
        f.visititems(visit)


def _is_ptu(path):
    """A PTU file, or a bare PicoHarp T2 stream named 0, 1, 2 ... (as pyscan writes them)."""
    try:
        with open(path, "rb") as f:
            head = f.read(6)
        if head == b"PQTTTR":
            return True
        return os.path.basename(path).isdigit() and os.path.getsize(path) % 4 == 0 and os.path.getsize(path) > 0
    except OSError:
        return False


def export(path, npz=False):
    d = read_ptu(path)
    if npz:
        out = path + "_timestamps.npz"
        np.savez(out, times_s=d["times_s"], channel=d["channel"])
    else:
        out = path + "_timestamps.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "detector"])
            w.writerows((f"{t:.12e}", c + 1) for t, c in zip(d["times_s"], d["channel"]))
    n0, n1 = int(np.sum(d["channel"] == 0)), int(np.sum(d["channel"] == 1))
    span = d["times_s"][-1] if len(d["times_s"]) else 0.0
    print(f"{path}: {len(d['times_s'])} photons (detector 1: {n0}, detector 2: {n1}) "
          f"over {span:.3f} s -> {out}")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    npz = "--npz" in sys.argv
    if not args:
        print(__doc__)
        sys.exit(1)
    for a in args:
        files = ([os.path.join(a, n) for n in sorted(os.listdir(a))] if os.path.isdir(a) else [a])
        for p in files:
            if os.path.isfile(p) and _is_ptu(p):
                export(p, npz)
