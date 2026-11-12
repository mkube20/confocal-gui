"""Saving Auto Scan output: scan images, scan data (CSV) and a results text file."""

import csv

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


AXIS_LABELS = {"marker": ("x along markers (um)", "y along markers (um)"),
               "piezo": ("piezo x (um)", "piezo y (um)")}
CSV_COORDS = {"marker": ("x_marker_um", "y_marker_um"), "piezo": ("x_piezo_um", "y_piezo_um")}


def check_folder_name(name):
    """Error message if ``name`` can't be used as a Windows folder name, else None."""
    if not name.strip():
        return "Enter a folder name."
    bad = sorted(set(name) & set('\\/:*?"<>|'))
    if bad:
        return "A folder name can't contain " + " ".join(bad)
    if name.strip() != name or name.endswith("."):
        return "A folder name can't start or end with a space, or end with a dot."
    return None


def axis_labels(frame_kind):
    return AXIS_LABELS.get(frame_kind, AXIS_LABELS["marker"])


def _f(v):
    return "" if v != v else f"{v:.6g}"   # blank for NaN


def save_raster_csv(res, path, perm=(0, 1, 2)):
    """One row per measured point, in the order measured. Stage/piezo columns are in
    the GUI frame: column k holds hardware axis perm[k]."""
    perm = list(perm)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["point", *CSV_COORDS.get(res.frame_kind, CSV_COORDS["marker"]),
                    "stage_x_mm", "stage_y_mm", "stage_z_mm",
                    "piezo_x_um", "piezo_y_um", "piezo_z_um",
                    "counts_cps", "detector1_cps", "detector2_cps"])
        for n, (j, i) in enumerate(res.order, start=1):
            w.writerow([n, f"{res.u[i]:.4f}", f"{res.v[j]:.4f}",
                        *(f"{v:.6f}" for v in res.stage_mm[j, i][perm]),
                        *(f"{v:.4f}" for v in res.piezo_um[j, i][perm]),
                        _f(res.counts[j, i]), _f(res.counts0[j, i]), _f(res.counts1[j, i])])


def save_zsweep_csv(res, path, perm=(0, 1, 2)):
    perm = list(perm)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["point", "stage_x_mm", "stage_y_mm", "stage_z_mm",
                    "piezo_x_um", "piezo_y_um", "piezo_z_um",
                    "counts_cps", "detector1_cps", "detector2_cps"])
        n = 0
        for k in range(len(res.z_abs)):
            if res.counts[k] != res.counts[k]:   # not measured (sweep stopped early)
                continue
            n += 1
            w.writerow([n, *(f"{v:.6f}" for v in res.stage_mm[k][perm]),
                        *(f"{v:.4f}" for v in res.piezo_um[k][perm]),
                        _f(res.counts[k]), _f(res.counts0[k]), _f(res.counts1[k])])


def _save_fig(fig, path):
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def draw_raster(ax, res, mark=None, clim=None):
    du = (res.u[1] - res.u[0]) / 2 if len(res.u) > 1 else 0.05
    dv = (res.v[1] - res.v[0]) / 2 if len(res.v) > 1 else 0.05
    im = ax.imshow(np.ma.masked_invalid(res.counts), origin="lower", cmap="inferno", aspect="equal",
                   interpolation="nearest",
                   extent=[res.u[0] - du, res.u[-1] + du, res.v[0] - dv, res.v[-1] + dv])
    if clim is not None:
        im.set_clim(*clim)
    if mark is not None:
        ax.plot(mark[0], mark[1], "c+", ms=14, mew=2)
    xl, yl = axis_labels(getattr(res, "frame_kind", "marker"))
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    return im


def save_raster_png(res, path, title, mark=None, clim=None):
    """``clim`` = (low, high) colour limits, e.g. the contrast set on screen."""
    fig = Figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = draw_raster(ax, res, mark, clim)
    fig.colorbar(im, ax=ax, label="counts/s")
    ax.set_title(title)
    _save_fig(fig, path)


def zsweep_axis(res):
    """x values and label for plotting a z sweep: piezo z if the stage did not move in z."""
    st = res.stage_mm[~np.isnan(res.stage_mm[:, 2]), 2]
    if len(st) and np.ptp(st) == 0:
        return res.piezo_um[:, 2], "piezo z (um)"
    return res.z_abs - res.z_abs[0], "z from sweep start (um)"


def draw_zsweep(ax, res, best_index=None):
    """Plot a z sweep; with ``best_index`` draw a red line at that measured point and
    label its piezo z and count rate. Returns (x, counts) of that point or None."""
    x, xlabel = zsweep_axis(res)
    ax.plot(x, res.counts, "o-", ms=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("counts/s")
    if best_index is None:
        return None
    xb, cb = float(x[best_index]), float(res.counts[best_index])
    ax.axvline(xb, color="r", lw=1.2)
    ax.text(0.02, 0.97, f"{xlabel.split(' (')[0]} = {xb:.3f} um\n{cb / 1000:,.1f} kcps",
            transform=ax.transAxes, va="top", color="r",
            bbox=dict(facecolor="white", edgecolor="r", alpha=0.85))
    return xb, cb


def save_zsweep_png(res, path, best_index=None):
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    draw_zsweep(ax, res, best_index)
    ax.set_title("Z sweep")
    _save_fig(fig, path)


def draw_power_sweep(ax, res):
    ok = ~np.isnan(res.counts)
    ax.plot(res.pct[ok], res.counts[ok], "o-", ms=4)
    ax.set_xlabel("laser diode current (% of max)")
    ax.set_ylabel("counts/s (both detectors)")


def save_power_sweep_png(res, path, title):
    fig = Figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111)
    draw_power_sweep(ax, res)
    ax.set_title(title)
    _save_fig(fig, path)


def save_power_sweep_csv(res, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["point", "laser_current_pct", "counts_cps", "detector1_cps", "detector2_cps"])
        n = 0
        for k in range(len(res.pct)):
            if res.counts[k] != res.counts[k]:
                continue
            n += 1
            w.writerow([n, f"{res.pct[k]:g}", _f(res.counts[k]), _f(res.counts0[k]),
                        _f(res.counts1[k])])
