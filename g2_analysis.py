"""g2 (second-order correlation) from the g2 run folders.

Reads every PicoHarp PTU file in a run folder (``0``, ``1``, ``2`` ... as written by
``devices.ph.run_correlation``), and:

1. builds the cross-correlation histogram of detector 2 vs detector 1 arrival times
   (tau = t_det2 - t_det1, every pair within +/- ``window``, bin ``bin_width``) and sums
   the raw coincidence counts over the runs (runs can be skipped by number, or
   automatically when the count rate dropped during the run, ``drift=``),
2. normalises it by the mean of the flat long-delay region (``norm_range``, |tau|) and
   compares that with the Poisson expectation R1 * R2 * T * dt (ratio ~1 for a
   non-blinking source),
3. fits g2(tau) = 1 - a exp(-|tau - t0| / t1) [+ b exp(-|tau - t0| / t2)] with Poisson
   weights; g2(0) = 1 - a [+ b] with its error from the fit covariance, and also
   g2(0) straight from the raw bins around t0,
4. optionally corrects for uncorrelated background: g2c = (g2 - (1 - rho^2)) / rho^2,
   rho = S / (S + B) with ``signal_plus_bkg`` = count rate on the emitter and ``bkg`` =
   count rate beside it (unreliable when rho < 0.5: a warning is printed),
5. saves NAME_g2.png (points with Poisson error bars, fit, dashed lines at 1 and 0.5,
   tau shifted so the dip is at 0, g2(0) +/- error) and NAME_g2.csv.

Notebook:
    from g2_analysis import analyse_folder
    r = analyse_folder(r"C:\\path\\to\\g2\\run_folder", bin_width=1e-9,
                       signal_plus_bkg=6000, bkg=1500)
    r["g2_0"], r["g2_0_err"], r["fit"]

Command line:
    python g2_analysis.py C:\\path\\to\\g2\\run_folder --bin 1 --window 500 \\
        --norm 200 500 --skip 3 7 --drift 0.7 --sb 6000 --b 1500 --bunching
(times in ns, rates in counts/s).
"""

import argparse
import csv
import os
import re
import warnings

import numpy as np

from ptu_timestamps import read_ptu, _is_ptu


# ----------------------------------------------------------------------------- files
def run_files(folder):
    """PTU files of a run folder, in run order (0, 1, 2, ...)."""
    files = [os.path.join(folder, n) for n in os.listdir(folder)]
    files = [f for f in files if os.path.isfile(f) and _is_ptu(f)]

    def key(p):
        m = re.search(r"(\d+)", os.path.basename(p))
        return (int(m.group(1)) if m else 10 ** 9, os.path.basename(p))
    return sorted(files, key=key)


def _run_number(path):
    m = re.fullmatch(r"(\d+)", os.path.splitext(os.path.basename(path))[0])
    return int(m.group(1)) if m else None


def drifted_runs(folder, threshold=0.7):
    """Run numbers whose count rate after the g2 fell below ``threshold`` x the rate
    after the optimisation before it (the notebook's meta_data.txt / max_counts.txt,
    or 'Max count rate' lines in logfile.txt). Empty if those files are missing."""
    after_g2 = before = None
    try:
        after_g2 = np.atleast_1d(np.loadtxt(os.path.join(folder, "max_counts.txt")))
    except (OSError, ValueError):
        pass
    try:
        before = np.atleast_2d(np.loadtxt(os.path.join(folder, "meta_data.txt")))[0]   # after_scan_counts0
    except (OSError, ValueError, IndexError):
        pass
    if before is None:
        try:
            with open(os.path.join(folder, "logfile.txt"), encoding="utf-8", errors="ignore") as f:
                vals = [float(line.split(":")[-1].split(",")[0]) for line in f if "Max count rate" in line]
            before = np.array(vals) if vals else None
        except (OSError, ValueError):
            pass
    if after_g2 is None or before is None:
        return []
    n = min(len(after_g2), len(before))
    return [k for k in range(n) if after_g2[k] < threshold * before[k]]


# ----------------------------------------------------------------------- correlation
def cross_correlate(t1, t2, window, bin_width):
    """Histogram of t2 - t1 over all pairs with |t2 - t1| <= window (times sorted, s).
    Returns (bin centres, counts)."""
    nb = int(round(window / bin_width))
    edges = (np.arange(-nb, nb + 2) - 0.5) * bin_width       # bins centred on -nb..nb x bin_width
    lo = np.searchsorted(t2, t1 + edges[0], side="left")
    hi = np.searchsorted(t2, t1 + edges[-1], side="left")
    k = hi - lo
    total = int(k.sum())
    if total == 0:
        return 0.5 * (edges[:-1] + edges[1:]), np.zeros(2 * nb + 1, dtype=np.int64)
    start = np.repeat(np.arange(len(t1)), k)
    offset = np.arange(total) - np.repeat(np.cumsum(k) - k, k)
    dtau = t2[np.repeat(lo, k) + offset] - t1[start]
    counts, _ = np.histogram(dtau, edges)
    return 0.5 * (edges[:-1] + edges[1:]), counts.astype(np.int64)


def correlate_run(path, window, bin_width):
    d = read_ptu(path)
    t, ch = d["times_s"], d["channel"]
    t1, t2 = np.sort(t[ch == 0]), np.sort(t[ch == 1])
    hdr = d["header"]
    acq = hdr.get("MeasDesc_AcquisitionTime")                # ms
    T = float(acq) / 1000.0 if acq else (float(t.max() - t.min()) if len(t) else 0.0)
    tau, counts = cross_correlate(t1, t2, window, bin_width)
    return {"tau": tau, "counts": counts, "n1": len(t1), "n2": len(t2), "T": T}


# ------------------------------------------------------------------------------- fit
def model(tau, t0, a, t1, b=0.0, t2=1.0):
    x = np.abs(tau - t0)
    return 1.0 - a * np.exp(-x / t1) + b * np.exp(-x / t2)


def _best_fit(tau, g, err, starts, lb, ub):
    """Weighted curve_fit from each start in ``starts``; the result with the lowest chi^2."""
    from scipy.optimize import curve_fit
    best = None
    for q0 in starts:
        q0 = [float(np.clip(v, lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo))) for v, lo, hi in zip(q0, lb, ub)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                q, qc = curve_fit(lambda x, *v: model(x, *v), tau, g, p0=q0, sigma=err,
                                  absolute_sigma=True, bounds=(lb, ub), maxfev=20000)
            except (RuntimeError, ValueError):
                continue
        c2 = float(np.sum(((g - model(tau, *q)) / err) ** 2))
        if best is None or c2 < best[0]:
            best = (c2, q, qc)
    return best


def fit_g2(tau, g, err, bunching=False, t0_guess=None, err_model=None):
    """Weighted fit of g2(tau) = 1 - a exp(-|tau-t0|/t1) [+ b exp(-|tau-t0|/t2)].
    Returns dict(params, p, cov, names, g2_0, g2_0_err, reduced_chi2) or None.

    |tau - t0| has a kink, so several starting values are tried. With ``bunching`` the
    dip is fitted first; then t0 stays within a few bins of it and t2 must be longer
    than the dip (otherwise b exp(-|tau|/t2) can imitate part of the dip).
    ``err_model(m)``: the Poisson error expected for model values m. When given, the fit
    is repeated with those errors (weights from each bin's own counts pull the fit
    toward the low-count bins in the dip and make g2(0) and t1 come out too high)."""
    if err_model is not None:
        f = fit_g2(tau, g, err, bunching, t0_guess)
        for _ in range(2):
            if f is None:
                return None
            e2 = err_model(model(tau, *f["p"]))
            f = fit_g2(tau, g, e2, bunching, f["params"]["t0"])
        return f
    ok = np.isfinite(g) & np.isfinite(err) & (err > 0)
    tau, g, err = tau[ok], g[ok], err[ok]
    if len(tau) < 6:
        return None
    step = float(np.median(np.diff(tau)))
    span = float(tau.max() - tau.min())
    sm = np.convolve(g, np.ones(5) / 5, mode="same")
    sm[:2] = sm[-2:] = np.inf                          # "same" pads the ends with zeros
    k = int(np.argmin(sm))
    t0 = tau[k] if t0_guess is None else t0_guess
    a0 = float(np.clip(1 - g[k], 0.05, 1.5))
    lb = [t0 - 10 * step - 5e-9, 0.0, step / 10]
    ub = [t0 + 10 * step + 5e-9, 3.0, span]
    starts = [[t0 + d * step, a0, t1s] for d in (-2, -1, 0, 1, 2) for t1s in (2 * step, 5e-9, 15e-9)]
    best = _best_fit(tau, g, err, starts, lb, ub)
    if best is None:
        return None
    names = ["t0", "a", "t1"]
    if bunching:
        _, (bt0, ba, bt1), _ = best
        lb = [bt0 - 3 * step, 0.0, bt1 / 3, 0.0, max(1.5 * bt1, 3 * step)]
        ub = [bt0 + 3 * step, 3.0, 3 * bt1, 10.0, 10 * span]
        starts = [[bt0, ba + bb, bt1, bb, t2s] for bb in (0.05, 0.3) for t2s in (5 * bt1, 30 * bt1, 0.3 * span)]
        best = _best_fit(tau, g, err, starts, lb, ub)
        if best is None:
            return None
        names += ["b", "t2"]
    c2, p, cov = best
    grad = np.zeros(len(p))
    grad[1] = -1.0
    if bunching:
        grad[3] = 1.0
    g0 = 1.0 - p[1] + (p[3] if bunching else 0.0)
    return {"params": dict(zip(names, map(float, p))), "p": p, "cov": cov, "names": names,
            "g2_0": float(g0), "g2_0_err": float(np.sqrt(max(0.0, grad @ cov @ grad))),
            "reduced_chi2": c2 / max(1, len(tau) - len(p))}


def raw_g2_0(tau, counts, norm, t0, halfwidth):
    """g2(0) from the summed raw bins with |tau - t0| <= halfwidth."""
    sel = np.abs(tau - t0) <= halfwidth + 1e-15
    if not sel.any():
        sel = np.zeros(len(tau), bool)
        sel[int(np.argmin(np.abs(tau - t0)))] = True
    n, c = int(sel.sum()), float(counts[sel].sum())
    return c / (n * norm), np.sqrt(max(c, 1.0)) / (n * norm), n


# ---------------------------------------------------------------------------- driver
def analyse_folder(folder, window=500e-9, bin_width=1e-9, norm_range=(200e-9, 500e-9), skip=(),
                   drift=None, bunching=False, signal_plus_bkg=None, bkg=None, raw_halfwidth=None,
                   plot_range=100e-9, name=None, save=True, out_dir=None, verbose=True, exclude=()):
    """Analyse every PTU run in ``folder`` (times in seconds, rates in counts/s).

    skip: run numbers to leave out. drift: e.g. 0.7 leaves out runs whose count rate after
    the g2 fell below 70 % of the rate after the optimisation. signal_plus_bkg / bkg:
    count rates on / beside the emitter for the background correction. exclude: tau ranges
    [(lo, hi), ...] (s, as measured, before shifting to t0) left out of the fit and the
    normalisation, e.g. the detector backflash / crosstalk peaks.
    Returns a dict with the histogram, normalisation, fits and g2(0) values."""
    say = print if verbose else (lambda *a, **k: None)
    files = run_files(folder)
    if not files:
        raise FileNotFoundError(f"no PTU files in {folder}")
    skip = set(int(s) for s in skip)
    if drift is not None:
        flagged = drifted_runs(folder, drift)
        if flagged:
            say(f"drift: leaving out runs {flagged} (count rate after the g2 < {drift:g} x before)")
        skip |= set(flagged)
    used, counts, expected, n1, n2, T = [], None, 0.0, 0, 0, 0.0
    for p in files:
        num = _run_number(p)
        if num is not None and num in skip:
            continue
        r = correlate_run(p, window, bin_width)
        counts = r["counts"] if counts is None else counts + r["counts"]
        tau = r["tau"]
        if r["T"] > 0:
            expected += r["n1"] * r["n2"] / r["T"] * bin_width       # R1 * R2 * T * dt
        n1, n2, T = n1 + r["n1"], n2 + r["n2"], T + r["T"]
        used.append(os.path.basename(p))
    if counts is None:
        raise ValueError("every run was skipped")
    say(f"{len(used)} runs used: {', '.join(used)}")
    say(f"detector 1: {n1} photons, detector 2: {n2} photons, {T:.1f} s in total "
        f"({n1 / T if T else 0:,.0f} / {n2 / T if T else 0:,.0f} cps)")

    ex = np.zeros(len(tau), bool)
    for lo, hi in exclude:
        ex |= (tau >= lo) & (tau <= hi)
    if ex.any():
        say("excluded from the fit / normalisation: " + ", ".join(f"{lo * 1e9:g} to {hi * 1e9:g} ns" for lo, hi in exclude))
    flat = (np.abs(tau) >= norm_range[0]) & (np.abs(tau) <= norm_range[1]) & ~ex
    if not flat.any():
        raise ValueError("norm_range has no bins inside the window")
    norm = float(counts[flat].mean())
    say(f"flat level {norm:.2f} coincidences/bin, expected R1*R2*T*dt = {expected:.2f} "
        f"(ratio {norm / expected if expected else float('nan'):.3f})")
    if norm <= 0:
        raise ValueError("no coincidences in the normalisation region")
    g = counts / norm
    err = np.sqrt(np.maximum(counts, 1)) / norm

    out = {"tau": tau, "counts": counts, "g2": g, "err": err, "norm": norm, "expected_per_bin": expected,
           "runs": used, "rate1": n1 / T if T else 0.0, "rate2": n2 / T if T else 0.0, "T": T, "excluded": ex}
    pois = lambda m: np.sqrt(np.maximum(m * norm, 1.0)) / norm    # Poisson error of a model value
    fit = fit_g2(tau, np.where(ex, np.nan, g), err, bunching, err_model=pois)
    out["fit"] = fit
    t0 = fit["params"]["t0"] if fit else float(tau[int(np.argmin(g))])
    hw = raw_halfwidth if raw_halfwidth is not None else bin_width
    out["t0"] = t0
    out["g2_0_raw"], out["g2_0_raw_err"], out["raw_bins"] = raw_g2_0(tau, counts, norm, t0, hw)
    if fit:
        say(f"fit: t0 = {t0 * 1e9:.2f} ns, t1 = {fit['params']['t1'] * 1e9:.2f} ns"
            + (f", b = {fit['params']['b']:.3f}, t2 = {fit['params']['t2'] * 1e9:.1f} ns" if bunching else "")
            + f", reduced chi2 = {fit['reduced_chi2']:.2f}")
        say(f"g2(0) from the fit: {fit['g2_0']:.3f} +/- {fit['g2_0_err']:.3f}")
    say(f"g2(0) from the {out['raw_bins']} raw bin(s) at t0: {out['g2_0_raw']:.3f} +/- {out['g2_0_raw_err']:.3f}")
    out["g2_0"] = fit["g2_0"] if fit else out["g2_0_raw"]
    out["g2_0_err"] = fit["g2_0_err"] if fit else out["g2_0_raw_err"]

    if signal_plus_bkg is not None and bkg is not None:
        rho = (signal_plus_bkg - bkg) / signal_plus_bkg
        out["rho"] = rho
        if rho <= 0:
            say(f"WARNING: background ({bkg:g}) >= signal + background ({signal_plus_bkg:g}): no correction possible")
        else:
            if rho < 0.5:
                say(f"WARNING: rho = S/(S+B) = {rho:.3f} < 0.5: the background correction divides by "
                    f"rho^2 = {rho ** 2:.3f} and is unreliable; reduce the background first")
            gc = (g - (1 - rho ** 2)) / rho ** 2
            ec = err / rho ** 2
            out.update(g2_corr=gc, err_corr=ec)
            fc = fit_g2(tau, np.where(ex, np.nan, gc), ec, bunching, t0_guess=t0,
                        err_model=lambda m: pois(m * rho ** 2 + (1 - rho ** 2)) / rho ** 2)
            out["fit_corr"] = fc
            if fc:
                out["g2_0_corr"], out["g2_0_corr_err"] = fc["g2_0"], fc["g2_0_err"]
            else:
                out["g2_0_corr"] = (out["g2_0_raw"] - (1 - rho ** 2)) / rho ** 2
                out["g2_0_corr_err"] = out["g2_0_raw_err"] / rho ** 2
            say(f"background corrected (rho = {rho:.3f}): g2(0) = {out['g2_0_corr']:.3f} "
                f"+/- {out['g2_0_corr_err']:.3f}")

    if save:
        name = name or os.path.basename(os.path.normpath(folder))
        out_dir = out_dir or folder
        out["png"] = os.path.join(out_dir, f"{name}_g2.png")
        out["csv"] = os.path.join(out_dir, f"{name}_g2.csv")
        save_csv(out, out["csv"])
        save_png(out, out["png"], plot_range, f"g2 {name}")
        say(f"saved {out['png']} and {out['csv']}")
    return out


# ---------------------------------------------------------------------------- output
def save_csv(r, path):
    corr = "g2_corr" in r
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        ex = r.get("excluded", np.zeros(len(r["tau"]), bool))
        w.writerow(["tau_ns", "tau_minus_t0_ns", "counts", "g2", "err"]
                   + (["g2_corrected", "err_corrected"] if corr else []) + ["excluded"])
        for k in range(len(r["tau"])):
            row = [f"{r['tau'][k] * 1e9:.4f}", f"{(r['tau'][k] - r['t0']) * 1e9:.4f}", int(r["counts"][k]),
                   f"{r['g2'][k]:.6g}", f"{r['err'][k]:.6g}"]
            if corr:
                row += [f"{r['g2_corr'][k]:.6g}", f"{r['err_corr'][k]:.6g}"]
            w.writerow(row + [int(ex[k])])


def save_png(r, path, plot_range=100e-9, title="g2"):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    corr = "g2_corr" in r
    g, e, fit = (r["g2_corr"], r["err_corr"], r.get("fit_corr")) if corr else (r["g2"], r["err"], r["fit"])
    g0, g0e = (r["g2_0_corr"], r["g2_0_corr_err"]) if corr else (r["g2_0"], r["g2_0_err"])
    x = (r["tau"] - r["t0"]) * 1e9
    sel = np.abs(x) <= plot_range * 1e9
    fig = Figure(figsize=(7, 4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ex = r.get("excluded", np.zeros(len(x), bool))
    keep = sel & ~ex
    if corr:
        ax.plot(x[keep], r["g2"][keep], ".", color="0.6", ms=3, label="raw")
    ax.errorbar(x[keep], g[keep], yerr=e[keep], fmt="o", ms=3, color="tab:blue", ecolor="tab:blue",
                elinewidth=0.6, alpha=0.8, label="background corrected" if corr else "data")
    if (sel & ex).any():
        ax.plot(x[sel & ex], g[sel & ex], "x", color="tab:orange", ms=4, label="excluded (backflash)")
    if fit:
        xf = np.linspace(x[sel].min(), x[sel].max(), 2000)
        ax.plot(xf, model(xf * 1e-9 + r["t0"], *fit["p"]), "r-", lw=1.5, label="fit")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlim(-plot_range * 1e9, plot_range * 1e9)
    ax.set_ylim(bottom=min(0.0, float(np.nanmin(g[keep])) - 0.05),
                top=max(1.6, float(np.nanmax(g[keep])) + 0.1))
    ax.set_xlabel(r"$\tau$ (ns)")
    ax.set_ylabel(r"$g^{(2)}(\tau)$")
    ax.set_title(title)
    text = f"$g^{{(2)}}(0)$ = {g0:.3f} $\\pm$ {g0e:.3f}"
    if corr:
        text += f"\n(raw {r['g2_0']:.3f} $\\pm$ {r['g2_0_err']:.3f}, $\\rho$ = {r['rho']:.2f})"
    ax.text(0.02, 0.04, text, transform=ax.transAxes, va="bottom",
            bbox=dict(facecolor="white", edgecolor="0.5", alpha=0.9))
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def main(argv=None):
    ap = argparse.ArgumentParser(description="g2 histogram, normalisation, fit and g2(0) from a g2 run folder")
    ap.add_argument("folder")
    ap.add_argument("--bin", type=float, default=1.0, help="bin width (ns), default 1")
    ap.add_argument("--window", type=float, default=500.0, help="+/- correlation window (ns), default 500")
    ap.add_argument("--norm", type=float, nargs=2, default=(200.0, 500.0), metavar=("MIN", "MAX"),
                    help="|tau| range (ns) used to normalise, default 200 500")
    ap.add_argument("--skip", type=int, nargs="*", default=[], help="run numbers to leave out")
    ap.add_argument("--drift", type=float, default=None,
                    help="leave out runs whose count rate after the g2 fell below this fraction")
    ap.add_argument("--bunching", action="store_true", help="add the bunching term b exp(-|tau|/t2)")
    ap.add_argument("--sb", type=float, default=None, help="count rate on the emitter (signal + background, cps)")
    ap.add_argument("--b", type=float, default=None, help="background count rate beside the emitter (cps)")
    ap.add_argument("--plot", type=float, default=100.0, help="+/- tau range of the plot (ns), default 100")
    ap.add_argument("--name", default=None, help="output name (default: the folder name)")
    ap.add_argument("--exclude", type=float, nargs=2, action="append", default=[], metavar=("LO", "HI"),
                    help="tau range (ns, as measured) to leave out of the fit / normalisation, e.g. a "
                         "backflash peak; can be given several times")
    a = ap.parse_args(argv)
    analyse_folder(a.folder, window=a.window * 1e-9, bin_width=a.bin * 1e-9,
                   norm_range=(a.norm[0] * 1e-9, a.norm[1] * 1e-9), skip=a.skip, drift=a.drift,
                   bunching=a.bunching, signal_plus_bkg=a.sb, bkg=a.b, plot_range=a.plot * 1e-9, name=a.name,
                   exclude=[(lo * 1e-9, hi * 1e-9) for lo, hi in a.exclude])


if __name__ == "__main__":
    main()
