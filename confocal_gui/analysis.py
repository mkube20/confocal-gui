"""Peak refinement (Gaussian fits) for scans and z sweeps."""

import warnings

import numpy as np

try:
    from scipy.optimize import curve_fit
except ImportError:  # fits are optional; raw maxima are used without scipy
    curve_fit = None


def _gauss1d(x, a, x0, s, b):
    return b + a * np.exp(-0.5 * ((x - x0) / s) ** 2)


def fit_peak_1d(x, y):
    """Fit a Gaussian + background. Returns dict with 'ok', 'center', 'sigma',
    'amplitude', 'background', 'peak'. 'ok' is False if the fit failed or the
    center fell outside the data."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = ~np.isnan(y)
    x, y = x[good], y[good]
    k = int(np.argmax(y)) if len(y) else 0
    out = {"ok": False, "center": float(x[k]) if len(x) else np.nan, "peak": float(y[k]) if len(y) else np.nan}
    if curve_fit is None or len(x) < 5:
        return out
    span = x.max() - x.min()
    p0 = [y[k] - np.median(y), x[k], max(span / 10, 1e-3), np.median(y)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = curve_fit(_gauss1d, x, y, p0=p0, maxfev=5000)
    except Exception:
        return out
    a, x0, s, b = p
    s = abs(s)
    if a > 0 and x.min() <= x0 <= x.max() and 0 < s < span:
        out.update(ok=True, center=float(x0), sigma=float(s), amplitude=float(a),
                   background=float(b), peak=float(a + b))
    return out


def _gauss2d(uv, a, u0, v0, su, sv, b):
    u, v = uv
    return b + a * np.exp(-0.5 * (((u - u0) / su) ** 2 + ((v - v0) / sv) ** 2))


def fit_peak_2d(u, v, img, half_window=None):
    """Fit a 2D Gaussian around the brightest pixel of ``img`` (shape len(v) x len(u)).
    Returns dict with 'ok', 'center' (u0, v0), 'sigma' (su, sv), 'peak'."""
    u, v, img = np.asarray(u, float), np.asarray(v, float), np.asarray(img, float)
    j, i = np.unravel_index(np.nanargmax(img), img.shape)
    out = {"ok": False, "center": (float(u[i]), float(v[j])), "peak": float(img[j, i])}
    if curve_fit is None or img.shape[0] < 3 or img.shape[1] < 3:
        return out
    step = abs(u[1] - u[0]) if len(u) > 1 else 0.1
    if half_window is None:
        half_window = max(3, int(round(1.0 / step)))   # ~1 um on each side
    j0, j1 = max(0, j - half_window), min(img.shape[0], j + half_window + 1)
    i0, i1 = max(0, i - half_window), min(img.shape[1], i + half_window + 1)
    sub = img[j0:j1, i0:i1]
    uu, vv = np.meshgrid(u[i0:i1], v[j0:j1])
    good = ~np.isnan(sub)
    if good.sum() < 8:
        return out
    b0 = np.nanmin(sub)
    p0 = [img[j, i] - b0, u[i], v[j], 2 * step, 2 * step, b0]
    try:
        with warnings.catch_warnings():      # noisy data: a failed fit is handled below
            warnings.simplefilter("ignore")
            p, _ = curve_fit(_gauss2d, (uu[good], vv[good]), sub[good], p0=p0, maxfev=5000)
    except Exception:
        return out
    a, u0, v0, su, sv, b = p
    su, sv = abs(su), abs(sv)
    inside = u[i0:i1].min() <= u0 <= u[i0:i1].max() and v[j0:j1].min() <= v0 <= v[j0:j1].max()
    if a > 0 and inside and su < 10 * (u[i1 - 1] - u[i0] + step) and sv < 10 * (v[j1 - 1] - v[j0] + step):
        out.update(ok=True, center=(float(u0), float(v0)), sigma=(float(su), float(sv)),
                   amplitude=float(a), background=float(b), peak=float(a + b))
    return out

def refine_peak_2d(u, v, img):
    """Sub-pixel position of the brightest spot: intensity-weighted centroid of the
    3 x 3 pixels around the brightest pixel (background = their minimum), limited to
    half a step from that pixel. Unlike a Gaussian fit this cannot wander off when the
    spot is only one or two pixels wide (coarse steps)."""
    u, v, img = np.asarray(u, float), np.asarray(v, float), np.asarray(img, float)
    j, i = np.unravel_index(np.nanargmax(img), img.shape)
    j0, j1 = max(0, j - 1), min(img.shape[0], j + 2)
    i0, i1 = max(0, i - 1), min(img.shape[1], i + 2)
    sub = img[j0:j1, i0:i1]
    w = np.nan_to_num(sub - np.nanmin(sub), nan=0.0)
    if w.sum() <= 0:
        return float(u[i]), float(v[j])
    uu, vv = np.meshgrid(u[i0:i1], v[j0:j1])
    cu, cv = float((w * uu).sum() / w.sum()), float((w * vv).sum() / w.sum())
    su = abs(u[1] - u[0]) / 2 if len(u) > 1 else 0.0
    sv = abs(v[1] - v[0]) / 2 if len(v) > 1 else 0.0
    return float(np.clip(cu, u[i] - su, u[i] + su)), float(np.clip(cv, v[j] - sv, v[j] + sv))
