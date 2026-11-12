"""Alignment-marker math.

Markers: TR, TL, BL, BR (top/bottom, left/right as seen in the sample's own frame).
The marker rectangle defines a "grid" frame:  u = distance to the right of the
bottom-left marker, v = distance up from it (both um, in the sample plane), and
dz = height above the focal plane fitted through the markers.
"""

import numpy as np

MARKER_LABELS = {"TR": "top right", "TL": "top left", "BL": "bottom left", "BR": "bottom right"}
MANUAL_ORDER = ["BL", "BR", "TR"]          # markers the user finds by hand
VERIFY_CYCLE = ["TL", "BL", "BR", "TR"]    # order visited during verification (TL is predicted first)

# For a parallelogram, each corner = adjacent1 + adjacent2 - opposite.
_PREDICT_FROM = {
    "TR": ("TL", "BR", "BL"),
    "TL": ("TR", "BL", "BR"),
    "BL": ("TL", "BR", "TR"),
    "BR": ("TR", "BL", "TL"),
}


def predict_marker(markers, name):
    """Predict marker ``name`` (abs xyz, um) from the other three."""
    a, b, opp = _PREDICT_FROM[name]
    return np.asarray(markers[a], float) + np.asarray(markers[b], float) - np.asarray(markers[opp], float)


class GridFrame:
    """Rigid (rotation + optional mirror) map between grid (u, v, dz) and absolute
    coordinates, plus a focal plane z = a*x + b*y + c through the markers."""

    kind = "marker"

    def __init__(self, markers):
        P = {k: np.asarray(markers[k], float) for k in ("TR", "TL", "BL", "BR")}
        self.markers = P
        right = 0.5 * ((P["BR"] - P["BL"]) + (P["TR"] - P["TL"]))[:2]
        up = 0.5 * ((P["TL"] - P["BL"]) + (P["TR"] - P["BR"]))[:2]
        self.width = float(np.linalg.norm(right))
        self.height = float(np.linalg.norm(up))
        cross = right[0] * up[1] - right[1] * up[0]
        self.handedness = 1.0 if cross >= 0 else -1.0
        a_r = np.arctan2(right[1], right[0])
        a_u = np.arctan2(up[1], up[0]) - self.handedness * np.pi / 2
        theta = np.arctan2(np.sin(a_r) + np.sin(a_u), np.cos(a_r) + np.cos(a_u))
        self.theta = float(theta)
        self.u_hat = np.array([np.cos(theta), np.sin(theta)])
        self.v_hat = self.handedness * np.array([-np.sin(theta), np.cos(theta)])
        # deviation of the measured corner angle from 90 degrees
        cosang = np.dot(right, up) / (self.width * self.height)
        self.skew_deg = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))) - 90.0)
        # (0, 0) is exactly the saved BL marker (x, y and focus); directions and the
        # focus tilt come from all four markers
        self.origin = P["BL"][:2].copy()
        A = np.array([[P[k][0], P[k][1], 1.0] for k in P])
        z = np.array([P[k][2] for k in P])
        a, b, _ = np.linalg.lstsq(A, z, rcond=None)[0]       # tilt of the focal plane
        self.plane = np.array([a, b, P["BL"][2] - a * P["BL"][0] - b * P["BL"][1]])
        self.grid_pos = {"BL": (0.0, 0.0), "BR": (self.width, 0.0),
                         "TL": (0.0, self.height), "TR": (self.width, self.height)}
        self.residuals = {k: P[k] - self.to_abs(*self.grid_pos[k]) for k in P}

    @property
    def rotation_deg(self):
        return float(np.degrees(self.theta))

    @property
    def mirrored(self):
        return self.handedness < 0

    def z_plane(self, x, y):
        a, b, c = self.plane
        return a * x + b * y + c

    def to_abs(self, u, v, dz=0.0):
        xy = self.origin + u * self.u_hat + v * self.v_hat
        return np.array([xy[0], xy[1], self.z_plane(xy[0], xy[1]) + dz])

    def to_grid(self, p):
        p = np.asarray(p, float)
        d = p[:2] - self.origin
        return np.array([d @ self.u_hat, d @ self.v_hat, p[2] - self.z_plane(p[0], p[1])])

    def tilt_um_per_um(self):
        """Focal-plane slope along u and v (um of z per um travelled)."""
        g = self.plane[:2]
        return float(g @ self.u_hat), float(g @ self.v_hat)

    def xy_direction(self, du, dv):
        """Sign of the absolute (x, y, z) motion for a step (du, dv) in the grid."""
        d = du * self.u_hat + dv * self.v_hat
        s = np.sign(np.where(np.abs(d) < 1e-6 * max(1.0, np.abs(d).max()), 0.0, d))
        return np.array([s[0], s[1], 0.0])

    def summary(self):
        res = np.array(list(self.residuals.values()))
        tu, tv = self.tilt_um_per_um()
        return {
            "width_um": self.width, "height_um": self.height,
            "rotation_deg": self.rotation_deg, "mirrored": self.mirrored,
            "skew_deg": self.skew_deg,
            "tilt_z_per_um_along_u": tu, "tilt_z_per_um_along_v": tv,
            "rms_residual_xy_um": float(np.sqrt(np.mean(np.sum(res[:, :2] ** 2, axis=1)))),
            "rms_residual_z_um": float(np.sqrt(np.mean(res[:, 2] ** 2))),
        }

    def to_dict(self):
        return {"markers_abs_um": {k: v.tolist() for k, v in self.markers.items()},
                "origin_abs_um": self.origin.tolist(), "u_hat": self.u_hat.tolist(),
                "v_hat": self.v_hat.tolist(), "plane_z_eq_ax_by_c": self.plane.tolist(),
                **self.summary()}


class AlignmentSession:
    """State machine for the marker alignment loop.

    1. The user finds BL, BR and TR by hand and saves each.
    2. Verification loop over TL, BL, BR, TR, TL, ...: the system predicts the next
       marker from the other three and moves there. The user accepts it or adjusts
       and saves. Any adjustment updates that marker and resets the counter.
    3. Done when four markers in a row are accepted without adjustment.
    """

    def __init__(self, accept_tol_um=0.05):
        self.tol = accept_tol_um
        self.measured = {}
        self.phase = "manual"
        self.manual_index = 0
        self.cycle_index = 0
        self.accepts_in_row = 0
        self.arrival = None    # where we actually ended up after moving to the prediction
        self.history = []

    @property
    def current(self):
        if self.phase == "manual":
            return MANUAL_ORDER[self.manual_index]
        if self.phase == "verify":
            return VERIFY_CYCLE[self.cycle_index % 4]
        return None

    @property
    def done(self):
        return self.phase == "done"

    def record_manual(self, pos):
        name = self.current
        self.measured[name] = np.asarray(pos, float)
        self.history.append({"marker": name, "action": "manual", "pos": list(map(float, pos))})
        self.manual_index += 1
        if self.manual_index >= len(MANUAL_ORDER):
            self.phase = "verify"
            self.cycle_index = 0
        return name

    def prediction(self):
        return predict_marker(self.measured, self.current)

    def set_arrival(self, pos):
        self.arrival = np.asarray(pos, float)

    def confirm(self, pos):
        """User is done with the current marker at ``pos``. Returns (adjusted, distance_um)."""
        pos = np.asarray(pos, float)
        name = self.current
        ref = self.arrival if self.arrival is not None else self.prediction()
        dist = float(np.linalg.norm(pos - ref))
        adjusted = dist > self.tol
        self.measured[name] = pos
        self.accepts_in_row = 0 if adjusted else self.accepts_in_row + 1
        self.history.append({"marker": name, "action": "adjusted" if adjusted else "accepted",
                             "pos": pos.tolist(), "moved_um": dist})
        self.arrival = None
        if self.accepts_in_row >= 4:
            self.phase = "done"
        else:
            self.cycle_index += 1
        return adjusted, dist

    def frame(self):
        m = dict(self.measured)
        if len(m) == 3:                       # the one not found by hand yet: predict it
            missing = next(k for k in ("TR", "TL", "BL", "BR") if k not in m)
            m[missing] = predict_marker(m, missing)
        return GridFrame(m) if len(m) == 4 else None

    def to_dict(self):
        return {"phase": self.phase, "manual_index": self.manual_index,
                "cycle_index": self.cycle_index, "accepts_in_row": self.accepts_in_row,
                "measured": {k: v.tolist() for k, v in self.measured.items()},
                "history": self.history, "accept_tol_um": self.tol}

    @classmethod
    def from_markers(cls, markers, accept_tol_um=0.05, start="BL", note="loaded"):
        """Skip the manual part: all four markers are known (e.g. a previous alignment,
        or a saved layout placed at a new BL marker); go straight to accept / adjust,
        starting with ``start``."""
        s = cls(accept_tol_um)
        s.measured = {k: np.asarray(markers[k], float) for k in ("TR", "TL", "BL", "BR")}
        s.manual_index = len(MANUAL_ORDER)
        s.phase = "verify"
        s.cycle_index = VERIFY_CYCLE.index(start)
        s.history = [{"marker": k, "action": note, "pos": s.measured[k].tolist()} for k in s.measured]
        return s

    @classmethod
    def from_dict(cls, d):
        s = cls(d.get("accept_tol_um", 0.05))
        s.phase = d["phase"]
        s.manual_index = d["manual_index"]
        s.cycle_index = d["cycle_index"]
        s.accepts_in_row = d["accepts_in_row"]
        s.measured = {k: np.asarray(v, float) for k, v in d["measured"].items()}
        s.history = d.get("history", [])
        return s

class PiezoFrame:
    """Scan frame along the piezo's x and y as shown in the GUI, with z fixed.

    u and v are the piezo x / y readouts (GUI frame) for as long as the stage does
    not move, so a point in a scan image can be typed straight into piezo Go-to.
    ``x_axis`` / ``y_axis`` are the hardware axes shown as GUI x / y.
    With ``plane`` (a GridFrame from the marker alignment) z follows the tilt of the
    marker plane instead, at the offset from that plane that ``at`` (abs x, y, z) has."""

    kind = "piezo"

    def __init__(self, stage_mm, piezo_sign, z_abs, x_axis, y_axis, plane=None, at=None):
        self.base = 1000.0 * np.asarray(stage_mm, float)
        self.sign = np.asarray(piezo_sign, float)
        self.z = float(z_abs)
        self.x_axis, self.y_axis = x_axis, y_axis
        self.plane = plane
        self.dz0 = 0.0
        if plane is not None:
            at = np.asarray(at, float)
            self.dz0 = float(at[2] - plane.z_plane(at[0], at[1]))

    @property
    def tilt(self):
        return self.plane is not None

    def _z_ref(self, x, y):
        return self.z if self.plane is None else self.plane.z_plane(x, y) + self.dz0

    def to_abs(self, u, v, dz=0.0):
        p = np.empty(3)
        p[self.x_axis] = self.base[self.x_axis] + self.sign[self.x_axis] * u
        p[self.y_axis] = self.base[self.y_axis] + self.sign[self.y_axis] * v
        p[2] = self._z_ref(p[0], p[1]) + dz
        return p

    def to_grid(self, p):
        p = np.asarray(p, float)
        return np.array([(p[self.x_axis] - self.base[self.x_axis]) / self.sign[self.x_axis],
                         (p[self.y_axis] - self.base[self.y_axis]) / self.sign[self.y_axis],
                         p[2] - self._z_ref(p[0], p[1])])

    def xy_direction(self, du, dv):
        d = np.zeros(3)
        d[self.x_axis] = np.sign(du * self.sign[self.x_axis])
        d[self.y_axis] = np.sign(dv * self.sign[self.y_axis])
        return d


def layout_offsets(markers):
    """Offsets (abs um) from the BL marker to every marker, for placing the same
    layout on another grid."""
    bl = np.asarray(markers["BL"], float)
    return {k: (np.asarray(markers[k], float) - bl) for k in ("TR", "TL", "BL", "BR")}


def markers_from_bl(bl, offsets):
    bl = np.asarray(bl, float)
    return {k: bl + np.asarray(offsets[k], float) for k in ("TR", "TL", "BL", "BR")}
