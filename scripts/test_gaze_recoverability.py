"""Self-test for the numerical core of gaze_recoverability.py (no GPU, no data)."""
import sys, tempfile, os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vjepa2"))

from gaze_recoverability import (
    Ridge, ridge_cv, r2, ChannelPCA, angular_error_deg, yawpitch_to_unit,
    GazeSeries, sample_indices,
)

ok = True
def check(name, cond, extra=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    ok &= bool(cond)

rng = np.random.default_rng(0)

# --- 1. Ridge recovers a known linear map (primal path: D <= N) ---------------
n, d, k = 400, 20, 3
X = rng.normal(size=(n, d))
Wtrue = rng.normal(size=(d, k))
Y = X @ Wtrue + 5.0 + 0.01 * rng.normal(size=(n, k))
pred = Ridge(1e-6).fit(X, Y).predict(X)
check("primal ridge recovers a linear map", r2(pred, Y) > 0.999, f"R2={r2(pred,Y):.5f}")

# --- 2. Dual path (D > N) agrees with primal on the same problem --------------
n2, d2 = 60, 300
X2 = rng.normal(size=(n2, d2))
Y2 = X2 @ rng.normal(size=(d2, 2)) + 1.0
a = 1.0
m_dual = Ridge(a).fit(X2, Y2)                    # d > n -> dual branch
# reference: explicit primal solve on the same centred problem
xm, ym = X2.mean(0, keepdims=True), Y2.mean(0, keepdims=True)
Xc, Yc = X2 - xm, Y2 - ym
W_ref = np.linalg.solve(Xc.T @ Xc + a * np.eye(d2), Xc.T @ Yc)
Xt = rng.normal(size=(10, d2))
p_ref = (Xt - xm) @ W_ref + ym
check("dual == primal", np.allclose(m_dual.predict(Xt), p_ref, atol=1e-6),
      f"max|diff|={np.abs(m_dual.predict(Xt)-p_ref).max():.2e}")

# --- 3. r2 == 0 when predicting the mean --------------------------------------
Ym = np.repeat(Y.mean(0, keepdims=True), n, axis=0)
check("r2(mean-predictor) == 0", abs(r2(Ym, Y)) < 1e-9, f"r2={r2(Ym,Y):.2e}")

# --- 4. ridge_cv prefers small alpha on clean data, large on pure noise --------
a_clean, _ = ridge_cv(X, Y, [1e-4, 1e0, 1e6], folds=4)
Ynoise = rng.normal(size=(n, k))
a_noise, _ = ridge_cv(X, Ynoise, [1e-4, 1e0, 1e6], folds=4)
check("CV picks small alpha on signal", a_clean <= 1e0, f"alpha={a_clean:g}")
check("CV picks large alpha on noise", a_noise == 1e6, f"alpha={a_noise:g}")

# --- 5. CV does not leak: R2 on unseen data is far below train R2 -------------
Xr, Yr = rng.normal(size=(80, 200)), rng.normal(size=(80, 2))
m = Ridge(1e-3).fit(Xr, Yr)
tr = r2(m.predict(Xr), Yr)
te = r2(m.predict(rng.normal(size=(80, 200))), Yr)
check("overfit detectable (train >> test on noise)", tr > 0.9 and te < 0.2,
      f"train={tr:.3f} test={te:.3f}")

# --- 6. ChannelPCA shape + variance -------------------------------------------
grids = rng.normal(size=(50, 16, 40))
p = ChannelPCA().fit(grids, 8)
t = p.transform(grids)
check("PCA output shape", t.shape == (50, 16, 8), str(t.shape))
check("PCA explained in (0,1]", 0 < p.explained <= 1.0 + 1e-9, f"{p.explained:.3f}")
# rank-limited input: 8 components should capture ~everything of a rank-8 signal
base = rng.normal(size=(50 * 16, 8)) @ rng.normal(size=(8, 40))
p2 = ChannelPCA().fit(base.reshape(50, 16, 40), 8)
check("PCA captures a rank-8 signal", p2.explained > 0.999, f"{p2.explained:.4f}")

# --- 7. Angular error ----------------------------------------------------------
yp = np.array([[0.0, 0.0], [0.3, -0.2]])
check("angular error zero for identical", np.allclose(angular_error_deg(yp, yp), 0, atol=1e-6))
e = angular_error_deg(np.array([[0.0, 0.0]]), np.array([[np.radians(10.0), 0.0]]))
check("10 deg yaw offset -> 10 deg error", abs(e[0] - 10.0) < 1e-4, f"{e[0]:.4f}")
check("unit vectors are unit", np.allclose(np.linalg.norm(yawpitch_to_unit(yp), axis=1), 1.0))

# --- 8. GazeSeries: nearest lookup, tolerance, validity ------------------------
tmp = tempfile.mkdtemp()
csv = os.path.join(tmp, "general_eye_gaze.csv")
ts = np.arange(0, 1_000_000, 10_000)             # 100 Hz for 1 s, in microseconds
pd.DataFrame({
    "tracking_timestamp_us": ts,
    "left_yaw_rads_cpf": np.linspace(0, 1, len(ts)),
    "right_yaw_rads_cpf": np.linspace(0, 1, len(ts)),
    "pitch_rads_cpf": np.linspace(-0.5, 0.5, len(ts)),
    "depth_m": np.full(len(ts), 1.5),
}).to_csv(csv, index=False)
gs = GazeSeries(csv, fps=30.0, tol_ms=50.0)
v = gs.at_ns(500_000 * 1000)                     # exactly on a sample
check("GazeSeries exact hit", v is not None and abs(v[0] - 0.5051) < 0.02, str(v))
check("GazeSeries beyond end -> None", gs.at_ns(5_000_000 * 1000) is None)
check("GazeSeries within tolerance", gs.at_ns((500_000 + 20) * 1000) is not None)

# tolerance only bites across a GAP — the series above is dense at 100 Hz, so
# every query lands within 5 ms of a sample. Drop a 300 ms hole and query inside it.
csv2 = os.path.join(tmp, "gapped_eye_gaze.csv")
keep = (ts < 400_000) | (ts > 700_000)
pd.DataFrame({
    "tracking_timestamp_us": ts[keep],
    "left_yaw_rads_cpf": np.zeros(keep.sum()),
    "right_yaw_rads_cpf": np.zeros(keep.sum()),
    "pitch_rads_cpf": np.zeros(keep.sum()),
    "depth_m": np.full(keep.sum(), 1.5),
}).to_csv(csv2, index=False)
gs2 = GazeSeries(csv2, fps=30.0, tol_ms=50.0)
check("GazeSeries mid-gap -> None", gs2.at_ns(550_000 * 1000) is None)
check("GazeSeries just outside gap -> hit", gs2.at_ns(390_000 * 1000) is not None)

# invalid rows (depth out of range) must be rejected, not silently filled
csv3 = os.path.join(tmp, "invalid_eye_gaze.csv")
pd.DataFrame({
    "tracking_timestamp_us": ts,
    "left_yaw_rads_cpf": np.zeros(len(ts)),
    "right_yaw_rads_cpf": np.zeros(len(ts)),
    "pitch_rads_cpf": np.zeros(len(ts)),
    "depth_m": np.full(len(ts), 50.0),          # > DEPTH_MAX -> gaze_valid False
}).to_csv(csv3, index=False)
check("GazeSeries rejects invalid rows",
      GazeSeries(csv3, fps=30.0).at_ns(500_000 * 1000) is None)

# --- 9. sample_indices respects the lead margin --------------------------------
idx = sample_indices(n_frames=3000, fps=30.0, max_lead_s=2.0, windows=5,
                     window_sec=8.0, per_window=4, rng=np.random.default_rng(1))
check("sample_indices count", 0 < len(idx) <= 20, f"n={len(idx)}")
check("sample_indices leaves lead margin", max(idx) < 3000 - 2.0 * 30 - 1, f"max={max(idx)}")
check("sample_indices sorted+unique", idx == sorted(set(idx)))
check("short video -> no samples", sample_indices(100, 30.0, 2.0, 5, 8.0, 4,
                                                  np.random.default_rng(1)) == [])

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
