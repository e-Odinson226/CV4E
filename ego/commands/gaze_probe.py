"""
T4: can gaze be read from the frozen encoder's features?

This tests hypothesis R in docs/EgoVault/1-introduction.md: the frozen image features
already contain gaze, so a gaze token gives the predictor no new information.

Method
------
A linear probe predicts gaze at time t + lead from the encoder's features of the frame
at time t. The script repeats this for each lead in --leads (0 to 2 s by default).

  * If the skill stays high as the lead grows, the features already contain future
    gaze, and a gaze token adds little.
  * If the skill falls as the lead grows, gaze carries information that the features
    lack at longer horizons.

The probe reads the frozen encoder, not the predictor. The predictor received gaze
during training, so gaze can be read from it trivially. The encoder never saw gaze.

The probe is ridge regression: a linear map with L2 regularization. A linear map cannot
build new features. It only reads information that the features hold in linear form.

The main readout keeps all 256 patches, because gaze is a place in the image. PCA
reduces the 1408 channels of each patch to --pca-dim. A readout that averages over the
patches is computed for comparison. The gap between the two shows how much of gaze
depends on where things are in the image.

Skill = 1 - MSE(probe) / MSE(guessing the training mean). 0 is chance and 1 is perfect.

Splits (--split)
----------------
  participant  Train on --train-participants and test on --test-participants. This is
               the main split. The probe cannot learn one person's habits.
  recording    Train and test on different recordings of the same people.
  random       Random frames. Near-identical frames end up in both sets, so the score
               is too high. Use it only for comparison.

The encoder features are cached in --cache. control-probe (T5) reads this
cache. The results are in docs/EgoVault/4-results.md, under T4.

Usage
-----
    python -m ego gaze-probe \
        --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
        --video-dir  data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
        --gaze-dir   data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
        --train-participants P01 P02 P03 P04 P05 \
        --test-participants  P06 P07 \
        --out results/gaze_recoverability
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ego.data import load_frames
from ego.linprobe import ChannelPCA, Ridge, angular_error_deg, probe_samples, r2, ridge_cv
from ego.model import encode_independent, load_models
from ego.runlog import Logger


def _fit_pca(pca, pca_buf, feats_grid, args, log, short=False):
    """
    Fit the channel PCA on buffered TRAIN frames, then flush the buffer into
    feats_grid *in order*.

    Buffered batches are always the earliest ones, so flushing them here — and
    appending nothing while buffering — keeps feats_grid index-aligned with
    feats_pool without needing placeholders.
    """
    buf = np.concatenate(pca_buf, 0)
    pca[0] = ChannelPCA().fit(buf, args.pca_dim)
    log(f"  [pca] fitted on {len(buf)} frames"
        f"{' (fewer than --pca-fit-frames; dataset is small)' if short else ''}, "
        f"{args.pca_dim}/{buf.shape[-1]} channels, {pca[0].explained:.1%} variance")
    for b in pca_buf:
        feats_grid.append(pca[0].transform(b).reshape(b.shape[0], -1))
    pca_buf.clear()


def collect(args, encoder, device, participants, split_name, pca, pca_buf, log):
    """Encode sampled frames and pair each with gaze at every lead. Returns arrays."""
    feats_grid, feats_pool, gaze_t, gaze_lead, meta = [], [], [], [], []
    rng = np.random.default_rng(args.seed)
    amp = torch.bfloat16 if (not args.no_amp and device.type == "cuda") else None

    for p, vp, _, keep, y_t, y_lead in probe_samples(
            args.video_dir, args.gaze_dir, participants, args.recordings, args.windows,
            args.window_sec, args.per_window, args.leads, args.tol_ms, rng, log):
        if not keep:
            continue

        frames = load_frames(str(vp), keep, size=args.img_size)      # (K,3,H,W)
        for s in range(0, len(keep), args.batch_size):
            fb = frames[s:s + args.batch_size].unsqueeze(1)          # (b,1,3,H,W)
            h = encode_independent(encoder, fb, device,
                                   normalize_reps=not args.no_normalize_reps,
                                   chunk=args.encode_chunk, amp_dtype=amp)
            g = h.cpu().numpy().astype(np.float32)                   # (b, tokens, D)
            feats_pool.append(g.mean(1))
            if pca[0] is None:
                # Still buffering: append NOTHING to feats_grid. The buffered
                # batches are the first ones, so flushing them in order the moment
                # PCA fits keeps feats_grid aligned with feats_pool. (Appending a
                # placeholder here instead would leave holes that never get filled.)
                pca_buf.append(g)
                if sum(b.shape[0] for b in pca_buf) >= args.pca_fit_frames:
                    _fit_pca(pca, pca_buf, feats_grid, args, log)
            else:
                feats_grid.append(pca[0].transform(g).reshape(g.shape[0], -1))
        gaze_t.append(np.stack(y_t)); gaze_lead.append(np.stack(y_lead))
        meta.extend([(p, vp.stem)] * len(keep))
        log(f"  {split_name} {p}/{vp.stem}: {len(keep)} samples")

    # Dataset smaller than the PCA quota: fit on whatever was buffered rather than
    # failing. --pca-fit-frames is a target, not a requirement.
    if pca[0] is None and pca_buf:
        _fit_pca(pca, pca_buf, feats_grid, args, log, short=True)

    if not feats_grid:
        raise RuntimeError(
            f"no samples collected for the {split_name} split — check --video-dir / "
            "--gaze-dir and the participant names, and that the gaze CSVs are extracted"
        )
    Xg = np.concatenate(feats_grid, 0)
    Xp = np.concatenate(feats_pool, 0)
    if len(Xg) != len(Xp):                    # would silently misalign features and targets
        raise RuntimeError(f"internal: grid/pooled feature counts differ "
                           f"({len(Xg)} vs {len(Xp)}) in the {split_name} split")
    return (Xg, Xp, np.concatenate(gaze_t, 0), np.concatenate(gaze_lead, 0), meta)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--gaze-dir", required=True)
    ap.add_argument("--train-participants", nargs="+", default=["P01", "P02", "P03", "P04", "P05"])
    ap.add_argument("--test-participants", nargs="+", default=["P06", "P07"])
    ap.add_argument("--leads", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0],
                    help="anticipation lead times in seconds")
    ap.add_argument("--recordings", type=int, default=6, help="max recordings per participant")
    ap.add_argument("--windows", type=int, default=8, help="sampling windows per recording")
    ap.add_argument("--window-sec", type=float, default=8.0)
    ap.add_argument("--per-window", type=int, default=4)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--pca-fit-frames", type=int, default=300)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--encode-chunk", type=int, default=32)
    ap.add_argument("--tol-ms", type=float, default=50.0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-normalize-reps", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/gaze_recoverability")
    ap.add_argument("--split", choices=["participant", "recording", "random"],
                    default="participant",
                    help="participant = the real experiment (held-out people); "
                         "recording = same people, held-out recordings (isolates "
                         "cross-scene from cross-person generalisation); "
                         "random = leaky control answering 'can this decode gaze at all?'")
    ap.add_argument("--cache", default="results/gaze_features.npz",
                    help="feature cache, shared across analysis variants so re-slicing "
                         "the data does not re-encode")
    ap.add_argument("--recollect", action="store_true",
                    help="rebuild the feature cache instead of reusing it")
    args = ap.parse_args()

    out = Path(args.out)
    log = Logger(out)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log(f"[setup] device={device}  leads={args.leads}")
    log(f"[setup] train={args.train_participants}  test={args.test_participants}")

    # Encoding is the expensive step (~25 min) and every analysis variant reuses the
    # same features, so cache them. Re-slicing the data a different way then costs
    # seconds instead of a re-encode.
    cache = Path(args.cache)
    if cache.exists() and not args.recollect:
        z = np.load(cache, allow_pickle=True)
        Xg_tr, Xp_tr, g0_tr, gl_tr = z["Xg_tr"], z["Xp_tr"], z["g0_tr"], z["gl_tr"]
        Xg_te, Xp_te, g0_te, gl_te = z["Xg_te"], z["Xp_te"], z["g0_te"], z["gl_te"]
        meta_tr, meta_te = z["meta_tr"], z["meta_te"]
        pca_expl = float(z["pca_explained"])
        log(f"[cache] reused features from {cache} (--recollect to rebuild)")
        if list(z["leads"]) != list(args.leads):
            raise SystemExit(f"cached leads {list(z['leads'])} != requested {args.leads}; "
                             "pass --recollect")
    else:
        t0 = time.time()
        encoder, _, _ = load_models(args.checkpoint, device, context_steps=1)
        log(f"[setup] encoder loaded in {time.time() - t0:.0f}s (predictor built but unused)")
        pca, buf = [None], []
        log("[collect] train split")
        Xg_tr, Xp_tr, g0_tr, gl_tr, mtr = collect(args, encoder, device,
                                                  args.train_participants, "train", pca, buf, log)
        log("[collect] test split")
        Xg_te, Xp_te, g0_te, gl_te, mte = collect(args, encoder, device,
                                                  args.test_participants, "test", pca, buf, log)
        # Per-sample "<participant>/<recording>" labels. Needed for the recording-level
        # split, and cheap to keep — omitting them last time cost a full re-encode.
        meta_tr = np.array([f"{p}/{s}" for p, s in mtr])
        meta_te = np.array([f"{p}/{s}" for p, s in mte])
        pca_expl = pca[0].explained
        np.savez(cache, Xg_tr=Xg_tr, Xp_tr=Xp_tr, g0_tr=g0_tr, gl_tr=gl_tr,
                 Xg_te=Xg_te, Xp_te=Xp_te, g0_te=g0_te, gl_te=gl_te,
                 meta_tr=meta_tr, meta_te=meta_te,
                 leads=np.array(args.leads), pca_explained=pca_expl)
        log(f"[cache] wrote {cache}")

    # Diagnostic split. "participant" is the real experiment — held-out people, so the
    # probe cannot memorise one person's gaze habits. "random" pools everything and
    # splits at random, which LEAKS across participants and recordings and is therefore
    # not a result: it is an internal control answering "can this pipeline decode gaze
    # AT ALL?". If random shows skill and participant does not, the finding is about
    # cross-person generalisation. If neither does, suspect the features.
    if args.split != "participant":
        Xg = np.concatenate([Xg_tr, Xg_te]); Xp = np.concatenate([Xp_tr, Xp_te])
        g0 = np.concatenate([g0_tr, g0_te]); gl = np.concatenate([gl_tr, gl_te])
        M = np.concatenate([meta_tr, meta_te])
        rng = np.random.default_rng(args.seed)

        if args.split == "random":
            perm = rng.permutation(len(Xg))
            cut = int(0.75 * len(Xg))
            tr, te = perm[:cut], perm[cut:]
            log("[split] RANDOM — leaks across participants AND within recordings: "
                "samples 5-per-6s-window put near-duplicate frames on both sides. "
                "Internal control only, NOT a reportable result.")
        else:                                     # recording
            recs = np.unique(M)
            rng.shuffle(recs)
            held = set(recs[int(0.75 * len(recs)):])
            te = np.array([i for i, m in enumerate(M) if m in held])
            tr = np.array([i for i, m in enumerate(M) if m not in held])
            log(f"[split] RECORDING — same people, {len(recs) - len(held)} recordings "
                f"train / {len(held)} held out. No within-recording leakage; still leaks "
                "across participants, so it isolates cross-SCENE from cross-PERSON "
                "generalisation. Diagnostic, not the headline result.")

        Xg_tr, Xp_tr, g0_tr, gl_tr = Xg[tr], Xp[tr], g0[tr], gl[tr]
        Xg_te, Xp_te, g0_te, gl_te = Xg[te], Xp[te], g0[te], gl[te]

    log(f"[collect] train n={len(Xg_tr)}  test n={len(Xg_te)}  "
        f"grid dim={Xg_tr.shape[1]}  pooled dim={Xp_tr.shape[1]}")

    def mse_yp(pred, true):
        return float(np.mean((pred[:, :2] - true[:, :2]) ** 2))

    rows = []
    for li, lead in enumerate(args.leads):
        Ytr, Yte = gl_tr[:, li, :], gl_te[:, li, :]

        # The chance floor, computed FIRST because everything else is scored against it.
        #
        # Predicting the TRAIN mean does NOT give R^2 = 0 here. R^2 measures residual
        # variance against the TEST set's own mean, and under a cross-participant split
        # the train mean sits somewhere else entirely — different person, different
        # kitchen, different resting gaze. So R^2 for this baseline is typically
        # NEGATIVE, and strongly so at small n. That is expected, not a bug.
        #
        # Hence `skill`: 1 - MSE_model / MSE_chance. It is 0 when the model does no
        # better than predicting the mean and 1 when perfect, and it is immune to the
        # train/test mean shift that makes raw R^2 unreadable here. Skill is the
        # headline number; R^2 is kept in the CSV for reference.
        chance = np.repeat(Ytr.mean(0, keepdims=True), len(Yte), axis=0)
        mse_chance = mse_yp(chance, Yte)

        def record(name, pred, alpha=np.nan, _Yte=Yte, _mc=mse_chance, _lead=lead):
            ang = angular_error_deg(pred, _Yte)
            rows.append(dict(
                lead_s=_lead, readout=name, alpha=alpha,
                skill=1.0 - mse_yp(pred, _Yte) / max(_mc, 1e-12),
                r2=r2(pred, _Yte), r2_yawpitch=r2(pred[:, :2], _Yte[:, :2]),
                ang_err_deg_median=float(np.median(ang)),
                ang_err_deg_mean=float(np.mean(ang)),
                depth_mae_m=float(np.mean(np.abs(pred[:, 2] - _Yte[:, 2]))),
                n_test=len(_Yte)))

        for name, Xtr, Xte in (("spatial", Xg_tr, Xg_te), ("pooled", Xp_tr, Xp_te)):
            a, _ = ridge_cv(Xtr.astype(np.float64), Ytr.astype(np.float64),
                            args.alphas, args.folds, seed=args.seed)
            record(name, Ridge(a).fit(Xtr.astype(np.float64), Ytr.astype(np.float64))
                                 .predict(Xte.astype(np.float64)), a)

        record("baseline_mean", chance)

        # Persistence: gaze(t+lead) = gaze(t). Uses a PRIVILEGED input the encoder never
        # gets, so it is a reference for "how predictable is gaze at this lead at all",
        # not a competitor. At lead 0 it must score exactly 1.0 / 0.00 degrees — that is
        # the end-to-end check that timestamp alignment and lead pairing are correct.
        record("baseline_persistence", g0_te.astype(np.float64))

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}.csv", index=False)

    log("\n" + "=" * 78)
    log("GAZE RECOVERABILITY — held-out participants, yaw/pitch")
    log("=" * 78)
    log("SKILL over the chance baseline  (0 = no better than predicting the mean, 1 = perfect)")
    log(df.pivot(index="lead_s", columns="readout", values="skill")
          .to_string(float_format=lambda v: f"{v:7.3f}"))
    log("\nMedian angular error, degrees")
    log(df.pivot(index="lead_s", columns="readout", values="ang_err_deg_median")
          .to_string(float_format=lambda v: f"{v:7.2f}"))
    log("\nRaw R^2 (negative for baseline_mean is EXPECTED under a cross-participant split)")
    log(df.pivot(index="lead_s", columns="readout", values="r2_yawpitch")
          .to_string(float_format=lambda v: f"{v:7.3f}"))

    # Sanity gate: persistence at lead 0 predicts gaze(t) from gaze(t). Anything other
    # than a perfect score means timestamp alignment or lead pairing is broken, and no
    # other number in this table can be trusted.
    p0 = df[(df.readout == "baseline_persistence") & (df.lead_s == min(args.leads))]
    if len(p0) and min(args.leads) == 0.0:
        s0 = float(p0.skill.iloc[0])
        log(f"\n[sanity] persistence @ lead 0 skill = {s0:.4f} "
            f"({'OK' if s0 > 0.999 else 'BROKEN — check VRS alignment'})")

    sp = df[df.readout == "spatial"].sort_values("lead_s")
    s_first, s_last, s_best = sp.skill.iloc[0], sp.skill.iloc[-1], sp.skill.max()
    log("\n" + "-" * 78)
    log(f"READING: spatial skill {s_first:+.3f} @ {sp.lead_s.iloc[0]:.2f}s "
        f"-> {s_last:+.3f} @ {sp.lead_s.iloc[-1]:.2f}s   (best {s_best:+.3f})")
    if len(Xg_tr) < 500:
        log(f"  !! n_train = {len(Xg_tr)} — far too few samples to fit "
            f"{Xg_tr.shape[1]} features. Do not interpret these numbers at all.")
        log("  Raise --recordings / --windows / --per-window and re-run.")
    elif s_best < 0.05:
        log("  NO SKILL at any lead — the probe never beats predicting the mean.")
        log("  Gaze is not LINEARLY recoverable from the encoder. Taken at face value this")
        log("  supports the conditioning signal carrying real information, but rule out an")
        log("  underpowered fit first (check n_train, and that alpha is not pinned to the")
        log("  top of the grid).")
    elif s_first - s_last > 0.05:
        log("  Skill FALLS with lead. Gaze carries information that the encoder lacks at")
        log("  longer horizons (about 1 - skill), so the null result at about 2 s may be due")
        log("  to the horizon. Test longer horizons next. See docs/EgoVault/6-next-steps.md.")
    elif s_last - s_first > 0.05:
        log("  Skill RISES with lead — unexpected, and no mechanism predicts it. Suspect")
        log("  noise (check n_test) or leakage before believing it.")
    else:
        log("  Skill is FLAT across lead. Consistent with REDUNDANCY: the encoder already")
        log("  carries future gaze about equally well at every horizon, so a gaze token")
        log("  adds little anywhere. This supports hypothesis R. See")
        log("  docs/EgoVault/1-introduction.md.")
    log("-" * 78)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for name, style in (("spatial", "-o"), ("pooled", "-s"),
                            ("baseline_persistence", "--^"), ("baseline_mean", ":x")):
            d = df[df.readout == name].sort_values("lead_s")
            ax[0].plot(d.lead_s, d.skill, style, label=name)
            ax[1].plot(d.lead_s, d.ang_err_deg_median, style, label=name)
        ax[0].set_xlabel("anticipation lead (s)")
        ax[0].set_ylabel("skill over chance (0 = mean predictor)")
        ax[0].set_title("Gaze recoverability from the frozen encoder")
        ax[0].axhline(0, color="k", lw=0.5); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        ax[1].set_xlabel("anticipation lead (s)"); ax[1].set_ylabel("median angular error (°)")
        ax[1].set_title("Angular error"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{out}.png", dpi=150)
        log(f"[out] {out}.png")
    except Exception as e:                                   # noqa: BLE001
        log(f"[warn] plot skipped: {e}")

    with open(f"{out}.json", "w") as f:
        json.dump(dict(config=vars(args), rows=rows,
                       n_train=len(Xg_tr), n_test=len(Xg_te),
                       pca_explained=pca_expl), f, indent=2, default=str)
    log(f"[out] {out}.csv  {out}.json  {out}.log")
    log.close()


if __name__ == "__main__":
    main()
