"""
Figures of the tests, for the notes (docs/figures/) and the presentation.

Every figure is drawn from the files that the test commands wrote. The one exception is the
error of every patch in Test 9, which the stored scores do not keep: the stage "patches" runs
every Test 9 model once more over the cached encoder features (one GPU, about 30 minutes) and
keeps the errors in results/test9/cache/patches/. The figures that need them are skipped until
then.

Usage
-----
    python -m ego figures                          # every figure whose inputs exist
    python -m ego figures --only test9 test8       # some groups
    python -m ego figures --stage patches \\
        --video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos \\
        --gaze-dir  data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze

Figures (group: file, what it shows)
------------------------------------
    overview: overview_effects.png      the size of each effect on the prediction error, on one axis
    picks:    picks_gaze_on_object.png  the gaze point on the object before a pick
    test3:    t3_probe.png              how well a linear probe reads gaze and palms from one frame
    test4:    t4_sensitivity.png        how far the prediction moves when an input changes
              t4_attention.png          attention to the gaze token in each block
    test8:    t8_accuracy.png           accuracy of naming the next object, by horizon
              t8_differences.png        the paired differences that decided Test 9
    test9:    t9_forms_diagram.png      where the gaze token sits and what it holds, per form
              t9_errors.png             the error of every model and seed
              t9_comparisons.png        the planned comparisons against the seed spread
              t9_training.png           training loss, the reproduction of ego_ft_v2, Δ in training
              t9_distance.png           gain over none by distance from the gaze point (patches)
              t9_gain_map.png           gain over none around the gaze point (patches)
    test11:   t11_training.png          held-out error, Δ within maps against maps − none, map strength
              t11_gains.png             gain of gaze against the matched model, Tests 9, 10 and 11
              t11_decomposition.png     the Δ within a model: value of the signals + cost of hiding them
              t11_references.png        the predictors without signals against the references, by horizon
              t11_distance.png          maps − none by distance from the gaze point and from the nearest palm
              t11_gain_map.png          maps − none and the cost of hiding, around the gaze point

The Test 11 figures read results/test11/ (python -m ego eval-ego); the last two need the error
of every patch (eval-ego --patches, results/test11/cache/patches.npz).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

FORMS = ["none", "angles", "pe", "rope", "pe+rope", "pe+rope shuffled", "future"]
COLORS = {"none": "#7f7f7f", "angles": "#0072B2", "pe": "#D55E00", "rope": "#009E73",
          "pe+rope": "#CC79A7", "pe+rope shuffled": "#B9A3B2", "future": "#000000"}
PATCH_DIR = Path("results/test9/cache/patches")
T = 8


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", name)


# ---------------------------------------------------------------------------
# Stage "patches": the error of every patch, for every Test 9 model (GPU)
# ---------------------------------------------------------------------------

def patch_errors(args):
    import torch
    from ego.commands import gaze_forms as gf
    from ego.runlog import Logger

    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    log = Logger(PATCH_DIR / "patches")
    ns = argparse.Namespace(video_dir=args.video_dir, gaze_dir=args.gaze_dir, participants=["P08", "P09"],
                            checkpoint=args.checkpoint, out="results/test9", device=args.device,
                            runs="checkpoints/test9", sd1p0="checkpoints/ego_sd1p0/best.pt",
                            ft_v2="checkpoints/ego_ft_v2/best.pt")
    clips = gf.test_clips(ns)
    feats, by_rec = gf.encoder_cache(ns, clips, log)
    gaze = np.concatenate([np.stack([c.sig[0][T - 1, 3:5] for c in cl]) for cl in by_rec.values()])
    recs = np.concatenate([[s] * len(cl) for s, cl in by_rec.items()])
    np.savez(PATCH_DIR / "clips.npz", gaze=gaze, recording=recs)
    device = torch.device(args.device)
    for name, form, seed, path, cfg in gf.find_models(ns):
        f = PATCH_DIR / f"{safe(name)}.npz"
        if f.exists() and f.stat().st_mtime > Path(path).stat().st_mtime:
            continue
        s = gf.score_model(path, cfg, form, feats, by_rec, device, patches=True)
        np.savez(f, patch=s["patch"].astype(np.float32), mse=s["mse"], recording=s["recording"],
                 form=form, seed=seed)
        log(f"[patches] {name} ({form}, seed {seed}): mean {s['mse'].mean():.4f}")
    log.close()


def run_dirs():
    """{model name as in the scores: training run folder} of the Test 9 forms."""
    dirs = {d.name: d for d in sorted(Path("checkpoints/test9").glob("*_s[0-9]"))}
    dirs["none_s0 (ego_sd1p0)"] = Path("checkpoints/ego_sd1p0")
    return dirs


def spiked_runs():
    """Runs whose logged gradient norm ever exceeded 1.0, the clipping threshold of training: a loss
    spike. In Test 9 these are two runs; every other run stays at or below 0.2."""
    out = set()
    for name, d in run_dirs().items():
        rows = [json.loads(l) for l in open(d / "metrics.jsonl") if l.strip()]
        if max(r["grad"] for r in rows if r.get("t") == "step") > 1.0:
            out.add(name)
    return out


def comparisons_without(names):
    """The planned comparisons, as gaze_forms computes them, without the runs in `names`."""
    import pandas as pd
    from ego.commands.gaze_forms import COMPARISONS
    from ego.stats import by_recording
    t = pd.read_csv("results/test9/gaze_forms.csv")
    t = t[(t.form != "reference") & ~t.model.isin(names)]
    sc = {m: dict(np.load(Path("results/test9/scores") / f"{safe(m)}.npz")) for m in t.model}
    rows = []
    for measure in ("mse", "near"):
        per = {f: (np.mean([sc[m][measure] for m in g.model], 0), [float(np.nanmean(sc[m][measure])) for m in g.model])
               for f, g in t.groupby("form")}
        recs = sc[t.model.iloc[0]]["recording"]
        for a, b in COMPARISONS:
            d = per[b][0] - per[a][0]
            ok = np.isfinite(d)
            r = by_recording(d[ok], recs[ok])
            rows.append({"measure": measure, "a": a, "b": b, "gain": r["mean"], "ci_lo": r["ci_lo"],
                         "ci_hi": r["ci_hi"], "seed_spread": max(np.std(per[a][1]), np.std(per[b][1]))})
    return pd.DataFrame(rows)


def load_patches(exclude=()):
    """({form: (n, 256) error averaged over seeds}, gaze (n, 2), recordings (n,)) or None."""
    if not (PATCH_DIR / "clips.npz").exists():
        return None
    clips = np.load(PATCH_DIR / "clips.npz")
    forms = {}
    for f in sorted(PATCH_DIR.glob("*.npz")):
        if f.name == "clips.npz":
            continue
        d = np.load(f)
        form = str(d["form"])
        if form == "reference" or f.stem in {safe(n) for n in exclude}:
            continue
        assert (d["recording"] == clips["recording"]).all()
        forms.setdefault(form, []).append(d["patch"])
    if not all(len(forms.get(k, [])) == 3 - sum(n.startswith(k.replace("+", "").replace(" shuffled", "shuf") + "_s")
                                                  for n in exclude) for k in FORMS):
        return None
    return {k: np.mean(v, axis=0) for k, v in forms.items()}, clips["gaze"], clips["recording"]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.6, "savefig.dpi": 200, "savefig.bbox": "tight",
        "legend.frameon": False})
    return plt


def save(plt, fig, out, name):
    fig.savefig(out / name)
    plt.close(fig)
    print(f"[figure] {out / name}")


def interval(values, recordings):
    from ego.stats import by_recording
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    r = by_recording(v[ok], np.asarray(recordings)[ok])
    return r["mean"], r["ci_lo"], r["ci_hi"]


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def fig_overview(plt, out):
    import pandas as pd
    t2 = pd.read_csv("results/rung_b1_contrasts.csv").set_index("contrast")
    sd = pd.read_csv("results/signal_dropout_contrasts.csv").set_index("contrast")
    t9 = pd.read_csv("results/test9/gaze_forms.csv")
    c9 = pd.read_csv("results/test9/gaze_forms_comparisons.csv")
    c9 = c9[c9.measure == "mse"].set_index(["a", "b"])
    ang = t9[t9.form == "angles"].mse

    def comp(a, b):
        r = c9.loc[(a, b)]
        return r.gain, r.ci_lo, r.ci_hi
    rows = [  # label, value, lo, hi, kind
        ("Fine-tuning on kitchen video\n(before − after, Test 2)",
         t2.loc["stock_ac:mask - finetuned:mask", "mean_delta"], None, None, "adaptation"),
        ("Seeds of one recipe: range of the three\nangles runs, one with a loss spike (Test 9)", ang.max() - ang.min(), None, None, "noise"),
        ("Positive control: signals of the\ntarget frame, future − none (Test 9)", *comp("future", "none"), "control"),
        ("ego_ft_v2 with its signals hidden,\nΔ (Test 2)", t2.loc["finetuned:mask - finetuned:real", "mean_delta"],
         None, None, "signals"),
        ("Gaze as pe, pe − none (Test 9)", *comp("pe", "none"), "signals"),
        ("Gaze and hand as angles: ego_sd1p0\nhidden − ego_ft_v2 real (Test 2)",
         sd.loc["sd1.0 masked - ft_v2 real", "mean_delta"], None, None, "signals"),
        ("Gaze as angles, angles − none (Test 9)", *comp("angles", "none"), "signals"),
    ]
    kinds = {"adaptation": ("#0072B2", "adaptation to the video"), "noise": ("#BBBBBB", "difference between training runs"),
             "control": ("#000000", "positive control"), "signals": ("#D55E00", "gaze and hand inputs")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.9), sharey=True, gridspec_kw={"width_ratios": [4, 1], "wspace": 0.05})
    y = np.arange(len(rows))[::-1]
    w = comparisons_without(spiked_runs()).query("measure == 'mse' and a == 'angles' and b == 'none'").iloc[0]
    a1.errorbar(w.gain, y[-1] - 0.3, xerr=[[w.gain - w.ci_lo], [w.ci_hi - w.gain]], fmt="d", ms=4.5,
                color="#555555", elinewidth=0.9, capsize=0)
    a1.text(w.ci_hi + 0.00005, y[-1] - 0.3, f"{w.gain:+.4f} without the angles run with a loss spike",
            va="center", fontsize=7, color="#555555")
    for yi, (label, v, lo, hi, kind) in zip(y, rows):
        for ax in (a1, a2):
            ax.barh(yi, v, color=kinds[kind][0], height=0.6)
        if lo is not None:
            a1.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="none", ecolor="#444444", capsize=2, lw=1)
        if kind == "adaptation":
            a2.text(v, yi, f" {v:.3f}", va="center", fontsize=8)
        else:
            a1.text(max(v, hi or v, 0) + 0.00005, yi + 0.3, f"{v:+.4f}", va="center", fontsize=7.5)
    a1.set_xlim(-0.0008, 0.0030)
    a1.set_xticks([-0.0005, 0, 0.0005, 0.001, 0.0015, 0.002, 0.0025])
    a2.set_xlim(0.1235, 0.1275)
    a2.set_xticks([0.125])
    a1.set_yticks(y, [r[0] for r in rows])
    a1.axvline(0, color="#333333", lw=0.8)
    a1.spines["right"].set_visible(False)
    a2.spines["left"].set_visible(False)
    a2.tick_params(left=False)
    for ax, xs in ((a1, (1,)), (a2, (0,))):          # break marks
        for x in xs:
            ax.plot([x - 0.012, x + 0.012], [-0.02, 0.02], transform=ax.transAxes, color="k", lw=0.8, clip_on=False)
    a1.set_xlabel("decrease of the prediction error (MSE), 0.27 s ahead")
    fig.suptitle("The size of each effect on the prediction error", x=0.55, y=1.0, fontsize=10.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in kinds.values()]
    a1.legend(handles, [l for _, l in kinds.values()], loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=7.5)
    fig.text(0.55, -0.06, "Test 2: 96 P08 clips, one run per model. Test 9: 600 P08 and P09 clips, mean of 3 seeds, "
             "95% intervals over recordings.\nThe error itself is about 0.49–0.50 after fine-tuning and 0.62 before.",
             ha="center", fontsize=7.5, color="#444444")
    save(plt, fig, out, "overview_effects.png")


# ---------------------------------------------------------------------------
# Gaze on the object before a pick
# ---------------------------------------------------------------------------

def fig_picks(plt, out):
    import pandas as pd
    g = pd.read_csv("results/gaze_projection/gaze_at_picks.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
    meas = json.loads(Path("results/gaze_projection/gaze_at_picks.json").read_text())["measures"]
    labels = [("gaze 2 s before the pick", 2), ("gaze 1 s before the pick", 1), ("gaze 0.5 s before the pick", 0.5),
              ("gaze 0.25 s before the pick", 0.25), ("at the pick", 0)]
    xs = [x for _, x in labels]
    m = [meas[k]["inside"]["mean"] for k, _ in labels]
    lo = [meas[k]["inside"]["ci_lo"] for k, _ in labels]
    hi = [meas[k]["inside"]["ci_hi"] for k, _ in labels]
    c = meas["chance (gaze at random times)"]["inside"]
    ch = (c["mean"], c["ci_lo"], c["ci_hi"])
    m, lo, hi = 100 * np.array(m), 100 * np.array(lo), 100 * np.array(hi)
    a1.axhspan(100 * ch[1], 100 * ch[2], color="#BBBBBB", alpha=0.5, lw=0)
    a1.axhline(100 * ch[0], color="#7f7f7f", ls="--", lw=1, label=f"chance: the gaze at random times, {100 * ch[0]:.1f}%")
    a1.fill_between(xs, lo, hi, color="#D55E00", alpha=0.2, lw=0)
    a1.plot(xs, m, "o-", color="#D55E00", label="the gaze point before the pick")
    for x, v in zip(xs, m):
        a1.annotate(f"{v:.1f}", (x, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5)
    a1.set_xlim(2.15, -0.1)
    a1.set_xticks([2, 1, 0.5, 0.25, 0], ["2", "1", "0.5", "0.25", "0"])
    a1.set_xlabel("seconds before the pick")
    a1.set_ylabel("gaze point in the object's box (%)")
    a1.set_ylim(0, 50)
    a1.set_title("Before the pick (the box is from the pick frame)")
    a1.legend(loc="upper left", fontsize=7.5)

    parts = sorted(g.participant.unique())
    at = [100 * (g[g.participant == p].dist == 0).mean() for p in parts]
    chance = [100 * g[g.participant == p].chance_inside.mean() for p in parts]
    x = np.arange(len(parts))
    a2.bar(x - 0.2, at, 0.4, color="#D55E00", label="at the pick")
    a2.bar(x + 0.2, chance, 0.4, color="#BBBBBB", label="chance")
    a2.set_xticks(x, parts)
    a2.set_ylabel("gaze point in the object's box (%)")
    a2.set_title("At the pick, per person")
    a2.legend(loc="upper right")
    a2.set_ylim(0, 60)
    fig.suptitle(f"The projected gaze point and HD-EPIC's pick boxes ({len(g):,} picks, "
                 f"{g.video.nunique()} recordings; 95% intervals over recordings). Before the pick, the "
                 "head and the object\nmay have moved, so the left panel mixes the gaze moving to the object "
                 "with the object moving in the image", fontsize=9.5)
    fig.tight_layout()
    save(plt, fig, out, "picks_gaze_on_object.png")


# ---------------------------------------------------------------------------
# Test 3
# ---------------------------------------------------------------------------

def fig_test3(plt, out):
    import pandas as pd
    splits = [("participant", "results/gaze_recov_participant.csv", "new people", "#D55E00"),
              ("recording", "results/gaze_recov_recording.csv", "new recordings", "#0072B2"),
              ("random", "results/gaze_recov_randomsplit.csv", "random frames", "#7f7f7f")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3))
    for _, path, label, color in splits:
        d = pd.read_csv(path)
        d = d[d.readout == "spatial"].sort_values("lead_s")
        a1.plot(d.lead_s, d.skill, "o-", color=color, label=label)
    a1.axhline(0, color="#333333", lw=0.8, label="0: guessing the average gaze")
    a1.set_xlabel("lead: the gaze this many seconds after the frame")
    a1.set_ylabel("skill (0 = guessing, 1 = perfect)")
    a1.set_title("Gaze read from one frame's features")
    a1.legend()
    c = pd.read_csv("results/control_recoverability.csv")
    c = c[(c.lead_s == 0) & (c.readout == "spatial")]
    targets = [("gaze_yawpitch", "gaze"), ("left_palm_xyz", "left palm"), ("right_palm_xyz", "right palm")]
    x = np.arange(len(targets))
    for i, (split, _, label, color) in enumerate(splits):
        v = [c[(c.split == split) & (c.target == t)].skill.iloc[0] for t, _ in targets]
        a2.bar(x + (i - 1) * 0.27, v, 0.27, color=color, label=label)
        for xi, vi in zip(x + (i - 1) * 0.27, v):
            a2.text(xi, max(vi, 0) + 0.01, f"{vi:.2f}", ha="center", fontsize=6.5)
    a2.set_xticks(x, [l for _, l in targets])
    a2.axhline(0, color="#333333", lw=0.8)
    a2.set_ylabel("skill at lead 0")
    a2.set_title("Control: the same probe reads the palms")
    a2.legend(loc="upper left")
    fig.suptitle("Test 3. Is gaze already in the image features? (ridge probe on the frozen encoder)", fontsize=10)
    fig.tight_layout()
    save(plt, fig, out, "t3_probe.png")


# ---------------------------------------------------------------------------
# Test 4
# ---------------------------------------------------------------------------

def fig_test4(plt, out):
    import pandas as pd
    s = pd.read_csv("results/sensitivity_ego_ft_v2.csv").groupby("condition").rel_last.mean()
    u = pd.read_csv("results/sensitivity_untrained.csv").groupby("condition").rel_last.mean()
    conds = [("gaze_yaw_1deg", "gaze yaw shifted by 1°"), ("gaze_yaw_10deg", "gaze yaw shifted by 10°"),
             ("gaze_yaw_45deg", "gaze yaw shifted by 45°"), ("gaze_swap", "gaze from another clip"),
             ("gaze_mask", "gaze hidden"), ("both_mask", "gaze and hand hidden"),
             ("video_swap", "video from another clip")]
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    y = np.arange(len(conds))[::-1]
    ax.barh(y + 0.18, [u[c] for c, _ in conds], 0.36, color="#BBBBBB", label="before fine-tuning")
    ax.barh(y - 0.18, [s[c] for c, _ in conds], 0.36, color="#0072B2", label="ego_ft_v2")
    for yi, (c, _) in zip(y, conds):
        ax.text(s[c] * 1.08, yi - 0.18, f"{s[c]:.4f}" if s[c] < 0.01 else f"{s[c]:.3f}", va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_yticks(y, [l for _, l in conds])
    ax.set_xlabel("change of the prediction, relative to its size (log scale)")
    ax.set_title("Test 4a. How far the prediction moves when one input changes (96 P08 clips)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", visible=False)
    save(plt, fig, out, "t4_sensitivity.png")

    a = pd.read_csv("results/attention_mass.csv")
    b = pd.read_csv("results/attention_mass_untrained.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    trained = a[(a.variant == "real") & a.trainable].block
    for ax, col, title in ((a1, "gaze_mass", "Mean over heads"),
                           (a2, "gaze_mass_max_head", "The head with the most attention on the gaze token")):
        ax.axvspan(trained.min() - 0.5, trained.max() + 0.5, color="#F0E442", alpha=0.25, lw=0)
        for d, variant, label, color, ls in ((b, "real", "before fine-tuning", "#7f7f7f", "-"),
                                             (a, "real", "ego_ft_v2, real signals", "#0072B2", "-"),
                                             (a, "gaze_masked", "ego_ft_v2, gaze hidden", "#0072B2", ":")):
            r = d[d.variant == variant].sort_values("block")
            ax.plot(r.block, 100 * r[col], ls, color=color, marker="o", ms=2.5, lw=1.2, label=label)
        ax.axhline(100 / 258, color="#D55E00", ls="--", lw=1, label="equal share, 0.39%")
        ax.set_yscale("log")
        ax.set_xlabel("block")
        ax.set_title(title)
    a1.set_ylabel("attention of image tokens\non the gaze tokens (%)")
    a1.text(trained.min() + 2.5, 0.5, "trained\nblocks", ha="center", fontsize=7.5, color="#7a6a00")
    a1.legend(loc="lower left", fontsize=7)
    fig.suptitle("Test 4b. Attention to the gaze token in each block (24 P08 clips)", fontsize=10)
    fig.tight_layout()
    save(plt, fig, out, "t4_attention.png")


# ---------------------------------------------------------------------------
# Test 8
# ---------------------------------------------------------------------------

T8_INPUTS = [("scene", "scene", "#7f7f7f", "o", "-"), ("angles", "+ angles", "#0072B2", "s", "-"),
             ("head point", "+ head point", "#009E73", "^", "-"),
             ("head history 3 s", "+ head history, 3 s", "#009E73", "^", ":"),
             ("gaze point", "+ gaze point", "#D55E00", "o", "-"),
             ("gaze history 3 s", "+ gaze history, 3 s", "#E69F00", "D", ":")]


def fig_test8(plt, out):
    import pandas as pd
    d = pd.read_csv("results/next_object/next_object.csv")
    hs = sorted(d.seconds_before.unique())
    x = np.arange(len(hs))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, m, title in ((axes[0], "top1", "Top-1: the right object"), (axes[1], "top5", "Top-5: among five guesses")):
        f = d[d.input == "frequency guess"].sort_values("seconds_before")
        ax.axhline(100 * f[m].iloc[0], color="#333333", ls="--", lw=1, label="frequency guess")
        for i, (key, label, color, marker, ls) in enumerate(T8_INPUTS):
            r = d[d.input == key].sort_values("seconds_before")
            off = (i - 2.5) * 0.06
            ax.errorbar(x + off, 100 * r[m], yerr=[100 * (r[m] - r[m + "_lo"]), 100 * (r[m + "_hi"] - r[m])],
                        color=color, marker=marker, ms=4, ls=ls, lw=1.2, capsize=0, elinewidth=0.8, label=label)
        ax.set_xticks(x, [f"{h:g} s" for h in hs])
        ax.set_xlabel("time before the pick")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(title)
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")
    n = int(d[d.seconds_before == 0.5].n_test.iloc[0])
    fig.suptitle(f"Test 8. Naming the next object from one frame (P08 and P09, {n:,} picks; "
                 "95% intervals over recordings)", fontsize=10)
    fig.tight_layout()
    save(plt, fig, out, "t8_accuracy.png")

    c = pd.read_csv("results/next_object/next_object_contrasts.csv")
    c = c[c.measure == "top1"]
    pairs = [("gaze point", "head point", "#D55E00", "o", "-"),
             ("gaze point", "angles", "#D55E00", "s", ":"),
             ("gaze history 3 s", "head history 3 s", "#E69F00", "o", "-"),
             ("gaze history 3 s", "angles", "#E69F00", "s", ":"),
             ("head point", "scene", "#009E73", "^", "-"),
             ("angles", "scene", "#0072B2", "^", "-")]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.axvspan(-0.4, 1.4, color="#F0E442", alpha=0.2, lw=0)
    ax.text(0.5, 8.3, "decision horizons", ha="center", fontsize=7.5, color="#7a6a00")
    ax.axhline(0, color="#333333", lw=0.8)
    for i, (a, b, color, marker, ls) in enumerate(pairs):
        r = c[(c.a == a) & (c.b == b)].sort_values("seconds_before")
        off = (i - 2.5) * 0.07
        ax.errorbar(x + off, 100 * r["diff"], yerr=[100 * (r["diff"] - r.lo95), 100 * (r.hi95 - r["diff"])],
                    color=color, marker=marker, ms=4, ls=ls, lw=1.1, elinewidth=0.8, label=f"{a} − {b}")
    ax.set_xticks(x, [f"{h:g} s" for h in hs])
    ax.set_xlabel("time before the pick")
    ax.set_ylabel("difference in top-1 (points)")
    ax.set_ylim(-2.5, 9)
    ax.set_title("Test 8. Paired differences in top-1, with 95% intervals over recordings")
    ax.legend(fontsize=7, loc="upper right")
    save(plt, fig, out, "t8_differences.png")


# ---------------------------------------------------------------------------
# Test 9
# ---------------------------------------------------------------------------

def fig_forms_diagram(plt, out):
    from matplotlib.patches import Rectangle
    gc, gr = 10.4, 6.6                                                     # an example gaze point
    forms = [("angles", "yaw, pitch, depth", False), ("pe", "sin and cos of the point\nat 5 frequencies, depth, flag", False),
             ("rope", "yaw, pitch, depth", True), ("pe+rope", "sin and cos of the point\nat 5 frequencies, depth, flag", True)]
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.4))
    for ax, (name, content, at_point) in zip(axes, forms):
        ax.set_xlim(0, 16); ax.set_ylim(16, 0)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for k in range(17):
            ax.axhline(k, color="#DDDDDD", lw=0.5); ax.axvline(k, color="#DDDDDD", lw=0.5)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#999999")
        ax.plot(gc, gr, "+", color="#D55E00", ms=12, mew=2)
        ax.text(gc + 0.6, gr - 0.6, "gaze point", color="#D55E00", fontsize=7.5)
        ax.add_patch(Rectangle((0, 0), 1, 1, color="#999999", zorder=2))
        tc, tr = (gc - 0.5, gr - 0.5) if at_point else (0, 0)
        ax.add_patch(Rectangle((tc, tr), 1, 1, fill=False, ec=COLORS[name], lw=2.2, zorder=4))
        lx, ly = (tc + 1.3, tr + 3.2) if at_point else (1.5, 2.2)
        ax.annotate("gaze token", (tc + 1, tr + 1), (lx, ly), fontsize=7.5, color=COLORS[name],
                    arrowprops=dict(arrowstyle="-", color=COLORS[name], lw=0.8))
        ax.text(1.4, 0.8, "hand token", fontsize=6.5, color="#7f7f7f", va="center")
        ax.set_title(name, color=COLORS[name], fontweight="bold")
        where = "at the gaze point" if at_point else "top-left patch"
        ax.set_xlabel(f"content: {content}\nposition in the attention: {where}", fontsize=7.5)
    fig.suptitle("Test 9. The four gaze forms on the 16 × 16 patch grid. Content: what the gaze token holds. "
                 "Position: where RoPE places it", fontsize=9.5)
    fig.tight_layout()
    save(plt, fig, out, "t9_forms_diagram.png")


def fig_test9(plt, out):
    import pandas as pd
    t = pd.read_csv("results/test9/gaze_forms.csv")
    c = pd.read_csv("results/test9/gaze_forms_comparisons.csv")
    spiked = spiked_runs()
    c2 = comparisons_without(spiked)
    c2.to_csv("results/test9/gaze_forms_comparisons_without_spikes.csv", index=False)

    # --- every model and seed ---
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    rng = np.random.default_rng(0)
    for ax, m, title in ((axes[0], "mse", "Whole frame"), (axes[1], "near", "Within 2 patches of the gaze point")):
        none_mean = t[t.form == "none"][m].mean()
        ax.axhline(none_mean, color=COLORS["none"], ls="--", lw=0.9)
        for i, f in enumerate(FORMS):
            r = t[t.form == f]
            v = r[m].to_numpy()
            x = i + rng.uniform(-0.12, 0.12, len(v))
            ax.scatter(x, v, s=22, color=COLORS[f], alpha=0.85, lw=0)
            sp = r.model.isin(spiked).to_numpy()
            ax.scatter(x[sp], v[sp], s=70, facecolor="none", edgecolor="#333333", lw=0.9)
            ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=COLORS[f], lw=2.2)
        ref = t[t.form == "reference"][m].iloc[0]
        ax.scatter(1.32, ref, marker="D", s=22, facecolor="none", edgecolor=COLORS["angles"], lw=1)
        ax.set_xticks(range(len(FORMS)), [f.replace(" shuffled", "\nshuffled") for f in FORMS])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("error 0.27 s ahead (MSE), lower is better")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, mfc="none", mec="#333333", label="run with a loss spike at the start"),
               plt.Line2D([], [], marker="D", ls="", ms=5, mfc="none", mec=COLORS["angles"], label="ego_ft_v2 (angles, earlier run)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Test 9. The error of every model: dots are seeds, bars their mean, the dashed line the mean "
                 "of none (600 P08 and P09 clips)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(plt, fig, out, "t9_errors.png")

    # --- the planned comparisons ---
    order = [("pe+rope", "angles"), ("rope", "angles"), ("pe", "angles"), ("angles", "none"), ("pe", "none"),
             ("rope", "none"), ("pe+rope", "none"), ("rope", "pe"), ("pe+rope", "pe+rope shuffled"), ("future", "none")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, m, title in ((axes[0], "mse", "Whole frame"), (axes[1], "near", "Within 2 patches of the gaze point")):
        ax.axvline(0, color="#333333", lw=0.8)
        for yi, (a, b) in zip(y, order):
            r = c[(c.measure == m) & (c.a == a) & (c.b == b)].iloc[0]
            g, lo, hi, sp = 1e3 * r.gain, 1e3 * r.ci_lo, 1e3 * r.ci_hi, 1e3 * r.seed_spread
            counts = (r.ci_lo > 0 or r.ci_hi < 0) and abs(r.gain) > r.seed_spread
            ax.add_patch(plt.Rectangle((-sp, yi - 0.3), 2 * sp, 0.6, color="#DDDDDD", lw=0))
            ax.plot([lo, hi], [yi, yi], color=COLORS[a], lw=1.6)
            ax.plot(g, yi, "o", ms=6, color=COLORS[a], mfc=COLORS[a] if counts else "white", mew=1.4)
            w = c2[(c2.measure == m) & (c2.a == a) & (c2.b == b)].iloc[0]
            if abs(w.gain - r.gain) > 1e-5:
                ax.plot([1e3 * w.ci_lo, 1e3 * w.ci_hi], [yi - 0.27] * 2, color="#555555", lw=0.9)
                ax.plot(1e3 * w.gain, yi - 0.27, "d", ms=4.5, color="#555555")
        for sep in (6.5, 2.5, 1.5, 0.5):
            ax.axhline(sep, color="#EEEEEE", lw=0.8)
        ax.set_xlabel("gain of A over B (10⁻³ MSE); > 0: A predicts better")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f"{a} − {b}" for a, b in order])
    axes[0].set_ylim(-0.7, len(order) - 0.3)
    handles = [plt.Line2D([], [], marker="o", ls="", color="#333333", ms=6, label="counts: interval excludes 0 and gain > seed spread"),
               plt.Line2D([], [], marker="o", ls="", color="#333333", mfc="white", ms=6, label="does not count"),
               plt.Rectangle((0, 0), 1, 1, color="#DDDDDD", label="± seed spread"),
               plt.Line2D([], [], marker="d", color="#555555", ms=5, lw=0.9,
                          label=f"without the {len(spiked)} runs with a loss spike (analysis after the run)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.08))
    fig.suptitle("Test 9. The planned comparisons, with 95% intervals over the 25 test recordings", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(plt, fig, out, "t9_comparisons.png")

    # --- training ---
    def val(run):
        rows = [json.loads(l) for l in open(Path(run) / "metrics.jsonl") if l.strip()]
        return [r for r in rows if r.get("t") == "val"]

    def epoch_loss(run):
        txt = (Path(run) / "train.log").read_text()
        return [float(v) for v in re.findall(r"\[epoch\s+\d+/\d+\]\s+loss=([0-9.]+)", txt)]
    runs = sorted(Path("checkpoints/test9").glob("*_s[0-9]"))
    form_of = {"angles": "angles", "future": "future", "none": "none", "pe": "pe", "rope": "rope",
               "perope": "pe+rope", "peropeshuf": "pe+rope shuffled"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    ax = axes[0]
    for run, label, ls in (("checkpoints/test9/angles_s0", "angles, seed 0 (new code)", "-"),
                           ("checkpoints/ego_ft_v2", "ego_ft_v2", ":")):
        v = val(run)
        e = [r["epoch"] for r in v]
        ax.plot(e, [r["mse_A"] for r in v], ls, marker="o", ms=3, color="#7f7f7f", label=f"{label}, hidden")
        ax.plot(e, [r["mse_B"] for r in v], ls, marker="s", ms=3, color=COLORS["angles"], label=f"{label}, real")
    ax.set_xlabel("epoch"); ax.set_ylabel("error on 96 P08 clips")
    ax.set_title("Reproduction of ego_ft_v2")
    ax.set_xticks([0, 1, 2, 3])
    ax.legend(fontsize=6.5)
    by_form = {}
    for run in runs:
        f = form_of[run.name.rsplit("_s", 1)[0]]
        by_form.setdefault(f, []).append((epoch_loss(run), val(run)))
        rows = [json.loads(l) for l in open(run / "metrics.jsonl") if l.strip()]
        st = [r for r in rows if r.get("t") == "step" and r["epoch"] == 1]
        hot = run.name in spiked
        axes[1].plot([r["step"] for r in st], [r["loss"] for r in st], color=COLORS[f] if hot else "#CCCCCC",
                     lw=1.4 if hot else 0.7, zorder=3 if hot else 1, label=f"{run.name} (loss spike)" if hot else None)
    for f in FORMS:
        if f in by_form and f != "none":
            ok = [v for (_, v), run in zip(by_form[f], [r for r in runs if form_of[r.name.rsplit("_s", 1)[0]] == f])
                  if run.name not in spiked]
            d = np.array([[r["delta"] for r in v] for v in ok])
            axes[2].plot(range(d.shape[1]), d.mean(0), "o-", ms=3, color=COLORS[f], label=f, lw=1.2)
    axes[1].plot([], [], color="#CCCCCC", lw=0.7, label="the other 18 runs")
    axes[1].set_xlabel("step of the first epoch"); axes[1].set_ylabel("training loss (one batch)")
    axes[1].set_title("Training loss in the first epoch")
    axes[1].legend(fontsize=6.5)
    axes[2].axhline(0, color="#333333", lw=0.8)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("Δ = hidden − real, mean of seeds")
    axes[2].set_title("Δ during training (96 P08 clips)")
    axes[2].set_xticks([0, 1, 2, 3])
    axes[2].legend(fontsize=6, ncol=2)
    fig.suptitle("Test 9. Training. Δ: mean of seeds without the runs with a loss spike; none is left out, "
                 "because it was never trained with signals", fontsize=9.5)
    fig.tight_layout()
    save(plt, fig, out, "t9_training.png")


def fig_test9_patches(plt, out):
    spiked = spiked_runs()
    loaded = load_patches(exclude=spiked)
    if loaded is None:
        print("[skip] t9_distance.png, t9_gain_map.png: run --stage patches first")
        return
    err, gaze, recs = loaded
    c = np.arange(16) + 0.5
    pc, pr = np.tile(c, 16)[None], np.repeat(c, 16)[None]                  # row-major patch centres
    dc, dr = pc - gaze[:, :1], pr - gaze[:, 1:]                            # (n, 256), in patches
    dist = np.hypot(dc, dr)

    # --- by distance ---
    edges = [0, 1, 2, 3, 4, 6, 8, 12, 23]
    mids = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.5), gridspec_kw={"width_ratios": [1, 1.5]})

    def ring_means(v):
        out = np.full((len(v), len(mids)), np.nan)
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            m = (dist >= a) & (dist < b)
            out[:, k] = np.where(m.any(1), (v * m).sum(1) / np.maximum(m.sum(1), 1), np.nan)
        return out
    e_none = ring_means(err["none"])
    st = [interval(e_none[:, k], recs) for k in range(len(mids))]
    a1.fill_between(mids, [s[1] for s in st], [s[2] for s in st], color=COLORS["none"], alpha=0.25, lw=0)
    a1.plot(mids, [s[0] for s in st], "o-", color=COLORS["none"], ms=3.5)
    a1.set_xlabel("distance from the gaze point (patches)")
    a1.set_ylabel("error of none (MSE)")
    a1.set_title("Where the error is")
    a2.axhline(0, color="#333333", lw=0.8)
    a2.axvspan(0, 2, color="#F0E442", alpha=0.2, lw=0)
    a2.text(1, 0.97, "near", transform=a2.get_xaxis_transform(), fontsize=7, color="#7a6a00", va="top", ha="center")
    for i, f in enumerate([f for f in FORMS if f != "none"]):
        g = ring_means(err["none"] - err[f])
        st = [interval(g[:, k], recs) for k in range(len(mids))]
        x = np.array(mids) + (i - 2.5) * 0.12
        a2.errorbar(x, [1e3 * s[0] for s in st], yerr=[[1e3 * (s[0] - s[1]) for s in st], [1e3 * (s[2] - s[0]) for s in st]],
                    color=COLORS[f], marker="o", ms=3.5, lw=1.2, elinewidth=0.7, label=f)
    a2.set_xlabel("distance from the gaze point (patches)")
    a2.set_ylabel("gain over none (10⁻³ MSE)")
    a2.set_title("Gain over none, by distance")
    a2.legend(fontsize=7, ncol=2)
    fig.suptitle("Test 9. The error of the next step around the gaze point of the last context frame. Mean of "
                 f"seeds without the {len(spiked)} runs with a loss spike; 95% intervals over recordings", fontsize=9.5)
    fig.tight_layout()
    save(plt, fig, out, "t9_distance.png")

    # --- around the gaze point ---
    R = 6
    bc, br = np.rint(dc).astype(int), np.rint(dr).astype(int)
    inside = (np.abs(bc) <= R) & (np.abs(br) <= R)
    idx = ((br + R) * (2 * R + 1) + (bc + R))[inside]
    count = np.bincount(idx, minlength=(2 * R + 1) ** 2).reshape(2 * R + 1, 2 * R + 1)

    def grid(v):
        s = np.bincount(idx, weights=v[inside], minlength=(2 * R + 1) ** 2).reshape(2 * R + 1, 2 * R + 1)
        g = s / np.maximum(count, 1)
        g[count < 300] = np.nan
        return g
    panels = [("future", "future − none"), ("pe", "pe − none"), ("rope", "rope − none"), ("angles", "angles − none")]
    gains = [1e3 * grid(err["none"] - err[f]) for f, _ in panels]
    vmax = np.nanpercentile(np.abs(np.stack(gains)), 98)
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), layout="constrained")
    ext = (-R - 0.5, R + 0.5, R + 0.5, -R - 0.5)
    im0 = axes[0].imshow(grid(err["none"]), extent=ext, cmap="viridis")
    axes[0].set_title("error of none (MSE)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, location="left", pad=0.02)
    for ax, g, (_, title) in zip(axes[1:], gains, panels):
        im = ax.imshow(g, extent=ext, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
    fig.colorbar(im, ax=list(axes[1:]), shrink=0.8, pad=0.01, label="gain (10⁻³ MSE); blue: better")
    for ax in axes:
        ax.grid(False)
        ax.plot(0, 0, "+", color="#D55E00" if ax is not axes[0] else "white", ms=9, mew=1.6)
        ax.add_patch(plt.Circle((0, 0), 2, fill=False, ls="--", lw=0.8, color="#333333" if ax is not axes[0] else "white"))
        ax.set_xticks([-6, -3, 0, 3, 6]); ax.set_yticks([-6, -3, 0, 3, 6])
        ax.set_xlabel("columns from the gaze point")
    axes[0].set_ylabel("rows from the gaze point")
    fig.suptitle("Test 9. Error and gain around the gaze point (+); the dashed circle is the 'near' region.\n"
                 f"Mean of seeds without the {len(spiked)} runs with a loss spike, 600 clips; cells with fewer than "
                 "300 patches are left blank", fontsize=9.5)
    save(plt, fig, out, "t9_gain_map.png")


# ---------------------------------------------------------------------------
# Test 11
# ---------------------------------------------------------------------------

T11 = {"maps": "#CC79A7", "none": "#7f7f7f"}
AC = "#0072B2"                                  # the V-JEPA 2-AC predictor of Tests 9 and 10
REF = {"blend of past frames": ("#555555", "--"), "repeat last frame": ("#AAAAAA", ":")}
T11_DIR = Path("results/test11")
T11_RUNS = Path("checkpoints/test11")


def t11_scores():
    if not (T11_DIR / "scores.npz").exists():
        return None
    return dict(np.load(T11_DIR / "scores.npz"))


def t11_gain(z, a, b, m, h):
    """Gain of a over b (error of b minus error of a) at horizon h: mean and 95% interval."""
    return interval(z[f"{b}|{m}"][:, h] - z[f"{a}|{m}"][:, h], z["recording"])


def t11_rows(run):
    rows = [json.loads(l) for l in open(T11_RUNS / run / "metrics.jsonl") if l.strip()]
    return [r for r in rows if r["t"] == "step"], [r for r in rows if r["t"] == "epoch"]


def fig_test11(plt, out):
    import pandas as pd
    z = t11_scores()
    if z is None:
        print("[skip] Test 11 figures: run `python -m ego eval-ego` first")
        return
    hz = [float(h) for h in z["horizons"]]

    # --- training ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    runs = {arm: t11_rows(f"{arm}_s0") for arm in ("maps", "none")}
    ax = axes[0]
    for arm, ls in (("none", "--"), ("maps", "-")):
        ep = runs[arm][1]
        ax.plot([r["epoch"] for r in ep], [r["val_score"] for r in ep], ls, marker="o", ms=3,
                color=T11[arm], lw=1.4 if arm != "none" else 1.8)
    ax.text(13.3, runs["maps"][1][-1]["val_score"], "maps and none\n(equal to 4 digits)", fontsize=7, va="center")
    ax.set_xlim(0.5, 17)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13])
    ax.set_xlabel("epoch"); ax.set_ylabel("held-out error (L1), lower is better")
    ax.set_title("Held-out error after each epoch", loc="left")

    ax = axes[1]
    within = [1e4 * np.mean([v[0][2] - v[0][3], v[1][2] - v[1][3]]) for v in (r["val"] for r in runs["maps"][1])]
    matched = [1e4 * np.mean([n[0][3] - m[0][3], n[1][3] - m[1][3]])
               for n, m in zip((r["val"] for r in runs["none"][1]), (r["val"] for r in runs["maps"][1]))]
    e = range(1, len(within) + 1)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.plot(e, within, "o-", ms=3.5, color=T11["maps"], lw=1.6)
    ax.plot(e, matched, "s--", ms=3.5, mfc="white", color="#333333", lw=1.2)
    ax.text(13.3, within[-1], "within maps:\nhidden − real", fontsize=7, va="center", color="#333333")
    ax.text(13.3, matched[-1], "maps − none", fontsize=7, va="center", color="#333333")
    ax.set_xlim(0.5, 17)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13])
    ax.set_xlabel("epoch"); ax.set_ylabel("gain (10⁻⁴ MSE), > 0: the points help")
    ax.set_title("The Δ within maps rises; maps − none does not", loc="left")

    ax = axes[2]
    spe = 327
    st = runs["maps"][0]
    x = [(r["epoch"] - 1) + r["step"] / spe for r in st]
    for sig, ls, label in (("gaze", "-", "gaze"), ("left", "--", "left palm"), ("right", ":", "right palm")):
        ax.plot(x, [r["signal_strength"][sig] for r in st], ls, color=T11["maps"], lw=1.5)
        ax.text(x[-1] + 0.2, st[-1]["signal_strength"][sig], label, fontsize=7, va="center")
    ax.set_xlim(0, 15.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("α‖e‖ (0 for none throughout)")
    ax.set_title("How strongly maps adds each map", loc="left")
    fig.suptitle("Test 11. Training of maps and none, seed 0, which start from the same weights and see the same "
                 "clips. Held-out: the 96 P08 clips of the check after each epoch, mean of the two horizons",
                 fontsize=9)
    fig.tight_layout()
    save(plt, fig, out, "t11_training.png")

    # --- gain of gaze against the matched model, Tests 9, 10 and 11 ---
    c10 = pd.read_csv("results/test10/gaze_forms_comparisons.csv")

    def old(a, b, m):
        r = c10[(c10.measure == m) & (c10.a == a) & (c10.b == b)].iloc[0]
        return r.gain, r.ci_lo, r.ci_hi, r.seed_spread
    measures = [("l1", "l1", "L1, whole frame"), ("near_l1", "near_l1", "L1, near the gaze point"),
                ("mse", "mse", "MSE, whole frame"), ("near", "near_mse", "MSE, near the gaze point")]
    rows = [("Test 9: pe − none, 0.27 s", COLORS["pe"], "o", lambda m, m11: old("pe", "none", m)),
            ("Test 10: pe l1 − none l1, 0.27 s", COLORS["pe"], "s", lambda m, m11: old("pe l1", "none l1", m)),
            ("Test 10: pe l1 full − none l1 full, 0.27 s", COLORS["pe"], "^", lambda m, m11: old("pe l1 full", "none l1 full", m)),
            ("Test 11: maps − none, 0.53 s", T11["maps"], "o", lambda m, m11: (*t11_gain(z, "maps_s0|present", "none_s0|present", m11, 0), np.nan)),
            ("Test 11: maps − none, 1.07 s", T11["maps"], "D", lambda m, m11: (*t11_gain(z, "maps_s0|present", "none_s0|present", m11, 1), np.nan)),
            ("Test 9: future − none, 0.27 s\n(positive control)", "#000000", "o", lambda m, m11: old("future", "none", m))]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (m, m11, title) in zip(axes, measures):
        ax.axvline(0, color="#333333", lw=0.8)
        for yi, (label, col, mk, get) in zip(y, rows):
            g, lo, hi, sp = (1e4 * v for v in get(m, m11))
            if np.isfinite(sp):
                ax.add_patch(plt.Rectangle((-sp, yi - 0.3), 2 * sp, 0.6, color="#E4E4E4", lw=0))
            ax.plot([lo, hi], [yi, yi], color=col, lw=1.6)
            ax.plot(g, yi, mk, ms=6, color=col)
        ax.axhline(2.5, color="#EEEEEE", lw=0.8)
        ax.axhline(0.5, color="#EEEEEE", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("gain (10⁻⁴); > 0: gaze helps")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [r[0] for r in rows])
    axes[0].set_ylim(-0.7, len(rows) - 0.3)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#E4E4E4", label="± spread between seeds (Tests 9, 10: 3 seeds)"),
                        plt.Line2D([], [], color="#333333", lw=1.6, label="95% interval over the 25 test recordings")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.06))
    fig.suptitle("Test 11 against Tests 9 and 10. The gain of gaze against the matched model, on four measures. "
                 "Test 11: one seed, so no seed spread;\nits clips are not those of Tests 9 and 10, but come from "
                 "the same 25 recordings of P08 and P09", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(plt, fig, out, "t11_gains.png")

    # --- the Δ within a model: value of the signals + cost of hiding them ---
    t2 = pd.read_csv("results/rung_b1_contrasts.csv").set_index("contrast")
    sd = pd.read_csv("results/signal_dropout_contrasts.csv").set_index("contrast")
    t10 = pd.read_csv("results/test10/gaze_forms.csv")

    def old_within(form, m):
        r = t10[t10.form == form]
        col = {"mse": ("mse_hide", "mse"), "near": ("near_hide", "near")}[m]
        return float((r[col[0]] - r[col[1]]).mean())

    def t11_within(m, h):
        return float(np.nanmean(z[f"maps_s0|hidden|{m}"][:, h] - z[f"maps_s0|present|{m}"][:, h]))
    dec = [  # label, color, {measure: (within, value)}
        ("Test 2: ego_ft_v2, angles, 0.27 s\n(96 P08 clips, one run)", "#0072B2",
         {"mse": (t2.loc["finetuned:mask - finetuned:real", "mean_delta"], sd.loc["sd1.0 masked - ft_v2 real", "mean_delta"])}),
        ("Test 9: pe, 0.27 s", COLORS["pe"], {m: (old_within("pe", m), old("pe", "none", m)[0]) for m in ("mse", "near")}),
        ("Test 10: pe l1, 0.27 s", COLORS["pe"], {m: (old_within("pe l1", m), old("pe l1", "none l1", m)[0]) for m in ("mse", "near")}),
        ("Test 11: maps, 0.53 s", T11["maps"], {m: (t11_within(m11, 0), t11_gain(z, "maps_s0|present", "none_s0|present", m11, 0)[0])
                                               for m, m11 in (("mse", "mse"), ("near", "near_mse"))}),
        ("Test 11: maps, 1.07 s", T11["maps"], {m: (t11_within(m11, 1), t11_gain(z, "maps_s0|present", "none_s0|present", m11, 1)[0])
                                               for m, m11 in (("mse", "mse"), ("near", "near_mse"))}),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    y = np.arange(len(dec))[::-1]
    for ax, m, title in ((axes[0], "mse", "MSE, whole frame"), (axes[1], "near", "MSE, near the gaze point")):
        ax.axvline(0, color="#333333", lw=0.8)
        for yi, (label, col, vals) in zip(y, dec):
            if m not in vals:
                ax.text(0.5, yi, "not measured", fontsize=7, color="#888888", va="center")
                continue
            w, v = (1e4 * x for x in vals[m])
            ax.plot([v, w], [yi, yi], color="#BBBBBB", lw=3, solid_capstyle="butt", zorder=1)
            ax.plot(w, yi, "o", ms=7, mfc="white", mec=col, mew=1.6, zorder=3)
            ax.plot(v, yi, "o", ms=7, color=col, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("10⁻⁴ MSE")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [d[0] for d in dec])
    axes[0].set_ylim(-0.6, len(dec) - 0.4)
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=7, mfc="white", mec="#333333", label="Δ within the model: signals hidden − real"),
                        plt.Line2D([], [], marker="o", ls="", ms=7, color="#333333", label="value of the signals: matched model − model with signals"),
                        plt.Line2D([], [], color="#BBBBBB", lw=3, label="the difference: the cost of hiding an input the model expects")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.08), fontsize=7.5)
    fig.suptitle("The Δ within a model overstates the value of the signals, in every test that has a matched model",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(plt, fig, out, "t11_decomposition.png")

    # --- the predictors without signals against the references, by horizon ---
    t9 = pd.read_csv("results/test10/gaze_forms.csv")
    r9 = pd.read_csv("results/test10/gaze_forms_references.csv").set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    for ax, key, ylab in ((axes[0], "corr", "correlation of predicted and true tokens\n(1 − MSE/2), higher is better"),
                          (axes[1], "l1", "error (L1), lower is better")):
        get = (lambda v: 1 - v / 2) if key == "corr" else (lambda v: v)
        m11 = "mse" if key == "corr" else "l1"
        for name, (col, ls) in REF.items():
            ys = [get(r9.loc[name, m11])] + [get(np.mean(z[f"{name}|-|{m11}"][:, h])) for h in (0, 1)]
            ax.plot([0.27] + hz, ys, ls, marker="o", ms=3.5, color=col, lw=1.3)
            ax.text(1.1, ys[-1], name, fontsize=7, va="center", color="#333333")
        for form, mk, label in (("none", "o", "MSE loss, last 6 blocks (Test 9)"), ("none l1", "s", "L1, last 6 blocks (Test 10)"),
                                ("none l1 full", "^", "L1, whole predictor (Test 10)")):
            v = t9[t9.form == form][m11].mean()
            ax.plot(0.27, get(v), mk, ms=6.5, color=AC, label=f"V-JEPA 2-AC, {label}", ls="")
        ys = [get(np.mean(z[f"none_s0|present|{m11}"][:, h])) for h in (0, 1)]
        ax.plot(hz, ys, "D-", ms=5, color=T11["maps"], lw=1.6, label="Test 11 predictor (none), L1, from the start")
        ax.set_xticks([0.27] + hz, ["0.27 s\n(Test 9 clips)", "0.53 s", "1.07 s"])
        ax.set_xlim(0.15, 1.45)
        ax.set_ylabel(ylab)
        ax.set_xlabel("horizon")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), fontsize=7.5)
    fig.suptitle("The predictors without signals against the references. At 0.27 s on the 600 clips of Tests 9 and 10; "
                 "at 0.53 s and 1.07 s on the 600 clips of Test 11 (same recordings)", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(plt, fig, out, "t11_references.png")


def fig_test11_patches(plt, out):
    f = T11_DIR / "cache" / "patches.npz"
    if not f.exists():
        print("[skip] t11_distance.png, t11_gain_map.png: run `python -m ego eval-ego --patches` first")
        return
    d = dict(np.load(f))
    recs, hz = d["recording"], [float(h) for h in d["horizons"]]
    gaze = d["points|present|gaze"]                                       # (n, 2) column, row
    hands, hval = d["points|present|hands"], d["points|present|hands_valid"]  # (n, 2, 2), (n, 2)
    c = np.arange(16) + 0.5
    pc, pr = np.tile(c, 16)[None], np.repeat(c, 16)[None]                  # row-major patch centres
    dc, dr = pc - gaze[:, :1], pr - gaze[:, 1:]
    dist = np.hypot(dc, dr)
    hd = np.stack([np.where(hval[:, k:k + 1], np.hypot(pc - hands[:, k, :1], pr - hands[:, k, 1:]), np.inf)
                   for k in (0, 1)]).min(0)                               # (n, 256) to the nearest palm
    has_hand = hval.any(1)
    none, maps, hide = d["none_s0|present"], d["maps_s0|present"], d["maps_s0|hidden"]   # (n, 2, 256)

    edges = [0, 1, 2, 3, 4, 6, 8, 12, 23]
    mids = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]

    def ring(v, dd):
        o = np.full((len(v), len(mids)), np.nan)
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            m = (dd >= a) & (dd < b)
            o[:, k] = np.where(m.any(1), (v * m).sum(1) / np.maximum(m.sum(1), 1), np.nan)
        return o

    def curve(ax, v, dd, rows, col, ls, mk, shift):
        r = ring(v[rows], dd[rows])
        st = [interval(r[:, k], recs[rows]) for k in range(len(mids))]
        x = np.array(mids) + shift
        ax.errorbar(x, [1e4 * s[0] for s in st], yerr=[[1e4 * (s[0] - s[1]) for s in st], [1e4 * (s[2] - s[0]) for s in st]],
                    color=col, ls=ls, marker=mk, ms=4, lw=1.3, elinewidth=0.7, mfc=col if mk != "D" else "white")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), gridspec_kw={"width_ratios": [1, 1.3, 1.3]})
    allc = np.ones(len(recs), bool)
    for h, ls, mk in ((0, "-", "o"), (1, "--", "D")):
        r = ring(none[:, h], dist)
        st = [interval(r[:, k], recs) for k in range(len(mids))]
        axes[0].plot(mids, [s[0] for s in st], ls, marker=mk, ms=3.5, color=T11["none"], mfc="white" if mk == "D" else T11["none"])
        curve(axes[1], none[:, h] - maps[:, h], dist, allc, T11["maps"], ls, mk, (h - 0.5) * 0.15)
        curve(axes[2], none[:, h] - maps[:, h], hd, has_hand, T11["maps"], ls, mk, (h - 0.5) * 0.15)
    axes[0].set_ylabel("error of none (MSE)")
    axes[0].set_title("Where the error is", loc="left")
    for ax, title in ((axes[1], "maps − none, by distance from the gaze point"),
                      (axes[2], f"maps − none, by distance from the nearest palm\n({has_hand.sum()} clips with a palm)")):
        ax.axhline(0, color="#333333", lw=0.8)
        ax.axvspan(0, 2, color="#F0E442", alpha=0.2, lw=0)
        ax.set_ylabel("gain (10⁻⁴ MSE), > 0: maps better")
        ax.set_title(title, loc="left")
    for ax, xl in zip(axes, ("distance from the gaze point (patches)", "distance from the gaze point (patches)",
                             "distance from the nearest palm (patches)")):
        ax.set_xlabel(xl)
    axes[0].legend(handles=[plt.Line2D([], [], color="#555555", ls="-", marker="o", ms=4, label=f"{hz[0]:.2f} s"),
                            plt.Line2D([], [], color="#555555", ls="--", marker="D", ms=4, mfc="white", label=f"{hz[1]:.2f} s")],
                   fontsize=7)
    fig.suptitle("Test 11. Where in the frame the maps change the prediction. Points of the last context frame; "
                 "600 clips, seed 0; 95% intervals over recordings", fontsize=9.5)
    fig.tight_layout()
    save(plt, fig, out, "t11_distance.png")

    R = 6
    bc, br = np.rint(dc).astype(int), np.rint(dr).astype(int)
    inside = (np.abs(bc) <= R) & (np.abs(br) <= R)
    idx = ((br + R) * (2 * R + 1) + (bc + R))[inside]
    count = np.bincount(idx, minlength=(2 * R + 1) ** 2).reshape(2 * R + 1, 2 * R + 1)

    def grid(v):
        s = np.bincount(idx, weights=v[inside], minlength=(2 * R + 1) ** 2).reshape(2 * R + 1, 2 * R + 1)
        g = s / np.maximum(count, 1)
        g[count < 300] = np.nan
        return g
    panels = [(none[:, 0] - maps[:, 0], f"maps − none, {hz[0]:.2f} s"), (none[:, 1] - maps[:, 1], f"maps − none, {hz[1]:.2f} s"),
              (hide[:, 0] - none[:, 0], f"none − maps with points hidden,\n{hz[0]:.2f} s: the cost of hiding")]
    gains = [1e4 * grid(v) for v, _ in panels]
    vmax = np.nanpercentile(np.abs(np.stack(gains)), 98)
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.5), layout="constrained")
    ext = (-R - 0.5, R + 0.5, R + 0.5, -R - 0.5)
    im0 = axes[0].imshow(grid(none[:, 0]), extent=ext, cmap="viridis")
    axes[0].set_title(f"error of none (MSE), {hz[0]:.2f} s")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, location="left", pad=0.02)
    for ax, g, (_, title) in zip(axes[1:], gains, panels):
        im = ax.imshow(g, extent=ext, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
    fig.colorbar(im, ax=list(axes[1:]), shrink=0.8, pad=0.01, label="gain (10⁻⁴ MSE); blue: the first is better")
    for ax in axes:
        ax.grid(False)
        ax.plot(0, 0, "+", color="white" if ax is axes[0] else "#D55E00", ms=9, mew=1.6)
        ax.add_patch(plt.Circle((0, 0), 2, fill=False, ls="--", lw=0.8, color="white" if ax is axes[0] else "#333333"))
        ax.set_xticks([-6, -3, 0, 3, 6]); ax.set_yticks([-6, -3, 0, 3, 6])
        ax.set_xlabel("columns from the gaze point")
    axes[0].set_ylabel("rows from the gaze point")
    fig.suptitle("Test 11. Error and gain around the gaze point of the last context frame (+); the dashed circle is "
                 "the 'near' region. 600 clips, seed 0; cells with fewer than 300 patches are blank", fontsize=9.5)
    save(plt, fig, out, "t11_gain_map.png")


GROUPS = {"overview": [fig_overview], "picks": [fig_picks], "test3": [fig_test3], "test4": [fig_test4],
          "test8": [fig_test8], "test9": [fig_forms_diagram, fig_test9, fig_test9_patches],
          "test11": [fig_test11, fig_test11_patches]}


def draw(args):
    plt = setup()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for group in args.only or GROUPS:
        for fn in GROUPS[group]:
            fn(plt, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["figures", "patches"], default="figures")
    ap.add_argument("--only", nargs="+", default=None,
                    help="groups: overview picks test3 test4 test8 test9 test11")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--video-dir")
    ap.add_argument("--gaze-dir")
    ap.add_argument("--checkpoint", default="data/model_checkpoints/vjepa2-ac-vitg.pt")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.stage == "patches":
        if not (args.video_dir and args.gaze_dir):
            ap.error("--stage patches needs --video-dir and --gaze-dir")
        return patch_errors(args)
    draw(args)


if __name__ == "__main__":
    main()
