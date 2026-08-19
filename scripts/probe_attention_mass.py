"""
A2 attention mass — WHERE in the predictor is gaze read, and how loudly?

WHY ATTENTION IS THE RIGHT INSTRUMENT (adoption plan step 15, measuring-signal-use.md A2)
------------------------------------------------------------------------------------------
Gaze enters this predictor as a TOKEN, not as a residual added to hidden states
(ego_predictor.py:185). A token can only influence a visual token through attention,
so the attention weight that visual queries place on the conditioning positions is
literally the bandwidth of the gaze channel. A1 says whether the channel carries
anything; this says through which of the 24 blocks, and how much.

THE REFERENCE NUMBER
--------------------
Per frame the sequence is [gaze, hand, 256 visual tokens], and the block-causal mask
lets a query at frame t see every token of frames 0..t — 258 keys per frame, of which
2 are conditioning. Uniform attention therefore puts exactly 2/258 = 0.775% on the
pair no matter which frame the query sits in, which makes it a clean floor:

  at or below 0.775%   the conditioning tokens are background
  clearly above        the model is actively querying them

Per-head maxima are reported alongside the mean, because one specialised head out of
sixteen is a real mechanism that a mean over heads would hide.

HOW IT IS MEASURED WITHOUT PERTURBING THE MODEL
-----------------------------------------------
F.scaled_dot_product_attention returns only the output, never the weights. Rather
than reimplement the block, this wraps that function: the value the network receives
is still the one the ORIGINAL kernel produced, and the explicit softmax is computed
alongside it purely for the statistics. The two are checked against each other on the
first call, so a mismatch surfaces as an error rather than as a plausible wrong number.

Usage
-----
    $PY scripts/probe_attention_mass.py \
        --checkpoint data/model_checkpoints/vjepa2-ac-vitg.pt \
        --predictor-checkpoint checkpoints/ego_ft_v2/best.pt \
        --video-dir data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
        --gaze-dir  data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
        --participants P08 --recordings 4 --clips 6 --out results/attention_mass
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vjepa2"))

from ego_common import load_models, encode_independent, load_frames
from eval_ego_mse import load_finetuned
from probe_sensitivity import discover, enumerate_clips, mask_gaze, mask_hand

_ORIG_SDPA = torch.nn.functional.scaled_dot_product_attention


class AttentionRecorder:
    """
    Wraps scaled_dot_product_attention. The wrapped call always returns the original
    kernel's output, so the forward pass is unchanged; the softmax recomputed here
    exists only to read the weights off.
    """

    def __init__(self, cond_tokens, hw, n_frames, head_chunk=4):
        self.A = cond_tokens
        self.HW = hw
        self.T = n_frames
        self.N = n_frames * (cond_tokens + hw)
        self.head_chunk = head_chunk
        self.enabled = False
        self.records = []
        self.checked = False
        self.max_abs_err = 0.0

    def reset(self):
        self.records = []

    def __call__(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        out = _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                         is_causal=is_causal, **kw)
        if self.enabled and q.dim() == 4 and q.size(-2) == self.N:
            self._record(q, k, v, attn_mask, out)
        return out

    @torch.no_grad()
    def _record(self, q, k, v, attn_mask, out):
        B, H, N, Dh = q.shape
        step = self.A + self.HW
        pos = torch.arange(N, device=q.device) % step
        is_cond = pos < self.A
        is_gaze = pos == 0
        is_hand = pos == 1
        vis_rows = ~is_cond
        scale = Dh ** -0.5

        per_head_gaze, per_head_hand = [], []
        for h0 in range(0, H, self.head_chunk):
            h1 = min(h0 + self.head_chunk, H)
            scores = (q[:, h0:h1].float() @ k[:, h0:h1].float().transpose(-2, -1)) * scale
            if attn_mask is not None:
                m = attn_mask
                if m.dtype == torch.bool:
                    scores = scores.masked_fill(~m, float("-inf"))
                else:
                    scores = scores + m
            attn = scores.softmax(-1)

            if not self.checked:                       # verify the recomputation once
                ref = out[:, h0:h1].float()
                got = attn @ v[:, h0:h1].float()
                self.max_abs_err = float((got - ref).abs().max())
                self.checked = True

            rows = attn[:, :, vis_rows, :]             # visual-token queries only
            per_head_gaze.append(rows[..., is_gaze].sum(-1).mean(-1))   # (B, h)
            per_head_hand.append(rows[..., is_hand].sum(-1).mean(-1))
            del scores, attn, rows

        g = torch.cat(per_head_gaze, dim=1)            # (B, H)
        hnd = torch.cat(per_head_hand, dim=1)
        self.records.append({
            "gaze_mean": float(g.mean()), "gaze_max_head": float(g.max()),
            "hand_mean": float(hnd.mean()), "hand_max_head": float(hnd.max()),
            "gaze_per_head": g.mean(0).cpu().numpy(),
        })


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--predictor-checkpoint", default=None)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--gaze-dir", required=True)
    ap.add_argument("--participants", nargs="+", default=["P08"])
    ap.add_argument("--recordings", type=int, default=4)
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--context-steps", type=int, default=8)
    ap.add_argument("--frame-stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--head-chunk", type=int, default=4)
    ap.add_argument("--no-normalize-reps", action="store_true")
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--encode-chunk", type=int, default=16)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/attention_mass")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    logf = open(f"{out}.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    device = torch.device(args.device)
    normalize_reps = not args.no_normalize_reps
    T, stride = args.context_steps, args.frame_stride
    # gaze_proj / hand_proj / the mask tokens are randomly initialised, so any arm
    # that uses the UNTRAINED predictor depends on this draw. Seeding keeps those
    # arms reproducible; checkpoint-loaded arms are deterministic regardless.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    encoder, predictor, _ = load_models(args.checkpoint, device, T, tubelet=2,
                                        encoder_key="target_encoder")
    if args.predictor_checkpoint:
        load_finetuned(predictor, args.predictor_checkpoint)
        log(f"[ckpt] fine-tuned predictor <- {args.predictor_checkpoint}")
    else:
        log("[ckpt] AC weights only — untrained projectors")
    predictor.eval()
    HW = predictor.grid_height * predictor.grid_width
    log(f"[setup] {time.time()-t0:.0f}s  T={T}  HW={HW}  seq={T*(2+HW)}")

    recs = discover(args.video_dir, args.gaze_dir, args.participants, args.recordings)
    clips = enumerate_clips(recs, T, stride, args.clips, args.seed,
                            not args.no_standardize, device, log)
    log(f"[clips] {len(clips)} clips")

    rec = AttentionRecorder(2, HW, T, head_chunk=args.head_chunk)
    torch.nn.functional.scaled_dot_product_attention = rec

    variants = {"real": lambda s: s, "gaze_masked": mask_gaze,
                "both_masked": lambda s: mask_hand(mask_gaze(s))}
    acc = {name: [] for name in variants}
    try:
        for i, clip in enumerate(clips):
            frames = load_frames(clip["rec"]["mp4"], clip["ctx_idx"])
            enc_ctx = encode_independent(encoder, frames.unsqueeze(0), device,
                                         normalize_reps, chunk=args.encode_chunk)
            for name, fn in variants.items():
                rec.reset(); rec.enabled = True
                with torch.no_grad():
                    predictor(enc_ctx, *fn(clip["sig"]))
                rec.enabled = False
                acc[name].append(rec.records)
            if (i + 1) % 5 == 0:
                log(f"  [{i+1}/{len(clips)}] {time.time()-t0:.0f}s")
    finally:
        torch.nn.functional.scaled_dot_product_attention = _ORIG_SDPA

    uniform = 1.0 / (2 + HW)
    log(f"\n[check] recomputed softmax vs the kernel's own output: max |diff| = "
        f"{rec.max_abs_err:.2e} ({'OK' if rec.max_abs_err < 1e-2 else 'MISMATCH — stats unreadable'})")
    log(f"[ref] uniform attention on ONE conditioning token = 1/{2+HW} = {uniform:.5f} "
        f"({100*uniform:.3f}%); on the pair = {2*uniform:.5f}")

    rows = []
    for name, per_clip in acc.items():
        n_blocks = len(per_clip[0])
        for b in range(n_blocks):
            gm = np.mean([c[b]["gaze_mean"] for c in per_clip])
            gx = np.mean([c[b]["gaze_max_head"] for c in per_clip])
            hm = np.mean([c[b]["hand_mean"] for c in per_clip])
            hx = np.mean([c[b]["hand_max_head"] for c in per_clip])
            rows.append({"variant": name, "block": b, "trainable": b >= 18,
                         "gaze_mass": gm, "gaze_mass_max_head": gx,
                         "hand_mass": hm, "hand_mass_max_head": hx,
                         "gaze_vs_uniform": gm / uniform,
                         "gaze_max_head_vs_uniform": gx / uniform})

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}.csv", index=False)

    d = df[df.variant == "real"].set_index("block")
    log("\n" + "=" * 92)
    log("A2 — share of a visual token's attention landing on the conditioning positions (real gaze)")
    log("   x/uniform: 1.0 = the token is indistinguishable from background")
    log("=" * 92)
    log(d[["gaze_mass", "gaze_vs_uniform", "gaze_mass_max_head", "gaze_max_head_vs_uniform",
           "hand_mass", "trainable"]]
        .to_string(float_format=lambda v: f"{v:10.5f}"))

    tr = d[d.trainable]
    fr = d[~d.trainable]
    log("\n" + "-" * 92)
    log(f"[summary] gaze mass, frozen blocks 0-17 : mean {fr.gaze_mass.mean():.5f} "
        f"({fr.gaze_vs_uniform.mean():.2f}x uniform)")
    log(f"[summary] gaze mass, trained blocks 18-23: mean {tr.gaze_mass.mean():.5f} "
        f"({tr.gaze_vs_uniform.mean():.2f}x uniform)")
    log(f"[summary] loudest block overall: {int(d.gaze_mass.idxmax())} "
        f"at {d.gaze_mass.max():.5f} ({d.gaze_vs_uniform.max():.2f}x uniform); "
        f"loudest single head {d.gaze_mass_max_head.max():.5f} "
        f"({d.gaze_max_head_vs_uniform.max():.2f}x uniform)")

    if d.gaze_vs_uniform.max() < 1.2:
        log("\nREADING: no block attends to gaze above the uniform floor. The token is background.")
    elif tr.gaze_vs_uniform.mean() > fr.gaze_vs_uniform.mean() * 1.2:
        log("\nREADING: the TRAINED blocks (18-23) attend to gaze more than the frozen ones do,")
        log("         so fine-tuning built the pathway where it had the freedom to.")
    else:
        log("\nREADING: gaze attention is not concentrated in the trained blocks — the frozen")
        log("         AC blocks already route attention to the conditioning slot, which they")
        log("         learned to do for robot actions in pretraining.")
    log("-" * 92)

    with open(f"{out}.json", "w") as f:
        json.dump({"config": vars(args), "uniform": uniform,
                   "sdpa_check_max_abs_err": rec.max_abs_err, "rows": rows},
                  f, indent=2, default=str)
    log(f"[out] {out}.csv  {out}.json  {out}.log")
    logf.close()


if __name__ == "__main__":
    main()
