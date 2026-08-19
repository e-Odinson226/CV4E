"""
Fine-tune VisionTransformerPredictorEgo on HD-EPIC with gaze + hand conditioning.

    loss = MSE( norm(predictor(enc_ctx, gaze, hand)) , norm(enc_target) )

Aligned with the original V-JEPA 2-AC droid training (see scripts/ego_common.py):
  * frames are encoded per-frame-independently with the EMA target_encoder,
  * the world model steps at a stride (~4 fps), not on adjacent frames,
  * reps are LayerNorm'd before the loss (normalize_reps),
  * every step is supervised: slot t predicts step t+1 (teacher forcing),
  * gaze/hand are randomly dropped (--signal-dropout) so the mask tokens are
    well trained and Condition A (no signal) is a FAIR baseline at eval time,
  * --shuffle-signals destroys the signal/frame correspondence while preserving
    everything else, giving the matched control arm the 2x2 rerun needs.

Usage:
    python scripts/finetune_ego.py \
        --checkpoint  data/model_checkpoints/vjepa2-ac-vitg.pt \
        --video-dir   data/epic-kitchen/ek100-hd/HD-EPIC/Videos \
        --gaze-dir    data/epic-kitchen/ek100-hd/HD-EPIC/SLAM-and-Gaze \
        --participants P01 P02 P03 P04 P05 P06 P07 \
        --out-dir     checkpoints/ego_finetune \
        --epochs 20 --batch-size 8

    # Dry-run (no gaze data needed — uses null signals):
    python scripts/finetune_ego.py ... --participants P01 --epochs 1 \
        --clips-per-recording 5 --batch-size 2 --out-dir checkpoints/dry_run
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vjepa2"))

from ego_common import (
    load_models, encode_independent, maybe_norm,
    load_frames, read_vrs_times, find_csvs, find_ts_csv,
)
from eval_ego_mse import eval_clip, _real_signals
from src.models.ego_finetune import (
    freeze_for_ego_finetune,
    get_ego_finetune_param_groups,
    log_finetuned_layers,
)
from src.datasets.ego_loaders import GazeTokenLoader, HandTokenLoader

log = logging.getLogger("finetune_ego")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(out_dir) / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(str(log_path), mode="a", encoding="utf-8")],
    )
    log.info(f"Log file: {log_path}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HDEpicClipDataset(Dataset):
    """
    Each item is one randomly sampled, time-strided clip:
        ctx_frames  (T,   3, H, W)   — context steps  f_0 .. f_{T-1}
        tgt_frames  (T,   3, H, W)   — next steps     f_1 .. f_T   (teacher forcing)
        gaze_vecs   (T, 3)           — signal at each context step
        gaze_valid  (T,)  bool
        hand_vecs   (T, 12)
        hand_lvalid (T,)  bool
        hand_rvalid (T,)  bool

    Frames are spaced `frame_stride` apart, so T steps span T*stride frames.
    Gaze/hand are aligned by ABSOLUTE VRS timestamp (mp4_to_vrs_time_ns.csv),
    the same path used at eval time. With prob `signal_dropout` a clip's signals
    are fully masked so the mask tokens learn a genuine "no signal" mode.
    """

    def __init__(self, video_dir, gaze_dir, participants,
                 context_steps=8, frame_stride=8, tubelet_size=2,
                 img_size=256, clips_per_recording=200, signal_dropout=0.4,
                 standardize=True):
        self.T = context_steps
        self.tubelet = tubelet_size
        self.stride = frame_stride
        self.img_size = img_size
        self.clips_per_rec = clips_per_recording
        self.signal_dropout = signal_dropout
        self.standardize = standardize

        self.recordings = []
        self._discover(video_dir, gaze_dir, participants)
        self._cache = {}   # per-recording (vrs_ts, gaze_loader, hand_loader)

        n_gaze = sum(1 for r in self.recordings if r["gaze_csv"])
        n_hand = sum(1 for r in self.recordings if r["hand_csv"])
        log.info(f"[dataset] participants={participants}")
        log.info(f"[dataset] recordings found:    {len(self.recordings)}")
        log.info(f"[dataset] with gaze CSV:        {n_gaze}/{len(self.recordings)}")
        log.info(f"[dataset] with hand CSV:        {n_hand}/{len(self.recordings)}")
        log.info(f"[dataset] context_steps={context_steps}  frame_stride={frame_stride}  "
                 f"(span {context_steps * frame_stride} frames/clip)")
        log.info(f"[dataset] clips/recording:      {clips_per_recording}")
        log.info(f"[dataset] signal_dropout:       {signal_dropout}")
        log.info(f"[dataset] total clips/epoch:    ~{len(self.recordings) * clips_per_recording}")
        if n_gaze == 0:
            log.warning("[dataset] No gaze CSVs found — training with NULL signals only. "
                        "Condition B is impossible until GAZE_HAND zips are downloaded.")

    def _discover(self, video_dir, gaze_dir, participants):
        for p in participants:
            for mp4 in sorted(Path(video_dir, p).glob("*.mp4")):
                ts_csv = find_ts_csv(str(mp4))
                if ts_csv is None:
                    continue
                gaze_csv, hand_csv = find_csvs(gaze_dir, p, mp4.stem)
                self.recordings.append({
                    "mp4": str(mp4), "ts_csv": ts_csv,
                    "gaze_csv": gaze_csv, "hand_csv": hand_csv,
                })

    def _get_cached(self, rec):
        key = rec["mp4"]
        if key not in self._cache:
            vrs_ts = read_vrs_times(rec["ts_csv"])
            gl = GazeTokenLoader(rec["gaze_csv"], 30.0, standardize=self.standardize) if rec["gaze_csv"] else None
            hl = HandTokenLoader(rec["hand_csv"], 30.0, standardize=self.standardize) if rec["hand_csv"] else None
            self._cache[key] = (vrs_ts, gl, hl)
        return self._cache[key]

    def __len__(self):
        return len(self.recordings) * self.clips_per_rec

    def __getitem__(self, idx):
        rec = self.recordings[idx % len(self.recordings)]
        try:
            return self._sample_clip(rec)
        except Exception as e:
            log.warning(f"[dataset] clip sample failed for {Path(rec['mp4']).name}: {e}")
            return self._null_item()

    def _sample_clip(self, rec):
        vrs_ts, gl, hl = self._get_cached(rec)
        n_frames = len(vrs_ts)
        span = self.T * self.stride                      # last target index = start + span
        if n_frames < span + 1:
            return self._null_item()

        start = int(np.random.randint(0, n_frames - span))
        ctx_idx = [start + i * self.stride for i in range(self.T)]
        tgt_idx = [start + (i + 1) * self.stride for i in range(self.T)]
        all_idx = sorted(set(ctx_idx + tgt_idx))

        frames = load_frames(rec["mp4"], all_idx, size=self.img_size)
        pos = {j: k for k, j in enumerate(all_idx)}
        ctx_frames = frames[[pos[j] for j in ctx_idx]]
        tgt_frames = frames[[pos[j] for j in tgt_idx]]

        drop = np.random.rand() < self.signal_dropout
        ctx_vrs = [int(vrs_ts[j]) for j in ctx_idx]
        gaze_v, gaze_val, hand_v, h_l, h_r = self._load_signals(gl, hl, ctx_vrs, drop)

        return ctx_frames, tgt_frames, gaze_v, gaze_val, hand_v, h_l, h_r

    def _load_signals(self, gl, hl, ctx_vrs_ns, drop):
        T = self.T
        gaze_vecs  = np.zeros((T, 3),  dtype=np.float32)
        gaze_valid = np.zeros(T,        dtype=bool)
        hand_vecs  = np.zeros((T, 12), dtype=np.float32)
        h_left     = np.zeros(T,        dtype=bool)
        h_right    = np.zeros(T,        dtype=bool)

        if not drop and gl is not None:
            for t, vrs_ns in enumerate(ctx_vrs_ns):
                gaze_vecs[t], gaze_valid[t] = gl.get_token_for_vrs_ns(vrs_ns)
        if not drop and hl is not None:
            for t, vrs_ns in enumerate(ctx_vrs_ns):
                hand_vecs[t], h_left[t], h_right[t] = hl.get_token_for_vrs_ns(vrs_ns)

        return (torch.from_numpy(gaze_vecs), torch.from_numpy(gaze_valid),
                torch.from_numpy(hand_vecs), torch.from_numpy(h_left), torch.from_numpy(h_right))

    def _null_item(self):
        T, sz = self.T, self.img_size
        return (torch.zeros(T, 3, sz, sz), torch.zeros(T, 3, sz, sz),
                torch.zeros(T, 3), torch.zeros(T, dtype=torch.bool),
                torch.zeros(T, 12), torch.zeros(T, dtype=torch.bool),
                torch.zeros(T, dtype=torch.bool))


# ---------------------------------------------------------------------------
# Collate — and the shuffled-signal control
# ---------------------------------------------------------------------------

def make_collate(shuffle_signals: str = "off"):
    """
    Stack a batch, optionally destroying the correspondence between a signal value
    and the frame it belongs to.

    WHY THIS EXISTS. Adding a gaze token adds two things at once: the information in
    gaze, and extra parameters, tokens and optimisation load. An arm that beats a
    no-signal baseline could owe its gain to either. Permuting the signals removes
    the information while holding everything else exactly fixed — identical tensor
    shapes, token count, parameter count, marginal distribution, validity statistics
    — so a shuffled arm is the only baseline that isolates the information itself.
    `real > shuffled` is the single comparison that licenses the sentence "the model
    used gaze information to predict the future".

    Modes:
      off    real signals, time-aligned. The normal training path.
      time   permute within the clip. Gaze and hand get the SAME permutation, so
             gaze/hand correspondence survives and only signal-to-frame alignment
             is destroyed. The permutation is redrawn per clip.
             A clip spans T*stride frames (64 at the defaults, ~2.1 s), and
             EXP-003 measured gaze persistence collapsing to skill -0.12 by 1 s, so
             a within-clip permutation genuinely decorrelates the signal rather
             than returning a near-copy of it.
      batch  every clip receives ANOTHER clip's signals — a different moment, and
             usually a different person and kitchen. The stronger control, at the
             cost of also changing the signal's marginal distribution per clip.

    Validity masks are permuted with their vectors, so per-batch validity statistics
    are untouched and a shuffled arm cannot be distinguished from a real one by how
    much signal it received.

    Randomness comes from torch, which the DataLoader seeds per worker from the
    global seed; numpy is NOT reseeded per worker, so it would hand every worker the
    same permutation stream.

    Validation is deliberately NOT shuffled: arms trained on real and on shuffled
    signals must be scored on the same held-out measurement to be comparable.
    """
    if shuffle_signals not in ("off", "time", "batch"):
        raise ValueError(f"unknown --shuffle-signals {shuffle_signals!r}")

    def collate(batch):
        ctx_f, tgt_f, gaze_v, gaze_val, hand_v, h_l, h_r = (
            torch.stack([b[i] for b in batch]) for i in range(7)
        )
        if shuffle_signals == "time":
            for b in range(gaze_v.size(0)):
                p = torch.randperm(gaze_v.size(1))
                gaze_v[b], gaze_val[b] = gaze_v[b][p], gaze_val[b][p]
                hand_v[b], h_l[b], h_r[b] = hand_v[b][p], h_l[b][p], h_r[b][p]
        elif shuffle_signals == "batch":
            p = torch.roll(torch.arange(gaze_v.size(0)), 1)   # a derangement for B > 1
            gaze_v, gaze_val = gaze_v[p], gaze_val[p]
            hand_v, h_l, h_r = hand_v[p], h_l[p], h_r[p]
        return ctx_f, tgt_f, gaze_v, gaze_val, hand_v, h_l, h_r

    return collate


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(encoder, predictor, loader, optimizer, device, epoch,
                    normalize_reps=True, amp_dtype=None, encode_chunk=16,
                    log_every=10, jsonl=None):
    predictor.train()
    HW = predictor.grid_height * predictor.grid_width

    total_loss = total_grad = 0.0
    ema = None                       # smoothed instantaneous loss
    n_batches = 0
    t_epoch = time.time()
    use_amp = amp_dtype is not None and device.type == "cuda"

    for step, batch in enumerate(loader, 1):
        ctx_f, tgt_f, gaze_v, gaze_val, hand_v, h_left, h_right = batch

        enc_ctx = encode_independent(encoder, ctx_f, device, normalize_reps,
                                     chunk=encode_chunk, amp_dtype=amp_dtype)
        enc_tgt = encode_independent(encoder, tgt_f, device, normalize_reps,
                                     chunk=encode_chunk, amp_dtype=amp_dtype)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = predictor(
                enc_ctx,
                gaze_v.to(device), gaze_val.to(device),
                hand_v.to(device), h_left.to(device), h_right.to(device),
            )
            pred = maybe_norm(pred, normalize_reps)
            loss = F.mse_loss(pred.float(), enc_tgt.float())   # supervise every step

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in predictor.parameters() if p.requires_grad], max_norm=1.0
        ).item()
        optimizer.step()

        inst = loss.item()
        ema = inst if ema is None else 0.9 * ema + 0.1 * inst
        total_loss += inst
        total_grad += grad_norm
        n_batches += 1

        if step % log_every == 0:
            avg = total_loss / n_batches
            elapsed = time.time() - t_epoch
            sps = step / elapsed
            eta = (len(loader) - step) / sps if sps > 0 else 0
            log.info(
                f"epoch {epoch}  step {step:4d}/{len(loader)}  "
                f"loss={avg:.4f}  grad={grad_norm:.3f}  "
                f"lr_proj={optimizer.param_groups[0]['lr']:.2e}  "
                f"lr_blocks={optimizer.param_groups[1]['lr']:.2e}  ETA={eta:.0f}s"
            )
            if jsonl:
                jsonl.write(json.dumps({"t": "step", "epoch": epoch, "step": step,
                                        "total_steps": len(loader), "loss": inst,
                                        "ema": ema, "avg": total_loss / n_batches,
                                        "grad": grad_norm, "it_s": sps, "eta": eta}) + "\n")
                jsonl.flush()

    return total_loss / max(n_batches, 1), total_grad / max(n_batches, 1), time.time() - t_epoch


# ---------------------------------------------------------------------------
# Held-out validation: paired MSE_A (masked) vs MSE_B (real signals)
# ---------------------------------------------------------------------------

def discover_recordings(video_dir, gaze_dir, participants, require_signal=True, limit=None):
    recs = []
    for p in participants:
        for mp4 in sorted(Path(video_dir, p).glob("*.mp4")):
            ts = find_ts_csv(str(mp4))
            if ts is None:
                continue
            g, h = find_csvs(gaze_dir, p, mp4.stem)
            if require_signal and not (g or h):
                continue
            recs.append({"mp4": str(mp4), "ts_csv": ts, "gaze_csv": g, "hand_csv": h})
            if limit and len(recs) >= limit:
                return recs
    return recs


@torch.no_grad()
def validate(encoder, predictor, recordings, device, T, stride, n_clips,
             normalize_reps, standardize, seed=12345):
    """
    Paired held-out eval. For a FIXED set of clips (seeded), score the future
    step with signals masked (A) and with real signals (B). Same clips every
    epoch, so the Delta trend is comparable across epochs.
    """
    was_training = predictor.training
    predictor.eval()
    rng = np.random.RandomState(seed)
    span = T * stride
    A, B = [], []

    for rec in recordings:
        vrs = read_vrs_times(rec["ts_csv"])
        n = len(vrs)
        if n < span + 1:
            continue
        gl = GazeTokenLoader(rec["gaze_csv"], 30.0, standardize=standardize) if rec["gaze_csv"] else None
        hl = HandTokenLoader(rec["hand_csv"], 30.0, standardize=standardize) if rec["hand_csv"] else None
        if gl is None and hl is None:
            continue
        for _ in range(n_clips):
            start = int(rng.randint(0, n - span))
            ctx_idx = [start + i * stride for i in range(T)]
            fut = start + T * stride
            try:
                frames = load_frames(rec["mp4"], ctx_idx + [fut])
                ctx, futf = frames[:T], frames[T]
                real = _real_signals(gl, hl, [int(vrs[j]) for j in ctx_idx], T, device)
                a, b = eval_clip(encoder, predictor, ctx, futf, device,
                                 real_signals=real, normalize_reps=normalize_reps)
                A.append(a); B.append(b)
            except Exception as e:
                log.warning(f"[val] clip error: {e}")

    if was_training:
        predictor.train()
    if not A:
        return None
    mA, mB = float(np.mean(A)), float(np.mean(B))
    return mA, mB, mA - mB, len(A)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",          required=True)
    ap.add_argument("--video-dir",           required=True)
    ap.add_argument("--gaze-dir",            required=True)
    ap.add_argument("--participants",        nargs="+",
                    default=["P01", "P02", "P03", "P04", "P05", "P06", "P07"])
    ap.add_argument("--out-dir",             default="checkpoints/ego_finetune")
    ap.add_argument("--epochs",              type=int,   default=20)
    ap.add_argument("--batch-size",          type=int,   default=8)
    ap.add_argument("--num-workers",         type=int,   default=4)
    ap.add_argument("--context-steps",       type=int,   default=8,
                    help="Temporal steps the predictor sees (droid used 8)")
    ap.add_argument("--frame-stride",        type=int,   default=8,
                    help="Frames between steps; 8 @30fps ~= droid's 4fps")
    ap.add_argument("--clips-per-recording", type=int,   default=200)
    ap.add_argument("--signal-dropout",      type=float, default=0.4,
                    help="Prob. of fully masking a clip's signals (trains mask tokens / fair Condition A)")
    ap.add_argument("--shuffle-signals",     choices=["off", "time", "batch"], default="off",
                    help="Destroy signal/frame alignment while holding shapes, token count, "
                         "parameter count and validity statistics fixed. The matched control "
                         "arm for the 2x2 rerun; validation is never shuffled.")
    ap.add_argument("--unfreeze-last-n",     type=int,   default=6)
    ap.add_argument("--lr-proj",             type=float, default=1e-3)
    ap.add_argument("--lr-blocks",           type=float, default=1e-4)
    ap.add_argument("--weight-decay",        type=float, default=1e-2)
    ap.add_argument("--no-normalize-reps",   action="store_true",
                    help="Disable LayerNorm on reps (config trained with it ON)")
    ap.add_argument("--no-amp",              action="store_true",
                    help="Disable bf16 autocast on encoder+predictor (fp32, ~2x slower)")
    ap.add_argument("--encode-chunk",        type=int,   default=16,
                    help="Frames per ViT-g encode sub-batch; raise to use more GPU (32-64)")
    ap.add_argument("--no-standardize",      action="store_true",
                    help="Disable gaze/hand input z-scoring")
    ap.add_argument("--val-participants",    nargs="+", default=None,
                    help="Held-out participants for per-epoch paired MSE_A/MSE_B/Delta (e.g. P08)")
    ap.add_argument("--val-recordings",      type=int,   default=4,
                    help="Max held-out recordings to validate on")
    ap.add_argument("--val-clips",           type=int,   default=8,
                    help="Clips per held-out recording (fixed across epochs)")
    ap.add_argument("--save-every",          type=int,   default=5)
    ap.add_argument("--log-every",           type=int,   default=10)
    ap.add_argument("--seed",                type=int,   default=0)
    ap.add_argument("--device",              default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    setup_logging(args.out_dir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    normalize_reps = not args.no_normalize_reps
    amp_dtype = None if args.no_amp else torch.bfloat16

    log.info("=" * 60)
    log.info("Ego predictor fine-tuning")
    log.info("=" * 60)
    for k, v in vars(args).items():
        log.info(f"  {k:<25} {v}")
    log.info(f"  normalize_reps            {normalize_reps}")
    log.info(f"  amp_dtype                 {amp_dtype}")
    log.info("=" * 60)

    encoder, predictor, (n_t, n_s) = load_models(
        args.checkpoint, device, args.context_steps, tubelet=2, encoder_key="target_encoder"
    )
    log.info(f"[ckpt] transferred {n_t} predictor keys, skipped {n_s}")

    freeze_for_ego_finetune(predictor, unfreeze_last_n_blocks=args.unfreeze_last_n)
    log_finetuned_layers(log, predictor, unfreeze_last_n=args.unfreeze_last_n)

    dataset = HDEpicClipDataset(
        video_dir=args.video_dir, gaze_dir=args.gaze_dir, participants=args.participants,
        context_steps=args.context_steps, frame_stride=args.frame_stride, tubelet_size=2,
        clips_per_recording=args.clips_per_recording, signal_dropout=args.signal_dropout,
        standardize=not args.no_standardize,
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=make_collate(args.shuffle_signals),
        pin_memory=(device.type == "cuda"), drop_last=True,
    )

    param_groups = get_ego_finetune_param_groups(predictor, lr_proj=args.lr_proj, lr_blocks=args.lr_blocks)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Held-out validation set (fixed clips, scored every epoch)
    val_recordings = []
    if args.val_participants:
        val_recordings = discover_recordings(
            args.video_dir, args.gaze_dir, args.val_participants,
            require_signal=True, limit=args.val_recordings,
        )
        log.info(f"[val] {len(val_recordings)} held-out recordings from {args.val_participants} "
                 f"({args.val_clips} clips each, fixed seed)")
        if not val_recordings:
            log.warning("[val] no held-out recordings with signals found — validation disabled")

    jsonl = open(Path(args.out_dir) / "metrics.jsonl", "a", encoding="utf-8")
    _prev_delta: dict = {"v": None}

    def run_val(tag, epoch_idx):
        if not val_recordings:
            return
        r = validate(encoder, predictor, val_recordings, device,
                     args.context_steps, args.frame_stride, args.val_clips,
                     normalize_reps, not args.no_standardize)
        if r:
            mA, mB, d, n = r
            prev = _prev_delta["v"]
            trend = "" if prev is None else f"  ({'+' if d > prev else ''}{d - prev:+.4f} vs prev)"
            _prev_delta["v"] = d
            flag = "signals HELP" if d > 0 else "no help"
            log.info(f"[val {tag}]  MSE_A(masked)={mA:.4f}  MSE_B(real)={mB:.4f}  "
                     f"Delta(A-B)={d:+.4f}  ({flag}, {n} clips){trend}")
            jsonl.write(json.dumps({"t": "val", "epoch": epoch_idx, "mse_A": mA,
                                    "mse_B": mB, "delta": d, "n_clips": n}) + "\n")
            jsonl.flush()

    log.info(f"[train] {len(loader)} steps/epoch  ({len(dataset)} clips, batch={args.batch_size})")
    log.info("[train] starting...")
    run_val("epoch0", 0)   # baseline Delta BEFORE any training

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        avg_loss, avg_grad, ep_time = train_one_epoch(
            encoder, predictor, loader, optimizer, device, epoch,
            normalize_reps=normalize_reps, amp_dtype=amp_dtype,
            encode_chunk=args.encode_chunk, log_every=args.log_every, jsonl=jsonl,
        )
        scheduler.step()

        elapsed = time.time() - t0
        eta = (elapsed / epoch) * (args.epochs - epoch)
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            torch.save({"predictor": predictor.state_dict(), "epoch": epoch, "loss": avg_loss,
                        "config": vars(args)}, Path(args.out_dir) / "best.pt")

        log.info(
            f"[epoch {epoch:3d}/{args.epochs}]  loss={avg_loss:.4f}{'*' if is_best else ' '}  "
            f"avg_grad={avg_grad:.3f}  lr_proj={optimizer.param_groups[0]['lr']:.2e}  "
            f"lr_blocks={optimizer.param_groups[1]['lr']:.2e}  time={ep_time:.0f}s  ETA={eta:.0f}s"
        )
        run_val(f"epoch{epoch}", epoch)

        if epoch % args.save_every == 0:
            ckpt = Path(args.out_dir) / f"epoch_{epoch:03d}.pt"
            torch.save({"predictor": predictor.state_dict(), "epoch": epoch, "loss": avg_loss,
                        "config": vars(args)}, ckpt)
            log.info(f"[save] {ckpt}")

    torch.save({"predictor": predictor.state_dict(), "epoch": args.epochs, "loss": best_loss,
                "config": vars(args)}, Path(args.out_dir) / "final.pt")
    log.info("=" * 60)
    log.info("Training complete")
    log.info(f"  Total time:  {(time.time() - t0)/60:.1f} min")
    log.info(f"  Best loss:   {best_loss:.4f}")
    log.info(f"  Checkpoints: {args.out_dir}/")
    log.info("=" * 60)
    jsonl.close()


if __name__ == "__main__":
    main()
