"""
Train the Test 11 predictor: gaze and hand as maps over the image tokens.

    python -m ego train-ego --checkpoint <vjepa2-ac.pt> \
        --video-dir <videos> --gaze-dir <gaze> --arm maps

What differs from `python -m ego train`. That command fine-tunes the pretrained V-JEPA 2-AC
predictor, which takes gaze in the slot its pretraining built for the robot's action, and
predicts one step of 8 frames ahead because that step is what the pretraining knows. This
command trains ego/ego_predictor.py from the start, gives the signals as maps over the
image tokens, and predicts several horizons in one forward pass. The encoder, the clip
sampling, the split and the signal reading are the same, so the gain of the signals stays
comparable with Tests 9 and 10 ([[3-method#Rules for comparing models]]).

The horizon. One time slot is `--frame-stride` frames, 0.27 s at the default of 8 frames and
30 frames per second. `--target-slots 2 4` therefore predicts about 0.53 s and 1.07 s after
the last observed frame: 0.5 s is where the gaze point told most in Test 8, and 1 s is the
horizon of the benchmarks. The slots are whole numbers because RoPE places every token at a
whole time slot. The log prints the horizon in seconds for the stride in use.

The arms (Test 11, docs/6-next-steps.md). Only the signals differ.

    maps      gaze and hand as maps: the design
    none      alpha held at 0: the matched model, exactly the same weights otherwise
    shuffled  the maps, with the points of another clip in the batch
    future    the points of the target frames: the positive control, the ceiling of the
              measure at each horizon

How much the model uses a signal is alpha * ||e||, printed each log step as |gaze|,
|left| and |right|. Every run starts at 0 there, because the vectors e start at zero: the
predictor begins as the matched model and the signals can only grow from it.

How long a run lasts. `--max-hours` is a wall-clock budget, checked after each epoch, and an
epoch is not started if it would pass the budget. `--patience` stops a run after that many
epochs with no improvement in the held-out error, which is the error with the real points
averaged over the horizons. `best.pt` is rewritten whenever that error improves, `epoch3.pt`
(from `--checkpoint-epochs`) is kept whatever it does, and `final.pt` is the last epoch. So a
run goes as far as it keeps improving and no further.

These four are the arms that make the new predictor comparable with the V-JEPA 2-AC
predictors of Tests 9 and 10: the same split, clips, encoder and controls, so the gain of
gaze can be read against theirs. `token` is the fifth arm, a gaze token per step placed at
the gaze point in RoPE as the rope form of Test 9, for comparing the two channels inside one
predictor. It needs a gaze token in ego/ego_predictor.py and raises until that exists.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ego import signals
from ego.data import find_recordings, load_frames, open_loaders, read_vrs_times
from ego.ego_predictor import ego_predictor, param_counts, signal_strength
from ego.gaze_geometry import GRID, projector_for
from ego.model import encode_independent, load_models, maybe_norm

log = logging.getLogger("train_ego")

ARMS = ("maps", "none", "shuffled", "future", "token")
LOSSES = {"l1": F.l1_loss, "mse": F.mse_loss}
VAL_SEED = 12345


def setup_logging(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    h = [logging.StreamHandler(), logging.FileHandler(Path(out_dir) / "train.log")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", handlers=h, force=True)


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------

def clip_indices(start, T, stride, target_slots):
    """Frame indices of the observed steps and of the frames at each target slot."""
    obs = [start + i * stride for i in range(T)]
    tgt = [start + (T - 1 + s) * stride for s in target_slots]
    return obs, tgt


def point_indices(obs_idx, stride, target_slots, future, n_frames):
    """
    The frames whose gaze and palms the predictor receives: the observed frames, or for the
    positive control every observed frame moved forward by the first target slot, so the last
    observed step carries the points of the first frame to predict.
    """
    idx = [j + target_slots[0] * stride for j in obs_idx] if future else list(obs_idx)
    return [min(j, n_frames - 1) for j in idx]


class EgoClipDataset(Dataset):
    """
    One item is one clip:

        obs_frames (T, 3, H, W)       the observed frames
        tgt_frames (n_tgt, 3, H, W)   the frames at the target slots
        gaze_pt    (T, 2)             the gaze point of each observed frame, in patches
        gaze_val   (T,)   bool
        hand_pt    (T, 2, 2)          the left then right palm, in patches
        hand_val   (T, 2) bool

    With `future_points`, the points are read at the frames of the first target slot in
    place of the observed frames: the positive control.
    """

    def __init__(self, video_dir, gaze_dir, participants, context_steps=8, frame_stride=8,
                 target_slots=(2, 4), img_size=256, clips_per_recording=200,
                 standardize=True, future_points=False, grid=GRID):
        self.T = context_steps
        self.stride = frame_stride
        self.target_slots = tuple(target_slots)
        self.img_size = img_size
        self.clips_per_rec = clips_per_recording
        self.standardize = standardize
        self.future_points = future_points
        self.grid = grid
        self.recordings = find_recordings(video_dir, gaze_dir, participants)
        self._cache = {}

        span = (self.T - 1 + max(self.target_slots)) * self.stride
        self.span = span
        log.info(f"[dataset] participants={participants}  recordings={len(self.recordings)}")
        log.info(f"[dataset] with gaze CSV: {sum(1 for r in self.recordings if r.gaze_csv)}  "
                 f"with hand CSV: {sum(1 for r in self.recordings if r.hand_csv)}")
        log.info(f"[dataset] context_steps={self.T}  frame_stride={self.stride}  "
                 f"target_slots={self.target_slots}  span={span} frames")
        log.info(f"[dataset] clips/recording={clips_per_recording}  "
                 f"total/epoch ~{len(self.recordings) * clips_per_recording}")

    def _cached(self, rec):
        if rec.mp4 not in self._cache:
            self._cache[rec.mp4] = (read_vrs_times(rec.ts_csv),
                                    *open_loaders(rec, self.standardize), projector_for(rec))
        return self._cache[rec.mp4]

    def __len__(self):
        return len(self.recordings) * self.clips_per_rec

    def __getitem__(self, idx):
        rec = self.recordings[idx % len(self.recordings)]
        try:
            return self._sample(rec)
        except Exception as e:
            log.warning(f"[dataset] clip failed for {Path(rec.mp4).name}: {e}")
            return self._null()

    def _sample(self, rec):
        vrs, gl, hl, proj = self._cached(rec)
        if len(vrs) < self.span + 1:
            return self._null()
        start = int(np.random.randint(0, len(vrs) - self.span))
        obs_idx, tgt_idx = clip_indices(start, self.T, self.stride, self.target_slots)

        all_idx = sorted(set(obs_idx + tgt_idx))
        frames = load_frames(rec.mp4, all_idx, size=self.img_size)
        pos = {j: k for k, j in enumerate(all_idx)}
        obs_frames = frames[[pos[j] for j in obs_idx]]
        tgt_frames = frames[[pos[j] for j in tgt_idx]]

        pt_idx = point_indices(obs_idx, self.stride, self.target_slots, self.future_points,
                               len(vrs))
        sig = signals.read(gl, hl, [int(vrs[j]) for j in pt_idx], self.T, proj)
        pts = signals.points(sig, proj, self.standardize, self.grid)
        return (obs_frames, tgt_frames, *(torch.from_numpy(x) for x in pts))

    def _null(self):
        T, sz, n = self.T, self.img_size, len(self.target_slots)
        return (torch.zeros(T, 3, sz, sz), torch.zeros(n, 3, sz, sz),
                *(torch.from_numpy(x) for x in signals.null_points(T)))


def fixed_val_clips(video_dir, gaze_dir, participants, T, stride, target_slots,
                    n_recordings, n_clips, img_size=256, grid=GRID, seed=VAL_SEED,
                    future_points=False):
    """
    The held-out clips of the check after each epoch. The same recordings, seed and sampler
    every epoch, so the trend of the gain is comparable across epochs. Video is not decoded
    here, only the indices and the points, so the list exists before the GPU is used.

    With `future_points` the points are those the future arm trains on (point_indices). The
    clip positions depend only on the seed, so the same clips come back either way.
    """
    rng = np.random.RandomState(seed)
    recs = find_recordings(video_dir, gaze_dir, participants)[:n_recordings]
    span = (T - 1 + max(target_slots)) * stride
    out = []
    for rec in recs:
        vrs = read_vrs_times(rec.ts_csv)
        if len(vrs) < span + 1:
            continue
        gl, hl = open_loaders(rec, True)
        if gl is None and hl is None:
            continue
        proj = projector_for(rec)
        for _ in range(n_clips):
            start = int(rng.randint(0, len(vrs) - span))
            obs_idx, tgt_idx = clip_indices(start, T, stride, target_slots)
            pt_idx = point_indices(obs_idx, stride, target_slots, future_points, len(vrs))
            sig = signals.read(gl, hl, [int(vrs[j]) for j in pt_idx], T, proj)
            out.append((rec, obs_idx, tgt_idx,
                        signals.points(sig, proj, True, grid)))
    log.info(f"[val] {len(out)} clips from {len(recs)} recordings of {participants}")
    return out


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------

def arm_points(arm, batch_points, device):
    """
    The points the predictor receives for this arm. `none` is handled by holding alpha at 0,
    so it needs no change here; `future` is handled by the dataset, which reads the points at
    the target frames.
    """
    gaze_pt, gaze_val, hand_pt, hand_val = (x.to(device) for x in batch_points)
    if arm == "shuffled":
        perm = torch.randperm(gaze_pt.size(0), device=device)
        gaze_pt, gaze_val = gaze_pt[perm], gaze_val[perm]
        hand_pt, hand_val = hand_pt[perm], hand_val[perm]
    return gaze_pt, gaze_val, hand_pt, hand_val


def hide_points(batch_points, device):
    """Every point missing: the maps add nothing, whatever alpha is."""
    gaze_pt, gaze_val, hand_pt, hand_val = (x.to(device) for x in batch_points)
    return (torch.full_like(gaze_pt, signals.NO_POINT), torch.zeros_like(gaze_val),
            torch.full_like(hand_pt, signals.NO_POINT), torch.zeros_like(hand_val))


def freeze_signals(predictor):
    """
    The matched model: every signal parameter held where it starts, so the maps add nothing
    for the whole run. The vectors e are zero at initialisation, so holding them there is
    exactly a predictor without signals, with the same weights everywhere else.
    """
    names = ("e_gaze", "alpha_gaze", "e_left", "e_right", "alpha_hand")
    with torch.no_grad():
        for n in names:
            p = getattr(predictor, n, None)
            if p is None:
                continue
            if n.startswith("e_"):
                p.zero_()
            p.requires_grad_(False)


# ---------------------------------------------------------------------------
# Train and check
# ---------------------------------------------------------------------------

def forward_loss(encoder, predictor, batch, device, args, amp_dtype, loss_fn, hide=False):
    obs_f, tgt_f = batch[0], batch[1]
    pts = batch[2:]
    enc_obs = encode_independent(encoder, obs_f, device, args.normalize_reps,
                                 chunk=args.encode_chunk, amp_dtype=amp_dtype)
    enc_tgt = encode_independent(encoder, tgt_f, device, args.normalize_reps,
                                 chunk=args.encode_chunk, amp_dtype=amp_dtype)
    g_pt, g_val, h_pt, h_val = (hide_points(pts, device) if hide
                                else arm_points(args.arm, pts, device))
    use_amp = amp_dtype is not None and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        pred = predictor(enc_obs, g_pt, g_val, h_pt, h_val, target_slots=args.target_slots)
        pred = maybe_norm(pred, args.normalize_reps)
        loss = loss_fn(pred.float(), enc_tgt.float())
    return loss, pred.detach(), enc_tgt


def per_horizon_errors(pred, target, grid):
    """L1 and MSE at each target slot, so the gain can be read per horizon."""
    HW = grid * grid
    n = pred.size(1) // HW
    out = []
    for i in range(n):
        p = pred[:, i * HW:(i + 1) * HW].float()
        t = target[:, i * HW:(i + 1) * HW].float()
        out.append((F.l1_loss(p, t).item(), F.mse_loss(p, t).item()))
    return out


def train_one_epoch(encoder, predictor, loader, optimizer, scheduler, device, epoch,
                    args, amp_dtype, loss_fn, jsonl=None):
    predictor.train()
    total = 0.0
    n = 0
    t0 = time.time()
    for step, batch in enumerate(loader, 1):
        loss, _, _ = forward_loss(encoder, predictor, batch, device, args, amp_dtype, loss_fn)
        optimizer.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            [p for p in predictor.parameters() if p.requires_grad], max_norm=1.0).item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item()
        n += 1
        if step % args.log_every == 0:
            sps = step / (time.time() - t0)
            st = signal_strength(predictor)
            log.info(f"epoch {epoch}  step {step:4d}/{len(loader)}  loss={total / n:.4f}  "
                     f"grad={grad:.3f}  "
                     + "  ".join(f"|{k}|={v:.4f}" for k, v in st.items())
                     + f"  sigma={predictor.sigma.item():.3f}  "
                     f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                     f"ETA={(len(loader) - step) / max(sps, 1e-9):.0f}s")
            if jsonl:
                jsonl.write(json.dumps({"t": "step", "epoch": epoch, "step": step,
                                        "loss": loss.item(), "avg": total / n, "grad": grad,
                                        "signal_strength": st,
                                        "sigma": predictor.sigma.item()}) + "\n")
                jsonl.flush()
    return total / max(n, 1), time.time() - t0


@torch.no_grad()
def validate(encoder, predictor, clips, device, args, amp_dtype):
    """
    The paired check on the held-out clips: the error with the points hidden (A) and with the
    real points (B), against the same targets, at each horizon. The gain is A - B.
    """
    predictor.eval()
    if not clips:
        return None
    acc = None
    for rec, obs_idx, tgt_idx, pts in clips:
        all_idx = sorted(set(obs_idx + tgt_idx))
        frames = load_frames(rec.mp4, all_idx, size=args.img_size)
        pos = {j: k for k, j in enumerate(all_idx)}
        obs_f = frames[[pos[j] for j in obs_idx]].unsqueeze(0)
        tgt_f = frames[[pos[j] for j in tgt_idx]].unsqueeze(0)
        batch = (obs_f, tgt_f, *(torch.from_numpy(x).unsqueeze(0) for x in pts))

        _, pred_hide, target = forward_loss(encoder, predictor, batch, device, args,
                                            amp_dtype, F.l1_loss, hide=True)
        _, pred_real, _ = forward_loss(encoder, predictor, batch, device, args,
                                       amp_dtype, F.l1_loss, hide=False)
        a = per_horizon_errors(pred_hide, target, predictor.grid)
        b = per_horizon_errors(pred_real, target, predictor.grid)
        if acc is None:
            acc = [[0.0] * 4 for _ in a]
        for i, ((l_a, m_a), (l_b, m_b)) in enumerate(zip(a, b)):
            acc[i][0] += l_a; acc[i][1] += l_b; acc[i][2] += m_a; acc[i][3] += m_b
    n = len(clips)
    return [[v / n for v in row] for row in acc]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="a V-JEPA 2-AC checkpoint; only its encoder is used")
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--gaze-dir", required=True)
    ap.add_argument("--participants", nargs="+",
                    default=["P01", "P02", "P03", "P04", "P05", "P06", "P07"])
    ap.add_argument("--out-dir", default="checkpoints/test11")
    ap.add_argument("--arm", choices=ARMS, default="maps")

    ap.add_argument("--context-steps", type=int, default=8)
    ap.add_argument("--frame-stride", type=int, default=8,
                    help="frames per time slot; 8 at 30 fps is 0.27 s, as in Tests 9 and 10")
    ap.add_argument("--target-slots", type=int, nargs="+", default=[2, 4],
                    help="slots after the last observed frame; 2 4 is about 0.53 s and 1.07 s")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--clips-per-recording", type=int, default=60)

    ap.add_argument("--pred-dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--num-heads", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=1.0, help="map width, in patches")
    ap.add_argument("--fixed-sigma", action="store_true", help="do not learn sigma")
    ap.add_argument("--no-hand", action="store_true", help="gaze maps only")

    ap.add_argument("--epochs", type=int, default=40,
                    help="a cap; --max-hours and --patience normally stop the run first")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="wall-clock budget for this run, 0 for none. Checked after each epoch")
    ap.add_argument("--patience", type=int, default=4,
                    help="stop after this many epochs with no improvement in the held-out error")
    ap.add_argument("--checkpoint-epochs", type=int, nargs="*", default=[3],
                    help="epochs to keep a copy of, whatever the held-out error does")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=500,
                    help="2 of 20 runs in Test 9 had a loss spike at the start")
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--loss", choices=["l1", "mse"], default="l1")
    ap.add_argument("--no-normalize-reps", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--encode-chunk", type=int, default=16)

    ap.add_argument("--val-participants", nargs="+", default=["P08"])
    ap.add_argument("--val-recordings", type=int, default=4)
    ap.add_argument("--val-clips", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.arm == "token":
        raise SystemExit("the token arm of Test 11 is not built yet: it needs a gaze token "
                         "in ego/ego_predictor.py")

    args.normalize_reps = not args.no_normalize_reps
    args.target_slots = tuple(args.target_slots)
    setup_logging(args.out_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    amp_dtype = None if args.no_amp else torch.bfloat16
    fps = 30.0
    horizons = [s * args.frame_stride / fps for s in args.target_slots]
    log.info(f"[arm] {args.arm}")
    log.info(f"[horizon] slots {args.target_slots} at {args.frame_stride} frames "
             f"= {', '.join(f'{h:.2f} s' for h in horizons)} after the last observed frame")
    log.info(f"[context] {args.context_steps} frames at {args.frame_stride} frames apart "
             f"= {args.context_steps * args.frame_stride / fps:.2f} s")

    encoder, _, _ = load_models(args.checkpoint, device, context_steps=args.context_steps)
    grid = args.img_size // 16
    predictor = ego_predictor(
        embed_dim=encoder.embed_dim, pred_dim=args.pred_dim, depth=args.depth,
        num_heads=args.num_heads, grid=grid, sigma=args.sigma,
        use_hand=not args.no_hand, learn_sigma=not args.fixed_sigma).to(device)
    if args.arm == "none":
        freeze_signals(predictor)
    trainable, total, sig_n = param_counts(predictor)
    log.info(f"[predictor] {total / 1e6:.1f}M parameters, {trainable / 1e6:.1f}M trainable, "
             f"{sig_n} of them the signal parameters; grid {grid}x{grid}")

    train_ds = EgoClipDataset(args.video_dir, args.gaze_dir, args.participants,
                              args.context_steps, args.frame_stride, args.target_slots,
                              args.img_size, args.clips_per_recording,
                              future_points=(args.arm == "future"), grid=grid)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_clips = fixed_val_clips(args.video_dir, args.gaze_dir, args.val_participants,
                                args.context_steps, args.frame_stride, args.target_slots,
                                args.val_recordings, args.val_clips, args.img_size, grid,
                                future_points=(args.arm == "future"))

    optimizer = torch.optim.AdamW(
        [p for p in predictor.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(args.epochs * len(loader), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=min(args.warmup_steps / total_steps, 0.3), anneal_strategy="cos")

    out = Path(args.out_dir)
    jsonl = open(out / "metrics.jsonl", "a")
    jsonl.write(json.dumps({"t": "config", **{k: str(v) for k, v in vars(args).items()}}) + "\n")

    def save(name, epoch, loss, score):
        torch.save({"predictor": predictor.state_dict(), "epoch": epoch, "loss": loss,
                    "val_score": score, "config": vars(args)}, out / name)

    t_start = time.time()
    best_score, best_epoch, stale = float("inf"), 0, 0
    stop = None
    for epoch in range(1, args.epochs + 1):
        loss, secs = train_one_epoch(encoder, predictor, loader, optimizer, sched, device,
                                     epoch, args, amp_dtype, LOSSES[args.loss], jsonl)
        rows = validate(encoder, predictor, val_clips, device, args, amp_dtype)

        # The held-out score is the error with the real points, averaged over the horizons.
        # For the `none` arm the points add nothing, so it is the plain prediction error.
        score = (sum(r[1] for r in rows) / len(rows)) if rows else loss
        improved = score < best_score - 1e-6
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            save("best.pt", epoch, loss, score)
        else:
            stale += 1
        if epoch in args.checkpoint_epochs:
            save(f"epoch{epoch}.pt", epoch, loss, score)

        elapsed = (time.time() - t_start) / 3600.0
        msg = [f"epoch {epoch} done  train_loss={loss:.4f}  {secs:.0f}s  "
               f"val={score:.5f}{'  (best)' if improved else f'  (best {best_score:.5f} @ {best_epoch}, stale {stale})'}  "
               f"elapsed={elapsed:.2f}h"]
        if rows:
            for h, (l_a, l_b, m_a, m_b) in zip(horizons, rows):
                msg.append(f"  +{h:.2f}s  L1 hide={l_a:.4f} real={l_b:.4f} "
                           f"delta={l_a - l_b:+.4f}  MSE delta={m_a - m_b:+.4f}")
        log.info("\n".join(msg))
        jsonl.write(json.dumps({"t": "epoch", "epoch": epoch, "train_loss": loss,
                                "seconds": secs, "horizons": horizons, "val": rows,
                                "val_score": score, "best_score": best_score,
                                "best_epoch": best_epoch, "stale": stale,
                                "elapsed_hours": elapsed}) + "\n")
        jsonl.flush()

        if args.max_hours and elapsed >= args.max_hours:
            stop = f"the budget of {args.max_hours:.2f} h is spent"
            break
        if args.patience and stale >= args.patience:
            stop = f"{stale} epochs with no improvement on the held-out error"
            break
        # the next epoch would run past the budget, so do not start it
        if args.max_hours and elapsed + secs / 3600.0 > args.max_hours:
            stop = f"another epoch would pass the budget of {args.max_hours:.2f} h"
            break
    else:
        stop = f"the cap of {args.epochs} epochs"

    save("final.pt", epoch, loss, score)
    jsonl.write(json.dumps({"t": "done", "epochs_run": epoch, "stopped_because": stop,
                            "best_score": best_score, "best_epoch": best_epoch,
                            "elapsed_hours": (time.time() - t_start) / 3600.0}) + "\n")
    jsonl.close()
    log.info(f"[done] stopped after epoch {epoch}: {stop}. "
             f"best held-out {best_score:.5f} at epoch {best_epoch} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
