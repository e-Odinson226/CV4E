"""
Tests for --shuffle-signals (finetune_ego.make_collate).

The control this flag implements is only valid if the shuffle destroys signal/frame
alignment and NOTHING else. These checks pin down exactly that: same shapes, same
multiset of values, same validity statistics, gaze and hand still aligned to each
other, frames untouched — and, for "time", that the alignment really is broken.

    $PY scripts/test_shuffle_signals.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from finetune_ego import make_collate

T, B = 8, 4
FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def make_batch(seed=0):
    """Signals carry their time index in every channel, so a permutation is visible."""
    g = torch.Generator().manual_seed(seed)
    batch = []
    for b in range(B):
        t = torch.arange(T, dtype=torch.float32)
        batch.append((
            torch.randn(T, 3, 8, 8, generator=g),          # ctx frames
            torch.randn(T, 3, 8, 8, generator=g),          # tgt frames
            (t[:, None] + 100 * b).repeat(1, 3),           # gaze  (T,3)
            (t % 2 == 0),                                  # gaze_valid
            (t[:, None] + 100 * b).repeat(1, 12),          # hand  (T,12)
            (t % 3 == 0),                                  # hand left valid
            (t % 4 == 0),                                  # hand right valid
        ))
    return batch


def main():
    torch.manual_seed(0)
    batch = make_batch()
    ref = make_collate("off")(make_batch())

    print("mode=off")
    off = make_collate("off")(make_batch())
    check("off leaves gaze untouched", torch.equal(off[2], ref[2]))
    check("off leaves validity untouched", torch.equal(off[3], ref[3]))

    print("mode=time")
    out = make_collate("time")(make_batch())
    check("shapes unchanged", [t.shape for t in out] == [t.shape for t in ref])
    check("frames untouched", torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1]))
    check("gaze values are a permutation of the originals",
          all(sorted(out[2][b, :, 0].tolist()) == sorted(ref[2][b, :, 0].tolist())
              for b in range(B)))
    check("validity statistics preserved per clip",
          all(int(out[3][b].sum()) == int(ref[3][b].sum()) for b in range(B)))
    # gaze and hand must move together: the control removes signal-to-FRAME alignment,
    # not the relationship between the two signals.
    check("gaze and hand permuted identically",
          torch.equal(out[2][:, :, 0], out[4][:, :, 0]))
    check("gaze_valid follows gaze",
          all(bool(out[3][b, i]) == (int(out[2][b, i, 0]) % 100 % 2 == 0)
              for b in range(B) for i in range(T)))
    check("hand validity follows hand",
          all(bool(out[5][b, i]) == (int(out[4][b, i, 0]) % 100 % 3 == 0)
              for b in range(B) for i in range(T)))
    check("no clip is left in its original order across 20 draws",
          any(not torch.equal(make_collate("time")(make_batch())[2], ref[2])
              for _ in range(20)))
    check("clips get independent permutations",
          any(not torch.equal(
              make_collate("time")(make_batch())[2][0, :, 0] -
              make_collate("time")(make_batch())[2][1, :, 0] % 100, torch.zeros(T))
              for _ in range(5)))

    print("mode=batch")
    out = make_collate("batch")(make_batch())
    check("every clip receives another clip's signals",
          all(int(out[2][b, 0, 0]) // 100 != b for b in range(B)))
    check("time order within a clip is preserved",
          all(torch.equal(out[2][b, :, 0] - out[2][b, 0, 0],
                          torch.arange(T, dtype=torch.float32)) for b in range(B)))
    check("frames untouched", torch.equal(out[0], ref[0]))
    check("validity follows its signal",
          all(bool(out[3][b, i]) == (int(out[2][b, i, 0]) % 100 % 2 == 0)
              for b in range(B) for i in range(T)))

    print("mode validation")
    try:
        make_collate("nonsense")
        check("unknown mode raises", False)
    except ValueError:
        check("unknown mode raises", True)

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
