# Egocentric Video Project — Context Handoff

> **Purpose of this file.** A self-contained briefing to paste into a fresh session so the
> assistant has full context on the project, the team, the data, the hypothesis, the current
> results, the supervisor's review, and the open decisions. Written for continuity, not as a
> formal report. Author: Erfan. Compiled from working discussions; project state ~mid-July 2026.

---

## 0. Who is who

- **Erfan (me)** — adapted the V-JEPA 2-AC predictor for human signals (replaced robot
  action/state projectors with gaze/hand encoders) and fine-tuned it on HD-EPIC P01–P07 with
  Ioana's projection-layer setup. My direction going forward is **language-aligned latent
  prediction**. I did not run new experiments in the last cycle — my contribution was study +
  direction.
- **Ioana** — literature review of recent V-JEPA / world-model work; hypotheses about encoder
  supervision and richer conditioning signals; built the projection-layer setup used in fine-tuning.
- **Parsa** — the hands-on experimental track: built and ran the EK100 probe evaluation of the
  behaviorally-trained predictor, with the control family. Reached the null result.
- **Ash** — supervisor. Reviewed the team report; his feedback is in §6.

---

## 1. What the project is

The base is **V-JEPA 2**, a video world model that learns by **prediction in embedding space,
not pixel space**. It masks part of a video and trains a **predictor** to guess the *latent
representation* of the missing part from the visible context. Nothing is rendered to pixels;
the target is another embedding (the "teacher" embedding).

Meta extended this to **V-JEPA 2-AC** (Action-Conditioned), which conditions the predictor on
**robot actions/states** (a 7D effector vector) — predicting the future latent *given* the
effector's motion.

**Our question is the human analogue:** can the predictor be conditioned on **human behavioral
signals** instead of robot actions — specifically:

- **Eye gaze** — leading indicator of intention; people fixate a target ~0.5–1 s before acting
  on it.
- **Hand pose** — 12D wrist+palm, the human counterpart of the robot end-effector.

…and does training with those signals produce representations that better encode **how human
actions (and ultimately intentions) unfold**?

**Shared goal (one line):** use human behavioral signals (gaze, hand) with V-JEPA-family world
models to build representations that encode how human actions and intentions unfold in
egocentric video.

---

## 2. The data

### 2.1 HD-EPIC — the training substrate (where signals exist)

Recorded with Project Aria glasses. Per-frame it carries:

- **Eye gaze** — real 2D eye-tracking fixation coordinates (not estimated saliency).
- **Hand pose** — 12D wrist+palm.
- **Rich text at multiple granularities** — recipe steps (goal-level) and fine-grained action
  descriptions (action-level).

Used **P01–P07** for fine-tuning the predictor. The **nested text hierarchy** (action → goal)
is the underexploited asset: it is exactly what a hierarchical action→goal model needs, and
exactly what a language-aligned space needs to be aligned to something richer than a verb-noun tag.

### 2.2 EK100 (EPIC-KITCHENS-100) — the evaluation benchmark

- 100 hours of unscripted kitchen activity, many kitchens/participants, densely narrated with
  **verb-noun pairs** (~97 verbs / ~300 nouns in the standard taxonomy — verify exact figures
  against the current release).
- **Task used: action anticipation** — observe up to time *t*, predict the action beginning at
  *t + τ*; standard anticipation time **τ = 1 s**.
- **Metric: class-mean recall@5** (top-5, because anticipation is genuinely ambiguous).
- **No gaze, no hand annotations.** ← this single fact shapes the whole protocol.

### 2.3 The train/eval split and its consequence

Train where the signals are (**HD-EPIC**), evaluate where the baselines are (**EK100**).
Because EK100 has no gaze/hand, at eval the predictor receives its learned
`gaze_mask` / `hand_mask` tokens instead of real signals. The predictor was trained with
**`signal_dropout: 0.4`** (40% of frames masked), so the fully-masked eval condition is
**in-distribution** — NOT a distribution shift.

**Implication:** the experiment measures **what behavioral *training* left in the weights**,
never **what live signals contribute at inference**. Any claim about live gaze/hand is out of
reach under this protocol.

**Terminology discipline (Parsa's):** the predictor outputs the "predicted latent
representation of the future token block," never "future frames"; signals at eval are
"mask-token substituted," never "absent."

---

## 3. The hypothesis and what it rests on

### 3.1 Hypothesis (falsifiable form)

> Conditioning a video world model's predictor on human behavioral signals (gaze, hand pose)
> during training produces latent representations that encode anticipatory structure — how
> actions unfold — better than the same architecture trained without those signals.

Decomposes into separable claims (this is why the control family + Phase 7 exist):

1. **The predictor adds something over the encoder** (does prediction training help at all?).
2. **Beyond token-count artifacts** (the confound that invalidated the first results cycle).
3. **Beyond persistence** (at ~1–2 s, "future ≈ present" is a strong null model).
4. **(Out of reach this cycle) Behavioral conditioning *specifically* helps** — vs. prediction
   training helping regardless of the signal. Only **Phase 7** (signal dropout 1.0) separates this.

### 3.2 Erfan's extension hypothesis

A **language-aligned** latent space gives behavioral conditioning room to help that a purely
visual space does not — because **concepts, not appearance, are what survive over long horizons**.

### 3.3 What the hypothesis rests on (the base)

**Load-bearing supports:**

- **Architectural precedent (strongest).** V-JEPA 2-AC already shows conditioning a latent
  predictor on effector actions/states works — for robots. Gaze+hand is the *human analogue* of
  that effector signal. Not a novel mechanism; a substitution into a mechanism shown to work.
- **Behavioral-science premise.** Gaze leads action by ~0.5–1 s (eye-hand span in natural
  tasks). This is why gaze should be *anticipatory*, not merely descriptive.
- **Embedding-space argument.** Latent prediction avoids the pixel-prediction pathology where
  averaging over plausible futures produces blur; in embedding space an averaged prediction at
  least stays semantically coherent. The family works on abstraction, not appearance.
- **Horizon evidence (Erfan's addition, from VL-JEPA).** VL-JEPA's EK100 anticipation advantage
  over V-JEPA 2 (same ViT-L-256px encoder) **widens monotonically with horizon**:
  +1.5 @1 s, +2.6 @2 s, +3.5 @4 s, +4.6 @10 s. Semantic/language-aligned structure helps most
  where visual continuity runs out — the regime where intent, not motion extrapolation, does the work.

**Assumptions that could break it (two already have consequences):**

- **Gaze carries info the visual embedding lacks.** If gaze is recoverable from the frozen
  encoder embedding (people look at salient objects, which are visible), the conditioning token
  is **redundant**. Untested premise underneath everything; cheapest to check. ← likely explains the null.
- **2 s is a long enough horizon to see the effect.** The pre-registered interpretation of the
  null says it isn't. Untested until the horizon sweep.
- **Behavioral training leaves a trace measurable without live signals.** Forced by the dataset
  split; unavoidable under this protocol.
- **A rich enough language supervision source exists.** VL-JEPA's gains lean on rich text;
  EK100 verb-noun pairs are terse. HD-EPIC recipe steps are the proposed answer; still the
  untested pivot of the language-alignment direction.

---

## 4. VL-JEPA — the anchor for the language-aligned direction

Erfan's direction is **language-aligned latent prediction**: keep predicting in embedding space,
but align that space with language so the model predicts **concepts** rather than appearance.
VL-JEPA is the clearest existing instantiation and the main influence (not the subject —
the direction generalizes beyond one paper).

**Core idea.** Instead of generating a text answer token-by-token, predict the **embedding of
the answer (Ŝ_Y)** in a continuous space, loss computed in embedding space. Fixes two VLM
weaknesses: training inefficiency (many token-disjoint correct answers) and an autoregressive
inference bottleneck unfit for streaming video. Roughly **halves trainable parameters**
(no heavy decoder in the loop).

**Four components.**
- Frozen V-JEPA 2 ViT-L **X-Encoder** → visual embeddings **S_V**.
- **Y-Encoder** (EmbeddingGemma-300M, trained slowly at ×0.05 LR) → text target **S_Y**.
- **Predictor** (last 8 layers of Llama-3.2-1B, causal mask disabled → vision & query attend
  bidirectionally) → predicted embedding **Ŝ_Y**.
- **Y-Decoder** — inference only, selective, reads an embedding out as text when needed.

Training: **bi-directional InfoNCE** (alignment + uniformity), two stages — query-free
pretraining then query-conditioned SFT.

**Evidence that makes the direction credible.**
- **Widening anticipation gap (the anchor)** — table above.
- **WorldPrediction-WM** — SOTA at choosing which clip explains a state transition, beating
  GPT-4o / Claude-3.5-Sonnet / Gemini-2.0. Embedding space captures transition structure
  language decoding misses.
- **Controlled comparison** — same encoder/data/schedule, only the loss differs; embedding
  prediction wins on performance and sample efficiency at ~half the parameters.
- **Y-Encoder quality** — joint training makes the text encoder more semantically discriminative.

**Where VL-JEPA falls short for our use (gaps to close):**
- Frozen, third-person-biased encoder — gaze/pre-grasp shape/wrist velocity are just motion patches.
- No goal-level annotation — targets are local actions; space organized around actions not goals.
- Action-oriented queries — "what is happening?", not "what does the person intend next?"
- Short temporal window (~32 frames) — no mechanism for minutes-long goal context.

---

## 5. What has been done — the experiment and the null

### 5.1 Method (Parsa)

Frozen-representation probing: freeze encoder + predictor, feed their tokens to a deliberately
weak attentive probe (1 cross-attention block, 1 query, 25-config LR×WD sweep), train on EK100
action anticipation, report recall@5. If predictor tokens carry anticipatory info beyond the
encoder's, the probe should score higher.

### 5.2 The confound that reshaped the experiment (all 2026-06-17)

An unseeded preview hit **5.83%** vs a 4.27% baseline — looked like a win, dismantled same day:

- **Token-count confound.** Treatment fed 1764 tokens; control fed 1568. The pooler
  cross-attends over all tokens, so 196 extra keys can raise the score regardless of content.
  → control split into a family:
  - **6a** encoder-only, 1568 (does the predictor block add anything?)
  - **6b** +196 zeros = 1764 (does token count alone inflate?)
  - **6c** +last temporal block repeated = 1764 (does prediction beat persistence?)
  - runtime asserts pin the counts.
- **Original Phase 7 impossible.** The robot AC predictor needs real 7D robot vectors and has no
  mask token → can't run the masked protocol. Redefined: retrain the same architecture with
  `signal_dropout: 1.0` (never sees a real signal).
- **"Mask-token mismatch" framing was wrong.** Config revealed dropout 0.4, so masked eval is
  in-distribution. **Claim ceiling set:** this cycle can only show *a behaviorally-trained
  predictor* helps vs. the encoder alone; attributing gains to behavioral conditioning
  *specifically* needs Phase 7.

Gap decomposition: `5−6a` = total predictor effect; `5−6b` = beyond token count;
`5−6c` = prediction beats persistence.

### 5.3 Result — a clean, pre-registered null at ~2 s

Four conditions × 3 seeds, shared frozen ViT-G encoder + probe + EK100 P01–P05; only the token
set differs. Best val action R@5:

| Condition | Tokens | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|---|
| 6a encoder only | 1568 | 3.70% | 3.44% | running |
| 6b + zeros | 1764 | 3.60% | 3.59% | not yet launched |
| 6c + last block repeated | 1764 | 3.80% | 3.31% | not yet launched |
| 5 + behavioral predictor | 1764 | 3.58% | 3.49% | running |

- Gaps: 5−6a = −0.04pp, 5−6b = −0.06pp, 5−6c = −0.02pp. All within 3.3–3.8%; between-condition
  differences ≈ within-condition seed noise (6c alone swings 3.80 → 3.31).
- Independent paired A/B at ~2 s: **Δ ≈ +0.001 MSE** — two measurements agreeing on ≈ no effect.
- **Pre-registered interpretation:** at ~2 s visual continuity alone predicts well, leaving
  behavioral conditioning little room; if the effect is real it should appear at longer horizons.
- **Must NOT be reported as results:** the 5.83% preview (confounded, unseeded); any Phase 5 vs
  Phase 2 comparison (encoder size, probe depth, clip length confounds).

**Why the null is worth something:** the claim ceiling was set *before* the outcome, so it's
reportable without goalpost-moving. The open question is whether the null means gaze is
**redundant** with visual info, or whether **2 s was too short** — and those are distinguishable cheaply (§7).

### 5.4 Phase 7 status (the missing causal link)

`signal_dropout: 1.0` matched retrain isolates: `7−6a` = prediction helps regardless of signal;
`5−7` = the behavioral signal itself matters. Pipeline received; **blocker: P02–P07 HD-EPIC data
(only P01 local)** — preferred path is the colleague running it. Null makes Phase 7 more urgent.

---

## 6. Supervisor (Ash) review + Erfan's answers

### 6.1 Ash's review (verbatim substance)

- Solid grasp of VL-JEPA; the widening anticipation gap is the right anchor; one-change-at-a-time
  discipline and Direction-1-first sequencing are correct.
- **Concern 1:** Direction 1 is nearly the same intervention Parsa just tested; the horizon-growth
  prediction is consistent with his, but built on his infrastructure/controls rather than in parallel.
- **Concern 2:** the 65% relative gain at 10 s is **11.7 vs 7.1** — small absolute numbers where
  relative gains flatter.
- **Concern 3:** VL-JEPA leans on rich text targets, but EK100 narrations are terse verb-noun
  pairs — **what is your language supervision source?**
- **Q1:** Can you already regress gaze position from the frozen visual embedding? If yes,
  conditioning tokens add nothing — measure that marginal information first.
- **Q2:** For Direction 2, HD-EPIC's recipe-step annotations are natural weak goal-level
  supervision — why invent geometric goal-reading before exhausting them?
- **Q3:** Is your goal target a point or a distribution? If a distribution, what loss?

### 6.2 The unifying read

The three concerns collapse onto **one linchpin (the language-supervision source)** and
**one cheap experiment that should come first (gaze recoverability)**. Concern 3 *gates* Concern 1:
if the Y-target is terse EK100 text, the space isn't meaningfully language-aligned and Direction 1
reduces to Parsa's null. The whole review is the same "prove the cheap thing before the expensive
thing" discipline applied one level deeper.

### 6.3 Erfan's answers (as sent / to send)

- **Concern 2 — conceded.** Drop the 65% relative figure; report absolute deltas and that they
  grow monotonically (+1.5 → +2.6 → +3.5 → +4.6). The trend is the real evidence; honest about low
  absolute performance at long horizons.
- **Concern 1 reposition.** Don't run in parallel — take Parsa's exact controls and change **one
  thing**: swap the plain-V-JEPA-2 latent space for the **language-aligned** one. Then the test is
  not "do behavioral tokens help" (his) but "does a language-aligned space give conditioning room
  to help that a plain space didn't." His infra becomes leverage, not overlap.
- **Concern 3 / supervision source.** Use **HD-EPIC's rich text** (recipe steps + fine-grained
  action descriptions) as the Y-target — that's the answer to "where does language alignment come
  from." Fine-tune the already-fine-tuned predictor *again* on this text for conceptual (not purely
  visual) grounding. (If nothing beats verb-noun pairs, Direction 1 IS a re-run of Parsa — say so and pivot.)
- **Q1 — run it first, strongest move.** Regress gaze (and hand) from the **frozen ENCODER
  embedding** (not the predictor — encoder ≠ predictor; probing the gaze-fed predictor is circular).
  Measure recoverability **as a function of anticipation lead time**, not one number. Prediction:
  high recoverability at zero lead (eyes on salient object), lower 0.5–1 s ahead (eyes on
  not-yet-salient target) → marginal info ≈ 1 − recoverability **grows with horizon**. Needs
  ground-truth gaze → runs on **HD-EPIC**, not EK100. ~a day of probing; kills or justifies
  Direction 1 cheaply and explains the null.
- **Q2 — conceded.** Recipe steps first as goal/sub-goal supervision; keep LeWM-style label-free
  geometric trajectory-reading only as a fallback if step labels prove insufficient.
- **Q3 — point now, distribution as the real target.**
  - **Current:** point target. Predictor outputs one embedding per masked token; **L2** to the
    teacher embedding. One prediction, one target, no distribution.
  - Defensible at short horizons (future constrained). Breaks at long horizons: a point predictor
    averages competing futures → blur in pixel space, but stays semantically coherent in embedding
    space.
  - **Why point first:** simplest use of recipe-step labels; exhaust it before the complex loss;
    keeps Direction 1's loss unchanged so any gain is attributable to gaze conditioning alone.
  - **Distributional extension (Direction 2):** the right next step for goal inference. Clean
    embedding-space form = **soft-label contrastive / cross-entropy over a fixed bank of candidate
    goal embeddings**; target distribution = **empirical data counts** (how often each prefix, e.g.
    "onions + oil," actually led to each dish). Alternative framings floated: diffusion-denoising or
    energy-based scoring over possible future embeddings.
  - **Switch trigger:** move to distributional the moment the point model is *confidently wrong* on
    ambiguous prefixes.

> Note on the sent email: point 3's justification for hierarchy should be "goals emerge from 
> sequences of finer-grained steps" (not "because the target is a distribution" — that belongs in Q3).

---

## 7. What could be done next (ordered by cost/value)

1. **Gaze-recoverability pretest (do first, ~1 day).** Frozen encoder embedding → regress gaze,
   as a function of lead time, on HD-EPIC. Distinguishes "gaze redundant" from "2 s too short" —
   the two explanations of the null — before building anything.
2. **Horizon sweep (Parsa, primary interpretive experiment).** Phase 5 vs 6a at 1/3/5/10 s,
   matched probe per horizon; 3 s and 5 s first. Tests whether the 2 s null is a horizon artifact.
3. **Phase 7.** signal_dropout 1.0 retrain; separates "prediction helps" from "the signal helps."
   Blocked on P02–P07 data transfer.
4. **Language-alignment direction (Erfan).** Fine-tune predictor on HD-EPIC recipe/action text;
   reposition as a substrate swap on Parsa's controls (one-variable change).
5. **Goal-level inference (Direction 2).** Recipe steps as weak goal supervision; point target
   first, distributional (soft-label CE over goal-embedding bank) when ambiguity bites.

**Cross-cutting discipline (Ash credited this):** change one thing at a time; match token counts
under permutation-invariant poolers; never report single unseeded sweep maxima; verify shapes/
configs empirically; set the claim ceiling before seeing the outcome.

---

## 8. Glossary / gotchas for a fresh session

- **Encoder ≠ predictor.** Encoder = frozen ViT that embeds frames → S_V. Predictor = the module
  we condition (takes S_V + gaze/hand tokens) → predicted future latent. In the V-JEPA-2 lineage
  "encoder" is sometimes used loosely; keep it strict. **Q1 regresses from the ENCODER output.**
- **EK100 = EPIC-KITCHENS-100.** Same dataset family as "EPIC-KITCHENS," just the shorthand for the
  100-hour version. Not a different database. HD-EPIC is a *separate* Aria-based dataset (with
  gaze/hand/text) used for training.
- **"Behavioral training left in the weights" vs "live signals."** Under the HD-EPIC-train /
  EK100-eval split with mask-token substitution, only the former is measurable.
- **The null is not a failure.** Pre-registered claim ceiling; the open question is redundancy vs.
  horizon, resolved cheaply by the gaze pretest.
- **Numbers to never cite as results:** 5.83% preview; Phase 5 vs Phase 2.
- **Figures to sanity-check against the current dataset release:** EK100 verb/noun counts;
  anticipation τ; exact HD-EPIC participant/annotation coverage.
