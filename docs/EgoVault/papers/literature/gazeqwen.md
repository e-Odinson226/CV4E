---
type: paper
status: running
created: 2026-08-03
updated: 2026-08-03
tags: [paper, gaze, mllm, vjepa, streaming, conditioning, annotated]
---

# GazeQwen: Lightweight Gaze-Conditioned LLM Modulation for Streaming Video Understanding

**Trong-Thang Pham$^1$*, Hien Nguyen$^2$, Ngan Le$^1$**
$^1$University of Arkansas $^2$University of Houston
`tp030@uark.edu` `hvnguy35@central.uh.edu` `thile@uark.edu`

> [!info] 📌 How to read this file
> The paper's own text is left **unmodified**. Everything added is inside a *callout* like this one, so annotation never blends into the authors' words. Four kinds:
>
> | Callout | Means |
> | --- | --- |
> | `[!note]` 📖 | Explanation of a term, mechanism, or piece of notation |
> | `[!question]` ❓ | An answer to a specific question you asked |
> | `[!warning]` ⚠️ | Where the paper is shaky, unclear, or possibly wrong |
> | `[!tip]` 🧵 | How this connects to *your* thesis |
>
> A `+` after the type means "starts expanded"; swap it to `-` to collapse every annotation and read the paper clean. Related vault notes are linked as `[[wikilinks]]`.

## Abstract

Current multimodal large language models (MLLMs) cannot effectively utilize eye-gaze information for video understanding, even when gaze cues are supplied via visual overlays or text descriptions. We introduce GazeQwen, a parameter-efficient approach that equips an open-source MLLM with gaze awareness through hidden-state modulation. At its core is a compact gaze resampler ($\sim$1–5 M trainable parameters) that encodes V-JEPA 2.1 video features together with fixation-derived positional encodings and produces additive residuals injected into selected LLM decoder layers via forward hooks. An optional second training stage adds low-rank adapters (LoRA) to the LLM for tighter integration. Evaluated on all 10 tasks of the StreamGaze benchmark, GazeQwen reaches 63.9% accuracy, a +16.1 point gain over the same Qwen2.5-VL-7B backbone with gaze as visual prompts and +10.5 points over GPT-4o, the highest score among all open-source and proprietary models tested. These results suggest that learning where to inject gaze within an LLM is more effective than scaling model size or engineering better prompts. All code and checkpoints are available at <https://github.com/phamtrongthang123/gazeqwen>.

> [!note]+ 📖 The abstract, term by term
> **MLLM** — *multimodal large language model*. An LLM with a vision encoder attached, so it consumes images/video **and** text and emits text. Qwen2.5-VL, GPT-4o and Claude are MLLMs.
>
> **Parameter-efficient** — they never retrain the 7-billion-parameter model. It stays frozen and only ~9 M *new* parameters are trained (≈0.13% of it). This is why it fits on one GPU.
>
> **Hidden-state modulation** — the core idea, and the whole paper in two words. Every transformer layer hands the next layer one vector per token; that vector is the *hidden state*. Rather than change the model's **input** (prompt or pixels), they reach in mid-computation and **add** to those vectors. "Modulation" = adjusting a signal in flight.
>
> **Resampler** — a small module that takes a *variable* number of input tokens and returns a *fixed* number of output tokens, by having a fixed set of learned query vectors attend over the inputs. From Perceiver / Flamingo. Read "re-sample" as *re-express this on a different grid*, not in the statistical sense.
>
> **Fixation-derived positional encodings** — the (x, y) point the eye rests on, expanded from 2 raw numbers into a 256-dimensional vector. Mechanism in §2.1.
>
> **Additive residuals** — the thing being added. Here `residual` simply means "a correction term added to what was already there", i.e. `h ← h + R`. Not a regression residual.
>
> **Forward hooks** — a PyTorch feature letting you register a function that fires whenever a given module runs, and rewrite its output. This is *how* `R` gets injected without editing a single line of Qwen's source code.
>
> **LoRA** — *Low-Rank Adaptation*. Fine-tuning that never touches the original weights: freeze `W`, learn two skinny matrices `A` (d×8) and `B` (8×d), and use `W + BA` instead. The 8 is the *rank*. Cheap to train, and detachable afterwards.

> [!note]+ 📊 The claimed numbers — and what "pp" means
> **pp = percentage points**, the plain arithmetic difference between two percentages, *not* a relative change. 47.8% → 63.9% is **+16.1 pp** (also a +34% *relative* increase — hence the notation, which exists to keep those two readings apart).
>
> Three numbers to carry forward:
>
> - **63.9%** — GazeQwen's unweighted mean accuracy over StreamGaze's 10 tasks.
> - **+16.1 pp** over Qwen2.5-VL-7B — *the identical backbone*, given gaze as a visual overlay.
> - **+10.5 pp** over GPT-4o.
>
> Every question is 4-way multiple choice, so **chance is 25%** and the human ceiling 82.7%.

> [!warning]+ ⚠️ Hold one caveat while reading "+16.1 pp"
> GazeQwen is **trained on StreamGaze's own training split**. Every baseline it is compared against is **off-the-shelf and zero-shot**. So the gap bundles three separate things: gaze, a whole second visual encoder, and ~9 M parameters fitted to this benchmark's distribution. The paper never runs the control that would separate them (same architecture, gaze zeroed). Come back to this at §3.2.

## 1. Introduction

Applications such as AR-glass assistants and embodied agents require models that interpret continuous video streams in real time [5, 7, 13]. In egocentric settings, eye gaze reveals the user’s focus of attention and likely next actions [10, 12, 22], yet how to make MLLMs actually use gaze data remains open.

> [!note]+ 📖 Setting the scene — "streaming", "egocentric", and why gaze at all
> **Streaming video understanding** — the model watches a *continuous, open-ended* feed and must answer at arbitrary moments, without ever having seen the end. Contrast with ordinary video QA, where the model receives a finished clip and can attend over all of it at once. Streaming forces **causality**: at query time you have only the past.
>
> **Egocentric** — first-person, head-mounted camera. The camera moves with the wearer, so the frame *is* their field of view. That is what makes gaze usable here: a 2-D point on the image plane corresponds directly to a thing the person is looking at. In third-person video, gaze would need a whole extra step to relate the eye to the scene.
>
> **Why gaze** — in natural manipulation the eye lands on the target object **300–600 ms before the hand starts moving toward it** (the *eye–hand span*). So gaze is not merely a description of what is happening; it is a forward-planning signal leaking out ahead of the action.

> [!tip]+ 🧵 This is the same premise your thesis runs on
> "Gaze reveals focus of attention and **likely next actions**" is the identical behavioural claim behind [[1-introduction|Path 2]]. Both projects bet on gaze being *anticipatory*. Where they diverge is what consumes the signal: GazeQwen feeds it to an LLM to answer questions; you feed it to a JEPA predictor as conditioning `z`.

The StreamGaze benchmark [11] quantifies this gap: 8,521 gaze-conditioned QA pairs across 285 egocentric videos. All tested MLLMs, including GPT-4o (53.4%) and Qwen2.5-VL-7B (47.8%), lag well behind humans (82.7%).

> [!note]+ 📖 StreamGaze is a *benchmark*, not a model
> **StreamGaze** [11] is a dataset of questions, not a system — 8,521 four-way multiple-choice questions built over 285 egocentric videos that carry synchronised eye-tracking. Each question is written so that knowing where the wearer looked is *relevant* to answering it. GazeQwen is a model evaluated **on** StreamGaze; the two are by different groups.
>
> The numbers quoted here are the benchmark authors' own baseline sweep, not GazeQwen's work:
>
> | | Accuracy | |
> | --- | --- | --- |
> | Chance | 25.0% | 4 options per question |
> | Qwen2.5-VL-7B | 47.8% | best open 7B model |
> | GPT-4o | 53.4% | best proprietary model |
> | **Human** | **82.7%** | the ceiling being chased |
>
> The ≈29 pp human-vs-GPT-4o gap is the paper's stated motivation. Keep in mind it is a gap in the *aggregate* — §3.2 shows it is distributed very unevenly across the 10 tasks, and that unevenness turns out to be the most interesting thing in the paper.

Notably, none of the three prompting strategies tested in that work (textual coordinates, visual overlays, saliency maps) reliably surpass the gaze-free baseline, suggesting that surface-level formatting is insufficient. Prior gaze-aware approaches either overlay gaze markers without training signal [16], or train task-specific models that do not generalize to streaming scenarios [9].

> [!note]+ 📖 The three failed prompting strategies — the paper's whole justification
> These are the obvious ways to hand an MLLM gaze *through its normal input*, all three tested by the StreamGaze authors:
>
> 1. **Textual coordinates** — append "the user is looking at (0.42, 0.67)" to the prompt. 2. **Visual overlays** — literally draw a dot or crosshair onto the frame before encoding it. 3. **Saliency maps** — a *saliency map* is a per-pixel heat map of visual importance; here, a blurred blob centred on the fixation, supplied as an extra image or channel.
>
> **The reported outcome is this paper's entire licence to exist: none of the three reliably beat the gaze-free baseline.** Adding gaze through the input sometimes made results *worse*. If input formatting had worked, none of the machinery in §2 would be needed.
>
> Worth noticing *why* it fails. None of these models was ever **trained** to interpret any of these formats. A red dot is out-of-distribution decoration to a model that has never once been rewarded for attending to red dots — so the failure is about missing supervision, not about gaze being useless. That reading is what motivates a *learned* pathway.

> [!note]+ 📖 The two prior approaches being dismissed
> **[16] (Peng et al., "In the eye of MLLM")** — overlays gaze markers and prompts a frozen model. "Without training signal" = nothing is ever fine-tuned, so the model must already know what the marker means. This is strategy 2 above, benchmarked more thoroughly.
>
> **[9] (AssistGaze / GazeVQA)** — the opposite trade: genuinely trained on gaze data, but as a small task-specific model on a different dataset. It appears in Table 1 at **24.0%**, i.e. *below the 25% chance floor* — see the warning at §3.2 about broken baselines.
>
> The gap being claimed: nobody has both **trained** the gaze pathway *and* kept a general-purpose streaming model. That is the slot GazeQwen fills.

We take a different approach with GazeQwen. Instead of modifying the input or retraining the backbone, we attach a small resampler module that directly steers the LLM’s internal representations based on gaze. It encodes fixation coordinates as sinusoidal positional signals, fuses them with frozen V-JEPA 2.1 video features [3], and injects the result as additive residuals at selected LLM layers via PyTorch forward hooks. An optional second stage adds LoRA adapters [8] for tighter integration. The design is low-cost ($\sim$5–9 M trainable parameters, single-GPU training), modular (gaze module attaches/detaches without reloading weights), and effective: GazeQwen scores 63.9% on StreamGaze, +16.1pp over the same Qwen2.5-VL backbone and +10.5pp over GPT-4o.

> [!note]+ 📖 The pitch, and the three claimed properties
> The load-bearing sentence is *"Instead of modifying the input or retraining the backbone, we attach a small resampler module that **directly steers the LLM's internal representations**."* There are only three places you can intervene, and the paper is taking the third:
>
> | Where you intervene | Cost | Their verdict |
> | --- | --- | --- |
> | The **input** (prompt / overlay / saliency) | free | doesn't work — see above |
> | **Retrain** the backbone on gaze data | enormous | never attempted |
> | The **hidden states**, mid-forward-pass | ~9 M params | the contribution |
>
> The three claimed properties, and what each actually rests on:
>
> - **Low-cost** — ~5–9 M trainable parameters, one GPU. For scale, 9 M is ≈0.13% of 7 B.
> - **Modular** — this follows *from the hooks specifically*. A hook can be `remove()`d, leaving
>   the model bit-for-bit original. So gaze becomes an optional attachment rather than a fork of the weights — you can serve one model and toggle gaze per request.
> - **Effective** — the 63.9% / +16.1 pp / +10.5 pp headline, with the caveat noted above.
>
> Note the sleight of hand in "$\sim$5–9 M" here versus "$\sim$1–5 M" in the abstract. The abstract counts only the resamplers; this counts resamplers + LoRA. Neither figure survives arithmetic — see the parameter-count warning in §2.2.

## 2. Method

GazeQwen equips an open-source MLLM with gaze awareness through hidden-state modulation (Figure 1). A compact gaze resampler encodes V-JEPA 2.1 video features together with fixation-derived positional encodings and produces additive residuals injected into selected LLM decoder layers via forward hooks.

> [!info]+ 🗺️ Orientation — the whole method on one page
> Read §2 with this picture in hand. Everything below is one of these boxes.
>
> ```
>  video ──► frozen V-JEPA (ViT-B/16, 384²) ──► F_t ∈ R^{HW×768}        §2.1 │ interpolate onto Qwen's token grid ▼ X = F_t·W_in ∈ R^{HW×256} │ gaze ──► active fixations A_t ──► mean of sinusoidal PE ──► g_t ∈ R^256   §2.1 │ ┌──────────────────────┴──────────────────────┐ │  32 learned latents, 2 cross-attn blocks     │   §2.2 │  (gaze added to the KEYS)                    │ │  reverse cross-attn back out to HW tokens    │ └──────────────────────┬──────────────────────┘ ▼ R_t = (…)·W_out ∈ R^{HW×3584}   (W_out zero-init) │ h_l[visual positions] += α_l · R_t   at l ∈ {6,13,20,27} of 28    §2.3 via PyTorch forward hooks
> ```
>
> Two things to hold onto: 1. **Nothing enters through the prompt.** The LLM's input is unchanged; the intervention is entirely mid-forward-pass. 2. **This diagram runs four times, independently** — one full resampler per injection depth.

### 2.1. Inputs

**Video features.** Given a video clip, we sample $T$ frames and pass them through a frozen V-JEPA 2.1 encoder [3] (ViT-B/16, $384\times384$). We use a *separate* visual encoder rather than reusing the MLLM’s own vision backbone because the host model’s visual features are optimized for language grounding and may discard fine-grained spatial detail needed for gaze alignment; V-JEPA’s self-supervised spatiotemporal representations retain this information without task-specific bias. For each temporal step $t \in \{1, \ldots, T\}$, the encoder produces a spatial feature grid $\mathbf{F}_t \in \mathbb{R}^{HW \times d_v}$, where $H, W$ are the spatial grid dimensions and $d_v=768$. The V-JEPA output is interpolated (trilinear temporal, bilinear spatial) to align with the host LLM’s visual token grid.

> [!note]+ 📖 Decoding "$\mathbf{F}_t \in \mathbb{R}^{HW \times d_v}$" and the encoder spec
> **Frozen** — the encoder's weights are never updated. It is used purely as a fixed feature extractor: video in, vectors out, no gradients flowing back into it.
>
> **ViT-B/16, 384×384** — a Vision Transformer, "Base" size (~86 M params), splitting the image into **16×16 pixel patches**. At 384×384 input that gives 384/16 = 24 patches per side, so **24 × 24 = 576 patches per frame**. Each patch becomes one token. This is what `H` and `W` are: the *patch-grid* dimensions (24 and 24), not pixel dimensions.
>
> **$\mathbf{F}_t \in \mathbb{R}^{HW \times d_v}$** — read as "a table with `HW` rows and `d_v` columns". One row per patch (576 of them), each row a **768-number vector** describing that patch. `d_v` = 768 is just the width of V-JEPA-Base's representation. So `F_t` is a *grid of descriptions*, one per location in the frame — the spatial layout is preserved.
>
> **The subscript `t`** — one such grid per sampled timestep, `t ∈ {1…T}`.

> [!note]+ 📖 What "interpolated (trilinear temporal, bilinear spatial)" is fixing
> A problem of **mismatched grids**. V-JEPA emits a 24×24 grid at its own frame rate; Qwen2.5-VL tokenizes video on a *different* grid (it uses dynamic resolution and merges 2×2 patch blocks). To add a residual to Qwen's visual token *number 137*, you need a V-JEPA vector that corresponds to the same patch of the same frame. The grids don't line up, so they resample:
>
> - **Bilinear (spatial)** — a new value at a point between grid cells is a weighted blend of the
>   4 surrounding cells, weights set by distance. "Bi-" = interpolating along 2 axes (x, y).
> - **Trilinear (temporal)** — the same idea across 3 axes, adding time: blend the 8 surrounding
>   samples in (x, y, t). Used because V-JEPA's tokens are **tubelets** — each token covers 2 adjacent frames, not one — so its time axis is coarser than the frame rate and needs stretching too.
>
> This is ordinary image-resizing arithmetic, applied to feature vectors instead of pixels.

> [!warning]+ ⚠️ "V-JEPA 2.1" is cited to the V-JEPA **1** paper
> Reference [3] is Bardes et al., *Revisiting feature prediction for learning visual representations from video* (arXiv 2404.08471, 2024) — that is **V-JEPA 1**. The paper says "V-JEPA 2.1" here, in §3.1 and in Table 2, and never cites a V-JEPA 2 paper at all.
>
> Since you maintain a vendored V-JEPA 2 fork, this is worth resolving against `github.com/phamtrongthang123/gazeqwen` before citing: which checkpoint is actually loaded changes what the (B) ablation in §3.3 is really comparing. See [[vjepa]].

> [!tip]+ 🧵 They use a ViT-B/16; you use ViT-g
> Same architectural family, very different scale — their `d_v` = 768 against your encoder's **1408** dims and 256 patch tokens ([[4-results#^t4|T4]]). Also note their stated reason for bolting on a *separate* encoder — that the host model's features "may discard fine-grained spatial detail needed for gaze alignment" — is an *assumption they never test*. Your EXP-003 is precisely a test of the neighbouring question (how much gaze information a frozen V-JEPA encoder linearly carries), and you have the number: **0.001 across unseen people**. Neither paper has that measurement.

**Gaze scanpath.** Eye gaze is represented as a scanpath $\mathcal{S} = \{(x_i, y_i, t_i, \Delta t_i)\}_{i=1}^N$, where $(x_i, y_i) \in [0, 1]^2$ are normalized fixation coordinates, $t_i$ is the midpoint timestamp, and $\Delta t_i$ is the duration. A fixation is active at frame time $t$ if $|t - t_i| \leq \Delta t_i/2$; we denote the active set as $\mathcal{A}_t$.

> [!note]+ 📖 Fixations, saccades, and what the notation says
> **Fixation vs saccade** — the eye does not pan smoothly; it *jumps*. A **saccade** is a jump (20–200 ms, during which vision is largely suppressed). A **fixation** is a pause between jumps where the eye is roughly still, typically **200–300 ms**, and it is during fixations that you actually see. A **scanpath** is the resulting sequence: fixation, jump, fixation, jump.
>
> So the raw eye-tracker signal (a sample every few ms) has already been reduced by a fixation detector into a list of *events*. Each event `i` in $\mathcal{S}$ is four numbers:
>
> | Symbol | Meaning |
> | --- | --- |
> | $(x_i, y_i) \in [0,1]^2$ | where, as a fraction of frame width/height. $[0,1]^2$ means "a pair, each between 0 and 1" — resolution-independent, so 0.5, 0.5 is always the centre |
> | $t_i$ | *midpoint* timestamp of the fixation |
> | $\Delta t_i$ | how long it lasted |
> | $N$ | how many fixations the whole clip contains |
>
> **The active test** — $|t - t_i| \le \Delta t_i / 2$ just says "does frame time `t` fall inside fixation `i`'s time span?" Storing the midpoint and duration means the span is $[t_i - \Delta t_i/2,\; t_i + \Delta t_i/2]$, and being within half a duration of the midpoint is exactly being inside it. $\mathcal{A}_t$ ("the active set") is the set of fixations live at that instant.

> [!warning]+ ⚠️ The averaging in Eq. (1) is almost certainly a no-op
> Fixations **partition time** — the eye is in exactly one fixation, or mid-saccade, at any instant. Detectors emit non-overlapping intervals. So for an instantaneous frame time `t`, $|\mathcal{A}_t|$ can only be **0 or 1**, and the "mean over active fixations" in Eq. (1) averages over at most one element.
>
> The paper nonetheless motivates its encoding choice by "the sparse and **variable-count** nature of fixation inputs," which only makes sense if $|\mathcal{A}_t| > 1$ can happen. Three ways it could: the frame represents a *window* rather than an instant (likely, given tubelets span 2 frames); the fixation detector emits overlapping candidates; or gaze comes from both eyes separately. **None is stated.** Worth checking in the code — it changes whether Eq. (1) is a real design decision or decoration.
>
> The `= 0` fallback when $\mathcal{A}_t = \emptyset$ is the more consequential branch: during every saccade, gaze conditioning is simply switched off. Given saccades are frequent, a meaningful fraction of timesteps get **no gaze signal at all**, and the model has no way to distinguish "mid-saccade" from "no eye tracker".

> [!tip]+ 🧵 Their gaze is 2-D image-plane; yours is 3-D head-relative
> They use $(x, y)$ — a point *on the frame*, which is directly a pointer into the patch grid. Yours is `(yaw, pitch, depth)` in [[3-method]] — a direction in 3-D head coordinates. The difference matters more than it looks: an image-plane coordinate can be turned into "which patch" by arithmetic, whereas yaw/pitch needs the camera intrinsics to become a patch index. Anything downstream must learn that conversion implicitly. Also: they get **fixation events** (already segmented, with durations); your pipeline uses **per-sample gaze** looked up by VRS timestamp. Theirs has the saccades stripped out; yours does not.

**Gaze encoding.** Active fixations are encoded using DETR-style sinusoidal positional encoding [4] with latent dimension $d_l=256$. Sinusoidal encodings are parameter-free and provide a smooth, continuous mapping from coordinates to high-dimensional space, making them well-suited for the sparse and variable-count nature of fixation inputs. Multiple active fixations are averaged:

$$
\mathbf{g}_t = \frac{1}{|\mathcal{A}_t|} \sum_{i \in \mathcal{A}_t} \text{PE}(x_i, y_i) \in \mathbb{R}^{d_l}. \quad (1)
$$

When $\mathcal{A}_t = \emptyset$, we set $\mathbf{g}_t = \mathbf{0}$.

> [!note]+ 📖 Sinusoidal positional encoding — what it is and why it beats raw numbers
> The job: turn 2 numbers `(x, y)` into a 256-number vector that a neural network can use well. The naive option is a learned `Linear(2 → 256)`, and the reason it is bad is instructive.
>
> **Sinusoidal PE** feeds the coordinate through sines and cosines at many different frequencies. For coordinate `x` and dimension index `k` (of `d` total):
>
> $$\text{PE}_{2k}(x) = \sin\!\left(\frac{x}{10000^{2k/d}}\right), \qquad \text{PE}_{2k+1}(x) = \cos\!\left(\frac{x}{10000^{2k/d}}\right)$$
>
> The frequencies are **geometrically spaced** — some components wrap many times across the image (fine detail), others barely once (coarse position). For 2-D, half the dimensions encode `x` and half `y`, then they are concatenated. Same construction as in the original Transformer; "DETR-style" [4] just means the standard 2-D image version of it.
>
> Three properties that matter:
>
> - **Parameter-free** — no weights to learn, so it cannot overfit and works from step 0.
> - **Smooth** — nearby coordinates map to nearby vectors, so the model generalizes across
>   positions it never saw exactly.
> - **High-frequency** — and this is the real win. A linear map on `(x, y)` produces a
>   **rank-2** signal: everything gaze can ever say lives in a fixed 2-D plane inside the 256-D space, and 0.50 vs 0.52 differ by a barely-perceptible nudge along one direction. The sinusoidal expansion spreads those two numbers across all 256 dimensions with the high-frequency components making *small* coordinate differences produce *large* vector differences. Networks have a known bias toward learning low frequencies first, and this sidesteps it. (Same reasoning as Fourier features in NeRF.)
>
> This is what §3.3's **(G)** ablation is measuring, and it is their largest reported effect.

> [!tip]+ 🧵 This is the cheapest change available to your predictor
> [ego_predictor.py:57](../../../../ego/ego_predictor.py) is `self.gaze_proj = nn.Linear(gaze_dim, predictor_embed_dim)` — a **`Linear(3 → 384)`**, which is exactly the naive option above. It is **rank ≤ 3**: the entire gaze channel occupies a fixed 3-dimensional subspace of the predictor's hidden space, for every input, permanently.
>
> And your inputs are *angles*, which makes it worse — a linear map is close to the worst parameterization of an angle, since 30° vs 32° is a tiny displacement along one fixed direction that 24 attention blocks then have to amplify into a meaningful distinction.
>
> Swapping to `Linear(fourier_features(gaze, n_bands≈32) → 384)` is self-contained, needs no pipeline changes, and is motivated by both this paper's largest ablation and the spectral-bias argument. Relevant to hypothesis H2 ("the model never learned to use it") in [[1-introduction#^hypotheses|the hypotheses table]].

### 2.2. Gaze-Conditioned Resampler

The resampler takes $\mathbf{F}_t$ and $\mathbf{g}_t$ and produces a residual in the LLM’s hidden-state space. V-JEPA tokens are projected to $\mathbf{X} = \mathbf{F}_t \mathbf{W}_{in} \in \mathbb{R}^{HW \times d_l}$. The gaze vector is broadcast to $\mathbf{G} \in \mathbb{R}^{HW \times d_l}$. $K=32$ learnable latents $\mathbf{L} \in \mathbb{R}^{K \times d_l}$ serve as an information bottleneck: by forcing all visual–gaze interactions through a small set of latent vectors, the resampler must learn to distill the scene into a compact, gaze-relevant summary rather than simply copying all spatial tokens.

> [!note]+ 📖 The three ingredients: $\mathbf{X}$, $\mathbf{G}$, and the latents $\mathbf{L}$
>
> **$\mathbf{X} = \mathbf{F}_t\mathbf{W}_{in}$** — a plain linear layer squeezing each patch vector from V-JEPA's 768 dims down to the resampler's working width $d_l = 256$. Purely dimensional bookkeeping; `HW` rows unchanged, so the spatial grid survives.
>
> **"The gaze vector is broadcast to $\mathbf{G} \in \mathbb{R}^{HW \times d_l}$"** — *broadcast* means copied, unchanged, into every row. $\mathbf{g}_t$ is one 256-vector; $\mathbf{G}$ is that same vector stacked `HW` times. **No row differs from any other.** Hold onto this — it is the hinge of the warning below.
>
> **$K = 32$ learnable latents $\mathbf{L}$** — 32 vectors of 256 numbers, initialised randomly and *trained*, not computed from the input. They are the same 32 vectors for every video. Think of them as 32 standing questions the module learned to ask of any scene ("what is being held?", "what is near the fixation?"), which get answered by attending to this particular frame.
>
> **"Information bottleneck"** — 576 spatial tokens must be squeezed through 32 latents. Because 32 ≪ 576, the module *cannot* pass everything through and is forced to select. The claim is that being forced to select is what makes it summarise rather than copy.
>
> [!note]+ 📖 Cross-attention from scratch — what Q, K and V actually do
> Needed to read Eqs. (2)–(4). Attention is a **soft lookup in a table.**
>
> Every attention op has three roles, each a different linear projection of some input:
>
> - **Query (Q)** — "what am I looking for?" One per output slot.
> - **Key (K)** — "what am I?" One per input item; the *address* used for matching.
> - **Value (V)** — "what do I hand over if selected?" One per input item; the *payload*.
>
> The computation, matching Eq. (4): 1. **Score** every query against every key: $\mathbf{Q}\mathbf{K}^\top$. A dot product, so it is large when a query and key point the same way — a similarity. 2. **Scale** by $\sqrt{d_l}$. Dot products of $d$-dimensional vectors grow like $\sqrt{d}$, and feeding large numbers to softmax saturates it into a near-one-hot argmax with vanishing gradients. Dividing keeps scores in a trainable range. This is why the $\sqrt{d}$ is there and not some other constant. 3. **Softmax** over the input axis, turning scores into weights that are positive and sum to 1 — "what fraction of my attention goes to each input?" 4. **Blend the values** with those weights: $\text{softmax}(\cdot)\mathbf{V}$.
>
> **Self- vs cross-attention:** in *self*-attention Q, K, V all come from one sequence. In *cross*-attention the queries come from one place and keys/values from another. Here queries come from the latents ($\mathbf{Q} = \mathbf{L}\mathbf{W}_Q$) and keys/values from $[\mathbf{X}; \mathbf{L}]$ — the image tokens *with the latents appended*, so latents can read both the image and each other in one op. (Short definitions: [[2-background#Terms|the terms list]].)

> [!question]- ❓ Deep Dive: Resampler Math & Architecture Questions
>
> **1. What is $[\mathbf{G}; \mathbf{0}]$?** `[Shape Math]`
> It is a vertical concatenation. $\mathbf{G}$ is the gaze vector broadcasted to match the number of image patches ($HW \times 256$). $\mathbf{0}$ is a block of zeros matching the number of latents ($K \times 256$). Stacking them into $[\mathbf{G}; \mathbf{0}]$ creates a single matrix of shape $(HW + K) \times 256$.
>
> **2. How are $[\mathbf{X}; \mathbf{L}]$ and $[\mathbf{G}; \mathbf{0}]$ related?** `[Alignment]`
> They are perfectly structurally aligned. $[\mathbf{X}; \mathbf{L}]$ stacks the $HW$ visual tokens on top of the $K$ latents. $[\mathbf{G}; \mathbf{0}]$ is the exact same size. When they are added together during the Key calculation, the alignment ensures that $\mathbf{X}$ (visuals) gets paired with $\mathbf{G}$ (gaze bias), and $\mathbf{L}$ (latents) gets paired with $\mathbf{0}$ (no gaze bias).
>
> **3. What are $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_G, \mathbf{W}_V$?** `[Parameters]`
> They are standard learnable weight matrices (Linear layers). $\mathbf{W}_Q$ projects inputs into Queries, $\mathbf{W}_K$ into Keys, and $\mathbf{W}_V$ into Values. $\mathbf{W}_G$ is an extra learned projection specifically for the gaze vectors before they are added to the keys. During training, gradient descent updates these matrices to optimize the attention lookups.
>
> **4. What is Forward vs. Reverse Cross-Attention?** `[Core Concept]`
> *Cross-Attention* generally is an operation where one sequence (Queries) looks into a *different* sequence (Keys/Values) to extract relevant info.
>
> - **Forward (Step 3):** The $K=32$ latents act as Queries to look at the 576 visual patches. It is a many-to-few operation that *compresses* the large image into a small summary.
> - **Reverse (Step 4):** The 576 visual patches act as Queries to look back at the 32 latents. It is a few-to-many operation that *expands* the summary back out so it matches the spatial image grid.
>
> **5. Why do Latents form Queries in Step 3, but Visuals ($\mathbf{X}$) form Queries in Step 4?** `[Architecture]`
> The *Queries* dictate the output size of an attention block. In Step 3, we want to compress 576 patches down to 32 latents. So the 32 latents act as Queries, resulting in 32 outputs. In Step 4, we need to inject the information back into the LLM, which expects a full grid of 576 patches. Thus, the 576 visual tokens ($\mathbf{X}$) must act as Queries to produce 576 outputs.
>
> **6. Why are the Forward and Reverse steps drawn differently?** `[Mechanism]`
> Both use Softmax! The flowchart just abbreviated Step 4 for space, but Reverse Cross-Attention still computes $\text{Softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d}) \mathbf{V}$. They differ in the *residual* step. Step 3 is a full Transformer block, so it updates the latents iteratively: $\mathbf{L}_{new} = \mathbf{L}_{old} + \text{Attention}$. Step 4 is just a single projection step to map the latents back to the image grid, so there is no residual update to $\mathbf{X}$—it just outputs the final residual $\mathbf{R}_t$.
>
> **7. What is `AttnOut` and the Skip Connection? Is the diagram correct?** `[Mechanism]`
> Yes, it is correctly drawn. `AttnOut` is the direct output of the attention mechanism (the weighted sum of Values). A "Skip Connection" (or residual connection) takes the original input before the attention block ($\mathbf{L}$) and adds it directly to the output ($\mathbf{L} + \text{AttnOut}$). This stabilizes training and allows gradients to flow smoothly through the network without getting blocked by the attention math.
>
> **8. What does "Summarized Latents" mean and why is it needed?** `[Information Bottleneck]`
> It means that all the useful visual information from the 576 image patches has been compressed down into just 32 vectors. This creates an **Information Bottleneck**. It is needed because computing attention across all 576 patches layer after layer is computationally expensive. By forcing the network to squeeze the scene into 32 latents, it is forced to learn to keep only the most important, gaze-relevant information and discard the rest.
>
> **$[\mathbf{X}; \mathbf{L}]$** — vertical concatenation: stack the 576 image rows on top of the 32 latent rows, giving 608 rows. **$\mathbf{L} \leftarrow \mathbf{L} + \ldots$** is a *residual update*: don't replace the latents, add to them, so information accumulates across the `B = 2` blocks instead of being overwritten.
>
> **LayerNorm + FFN ($4d_l$ hidden)** — the standard transformer-block tail. LayerNorm rescales each vector to zero mean and unit variance for training stability; the FFN is a two-layer MLP that expands 256 → 1024 → 256 (the "4×" is conventional), giving per-token nonlinear processing between attention steps.

Over $B=2$ cross-attention blocks, the latents attend over $[\mathbf{X}; \mathbf{L}]$ with gaze-biased keys. Adding the gaze encoding to the *keys* rather than the queries or values steers the attention distribution: tokens near the fixation point receive higher attention weight, while the aggregated content (values) remains unaltered, preserving visual semantics while spatially re-weighting them:

$$
\mathbf{Q} = \mathbf{L}\mathbf{W}_Q, \quad \mathbf{K} = [\mathbf{X}; \mathbf{L}]\mathbf{W}_K + [\mathbf{G}; \mathbf{0}]\mathbf{W}_G, \quad (2)
$$
$$
\mathbf{V} = [\mathbf{X}; \mathbf{L}]\mathbf{W}_V, \quad (3)
$$
$$
\mathbf{L} \leftarrow \mathbf{L} + \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_l}}\right)\mathbf{V}, \quad (4)
$$

> [!warning]+ ⚠️ Eq. (2) cannot do what the paragraph above it claims
> This is the paper's central mechanistic argument, and as printed the math does not support it.
>
> The claim: *"tokens near the fixation point receive higher attention weight […] spatially re-weighting them."* Work through it. For query $\mathbf{q}_i$ and a **spatial** token `j`, Eq. (2) gives the key $\mathbf{k}_j = \mathbf{x}_j\mathbf{W}_K + \mathbf{g}\mathbf{W}_G$, so the attention logit is
>
> $$\mathbf{q}_i \cdot \mathbf{k}_j \;=\; \underbrace{\mathbf{q}_i\cdot\mathbf{x}_j\mathbf{W}_K}_{\text{depends on } j} \;+\; \underbrace{\mathbf{q}_i\cdot\mathbf{g}\mathbf{W}_G}_{\text{call it } c_i,\ \textbf{same for every } j}$$
>
> The second term carries no `j` — because $\mathbf{G}$ was **broadcast**, every spatial token receives the *identical* key bias. And a constant added to every logit **cancels in the softmax**: $\text{softmax}(s_j + c) = \text{softmax}(s_j)$. So the relative attention *among* spatial tokens is exactly unchanged by gaze.
>
> What does survive: the latent block gets $[\mathbf{G}; \mathbf{0}]$ — a **zero** bias, not $c_i$. So the shift is between the two blocks, scaling spatial-token mass relative to latent mass by $e^{c_i}$. Gaze therefore acts as a **per-query scalar gate** on "how much do I read from the image versus from myself", **not** as a spatial selector.
>
> To actually get the claimed behaviour the bias must depend on position — e.g. $\mathbf{k}_j \mathrel{+}= \text{PE}(\text{pos}_j - \text{gaze})$, or an explicit proximity scalar. Broadcasting cannot produce it.
>
> **A consistency check that supports this reading:** §3.3's **(G)** ablation finds Coord-PE *beats* Gaussian heatmaps. Heatmaps are the natural way to get spatial selectivity — if spatial re-weighting were the mechanism, they should win, not lose. That coord-PE wins is evidence the module is exploiting a **global gaze-state signal**, which is what this analysis predicts.
>
> Two caveats before treating this as a bug: the equation may be simplifying the code (the repo is public), and a gate is a *real* channel — gaze still reaches the LLM and could plausibly drive the gains. What is unsupported is the *stated reason why* it works.

> [!note]+ 📖 Why a "reverse cross-attention" is needed at all
> A shape problem the bottleneck created. After the blocks you hold 32 latents — a *summary of the whole frame* with no spatial extent. But Eq. (6) must add a residual to **every one of Qwen's visual tokens**, each tied to a specific screen location. 32 ≠ 576, and the latents have no positions.
>
> So the attention is run **backwards**: now the 576 spatial tokens $\mathbf{X}$ are the queries and the 32 latents are the keys/values. Each spatial position asks "which parts of the summary are relevant to *me*?" and receives its own blend. Position-specific residuals, restored.
>
> Compress to 32 → process → expand back to 576. The compression forces summarisation; the expansion makes the summary addressable.

> [!note]+ 📖 Zero-initialisation — a small trick that matters a lot
> $\mathbf{W}_{out}$ maps 256 → $d_{llm} = 3584$ (Qwen2.5-VL-7B's hidden width) and starts as **all zeros**. So at step 0, $\mathbf{R}_t = \mathbf{0}$ and Eq. (6) adds nothing: the module is a guaranteed **no-op**, and the model behaves *exactly* like stock Qwen.
>
> Why this is not a detail. With random init, a fresh module injects **structured noise** into a pretrained network, degrading it immediately. The fastest way for training to recover is to **suppress the new channel** — drive its weights toward zero — and a pathway that has been pushed to zero to stop the damage may never come back. Zero-init removes the choice: the only direction that changes anything is one that *helps*. Gradients still flow (the input to $\mathbf{W}_{out}$ is non-zero, so its gradient is non-zero) — it is only the *output* that starts silent.
>
> Same trick as ControlNet's zero-convolutions, LoRA's `B = 0`, and AdaLN-Zero in DiT.

> [!tip]+ 🧵 Your projectors are randomly initialised, and this may matter
> [ego_predictor.py:57–62](../../../../ego/ego_predictor.py) initialises `gaze_proj` and `hand_proj` via `trunc_normal_`, and the mask tokens as `randn * 0.02`. So at step 0 you inject structured noise into a pretrained predictor — and per [ego_finetune.py:88](../../../../ego/ego_finetune.py) those params train at `lr_proj`, *higher* than everything else, because they are "randomly init, must learn fast". That is precisely the configuration where suppression-before-learning is most likely.
>
> Zero-initialising the projector output plus a learned scalar gate `α_gaze` (init 0) makes "use gaze" strictly-improving from a no-op start. **[corrected 2026-08-18 — do not do both.** With `W = 0` and `α = 0`, the projector's gradient is `α · x = 0` exactly, and `α`'s gradient is input-independent: the pathway cannot learn to use gaze. And zero-init does not port here anyway — our gaze is a *token*, not a residual, so a zero projector yields a zero token rather than a no-op. Gate toward `gaze_mask` with the projector at normal init: see [[3-method]].** **And the scalar is free instrumentation:** log `α_gaze` over training and you can read off whether the model ever engaged the channel. Right now nothing in the architecture distinguishes *"gaze was tried and didn't help"* from *"gaze was never engaged"* — which are hypotheses H1 and H2 in [[1-introduction#^hypotheses|the hypotheses table]], currently indistinguishable.

> [!warning]+ ⚠️ The parameter counts do not survive arithmetic
> The abstract says the resampler is "$\sim$1–5 M trainable parameters"; §1 says "$\sim$5–9 M" overall; §2.4 says LoRA is $\sim$3.5 M. Since 5–9 minus 3.5 ≈ 1.5–5.5, "1–5 M" reads as the total for **all four** resamplers. Count one:
>
> | Component | Size | Params |
> | --- | --- | --- |
> | $\mathbf{W}_{in}$ | 768 × 256 | 197 K |
> | 2 blocks × {$\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V,\mathbf{W}_G$} | 4 × 256 × 256 each | 524 K |
> | 2 blocks × FFN | 256→1024→256 | 1.05 M |
> | Reverse cross-attention | ~3 × 256 × 256 | 197 K |
> | Latents $\mathbf{L}$ | 32 × 256 | 8 K |
> | **$\mathbf{W}_{out}$** | **256 × 3584** | **918 K** |
> | | **one resampler** | **≈ 2.9 M** |
>
> Four independent resamplers ⇒ **≈ 12 M**, well outside "1–5 M". $\mathbf{W}_{out}$ alone is 918 K × 4 = **3.7 M**, already past the bottom of the claimed range before anything else exists.
>
> Either $\mathbf{W}_{out}$ is shared across depths — which would undercut §2.3's argument that per-depth independence is essential, and §3.3's **(S)** ablation with it — or the reported counts are per-module and both the abstract and §1 are wrong. Resolvable from the repo.

followed by LayerNorm + FFN ($4d_l$ hidden). After the blocks, spatial tokens read from the updated latents via reverse cross-attention. This reverse step is necessary because the forward pass compresses $HW$ spatial tokens into only $K=32$ latents; the reverse cross-attention expands the gaze-modulated representation back to the original spatial resolution so that each visual token in the LLM receives a position-specific residual. A zero-initialized output projection maps to $d_{llm}=3,584$, ensuring the module has no effect at the start of training and the pretrained LLM representations are not disrupted:

$$
\mathbf{R}_t = \text{XAttn}(\mathbf{X}, \mathbf{L}) \mathbf{W}_{out} \in \mathbb{R}^{HW \times d_{llm}}. \quad (5)
$$

### 2.3. Hook-Based LLM Injection

We inject gaze information at multiple depths of the LLM rather than at a single point, because different decoder layers encode progressively more abstract representations: early layers capture low-level spatial layout while later layers handle high-level semantics. Injecting at four evenly-spaced layers ($l \in \{6, 13, 20, 27\}$ out of 28) via PyTorch forward hooks lets gaze influence all levels of abstraction without modifying the model’s forward pass code:

$$
\mathbf{h}_l^{(v)} \leftarrow \mathbf{h}_l^{(v)} + \alpha_l \cdot f_{\theta}^{(l)}(\mathbf{F}_t, \mathbf{g}_t), \quad (6)
$$

> [!note]+ 📖 Reading Eq. (6), symbol by symbol
>
> | Symbol | Meaning |
> | --- | --- |
> | $\mathbf{h}_l$ | the hidden states leaving decoder layer `l` — one vector per token |
> | superscript $(v)$ | **visual token positions only**. The sequence also holds text tokens; those are untouched, as the last line of §2.3 says |
> | $\leftarrow$ | assignment — overwrite with the right-hand side |
> | $\alpha_l$ | a single learned number per layer, scaling the whole residual |
> | $f_\theta^{(l)}$ | resampler number `l` — the entire §2.2 pipeline, with its own weights $\theta$ |
>
> So: *"take the visual hidden states at layer `l`, add $\alpha_l$ times what resampler `l` computed from this frame's V-JEPA features and gaze."*
>
> **Why layers {6, 13, 20, 27} of 28** — evenly spaced, ≈ every 7th. The argument is that depth corresponds to abstraction: early layers hold spatial/low-level structure, later layers hold semantics. Injecting once would pick one level arbitrarily; injecting at four covers the range. (Note this is asserted, not measured — there is no ablation over *which* or *how many* depths. §3.3's (S) axis varies whether resamplers are **shared**, not where they attach.)

> [!note]+ 📖 What a forward hook is, and why "eager attention" is required
> In PyTorch, `module.register_forward_hook(fn)` attaches `fn` to a module so it runs every time that module produces output — and if `fn` returns a value, that value **replaces** the output. The wrapped module never knows. So you can rewrite layer 6's output without editing a line of Qwen's source, and `handle.remove()` restores the original exactly. This is the entire basis of the "modular, attaches/detaches" claim in §1.
>
> **"Eager attention"** (from §3.1) — modern implementations swap the attention math for fused kernels (SDPA, FlashAttention) that compute several steps in one GPU operation for speed. Fewer distinct sub-modules run, so the hook points you want may not exist as separate modules. *Eager* mode means the plain, unfused implementation with every sub-module executed individually — slower, but every internal boundary is hookable. A real cost of this design.
>
> **"Additional hooks on the visual encoder and model input locate the visual-token positions"** — solving a bookkeeping problem: Qwen2.5-VL interleaves text and image tokens in one sequence at positions that vary per prompt, so before adding to $\mathbf{h}_l^{(v)}$ you must know *which* sequence indices are visual. They recover that by observing the model's own input processing.

> [!tip]+ 🧵 $\alpha_l$ is the diagnostic your architecture lacks
> A learned scalar gating a conditioning signal is directly readable after training: if gaze were useless, gradient descent drives $\alpha_l \to 0$. It converts "did the model use the signal?" from an inference into a measurement — and the paper, having built it, **never reports its values**, which is a missed opportunity in a paper whose whole thesis is about *where* to inject.
>
> For you the point is sharper, because that question is live: hypothesis H2 in [[1-introduction#^hypotheses|the hypotheses table]] is "the model never learned to use it," and you currently have no instrument for it. One scalar per signal, logged each epoch, costs nothing and would settle it. See also [[4-results#^t2|T2]].

where $\alpha_l$ is a learned per-layer amplitude scalar that lets the model control how strongly gaze modulates each depth. Each layer has an *independent* resampler so that the gaze signal can be tailored to the representation space at each depth; a shared resampler would be forced to produce a single residual that is simultaneously appropriate for all layers. Additional hooks on the visual encoder and model input locate the visual-token positions in the LLM sequence. Text tokens are not modified.

### 2.4. Training

**Two-stage training.** Training proceeds in two stages to decouple gaze alignment from LLM adaptation:

- *Stage 1*: Only the four resamplers (~1–5 M params) are trained with 4-way cross-entropy over answer-token logits (A/B/C/D). This stage teaches the resamplers to produce useful gaze residuals while the LLM remains fully frozen, preventing catastrophic forgetting.
- *Stage 2*: Rank-8 LoRA adapters [8] ($\alpha=16$, ~3.5 M params) on LLM Q/V projections are added and trained jointly with the resamplers, allowing the LLM to better incorporate the gaze signal by adjusting its own attention patterns.

Both stages use AdamW (lr $3 \times 10^{-4}$, weight decay $10^{-2}$), 20 warmup steps, gradient accumulation of 8, and up to 20 epochs. Data is split 70/15/15 by video to prevent leakage.

> [!note]+ 📖 "4-way cross-entropy over answer-token logits" — the loss, concretely
> They never make the model *generate text*. Each question is multiple choice, so:
>
> 1. Run the forward pass once. At the final position the LLM produces a **logit** (an unnormalised score) for every token in its ~150 K vocabulary.
> 2. Pull out just **four** of those numbers — the logits for the tokens `A`, `B`, `C`, `D`.
> 3. Softmax over only those four → a probability distribution over the options.
> 4. **Cross-entropy** loss: $-\log p(\text{correct option})$. Zero when the right answer gets probability 1, growing as that probability falls.
>
> This is why it is "4-way" rather than vocabulary-wide, and it makes training and evaluation identical in form — §3.1's protocol is step 2 plus an argmax. Cheap and unambiguous: no parsing of free text, no format failures. (Which makes the sub-chance baselines in Table 1 more damning, not less — see the warning there.)

> [!note]+ 📖 The two stages, and the training hyperparameters
> **Why two stages.** *Stage 1* trains only the resamplers against a fully frozen LLM. The gaze pathway must learn to produce something useful while its consumer is a fixed target — if both moved at once from random init, the LLM could adapt to noise. **Catastrophic forgetting** is the named risk: updating a pretrained network on a narrow task overwrites unrelated capability. A frozen LLM cannot forget anything.
>
> *Stage 2* adds LoRA so the LLM can meet the signal halfway. **Rank-8, $\alpha = 16$, on Q/V projections:** rank 8 is the width of the bottleneck (`W + BA` with `A` being d×8, `B` 8×d); the LoRA $\alpha$ is a **scaling factor**, with the update applied as $\frac{\alpha}{r}BA$ — here $16/8 = 2$, so the adapter's effect is doubled. Note this $\alpha$ is unrelated to $\alpha_l$ in Eq. (6); the paper reuses the symbol. "Q/V projections" means adapters go on the query and value matrices of the LLM's own attention and not on K or the MLPs — the common default from the LoRA paper, which found Q/V sufficient.
>
> **The rest:** *AdamW* is Adam with decoupled weight decay (the $10^{-2}$). *20 warmup steps* ramp the learning rate from 0 to $3\times10^{-4}$ so early large updates don't destabilise things. **Gradient accumulation of 8** is forced by the protocol: videos have different lengths so batch size must be 1, but a batch of 1 gives noisy gradients — so they sum gradients over 8 examples before each optimizer step, giving an *effective* batch of 8 at the cost of 8× the wall-clock per update.

> [!warning]+ ⚠️ "70/15/15 by video to prevent leakage" is weaker than it sounds
> Splitting by video does stop the same *clip* appearing in train and test. It does **not** stop the same **person** appearing on both sides — 285 videos come from far fewer participants, and the paper never reports how many.
>
> You have measured exactly how much that matters. [[4-results#^t4|T4]] ran the same probe under three splits:
>
> | Split | Leaks | skill @ 0 s |
> | --- | --- | --- |
> | random | across people **and** within recordings | +0.273 |
> | **recording** (same people, unseen clips) | **across people only** | **+0.116** |
> | participant (unseen people) | nothing | +0.001 |
>
> The middle row is StreamGaze's split design, and it retains **~100×** the recoverable gaze signal of the honest one. A model can learn person-specific gaze-scene habits — where *this* individual looks in *their* kitchen — and be scored on the same person's other videos.
>
> This does not invalidate GazeQwen's gains, which are far too large to be only that. But it means the reported numbers are an **upper bound** on cross-person performance, and the paper offers no way to know how loose. Your participant-level split is the stricter protocol, and this is a concrete methodological point you hold over the published work.

## 3. Experiments

### 3.1. Setup

**Benchmark.** We evaluate on StreamGaze [11], a gaze-conditioned video QA benchmark containing 8,521 multiple-choice questions (4 options each) derived from 285 egocentric videos with eye-tracking data. The benchmark defines 10 task types grouped into three temporal categories [11]: *Past tasks* (NFI, OTP, SR, GSM) require reasoning over the full viewing history; *Present tasks* (OI-Easy, OI-Hard, OAR, FAP) operate within a 60-second context window around the query time; *Proactive tasks* (GTA, OAA) require anticipating future events or alerting the user based on current gaze patterns. We use a 70/15/15 video-level split, yielding 1,055 test QA pairs.

> [!note]+ 📖 The 10 task acronyms — and which the paper never defines
> You need these to read Table 1, and **the paper expands only six of them**. Marked below by where each comes from, because guessing silently would be worse than admitting the gap.
>
> | | Acronym | Expansion | Source |
> | --- | --- | --- | --- |
> | **Past** | NFI | Never-Fixated Item | *inferred* — §3.3 says it is about "which objects were **never attended**" |
> | | OTP | Object Tracking in Past | stated, §3.2 |
> | | SR | Scene Recall | stated, §3.2 |
> | | GSM | — | **not defined anywhere in this paper**; check StreamGaze [11] |
> | **Present** | OI-Easy | Object Identification (Easy) | stated, §3.2 |
> | | OI-Hard | Object Identification (Hard) | stated, §3.2 |
> | | OAR | Object Attribute Recognition | stated, §3.2 |
> | | FAP | Future Action Prediction | *inferred* — §3.2 says it is "predicting future actions" |
> | **Proactive** | GTA | Gaze Target Anticipation (?) | **not defined**; expansion is a guess |
> | | OAA | Object Appearance Alert | stated, §3.2 |
>
> **The three categories are defined by how much context the model gets, not by what is asked:**
>
> - *Past* — reasoning over the **full viewing history**.
> - *Present* — a **60-second window** around the query time.
> - *Proactive* — anticipating what comes next, or alerting the user.
>
> That distinction resolves what otherwise looks like a taxonomy error: **FAP asks about the future but sits under *Present***, because it is granted only the 60-second window. So the categories describe the *input budget*; the task names describe the *question*. FAP is a future-facing question on a present-sized context.

> [!tip]+ 🧵 FAP is the task that maps onto your thesis
> Of all 10, **FAP (Future Action Prediction)** is nearest to what [[1-introduction|Path 2]] and [[4-results#^t3|T3]] actually measure — anticipation from egocentric video. GTA is next. Watch both closely in §3.2; the result there is the single most relevant number in the paper for you, and it is not the headline.

**Backbone.** Our base MLLM is Qwen2.5-VL-7B-Instruct [2], loaded with eager attention to allow PyTorch forward hook registration on individual decoder layers. Visual features come from a frozen V-JEPA 2.1 encoder (ViT-B/16, $384\times384$ input resolution) [3], whose spatiotemporal features are interpolated (trilinear temporal, bilinear spatial) to match Qwen’s visual token grid. Both the MLLM and V-JEPA backbones remain frozen throughout training; only the resampler modules and optional LoRA adapters are updated.

**Baselines.** We compare against four categories of models. *Closed-source MLLMs*: GPT-4o [1] and Claude Sonnet/Opus 4 [15], evaluated with gaze supplied as textual coordinate prompts. *Open-source MLLMs*: Qwen2.5-VL [2], InternVL3.5 [18], VITA 1.5 [6], MiniCPM-V [20], and Kangaroo [14]. *Streaming MLLMs*: ViSpeak [7], Dispider [17], Flash-VStream [21], and VideoLLM-online [5]. *Gaze-specialized*: AssistGaze [9], a fine-tuned model trained on GazeQA data. All baseline numbers are taken from StreamGaze [11].

**Evaluation protocol.** Each QA pair is evaluated independently (batch size 1) to accommodate variable video lengths. The predicted answer is the option (A/B/C/D) with the highest logit at the last token position. We report per-task accuracy and the unweighted mean across all 10 tasks.

> [!note]+ 📖 The four baseline families, and what "all numbers are taken from [11]" means
>
> - **Closed-source MLLMs** — GPT-4o, Claude Sonnet 4 / Opus 4. Gaze supplied as *text
>   coordinates*, the only channel available through an API.
> - **Open-source MLLMs** — general video-capable models, evaluated off-the-shelf.
> - **Streaming MLLMs** — architectures purpose-built for continuous feeds (persistent memory,
>   deciding *when* to speak). Their "Frames" column shows a **rate** (1 fps, 2 fps) rather than a count, because they consume video continuously.
> - **Gaze-specialized** — AssistGaze [9], trained on gaze data but for a different dataset/task.
>
> **The critical sentence is the last one: "All baseline numbers are taken from StreamGaze [11]."** The authors did not run these models. They copied a table. So every baseline is *zero-shot, as evaluated by someone else, under that paper's harness* — while GazeQwen is trained on this data by these authors. That asymmetry is the confound flagged at the abstract, and it is also why the broken numbers below are inherited rather than introduced.

> [!warning]+ ⚠️ Sample size — the number that makes §3.3 unreadable
> **1,055 test QA pairs ÷ 10 tasks ≈ 105 per task.** Work out the noise floor from that.
>
> For an accuracy `p` measured on `n` items, the binomial standard error is $\sqrt{p(1-p)/n}$. At `p ≈ 0.6`, `n = 105`:
>
> $$\text{SE} = \sqrt{\frac{0.6 \times 0.4}{105}} \approx 0.048 = \textbf{4.8 pp}$$
>
> So a *single* per-task cell in Table 1 carries ≈ ±4.8 pp of noise, and the **difference** between two conditions on one task has SE ≈ 6.8 pp if independent. Consequences:
>
> - The **overall** column (mean over all 1,055 items) is solid — SE ≈ 1.5 pp. The +16.1 pp
>   headline is far outside noise, whatever else confounds it.
> - **Per-task deltas under ~10 pp are not distinguishable from noise.** This is what guts the
>   ablation table — see the warning at §3.3.
> - The large per-task gains (+34.1, +29.4, +26.1, +20.8) are real; they are 3–7 SE.
>
> No confidence intervals, error bars, or significance tests appear anywhere in the paper, and single-seed training means run-to-run variance is also unmeasured.

### 3.2. Main Results

Table 1 summarizes accuracy on each task. GazeQwen reaches **63.9%** overall, the highest among all systems including proprietary ones:

- **+10.5pp** over GPT-4o (53.4%), despite using a 7B open-weight model versus a much larger proprietary system with gaze supplied as text coordinates.
- **+16.1pp** over the same Qwen2.5-VL backbone with gaze supplied as visual prompts (47.8%), demonstrating that internal hidden-state modulation is far more effective than surface-level input formatting.

**Where does gaze help most?** The per-task breakdown reveals that improvements concentrate on tasks where spatial gaze information directly reduces ambiguity. OAA (Object Appearance Alert) gains +34.1pp, as proactive alerts about object appearances benefit directly from knowing which objects the user is currently attending to. OTP (Object Tracking in Past) gains +29.4pp, as the accumulated fixation history provides a strong temporal signal for tracking attended objects over time. OAR (Object Attribute Recognition) gains +26.1pp because knowing exactly which object the user fixates narrows the relevant region, making attribute recognition straightforward even when multiple objects with different attributes coexist in the scene. OI-Hard (Object Identification Hard) gains +20.8pp since gaze disambiguates the target from visually similar nearby objects that confuse the model without spatial guidance.

> [!warning]+ ⚠️ What the headline deltas actually compare
> Both bullets above are **trained-on-this-data vs zero-shot**. GazeQwen saw StreamGaze's 70% training split; Qwen2.5-VL and GPT-4o saw none of it. So "+16.1 pp over the same backbone" bundles at least three distinct causes:
>
> 1. **Gaze**, injected into hidden states — the claimed cause.
> 2. **A second visual encoder.** V-JEPA features are injected at four depths *regardless of gaze*. That is extra visual information the baseline never receives.
> 3. **~9 M parameters + LoRA fitted to this benchmark's distribution** — question phrasing, option formatting, answer priors, video domain.
>
> **The paper never runs the control that separates them.** There is no row for "same architecture, $\mathbf{g}_t = \mathbf{0}$", no shuffled-scanpath control, no V-JEPA-only ablation. Every one of those is a single training run on hardware they already used. Without them, "gaze causes the gain" is an assumption, not a result — which is uncomfortable for a paper whose thesis is *learning where to inject gaze* beats prompting.
>
> The §3.3 ablations do not fill the gap: all four axes vary *how* gaze is encoded or injected, and every arm has gaze. None asks whether gaze matters at all.

> [!tip]+ 🧵 This is exactly your Phase 7 — and you have it, they don't
> The missing control is `--signal-dropout 1.0`: same architecture, behavioural signal zeroed. It was later run as [[4-results#^t11|T11]].
>
> Worth registering plainly: **on the methodological point that matters most, your protocol is ahead of a published paper's.** That is a related-work sentence, and an argument for finishing Phase 7 rather than treating it as overdue housekeeping.

**Where does gaze not help?** OI-Easy (+0.4pp) and FAP (+1.7pp) show negligible gains, suggesting that easy object identification is already achievable without gaze disambiguation, and predicting future actions requires higher-level temporal reasoning beyond where the user is currently looking. SR (Scene Recall, +9.8pp) shows a moderate gain; the task asks about previously seen background context, where gaze localization is less critical since the answer depends on global scene memory rather than fixation targets.

> [!tip]+ 🧵 **The most important paragraph in this paper, for your thesis**
> Set the headline aside and read what just happened on the future-facing tasks.
>
> | Task | Qwen2.5-VL | GazeQwen | Δ | Human | Best in table |
> | --- | --- | --- | --- | --- | --- |
> | **FAP** (future action prediction) | 0.391 | 0.408 | **+1.7** | 0.840 | Claude Sonnet 4 (0.439) |
> | **GTA** (anticipation) | 0.486 | 0.603 | +11.7 | 0.765 | ViSpeak (0.635) |
> | OAR (present-tense attribution) | 0.548 | 0.809 | **+26.1** | 0.800 | **GazeQwen** |
> | OAA (current attention) | 0.407 | 0.748 | **+34.1** | 0.780 | **GazeQwen** |
>
> A system with a +16.1 pp overall gain, discriminative supervision that *cannot be satisfied without gaze*, and in-domain training, buys **+1.7 pp on predicting future actions** — and is beaten there by an off-the-shelf Claude. On both future-facing tasks it fails to top the table.
>
> **This is not a ceiling effect.** Humans reach 0.840 on FAP while the model sits at 0.408: 43 pp of headroom that gaze does not touch. Meanwhile on present-tense attribution the *same mechanism* adds +26 to +34 pp and reaches human parity (0.809 vs 0.800).
>
> **Why this matters to you.** It is an independent replication of the *shape* of your null — different architecture, objective, dataset, and research group. Their own explanation: predicting future actions "requires higher-level temporal reasoning beyond where the user is currently looking."
>
> For the workshop paper planned in [[6-next-steps]], this moves the finding from *"our pipeline may be broken"* toward *"gaze is strongly informative about the present and weakly informative about the future, and two independent systems now show it."* It is the cleanest external corroboration available.
>
> **It does not contradict [[4-results#^t4|T4]].** Recoverability falling with lead is about how much *marginal information* gaze carries; this is about how much of that information is *predictively useful*. Both can hold at once — and together they sharpen hypothesis 1 in that note's Takeaway ("informative ≠ predictive") into the leading explanation.

> [!question]+ ❓ So does gaze help "the present" only because the questions are about gaze?
> Worth being careful here, because it is the obvious objection and it partly lands. OAA, OAR and OI-Hard are *constructed* to be gaze-dependent — "which object is the user looking at" is near-unanswerable without gaze, so a large gain is close to guaranteed once the pathway exists. That inflates the present-tense numbers relative to any natural task distribution.
>
> But it **strengthens** rather than weakens the FAP reading. FAP was written by the same authors, for the same benchmark, with the same intent that gaze be relevant. If the near-null there were just "the question doesn't need gaze", that would itself be the finding: even a benchmark *designed* to make gaze matter cannot make it matter for future actions.

### 3.3. Design Space Analysis

We swept four design axes during development and report the marginal effect of each choice in Table 2.
**(G) Gaze encoding:** Coord-PE (sinusoidal encoding of raw fixation coordinates) outperforms both heatmap-$\tau$ (Voila-style [19] Gaussian heatmap with learnable temporal decay) and heatmap-dur (duration-weighted Gaussian heatmap). The key advantage is that Coord-PE preserves sub-pixel spatial precision, whereas heatmap encodings discretize gaze onto the LLM’s coarse spatial grid, losing fine-grained positional information. The largest gap appears on NFI (9.8pp), where precise fixation history is critical for determining which objects were never attended.
**(B) Visual backbone:** V-JEPA 2.1 consistently outperforms DINOv2 across matched configurations. V-JEPA’s joint spatiotemporal embeddings (via 3D tubelet patches) capture motion and temporal continuity that DINOv2’s frame-independent features miss. The gap is widest on fixation-sequence tasks (NFI: 8.1pp, OTP: 5.3pp), where temporal context is essential.
**(S) Layer sharing:** Per-layer resamplers (four independent modules) outperform a single shared resampler by 5.2pp on OAR. This confirms that different LLM depths benefit from specialized gaze representations: early layers may need broad spatial modulation while deeper layers require fine-grained object-level focus.
**(A) LLM adaptation:** Adding rank-8 LoRA adapters in stage 2 provides a further 5.2pp gain on OI-Hard. LoRA lets the LLM’s own attention patterns adapt to better incorporate the injected gaze residuals, rather than relying solely on additive modulation of frozen representations.

> [!note]+ 📖 The four axes, and the terms in them
> **(G) Gaze encoding** — three ways to turn fixations into a vector:
>
> - *Coord-PE* — sinusoidal encoding of raw `(x, y)`, as in §2.1. The winner.
> - *heatmap-τ* — draw a **Gaussian blob** centred on the fixation over the spatial grid (bright
>   at the fixation, fading outward), with a *learnable temporal decay* τ so older fixations fade. "Voila-style" refers to Voila-A [19], which aligns VLMs to gaze this way.
> - *heatmap-dur* — the same blob, weighted by how long each fixation lasted.
>
> The stated advantage is that a heatmap must be **rasterised onto the coarse token grid** (24×24, so each cell is ~4% of the frame width), quantising the fixation to a cell, whereas coord-PE keeps the exact real-valued coordinate. "Sub-pixel precision" overstates it — the input is normalised floats, so it is sub-*cell* precision — but the mechanism is right.
>
> **(B) Visual backbone** — V-JEPA vs **DINOv2**, a strong self-supervised *image* model. The stated difference is real and important: DINOv2 processes **each frame independently**, so any notion of motion must be reconstructed downstream, while V-JEPA uses **3-D tubelet patches** (each token spans 2 frames × 16 × 16 pixels), making motion native to the representation. That the gap is widest on *fixation-sequence* tasks is at least consistent with that story.
>
> **(S) Layer sharing** — four independent resamplers vs one shared across all depths.
>
> **(A) LLM adaptation** — with vs without stage-2 LoRA.

> [!warning]+ ⚠️ Table 2's "spread" is a maximum over 10 tasks — it measures its own noise
> Read the caption carefully: the spread is *"measured on the task with largest difference."* That is a **maximum selected over 10 tasks**, and every parenthesis in the table confirms it (NFI, NFI, OAR, OI-H — a different task each row).
>
> This is the classic max-selection trap. From the §3.1 warning, a per-task difference between two conditions has SE ≈ 5–7 pp at `n ≈ 105`. The expected **maximum** absolute difference across 10 such tasks, *under a pure null where the two conditions are identical*, is roughly 1.5 SE ≈ **8–10 pp**. So:
>
> | Reported spread | vs null expectation |
> | --- | --- |
> | (G) 9.8 pp | inside |
> | (B) 8.1 pp | inside |
> | (S) 5.2 pp | well inside |
> | (A) 5.2 pp | well inside |
>
> **Every reported spread is at or below what noise alone produces under this procedure.** The honest readout would be the *overall* accuracy per arm, on all 1,055 items, where SE is ≈1.5 pp — but overall numbers for the ablation arms are never given.
>
> This does not prove the design choices are worthless; it means **Table 2 provides no evidence either way**. Which is a shame, because (G) — coordinate PE vs heatmaps — is the one result here with direct transfer value to your predictor.

> [!question]+ ❓ Then should I still take the coord-PE recommendation seriously?
> Yes, but on the *argument*, not on Table 2. The empirical support is inside the noise floor. The mechanistic case is independent and strong: a `Linear(3 → 384)` on angles is rank-3 and low-frequency, and Fourier expansion is the standard fix, with the same justification used in NeRF and in the original Transformer's positional encodings. Treat it as **a well-motivated cheap change worth testing**, not as a result this paper established.

All four axes contribute comparably (5–10pp spread), and the gains are largely complementary: removing any single component degrades performance, confirming that the full combination is needed for the best result.

## 4. Conclusion

GazeQwen shows that a small resampler ($\sim$1–5 M params) with LoRA adapters ($\sim$3.5 M params) injecting gaze residuals into a 7B MLLM can outperform much larger systems on gaze-conditioned video QA. The approach requires no architecture surgery, trains on a single GPU, and the gaze module can be detached for gaze-free operation. Our findings suggest the bottleneck in gaze-aware video understanding is not model capacity but the absence of a learned pathway from gaze to internal representations. We release all code to support future work on gaze-guided streaming models.

## Tables

**Table 1. Results on the StreamGaze benchmark.** We report accuracy on all 10 task types. Baseline numbers are from [11]; GazeQwen is our evaluation on the Qwen2.5-VL-7B backbone. Overall is the mean over the 10 tasks shown. Best per-task (excluding Human) is bolded.

| Method | Params | Frames | Past NFI | Past OTP | Past SR | Past GSM | Present OI (E) | Present OI (H) | Present OAR | Present FAP | Proactive GTA | Proactive OAA | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Human | – | – | 0.700 | 0.889 | 0.903 | 0.707 | 0.960 | 0.920 | 0.800 | 0.840 | 0.765 | 0.780 | 0.827 |
| **Closed-Source MLLMs** | | | | | | | | | | | | | |
| GPT-4o [1] | – | 16 | 0.601 | 0.449 | 0.535 | 0.580 | **0.729** | 0.730 | 0.596 | 0.370 | 0.597 | 0.149 | 0.534 |
| Claude Sonnet4 [15] | – | 16 | 0.500 | 0.554 | 0.425 | 0.325 | 0.521 | 0.533 | 0.561 | **0.439** | 0.535 | 0.350 | 0.474 |
| Claude Opus4 [15] | – | 16 | 0.372 | 0.392 | 0.460 | 0.166 | 0.431 | 0.436 | 0.430 | 0.351 | 0.466 | 0.490 | 0.399 |
| **GazeQA-based Fine-tuned Models** | | | | | | | | | | | | | |
| AssistGaze [9] | 26M | 32 | 0.294 | 0.131 | 0.310 | 0.294 | 0.278 | 0.250 | 0.109 | 0.254 | N/A | N/A | 0.240 |
| **Open-Source MLLMs** | | | | | | | | | | | | | |
| Qwen2.5-VL [2] | 7B | Adapt. | 0.518 | 0.350 | 0.450 | 0.483 | 0.590 | 0.558 | 0.548 | 0.391 | 0.486 | 0.407 | 0.478 |
| InternVL3.5 [18] | 8B | Adapt. | 0.490 | 0.311 | **0.573** | 0.548 | 0.627 | 0.628 | 0.466 | 0.372 | 0.373 | 0.051 | 0.444 |
| VITA 1.5 [6] | 7B | 16 | 0.474 | 0.365 | 0.346 | 0.378 | 0.455 | 0.396 | 0.437 | 0.370 | 0.351 | 0.267 | 0.384 |
| MiniCPM-V [20] | 8B | 32 | 0.430 | 0.374 | 0.354 | 0.296 | 0.334 | 0.379 | 0.438 | 0.345 | 0.480 | 0.216 | 0.365 |
| Kangaroo [14] | 7B | 64 | 0.363 | 0.365 | 0.319 | 0.275 | 0.454 | 0.484 | 0.402 | 0.412 | 0.242 | 0.198 | 0.351 |
| **Open-Source Streaming MLLMs** | | | | | | | | | | | | | |
| ViSpeak [7] | 7B | 1 fps | 0.463 | 0.358 | 0.417 | 0.473 | 0.572 | 0.581 | 0.406 | 0.309 | **0.635** | 0.458 | 0.467 |
| Dispider [17] | 7B | 1 fps | 0.366 | 0.365 | 0.381 | 0.263 | 0.336 | 0.338 | 0.353 | 0.321 | 0.252 | 0.261 | 0.324 |
| Flash-VStream [21] | 7B | 1 fps | 0.249 | 0.202 | 0.336 | 0.220 | 0.289 | 0.147 | 0.044 | 0.280 | 0.443 | 0.217 | 0.243 |
| VideoLLM-online [5] | 8B | 2 fps | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 | 0.006 | 0.002 | 0.000 | 0.458 | 0.333 | 0.081 |
| **Ours** | | | | | | | | | | | | | |
| **GazeQwen** | 7B | Adapt. | **0.657** | **0.644** | 0.548 | **0.609** | 0.594 | **0.766** | **0.809** | 0.408 | 0.603 | **0.748** | **0.639** |
| $\Delta$ vs. Qwen2.5-VL | | | *+13.9* | *+29.4* | *+9.8* | *+12.6* | *+0.4* | *+20.8* | *+26.1* | *+1.7* | *+11.7* | *+34.1* | *+16.1* |

> [!note]+ 📖 Reading Table 1 — the columns, and how "Overall" is computed
> **Params** — total model size, not trainable size. GazeQwen's "7B" is the frozen Qwen backbone; only ~9 M of it is ever updated. AssistGaze's 26 M is a genuinely small model.
>
> **Frames** — how much video the model consumes: a fixed count (16, 32, 64), a **rate** for streaming models (1 fps, 2 fps), or **"Adapt."** for adaptive/dynamic sampling, where the count varies with video length. Not held constant across rows, so it is a confound in every comparison — a 64-frame model and a 16-frame model are not seeing the same input.
>
> **Overall** — the caption says "the mean over the 10 tasks shown", i.e. an **unweighted mean of per-task accuracies**, *not* accuracy over all 1,055 items. These differ whenever tasks have unequal sample counts: a task with 40 items counts as much as one with 200. The per-task counts are never reported, so the two cannot be reconciled.
>
> **Scorecard:** GazeQwen takes best-in-column on **6 of 10** — NFI, OTP, GSM, OI-Hard, OAR, OAA. It loses SR (InternVL3.5, 0.573), OI-Easy (GPT-4o, 0.729), FAP (Claude Sonnet 4, 0.439) and GTA (ViSpeak, 0.635). Note it also *exceeds the human row* on OAR (0.809 vs 0.800) — less a sign of superhuman ability than that the task becomes near-mechanical once gaze is available.

> [!warning]+ ⚠️ Several baselines score **below chance** — those rows are broken evals
> Every question is 4-way multiple choice, so **random guessing scores 0.25**. Anything far below that is not a weak model; it is a model whose answer was never correctly extracted.
>
> | Row | Score | Problem |
> | --- | --- | --- |
> | VideoLLM-online | **0.000** on seven tasks | ~105 items each; guessing randomly gives 0.000 with probability $0.75^{105} \approx 10^{-13}$ |
> | InternVL3.5, OAA | 0.051 | 5× below chance |
> | Flash-VStream, OAR | 0.044 | 6× below chance |
> | AssistGaze | 0.240 overall, 0.109 on OAR | at/below chance throughout |
> | Claude Opus 4, GSM | 0.166 | and Opus 4 scores *below* Sonnet 4 overall (0.399 vs 0.474) |
>
> Consistently-below-chance results mean systematic answer-extraction failure — refusals, format mismatches, or a parser that scores a non-answer as wrong. **These numbers measure a harness, not a capability.**
>
> Two consequences. First, the *bottom* of Table 1 is uninformative, so "highest score among all models tested" is a weaker claim than it sounds — much of the field it beats was never functioning. Second, the failures are **inherited** from StreamGaze [11], not introduced here; §3.1 says all baseline numbers were copied. Reproducing them was not attempted, and a paper whose own protocol reads 4 answer logits directly (§2.4) was well placed to notice that free-text baselines were failing to parse.
>
> The comparisons that survive are the ones near the top — GPT-4o at 0.534 and Qwen2.5-VL at 0.478 are plausible — and those are still zero-shot against a trained model.

**Table 2. Ablation over design axes.** Each row shows the best option for that axis and the accuracy spread between best and worst options (measured on the task with largest difference).

| Axis | Best option | Spread |
| :--- | :--- | :--- |
| (G) Gaze encoding | Coord-PE | 9.8pp (NFI) |
| (B) Visual backbone | V-JEPA 2.1 | 8.1pp (NFI) |
| (S) Layer sharing | Per-layer (4 ind.) | 5.2pp (OAR) |
| (A) LLM adaptation | LoRA (rank 8) | 5.2pp (OI-H) |

> [!warning]+ ⚠️ What Table 2 omits
> Three columns you would need and do not get: the **overall accuracy of each arm**, the **worst-performing option** whose subtraction produced the spread, and any **sample size**. The table reports a difference without either endpoint, computed on a task chosen *because* the difference there was largest. See the max-selection warning at §3.3.
>
> Also absent: the axis a reader most wants, **gaze vs no gaze**. All four rows compare ways of using gaze; none tests whether it is doing the work.

> [!question]+ ❓ Summary — what should I actually take from this paper?
>
> **Solid and worth taking:**
>
> - **The hook-based injection pattern.** Cheap, modular, no architecture surgery, genuinely
>   detachable. The engineering is the real contribution.
> - **Zero-initialisation of the output projection** — small, standard, and probably load-bearing.
>   Directly applicable to your projectors.
> - **A learned per-depth amplitude $\alpha_l$**, which doubles as instrumentation for "did the
>   model use the signal?" — the diagnostic your setup lacks.
> - **Coordinate PE over a raw linear map** — on the mechanistic argument, not their evidence.
> - **The FAP near-null.** The most valuable result in the paper for you, and not one the authors
>   foreground.
>
> **Do not take at face value:**
>
> - **+16.1 pp as a gaze effect** — it is trained-vs-zero-shot, with no gaze-ablated control.
> - **Table 2** — every spread sits inside the noise floor of its own selection procedure.
> - **"Highest score among all models tested"** — much of the field it beats had broken evals.
> - **The stated mechanism for Eq. (2)** — broadcast gaze cannot spatially re-weight (§2.2).
> - **The parameter counts**, which do not close by a factor of ~3.
> - **"V-JEPA 2.1"**, cited to the V-JEPA 1 paper.
>
> **For [[6-next-steps|Path 3]]:** this is a hybrid your three-position table has no row for — a *non-linguistic conditioning pathway* into an LLM whose output is still a token logit, so it keeps the inference-time latency you are trying to escape. It is not a JEPA and does not threaten the thesis. But the hook mechanism is substrate-agnostic, so if Ash's Concern 3 bites and HD-EPIC recipe steps prove no richer than verb-noun pairs, "inject gaze into an already-language-aligned space" is a cheaper fallback than training VL-JEPA-style alignment.
>
> **Positioning:** non-competing. Different benchmark, objective, and consumer of the signal. *"Concurrent work injects gaze into an MLLM's hidden states and reports large gains on present-tense gaze attribution; we ask the orthogonal question of whether gaze conditioning shapes a predictive latent space, and find its benefit concentrated away from anticipation — consistent with their near-null on future action prediction."*

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023. 3, 4
[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen2.5-vl technical report. *arXiv preprint arXiv:2511.21631*, 2025. 3, 4
[3] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. *arXiv preprint arXiv:2404.08471*, 2024. 1, 3
[4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In *European conference on computer vision*, pages 213–229. Springer, 2020. 2
[5] Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. Videollm-online: Online video large language model for streaming video. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 18407–18418, 2024. 1, 3, 4
[6] Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Haoyu Cao, Zuwei Long, Heting Gao, Ke Li, et al. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction. *arXiv preprint arXiv:2501.01957*, 2025. 3, 4
[7] Shenghao Fu, Qize Yang, Yuan-Ming Li, Yi-Xing Peng, Kun-Yu Lin, Xihan Wei, Jian-Fang Hu, Xiaohua Xie, and Wei-Shi Zheng. Vispeak: Visual instruction feedback in streaming videos. pages 21778–21788, 2025. 1, 3, 4
[8] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: low-rank adaptation of large language models. arxiv preprint. *arXiv preprint arXiv:2106.09685*, 2021. 1, 3
[9] Muhammet Ilaslan, Chenan Song, Joya Chen, Difei Gao, Weixian Lei, Qianli Xu, Joo Lim, and Mike Shou. Gazevqa: A video question answering dataset for multiview eye-gaze task-oriented collaborations. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 10462–10479, 2023. 1, 3, 4
[10] Bolin Lai, Fiona Ryan, Wenqi Jia, Miao Liu, and James M Rehg. Listen to look into the future: Audio-visual egocentric gaze anticipation. In *European Conference on Computer Vision*, pages 192–210. Springer, 2024. 1
[11] Daeun Lee, Subhojyoti Mukherjee, Branislav Kveton, Ryan A Rossi, Viet Dac Lai, Seunghyun Yoon, Trung Bui, Franck Dernoncourt, and Mohit Bansal. Streamgaze: Gaze-guided temporal reasoning and proactive understanding in streaming videos. *arXiv preprint arXiv:2512.01707*, 2025. 1, 3, 4
[12] Yin Li, Miao Liu, and James M Rehg. In the eye of the beholder: Gaze and actions in first person video. *IEEE transactions on pattern analysis and machine intelligence*, 45(6): 6731–6747, 2021. 1
[13] Junming Lin, Zheng Fang, Chi Chen, Zihao Wan, Fuwen Luo, Peng Li, Yang Liu, and Maosong Sun. Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. *arXiv preprint arXiv:2411.03628*, 2024. 1
[14] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input: J. liu et al. *International Journal of Computer Vision*, 134(3):114, 2026. 3, 4
[15] Anthropic PBC. Introducing claude 4. Online blog post, 2025. 3, 4
[16] Taiying Peng, Jiacheng Hua, Miao Liu, and Feng Lu. In the eye of mllm: Benchmarking egocentric video intent understanding with gaze-guided prompting. *arXiv preprint arXiv:2509.07447*, 2025. 1
[17] Rui Qian, Shuangrui Ding, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Dispider: Enabling video llms with active real-time interaction via disentangled perception, decision, and reaction. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 24045–24055, 2025. 3, 4
[18] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. *arXiv preprint arXiv:2508.18265*, 2025. 3, 4
[19] Kun Yan, Zeyu Wang, Lei Ji, Yuntao Wang, Nan Duan, and Shuai Ma. Voila-a: Aligning vision-language models with user’s gaze attention, 2024. 3
[20] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. *arXiv preprint arXiv:2408.01800*, 2024. 3, 4
[21] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Flash-vstream: Memory-based real-time understanding for long video streams, 2024. 3, 4
[22] Mengmi Zhang, Keng Teck Ma, Joo Hwee Lim, Qi Zhao, and Jiashi Feng. Deep future gaze: Gaze anticipation on egocentric videos using adversarial networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 4372–4381, 2017. 1
