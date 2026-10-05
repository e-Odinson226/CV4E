---
type: paper
status: running
created: 2026-08-25
updated: 2026-08-25
tags: [paper, own-work, final-report, gaze, hand-pose, conditioning, null-result, vjepa2, ek100, hd-epic, annotated]
---

# Behavioral Conditioning of Video World-Model Predictors for Egocentric Action Anticipation

**Seyyed Parsa Sharifi** — University of Rostock — `seyyed.sharifi@uni-rostock.de`
**Erfan Yekehzare** — University of Rostock — `erfan.yekehzare@uni-rostock.de`

> **Code:** `https://github.com/EgocentricPerceptions/CV4Egocentric_Semantic_Intention_Prediction` · **Project:** EgoProject 2026 (Univ. Rostock × UBB Cluj)

> [!info] 📌 How to read this file
> The paper's own text is left **unmodified** below. It matches the LaTeX source in `papers/overleaf/`. The paper is derived from the notes and the code in this repository; where they differ, the paper is changed. Everything added is inside a *callout* like this one, so annotation never blends into the authors' words. Four kinds:
>
> | Callout | Means |
> | --- | --- |
> | `[!note]` 📖 | Explanation of a term, mechanism, or piece of notation |
> | `[!question]` ❓ | An answer to a specific question you asked |
> | `[!warning]` ⚠️ | Where the paper is shaky, unclear, or possibly wrong |
> | `[!tip]` 🧵 | How this connects to *your* thesis |
>
> A `+` after the type means "starts expanded"; swap it to `-` to collapse every annotation and read the paper clean. This first pass is **extraction only** — no annotations added yet. The open problems are listed in [[project#Open problems|the project note]]. Related: [[gazeqwen]], [[vl-jepa]], [[vjepa]].

## Abstract

Egocentric action anticipation depends on reading preparatory cues before an action becomes visible. Human behavioral signals are an obvious place to look for those cues: gaze tends to precede manipulation, and the hands are what turn intention into action. We ask whether conditioning a frozen V-JEPA 2 predictor on gaze and hand pose improves short-horizon anticipation on EPIC-KITCHENS-100. It does not, and the aim of this paper is to say why with some precision. We turn the existing peer-token conditioning pathway into a bit-exact control, add a guard against three failure modes that can each counterfeit a null, and run five diagnostics against the frozen predictor and the behavioral signal itself. We then repeat the predictor's own training with the signal ablated, which separates the value of the information from the cost of withholding an input the model expects. The results agree: the signal reaches the predictor's output, the predictor is indifferent to its content, a predictor that never saw the signal matches one trained with it, and on its own the signal beats a class prior by under two points. We therefore retired a stronger, GazeQwen-informed pathway at the design stage rather than after paying to build it. One effect does survive. Behavioral signals favour verbs, which encode motion, while vision favours nouns, which encode object identity, and the same split organises the two strongest published systems on this benchmark. But once the vision comparator is fitted fairly, the effect holds only in direction and never rises above the verb prior, and the constraint that bounds it is single-participant sample size rather than any floor in the probe's dynamic range. We read the null as evidence that feeding these signals in at the input is the wrong way to use them, and argue for supervising the representation with behavioral signals toward goal-level targets instead. The anticipation results here are bounded to one participant at a one-to-two-second horizon; the training-side ablations that carry the null are cross-participant.

## 1 Introduction

Anticipating what a person will do next from egocentric video is hard for a structural reason: the most discriminative evidence, the action itself, has not happened yet. The model has to work from preparatory cues instead. Human behavioral signals are a natural candidate for those cues. Gaze leads manipulation by something like half a second to a second, and the hands are the effector that carries out intention, so conditioning a predictive model on gaze and hand pose ought to help it anticipate.

V-JEPA 2 (Assran et al., 2025) makes that idea testable. It learns a frozen encoder and a latent predictor by predicting masked spatiotemporal features, and its action-conditioned variant conditions the predictor on robot actions. We ask whether the same predictor can be conditioned on *human* signals instead, gaze and hand pose from egocentric recordings, and whether that improves action anticipation on EPIC-KITCHENS-100 (EK100) (Damen et al., 2022). Earlier phases of the project had already returned a stubborn null at the two-second horizon, and a control family had suggested that what looked like an effect of conditioning was mostly an effect of feeding the model more tokens. That left one question unanswered, with two answers that point in opposite directions. Perhaps the injection pathway is too weak to expose whatever the signal carries; perhaps the signal simply does not carry action-relevant information at this horizon. The first answer argues for a stronger architecture. The second argues for giving up on input-level conditioning altogether.

This paper sets out to decide between those two rather than assume either. The contribution is diagnostic, and we state it plainly, null included:

- We make the existing peer-token pathway a bit-exact control and guard against three silent-failure modes that can each fake a null, so that the null we report is a property of the signal and the pathway, not an artifact of the evaluation code (§3.2).
- We run five diagnostics directly against the frozen predictor and the behavioral signal. They tell a consistent story: the signal reaches the output, the predictor ignores its content, and on its own the signal anticipates the next action barely above a class prior (§4.3). Varying the training rather than the input confirms it: a predictor whose training never included the signal matches one trained with it (§4.4). That evidence let us retire a stronger, GazeQwen-informed (Pham et al., 2026) pathway at the design stage instead of after building it (§4.5).
- We isolate the one effect that does survive, a motion-specific advantage of behavioral signals on verbs over vision, and show that under a fair comparator it holds only in direction, never above the verb prior, and that the real constraint is single-participant sample size, not a floor in dynamic range (§4.6–4.7).

The result is a null we can explain rather than one we merely report, and the explanation points past input-level conditioning toward representation-level supervision of behavioral signals (§7). We keep every anticipation claim bounded to one participant at roughly one to two seconds; the training-side ablations reach across participants, and we flag which is which where it matters.

## 2 Background and related work

### 2.1 Joint-embedding predictive architectures for video

**Latent prediction.** The lineage this work builds on replaces pixel reconstruction with prediction in representation space. I-JEPA (Assran et al., 2023) established the pattern for images: from a context block, predict the *embeddings* of masked target blocks rather than their pixels, so the objective is never forced to model detail that carries no semantic content. V-JEPA (Bardes et al., 2024) carried the idea to video, and V-JEPA 2 (Assran et al., 2025) scaled it. Its three components fix the vocabulary we use throughout: a student encoder that sees the visible patches, a teacher encoder updated by exponential moving average that supplies target embeddings under a stop-gradient, and a *predictor* that maps the student's context embeddings plus positional queries for the masked positions onto what the teacher would emit there. The entire training signal is a distance between predicted and target latents; nothing is ever rendered to pixels. V-JEPA 2.1 (Mur-Labadia et al., 2026) revises the recipe toward denser features, and is the backbone of both frozen-probe systems that currently lead EK100 anticipation (§2.2).

**Frozen features as a world model.** The encoder/predictor split matters because the two can be used separately. DINO-WM (Zhou et al., 2024) builds a world model on top of frozen visual features and plans by optimising action sequences against predicted future latents, which is evidence that a frozen representation plus a learned latent predictor is already enough to support prediction-based control. Our protocol rests on the same assumption in a weaker form: we never fine-tune the representation, and read anticipation out of it with a deliberately small probe (§3.1).

**The action-conditioned predictor, and what we re-purpose.** V-JEPA 2 also ships an action-conditioned variant (V-JEPA 2-AC), in which the predictor is additionally conditioned on robot end-effector action and state, each encoded to a token and concatenated as a peer alongside the visual tokens *inside* the predictor. That token layout is precisely the interface this paper re-purposes: we substitute human gaze and hand pose for the robot's action and state and change nothing else about the pathway (§3.2). EgoAgent (Chen et al., 2025b) makes an adjacent move in the egocentric setting, learning representation, prediction and action jointly rather than treating them as separate stages.

**Language-aligned latent prediction.** A final branch of the lineage moves the *target* of latent prediction from appearance toward meaning. VL-JEPA (Chen et al., 2025a) predicts the embedding of an answer rather than generating it token by token, so that two paraphrases of the same content land near each other instead of being orthogonal in token space, and the semantics are available without waiting for a decode. Its relevance here is directional rather than architectural: it demonstrates that the JEPA objective survives a change of target, which is exactly the move our own results argue for in place of input-level conditioning — supervising the representation toward goal- or intention-level targets rather than concatenating a signal at the input (§7).

### 2.2 Conditioning prediction on behavioral signals

A parallel line of work asks how gaze and body signals should enter a predictive model, and it divides usefully by where the signal is injected. At one end sit input concatenation methods, the peer-token approach this paper diagnoses. At the other, signals supervise the representation during training and are discarded at inference. GABRIL (Banayeeanzade et al., 2025) regularises an imitation-learning representation toward human gaze to mitigate causal confusion; the gaze-regularised vision-language models of Pani and Yang (2026) align model attention to gaze via a regularisation term and report improved future-event prediction without gaze at inference; and Roy and Fernando (2022) model an abstract goal distribution and a goal-consistency objective for next-action selection. A related line conditions anticipation explicitly on intention or goal: the intention-conditioned VAE of Valls Mascaró et al. (2023), the goal-conditioning isolated by AntGPT (Zhao et al., 2024), the intention-guided cognitive reasoning of Chu et al. (2026b), and the gaze-guided, intention-conditioned graph network of Ozdel et al. (2024) all report that an intention or goal signal improves long-term anticipation, a pattern our own results echo in pointing away from input-level conditioning and toward goal-level targets. Anticipating the behavioral signal itself has also been studied directly, as in the audio-visual egocentric gaze anticipation of Lai et al. (2024). Between these, GazeQwen (Pham et al., 2026) modulates a frozen video-language model with gaze through a lightweight resampler that adds residuals at decoder layers, and argues that the obstacle is a missing learned pathway from gaze into the model's internal representations rather than a deficiency of the signal. That framing organises our own investigation: we adapt GazeQwen's mechanisms as the template for a stronger pathway, but test whether such a pathway could help before building it (§3.3).

On the benchmark itself, the two strongest published EK100 anticipation systems are both frozen-V-JEPA-2.1 attentive probes (Chu et al., 2026a; Wang and Xu, 2026). Neither conditions on behavioral signals, yet both split their probes into separate verb and noun heads on the reasoning that verbs track motion and nouns track object appearance, a split that reappears in our own results (§4.6). They mark the vision-only ceiling of the frozen-probe regime, and they leave our question, whether behavioral conditioning adds anything on top, untested.

### 2.3 Project origin

This study grew out of an earlier attempt to carry V-JEPA 2's action-conditioned predictor from robot manipulation to egocentric human video, replacing the robot's end-effector action and state tokens with human gaze and hand-pose tokens. (An even earlier exploratory step had used gaze only to bias *context selection* for masked reconstruction, and found no benefit at high coverage; it did not lead anywhere and we mention it only for completeness.) The conditioning effort was organised as a five-stage plan: reproduce the published EK100 anticipation baseline, train an own probe to own that baseline, map the action-conditioned predictor's architecture, post-train the predictor on HD-EPIC with gaze and hand tokens, and re-probe on EK100 to measure the change.

The plan did not survive contact with a deadline, and the first behavioral comparison was confounded by design. The from-scratch behavioral predictor trained on HD-EPIC was a ViT-L model, but the only fine-tuned behavioral checkpoint available in time was the ViT-G model from the project's own conditioning track, trained separately and with a differently-shaped (three-dimensional) gaze input. Rather than wait, the comparison was run across encoders, knowingly unfairly: a ViT-L vision baseline reached 4.27% action R@5 on EK100 while the ViT-G behavioral arm reached 2.81%, with the behavioral arm also behind on verb and noun recall, feature silhouette, and nearest-neighbour retrieval. Both sides were trained for a single probe epoch and were underfit. The number is therefore directional only, and we treat it as a confounded pilot rather than a result: the encoders differ, the predictors come from different training runs, and gaze is masked at EK100 evaluation. What the pilot did establish was the question this paper takes up. Behavioral conditioning had not helped, and it was not yet clear whether the fault lay with the injection pathway, the confounds, or the signal itself. The controlled study that follows was built to separate those explanations.

## 3 Method

Our method is a frozen-representation probe of whether egocentric behavioral signals carry action-anticipatory content, together with a controlled pathway by which those signals are injected into the predictor. We first describe the frozen-probe setup and evaluation protocol (§3.1), then the two conditioning pathways: a weak one that is built and verified, and a stronger one that is deliberately specified but left unimplemented (§3.2–3.3). Last comes the direct, signal-present readout used to measure the signal's information content independent of any pathway (§3.4).

### 3.1 Frozen-representation probing

We build on V-JEPA 2 (Assran et al., 2025), a self-supervised video model whose encoder and latent predictor are pretrained to predict masked spatiotemporal features rather than pixels. Throughout the anticipation probe, both the encoder and the predictor are kept frozen; the only trained component is a lightweight attentive probe that reads the predictor's output tokens and emits verb, noun, and action logits for the EK100 anticipation task.

**Backbone and projectors.** Two predictor instantiations appear in this work, and we are explicit about which carries which result. A ViT-L predictor (12 blocks, width 384) is used only for the Phase-4 feasibility fine-tuning of §4.1; every anticipation result reported below uses the ViT-G predictor (24 blocks, internal width 1024, projected back to the 1408-dimensional encoder space on exit), on a $224/16 = 14$ patch grid ($N = 196$ tokens per frame) over $T = 8$ temporal positions. Behavioral signals enter through two single linear projectors with no activation or normalisation: a gaze projector ($3 \rightarrow 1024$) and a hand projector ($12 \rightarrow 1024$), each paired with a learned mask-token parameter used when the signal is invalid. Gaze carries one validity flag; the two-hand signal carries two, combined by disjunction, so the mask substitutes only when both hands are invalid and a one-hand-valid frame passes its full twelve-dimensional vector (with the invalid half zero-filled). Of the predictor's 24 blocks, the last six (blocks 18–23) are unfrozen during conditioning. That six is the library default rather than an explicit argument, confirmed empirically by the gradient guard (Appendix C).

Freezing the backbone is a deliberate design choice rather than a shortcut: it keeps the limited supervised capacity focused on the task's label space, limits overfitting on rare classes, and makes the comparison between conditioning pathways a comparison of *inputs to a fixed predictor* rather than of two separately fine-tuned models. The same frozen-probe recipe underlies the two strongest published EK100 anticipation systems (Chu et al., 2026a; Wang and Xu, 2026), which lends the choice external support.

**Masked evaluation is in-distribution.** Several diagnostics in §4.3 compare the predictor's behaviour on the real signal against its behaviour when the signal is replaced by a mask token. This masked condition is not out-of-distribution: the warm-start predictor checkpoint was trained with a signal-dropout rate of 0.4, so masked behavioral slots were seen during training and the model has a defined response to them. Comparisons against the mask token are therefore not measuring a distribution shift. They do, however, conflate two effects — the value of the signal's content and the cost of withholding an input the model was trained to expect — and only a predictor trained without the signal at all can separate them. We report such a control in §4.4; it is null, which is what licenses the content reading we take here. This dropout rate is the checkpoint's own training setting: the warm-start predictor was fine-tuned within the project on HD-EPIC P01–P07 with P08 held out, for three epochs at a signal-dropout rate of 0.4, with the last six predictor blocks unfrozen.

### 3.2 The weak pathway: peer-token concatenation

The weak conditioning pathway is inherited from the project's earlier phases. Gaze and hand are each encoded to a single token per frame and concatenated alongside the visual tokens inside the predictor, mirroring the action–state–visual token layout of the original action-conditioned predictor. This widening is internal and transient: per frame the two behavioral tokens take the predictor's working sequence from 1568 ($T \cdot N = 8 \times 196$) to 1584 ($T \cdot (N{+}2) = 8 \times 198$), and they are stripped again before the predictor returns, so its output is once more 1568 tokens. The behavioral signal therefore conditions the prediction without appearing in the predictor's output.

**Bit-exact equivalence as a control.** For the weak pathway to serve as a baseline, it must reproduce the established Phase-5 evaluation route exactly; otherwise a later comparison would confound the pathway with an incidental change in evaluation code. We verify an exact match, with identical mean, standard deviation, and extrema, and a maximum absolute difference of zero to eight decimal places over the full $[1, 1568, 1408]$ output (Appendix C). The weak pathway is thus a faithful control, not an approximate one.

**Guarding against silent nulls.** Three mechanisms in this codebase can each produce the appearance that "conditioning does not help" without the conditioned modules ever being trained, and all three fail silently: a `no_grad` scope around the predictor forward, a parameter-group misclassification that routes new modules into a low-learning-rate group, and default weight initialisation in place of the codebase's scheme. Because the last of these reproduces the very weight-stagnation symptom the project set out to explain, we verify after a single backward pass that every unfrozen module receives a non-zero gradient; all ten watched modules pass and the frozen blocks correctly receive none (Appendix C). This guard is a precondition for any future implementation of the stronger pathway.

### 3.3 The strong pathway: designed, not built

The predictor exposes a second conditioning mode behind the same switch, reserved for a stronger pathway that *adapts* mechanisms from GazeQwen (Pham et al., 2026) to our architecture: biasing the visual tokens' keys with the behavioral signal, a bottleneck resampler that fuses gaze and hand, and a learned per-block gate over the fine-tuned predictor blocks. We stress that this is an adaptation rather than a reproduction: GazeQwen injects its signal externally, as a resampler residual added onto language-model decoder layers, and performs no in-place modification of a vision transformer's attention; the in-attention key-biasing here is our own design choice, informed by their mechanisms but structurally distinct from their implementation. **This pathway is deliberately left unimplemented.** Construction of the switch succeeds, but its forward pass raises rather than silently running a partial pathway (Appendix C); no result in this paper is produced by the strong pathway, and it is never trained or evaluated.

The ordering is intentional. Following supervision, we adopted the mechanism-adaptation reading of GazeQwen, adapting its key-biasing and resampler mechanisms into our own predictor, rather than reproducing its pipeline wholesale (its Qwen2.5-VL host, its benchmark, and live gaze), which was judged to offer little.[^1] A stronger pathway is worth building only if a stronger pathway could help; §4.3 tests exactly that before any implementation cost is incurred, and §4.5 records the resulting decision to retire it at the design stage. The design stays on record, and because its zero-initialised gate makes it runnable as a falsifier, the choice can be revisited if the sample-size constraint of §4.7 is relieved.

### 3.4 Direct readout of signal content

Independent of any injection pathway, we measure how much anticipatory information the behavioral signal carries by reading it out directly, adapting the whole-body-conditioned prediction protocol of PEVA (Bai et al., 2025) to a signal-present readout on HD-EPIC P01. We distinguish two questions the readout can answer. The first, an average-error comparison (PEVA-1), asks whether the predictor's output is on average closer to the target when the real signal is present than when it is masked, which tests detectability. The second, a discrimination comparison (PEVA-2), asks whether the signal opens a usable margin between correct and incorrect candidates, which tests whether any detectable effect is large enough to change a decision. As §4.3 reports, the signal is detectable under the first test and negligible under the second. That decoupling, a real but decision-irrelevant effect, is what the diagnostics turn on.

## 4 Experiments

This section works through one question: does conditioning a frozen V-JEPA 2 predictor on egocentric gaze and hand improve short-horizon action anticipation, and if not, why? We fix the protocol (§4.1), build the controls that separate a token-count confound from a real content effect (§4.2), and then report the null and the five diagnostics that characterise it (§4.3). A companion set of experiments then varies the training rather than the input (§4.4). Together they license the design-stage decision of §4.5. The one residual effect and its limits follow (§4.6), then the sample-size mechanism that binds the picture together (§4.7). A closing horizon analysis (§4.8) characterises how far ahead the behavioral signal is legible in the frozen representation at all.

**Three venues, and which one carries which claim.** The work spans three evaluation venues that answer different questions, and keeping them apart is essential to reading the results correctly. The EK100 result (§4.2) is *signal-absent*: EK100 provides no gaze or hand at evaluation, so the behavioral slots are masked, and a null there tells us whether a behaviorally-trained predictor transfers downstream, nothing more. The diagnostics, the dissociation, and the sample-size analysis (§4.3–4.7) all live in the *signal-present* venue, a direct probe on HD-EPIC P01, which is the only setting that can test the gaze/hand-to-motion mechanism directly. The third venue is the conditioning objective itself (§4.4, §4.8): feature-space prediction error, trained on HD-EPIC P01–P07 and evaluated on held-out participants. It is the only venue here that is cross-participant, and it is where the ablations that vary training rather than input are run. The decision to retire the stronger pathway (§4.5) rests on the signal-present diagnostics and on those ablations; the EK100 null is corroborating context, not the basis for it. We do not read a signal-absent null as evidence about the signal's content.

### 4.1 Data and protocol

The ViT-L feasibility fine-tuning below is trained on HD-EPIC (Perrett et al., 2025) P01 (the ViT-G checkpoint carrying every anticipation result was trained on P01–P07 and validated on P08; §3.1), whose Aria-quality annotations supply per-frame three-dimensional gaze (yaw, pitch, depth) and a twelve-dimensional two-hand pose signal. The anticipation probe is evaluated on the EPIC-KITCHENS-100 (EK100) action-anticipation task (Damen et al., 2022), which asks a model to predict the verb, noun, and verb–noun action of a segment from a context clip ending a fixed interval before onset; performance is reported as mean-class recall@5.

**A single-participant evaluation scope.** We state at the outset a framing constraint that governs the interpretation of every result below. Although the EK100 validation annotations for participants P01–P05 comprise 2213 clips, the evaluation in practice ran on P01 alone: of the P01–P05 videos, only the thirteen P01 videos were present on disk at run time, and the loader's missing-file filter silently drops the remainder.[^2] The evaluated set is therefore **870 validation and 2401 training clips, P01 only**; the "P01–P05 (2213)" figure describes the annotation subset, not the data actually seen. This is not a cosmetic correction. It makes the conditioning null a single-participant result, accounts directly for the small between-seed spread reported below, and is the mechanism to which we return in §4.7 and in the limitations.

**Compute.** The work ran across two clusters. The earlier conditioning experiments, including the confounded pilot of §2.3, were carried out on a single NVIDIA L40S GPU (≈46 GB) under a `tcsh` environment. The project data and checkpoints were then transferred to a second cluster, on which the frozen-probe training and evaluation reported in this paper were run.[^3] All reported anticipation numbers come from the second cluster; no result in this paper depends on the compute environment, and the binding constraint on the results is sample size rather than compute (§4.7).

**Feasibility of the conditioning target.** Before probing anticipation, we confirm the behavioral predictor was trainable at all. Fine-tuning the V-JEPA 2 predictor on HD-EPIC P01 (ViT-L, 20 epochs) reduces the per-epoch mean smooth-$L_1$ prediction loss from 1.207 to 1.019, with the curve essentially flat after roughly the fifth epoch (epoch 5 mean 1.063; epoch 15 mean 1.034).[^4] The signal is learnable, then, but its marginal return saturates early. That is an honest feasibility ceiling, not a training failure.

> [!note]- 🖼️ Figure 1 (data, not a rendered image)
> **Caption:** Per-epoch mean prediction loss for the behavioral predictor (HD-EPIC P01, ViT-L, 20 epochs). The loss falls from 1.207 to 1.019 but is essentially flat after epoch 5, indicating an early feasibility ceiling rather than a training failure. Intermediate points (1.063, 1.042, 1.034) are the epoch 5/10/15 means. The curve is drawn from the five recorded checkpoints.
>
> Plotted points (epoch, loss): (1, 1.2074), (5, 1.0632), (10, 1.0421), (15, 1.0344), (20, 1.0189).

### 4.2 Controls: separating token count from content

A conditioned run differs from an unconditioned one in more than the presence of the behavioral signal. To expose anticipatory content, the probe is fed not only the encoder's 1568 output tokens ($T \cdot N = 8 \times 196$) but also the predictor's last predicted-frame block appended alongside them, taking the probe input to 1764 tokens ($(T{+}1) \cdot N = 9 \times 196$). A naive conditioned-versus-unconditioned comparison therefore confounds any content effect with the effect of simply giving the probe more tokens. We break the confound with a control family that holds the target constant and varies only what fills the appended block:

- **6a, encoder only** (1568 tokens): no appended block.
- **6b, zero padding** (1764 tokens): the appended block is zeros.
- **6c, content-free repeat** (1764 tokens): the appended block repeats an existing frame, matching the token count without adding information.
- **Behavioral** (1764 tokens): the appended block is the real prediction from the behaviorally-conditioned predictor.

The attentive pooler used for readout carries no positional encoding in its attention path at any probed depth, so it is permutation-invariant over the token axis, which is what makes 6b and 6c valid token-count-matched controls. We show the controls before the headline comparison because the null only means something once the token-count effect has been subtracted out.

The comparison is in Table 1, and the headline reading is best-epoch, where the behavioral arm does not separate from its token-count-matched controls: the behavioral-minus-6a gap is −0.013 pp. Whatever advantage conditioning appears to give is a token-count effect, not a content effect. The final-epoch column tells a superficially different story that we address directly below, so that it is not misread.

**Table 1** — Control family for the input-conditioning null (EK100 action R@5, %, P01, signal-absent). Best-epoch values headline, with final-epoch (epoch 10) in parentheses.

| Condition | Tokens | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|---|---|---|---|---|
| 6a encoder only | 1568 | 3.69 (3.38) | 3.44 (3.16) | 3.72 (3.54) | 3.62 (3.36) |
| 6b zero padding | 1764 | 3.61 (3.61) | 3.59 (3.22) | — | 3.60 (3.42) |
| 6c content-free repeat | 1764 | 3.77 (3.46) | 3.31 (3.13) | — | 3.54 (3.29) |
| Behavioral | 1764 | 3.58 (3.48) | 3.49 (3.47) | 3.75 (3.75) | 3.61 (3.57) |

> *Caption note (verbatim):* Best-epoch is the honest comparison, though it is selected on the reported validation set: the behavioral arm does not separate from its token-count-matched controls (behavioral − 6a = −0.013 pp). On final-epoch the behavioral arm leads by 0.15–0.27 pp, but only because it is the one arm that does not decay after epoch 8; that is a decay asymmetry, not a sign of the signal. The 6b and 6c conditions were run at two seeds, the third having been dropped once the direction was closed.

### 4.3 Five diagnostics on the frozen predictor

Two readings of the null remained open after the controls: either the injection pathway is too weak to expose whatever the signal carries, or the signal does not carry action-relevant information at this horizon. The first recommends a stronger architecture; the second recommends abandoning input-level conditioning altogether. Rather than build a stronger pathway on the assumption that it would help, we ran five diagnostics against the frozen predictor and the behavioral signal directly (Table 2). They converge.

**Table 2** — Diagnostics bearing on whether a stronger pathway could recover anticipatory information. Full outputs in Appendix B.

| Diagnostic | Measurement | Interpretation |
|---|---|---|
| Output shift (real vs. masked) | relative $L_2$ = 9.9% | the signal reaches the output |
| Average-error gap (PEVA-1) | 0.055%; real>masked 172/220, $p \approx 3\times10^{-19}$ | detectable but negligible |
| Discrimination (PEVA-2) | top-1 gap 0.0000; 0/220 discordant | no discriminative margin |
| Attention on the slot (Test A) | 28.6× (gaze) / 35.4× (hand) uniform | the slot is heavily attended… |
| Attention change, real vs. masked | ≤ 0.1%, negative in sign | …but indifferent to content |
| Signal-alone anticipation (Test B) | 12.2% vs. 10.4% prior (+1.8 pp) | faint at source |

The attention result is the pivotal one. The predictor attends to the gaze and hand slots at roughly twenty-nine to thirty-five times the uniform rate, so the signal is clearly not being ignored on positional grounds. But replacing the real signal with the mask token barely moves the attention; if anything the mask draws marginally more. The model attends to where the signal sits and stays indifferent to what it says. §4.4 shows that this routing is inherited from the action-conditioned pretraining rather than learned here, which makes the absence of headroom a property of the architecture rather than of this particular run. Key-biasing, the first mechanism a stronger, GazeQwen-style pathway (Pham et al., 2026) would add, needs headroom in exactly this attention magnitude, and there is none to work with. Put beside the direct probe, where the signal beats a class prior by only 1.8 points, the reading is consistent: the signal is registered but faint, and no rearrangement of the pathway can recover information that a direct probe can barely find.

> [!note]- 🖼️ Figure 2 (data, not a rendered image)
> **Caption:** Attention on the behavioral slots, real versus masked signal, against the uniform baseline (dashed, 0.0017). Both slots draw roughly 29–35× uniform attention, yet the real and masked bars are indistinguishable (gaze 0.0491 vs. 0.0492; hand 0.0607 vs. 0.0610): the predictor attends to where the signal sits, not to what it says.
>
> Bars (real, masked, uniform dashed): gaze 0.049080 / 0.049155 / 0.001716 (≈29×); hand 0.060709 / 0.061035 / 0.001716 (≈35×).

### 4.4 Varying the training instead of the input

Every diagnostic in §4.3 holds one predictor fixed and varies what is fed to it. That instrument can establish whether the model responds to the signal, but it cannot separate the value of the information from the cost of withholding an input the model was organised around: both appear as the same difference. Separating them requires varying the *training*. The experiments below do exactly that, and they do it on the same warm-start checkpoint that carries every anticipation result above (§3.1): its training run is re-executed with the behavioral signal ablated, and the two models are compared directly. They are scored on the conditioning objective, feature-space prediction error over held-out participant P08, on a fixed set of 96 clips, paired throughout.

**The pathway is alive, graded, and small.** Holding the video fixed and perturbing the gaze input by a known angle gives a response that is monotone and near-linear from $1^\circ$ to $10^\circ$ (0.0012 → 0.0024 → 0.0061 → 0.0130, a factor of eleven in output movement for a factor of ten in angle) and super-linear beyond, reaching 0.081 at $45^\circ$, about 14% of the movement caused by replacing the entire visual context. An identity control, the same forward pass run twice, is exactly zero on all 96 clips, so every non-zero figure above is signal rather than nondeterminism. The first reading of the null, that the signal never reaches the predictor or that the pathway failed to train, is therefore false. The channel functions; its gain is simply small.

**Attention, quantified.** On a 258-token sequence the uniform share of a visual token's attention falling on the gaze token is $1/258 = 0.388\%$. Measured, visual tokens spend 12–26% of their attention there, and in one head the gaze token takes essentially the entire attention mass, 258× uniform. The same measurement on the *untrained* model (action-conditioned weights, randomly initialised projectors, no fine-tuning) already gives roughly 38× uniform, and three epochs of training move it only to about 42×. The routing is inherited from the action-conditioned pretraining that placed robot action and state tokens in that position; training barely touches it.

**Training built the pathway up, not down.** A natural explanation for a null is that optimisation suppressed the new channel. It did the opposite: the gaze projector's weight norm *grew* 15.7% against its initialisation while every other trained parameter under the same weight decay stayed within 0.05% of its own.[^5]

**Most of what the projector learned is a constant.** The mechanism behind the null is visible in the projector's parameters. Its bias moved from exactly 0.000 to 0.572 against a weight norm of 1.270, so a large share of what it learned does not depend on gaze at all. That accounts for the pattern the input-side diagnostics see: masking gaze moves the prediction 5.2× further than replacing it with a *different* clip's gaze (0.084 versus 0.016). The model separates "some gaze" from "no gaze" far more sharply than one gaze from another, which is how heavy attention coexists with small sensitivity: the heavily attended token is mostly a constant.[^6]

**What the fine-tune was actually worth.** Architecture and domain adaptation can be separated from conditioning too. The stock action-conditioned predictor is 0.125 MSE worse than the fine-tuned model running with *no* conditioning at all, on 96 of 96 clips, while gaze and hand together are worth 0.0011. Conditioning accounts for roughly 0.9% of what fine-tuning achieved; the remaining 0.125 of 0.126 is domain adaptation.

**The matched information control.** Finally, the warm-start checkpoint's own training run was repeated with the behavioral signal dropped at every step, producing a predictor that has never seen gaze or hand and is otherwise matched configuration-for-configuration. The two were scored on the identical clips. Run with real signals, the signal-trained model beats its own masked condition by $\Delta = 0.0011$ ($p = 0.0015$); measured against the never-conditioned model, the same real-signal predictor gains $\Delta = 0.0003$ ($p = 0.40$, n.s.). The first quantity is the cost of withholding an expected input; only the second is the value of the information, and it is null. Two details sharpen it. Under masked evaluation the never-conditioned model is in fact slightly *better* (−0.0008, $p = 0.015$), and feeding real gaze to a model that never saw one costs +0.082 on 96 of 96 clips. The signal is a large perturbation to a network not built around it, and no help to one that was.

### 4.5 Decision: retiring the strong pathway at the design stage

The diagnostics of §4.3 and §4.4 settle the weak-pathway-versus-faint-signal question in favour of the faint signal, and we acted on that by retiring the stronger pathway before building it. The ordering was deliberate and followed supervision. We took the mechanism-adaptation reading of GazeQwen (Pham et al., 2026), adapting its key-biasing and bottleneck-resampler mechanisms into our own predictor, rather than reproducing its pipeline wholesale, and we ran the discriminating diagnostic before spending anything on implementation.[^7]

There is a useful consequence to keeping the design around. A gate initialised at zero, left to train, would settle at zero precisely if the model found the signal unhelpful, so the retired design doubles as a falsifier rather than a dead end (§3.3). One qualification follows from §4.4: a settled-at-zero gate is not the only informative outcome, since the existing projector was not suppressed by training but grew, and still bought nothing. A gate could likewise open onto a channel that carries little.

### 4.6 A motion-specific residual, and its limits

The aggregate signal is faint, but the direct probe is not uniform across targets. Split by verb and noun, gaze and hand beat vision on *verbs*, which encode motion, and lose to it on *nouns*, which encode object identity: vision reaches 0.480 on verbs against the signal's 0.594, and 0.318 on nouns against the signal's 0.231. Put plainly, the behavioral signal anticipates how the hand will move, not what it will act on, and object identity is what makes the EK100 action space hard. The same verb/noun split turns out to be the organising principle of the two strongest published EK100 systems, both frozen-V-JEPA-2.1 probes, which separate their verb and noun heads on the same reasoning that verbs track motion and nouns track object appearance (Chu et al., 2026a; Wang and Xu, 2026). So the dissociation is not a quirk of our setup.

This one encouraging result warranted a more careful test than the exploratory probe that produced it. We evaluated it as a held-out verb-level anticipation probe on HD-EPIC P01, comparing the behavioral arm against a vision arm and against the verb class prior across horizons. An initial version was discarded on two grounds: the seeds were inert, because zero-initialised weights under full-batch deterministic optimisation consume no randomness, and model selection read the reported validation set. The corrected version restores genuine seed variance through bootstrap resampling and reports on a held-out test split; a final (fair-vision) version additionally fits the vision arm under an a-priori grid with a prior-mimicry guard, so the comparator cannot pass by collapsing onto the class marginal. Results are in Table 3.

**Table 3** — Verb-level behavior-minus-vision margins on held-out test data (fair-vision, v3). The one-second row is a separate run with its own subsample and prior and is not directly comparable to the longer horizons. An earlier dissociation figure, computed against an under-fit vision comparator, is superseded by these values and is not reported.

| Horizon | behavior − vision | 95% CI | behavior vs. prior |
|---|---|---|---|
| 1 s | +0.035 | [+0.008, +0.118] | −0.051 (below) |
| 3 s | +0.054 | [−0.029, +0.099] | −0.092 (below) |
| 5 s | +0.103 | [+0.017, +0.144] | −0.060 (below) |
| 10 s | +0.121 | [+0.054, +0.169] | −0.059 (below) |

Two qualifications are decisive. First, once the vision arm is fitted fairly, a substantial part of what had looked like a behavioral advantage turns out to have been an under-fit comparator: at the three-second horizon roughly half the exploratory margin disappears, and the earlier headline figure should not be quoted. Second, and more fundamentally, the behavioral arm never exceeds the verb class prior at any horizon; the margins against the prior are uniformly negative. This is a dissociation below the level of useful anticipation, not anticipation itself. The margin does widen with horizon, but the widening should not be read as the signal strengthening. From three to ten seconds it grows by 0.067, of which roughly half is the vision comparator decaying (0.530 → 0.495) and roughly half the behavioral arm rising (0.583 → 0.616). Both arms move, and the behavioral arm stays beneath its own class prior at every horizon, which is the fact that bounds the claim.

### 4.7 The binding constraint: sample size, not dynamic range

A natural objection is that probe accuracies so far below published EK100 baselines leave too little dynamic range to detect anything, so that we are seeing a floor artifact. The evidence points the other way, to a ceiling in sample size. The HD-EPIC verb probe trains on roughly 1900 examples, the verb-labelled subsample of P01 used for this analysis, which is a different and much smaller set than the EK100 clip counts of §4.1. On that set both arms interpolate their training data: vision reaches a training recall near unity, the behavioral arm around 0.80. This is the familiar high-dimension, low-sample regime, where no amount of optimiser tuning buys generalisation. Two observations rule out a pure floor. First, the noun contrast reverses the sign of the verb contrast, and a uniform floor cannot produce a sign reversal. Second, as §4.1 established, the evaluation is single-participant, which both explains the narrow between-seed spread and points to the concrete fix. The honest limitation is single-participant sample size, and §7 takes up the remedy.

### 4.8 Horizon analysis: how far ahead the signal is legible

The diagnostics so far ask what the predictor does with the behavioral signal at one horizon. This subsection asks a complementary question about the signal itself: over what lead time is behavioral information present in the frozen representation at all? The measurement is not anticipation recall and must not be read against Table 1. From the frozen encoder's features at time $\tau$ we regress a behavioral target at $\tau + t$ with a ridge probe, and report *skill*, the fractional reduction in squared error against a predict-the-mean baseline: zero is chance, one is exact. Data are HD-EPIC (Perrett et al., 2025) recordings. Because each participant has their own kitchen, we separate a split across held-out *recordings* (same people, unseen scenes) from a split across held-out *participants* (unseen people), and carry a hand-pose target alongside gaze as a positive control.

**Table 4** — Recoverability of behavioral targets from the frozen encoder as a function of lead $t$ (skill; 0 = chance). Gaze is near chance across held-out participants at every lead, while palm position, regressed from the same features under the same split, transfers and then decays. Whatever behavioral information the representation carries is about the present.

| Lead $t$ | gaze — held-out participants | gaze — held-out recordings | hand (control) — held-out participants |
|---|---|---|---|
| 0.00 s | +0.001 | +0.116 | +0.346 |
| 0.25 s | +0.004 | +0.071 | +0.287 |
| 0.50 s | −0.015 | +0.055 | +0.207 |
| 1.00 s | −0.007 | −0.000 | +0.079 |
| 2.00 s | −0.027 | +0.015 | −0.032 |

Three readings follow, in order of how much they constrain the rest of the paper.

**Gaze is not redundant with what the encoder already represents.** A natural way to dismiss the null of §4.2 is to say that people look at salient objects, salient objects are visible, and so the gaze token tells a visual encoder nothing it does not already have. Table 4 rules that out: across held-out participants gaze is recoverable at skill 0.001 even at zero lead, indistinguishable from chance. The conditioning signal is not redundant, which makes the null harder to explain away rather than easier.

**The collapse is specific to gaze, not an artifact of the probe or the domain gap.** Under the identical split, features and probe, palm position retains skill 0.346 across unseen participants. Participant identity is itself recoverable from these features at 88.7% (seven-way, chance 14.3%), so the domain gap between kitchens is real and large — and a behavioral target crosses it anyway. The probe can decode behavior from this representation; it is gaze specifically that does not survive a change of person.

**The horizon reading proper: behavioral legibility is a present-tense property.** Palm skill falls from 0.346 at zero lead to 0.079 at one second and to chance by two. Gaze, in the one split where it is measurable at all, falls from 0.116 to zero over the same interval. Whatever the frozen representation knows about behavior, it knows about now, and that knowledge is largely gone by the one-to-two-second mark — the horizon at which this paper's anticipation experiments run.

We are explicit about what this does not show. It characterises the decay of behavioral legibility with lead; it is not evidence that behavioral conditioning becomes *more* valuable at longer horizons. The widening margin of §4.6 is a different measurement with its own explanation, and neither published challenge system sweeps horizon at all (Chu et al., 2026a; Wang and Xu, 2026), so there is no external support for a growth reading. If anything the evidence here runs the other way: the signal is most legible at the shortest leads, which bounds what any input-level pathway could have delivered at the horizon we probe.

## 5 Discussion

**What the null does and does not say.** The central result is a null with a mechanism behind it. Conditioning a frozen V-JEPA 2 predictor on gaze and hand at the input does not improve short-horizon anticipation, and the five diagnostics say why: the signal reaches the predictor's output, the predictor does not respond to its content, and on its own the signal predicts the next action barely above a class prior. So the claim we are entitled to is narrow. Not that gaze and hand are uninformative, but that at this horizon, this sample size, and through this pathway they carry too little action-relevant information to move the prediction. That narrower claim matters, because it points to a different fix than "build a bigger pathway."

**Detectability versus decision-relevance.** A single accuracy number hides a distinction the PEVA-style readout makes visible. One question is whether the signal is detectable at all; another is whether it is large enough to change a decision. Here the signal is detectable in the statistical sense (it beats the masked version on 172 of 220 clips) yet produces no discriminative margin (not one discordant pair). An effect that is real but decision-irrelevant is easy to report as a success if only average error is measured, which is one reason we ran both stages of the readout rather than the first alone.

**The motion/object dissociation, in context.** The one piece of residual structure is that gaze and hand favour verbs while vision favours nouns. The natural reading is that behavioral signals encode how the body is about to move rather than what it will act on, and that EK100 is hard mainly because of object identity. As a performance result this carries little weight, since it sits below the verb prior. As corroboration it carries more: the two strongest published EK100 systems independently split their probes along the same verb/noun line and defend it with the same motion-versus-appearance argument (Chu et al., 2026a; Wang and Xu, 2026). When three separate efforts land on the same boundary, it is likely a property of the task rather than of our setup.

**Why input-level conditioning may be the weak use of these signals.** Set against the recent literature, our null looks less like a surprise and more like one more instance of a pattern. Methods that use gaze to *supervise* a representation during training and then drop it at inference tend to report gains where input concatenation, as here, does not: the gaze regulariser of GABRIL (Banayeeanzade et al., 2025), the gaze-regularised vision-language models of Pani and Yang (2026), and the goal-consistency objective of Roy and Fernando (2022) all take this route. GazeQwen puts the point directly, arguing that the obstacle is a missing learned pathway from gaze into the model's internal representations rather than any deficit in the signal. Our diagnostics fit that argument at the input level almost exactly: the model attends to where the signal sits but not to what it says, which is what a missing learned pathway would look like.

**Relation to the horizon analysis.** The horizon sweep of §4.8 measures something different from the margins above: how far ahead a behavioral target is recoverable from the frozen representation at all. It finds that behavioral legibility is a present-tense property, largely gone by one to two seconds, which bounds what any input-level pathway could have delivered at the horizon we probe. One caution keeps the two analyses consistent. The behavior-minus-vision margin in §4.6 does widen with horizon, but the widening splits roughly evenly between a decaying vision comparator and a rising behavioral arm, and the behavioral arm remains below its own class prior at every horizon. Since neither published challenge system sweeps horizon, there is no outside evidence for a "signal grows with horizon" reading either. The horizon numbers describe the regime; they are not evidence that a longer horizon rescues the signal.

## 6 Limitations

We put the limitations before the discussion's conclusions travel too far, because several of them set exactly how far those conclusions can go.

**Single-participant evaluation.** The binding limitation is scope. The EK100 probe was evaluated on P01 alone (870 validation clips), and the HD-EPIC direct probes use P01 as well. §4.1 explains that this followed from which videos were on disk at run time rather than from a deliberate choice. Every anticipation claim in the paper is therefore a claim about one participant. Two analyses are exceptions and should be read as such: the training-side ablations of §4.4 and the recoverability probes of §4.8 train on P01–P07 and evaluate on held-out participants, so where our conclusion rests on those it is not single-participant. We make no assertion about cross-participant generalisation, and the small spread between seeds should be read as a single-participant artifact, not as evidence of stability across the dataset.

**Sample size below the generalisation threshold.** Participant count aside, the direct probe runs on about 1900 training examples against a high-dimensional input, and both arms interpolate that training set (§4.7). In that regime held-out performance is set by sample size, not by model or optimiser choices. This is why we call the result a faint signal under a sample ceiling rather than no signal at all: the experiment as run cannot separate a genuinely absent effect from a real one too small to show at this $n$.

**Mechanism claims rest on the direct probe, not the benchmark.** The dissociation and the signal-content diagnostics come from the signal-present HD-EPIC venue. The EK100 result is signal-absent by construction, since the behavioral slots are masked at evaluation, so it speaks to whether a behaviorally-trained predictor transfers, not to the gaze/hand-to-motion mechanism. We do not carry evidence across that line. The cost of that discipline is that our mechanism claims are never checked on the benchmark task itself.

**The vision comparator sits below its own prior.** Even fitted fairly (§4.6), the vision arm does not clear the verb class prior non-degenerately, and the behavioral arm never clears it at all. The dissociation is established below the level of useful anticipation. A stronger comparator, or a task with more headroom above the prior, might change its size, though on this evidence not its sign.

**The stronger pathway was not run.** We retired the GazeQwen-informed pathway at the design stage (§4.5) on the strength of the diagnostics, without implementing it. The diagnostics argue convincingly that it would not recover the signal, but the argument is inferential. We did not falsify it directly, and a zero-initialised gated implementation is still the clean way to do so if the sample-size constraint is ever lifted.

**The signal-drop control is scored on the conditioning objective, not on EK100.** The matched never-conditioned control of §4.4 is a full-rate ablation of the very checkpoint that carries every anticipation result here, which is stronger than a comparison across models. It is scored, however, on feature-space prediction error over a held-out participant rather than through the EK100 probe itself. What remains unrun is the same ablation carried all the way through to anticipation recall; we therefore report the null in signal content as established on the conditioning objective and inherited, rather than independently re-measured, downstream.

**The checkpoint is under-trained relative to the intended scale.** Every anticipation number here rests on a predictor fine-tuned for three epochs at thirty clips per recording (§3.1). That is the small configuration, not the scale the conditioning study was designed around, and no full-scale run ever completed. This matters in a specific direction: a null measured on an under-trained predictor is weaker evidence than it reads as, because a model that has not finished learning the task has not finished learning what to do with an auxiliary input either. Two things limit how much weight this objection can carry. The feasibility curve of §4.1 is essentially flat after the fifth epoch, and the fine-tune is demonstrably worth 0.125 MSE over the stock predictor (§4.4), so the training did take. But we cannot rule out that a longer schedule would give the conditioning pathway something it currently lacks, and we do not claim otherwise.

**Goal-level probing is infeasible on one participant.** The representation-level test we wanted to run, a recipe-goal probe on HD-EPIC P01, does not work at single-participant scale. Under the hand-validity session split, four of eight recipes have no held-out examples, one recipe accounts for about three-quarters of the held-out set, and recipe identity is nearly collinear with session identity. That is a characterised data requirement rather than a failed experiment, and it is what motivates the multi-participant direction of §7.

## 7 Conclusion

We asked whether conditioning a frozen V-JEPA 2 predictor on egocentric gaze and hand improves short-horizon action anticipation, and when it did not, we asked why. For one participant at a one-to-two-second horizon on the benchmark, and across held-out participants on the conditioning objective, the answer is that the signal reaches the predictor but does not change its output in any content-dependent way. Feeding these signals in at the input is the weak way to use them. We retired a stronger conditioning pathway at the design stage on converging diagnostic evidence, described a motion-specific residual that lives below the level of useful anticipation, and traced the binding constraint to single-participant sample size rather than to any floor in the probe's dynamic range.

**Future work.** The evidence points not toward a stronger input pathway but toward a different use of the signal: supervising the representation with gaze and hand during training and discarding them at inference, aimed at goal- or intention-level targets rather than the next action. To our knowledge that intersection is still open, a frozen-JEPA latent predictor trained with a representation-level auxiliary loss on behavioral signals against a goal-level target, and it is the natural next step from here. It also needs the goal probe run across participants rather than within one. HD-EPIC P01–P07 already underpins the conditioning experiments reported here (§4.4); extending the recipe-level probe to that range is what our single-participant attempt shows to be a prerequisite rather than a convenience.

## Acknowledgments and Disclosure of Funding

We thank Jun.-Prof. Dr.-Ing. Stefan Lüdtke for overseeing the project. We also thank Ashwin Nedungadi, and Klára Orbán for supervising the Rostock and Cluj teams, respectively, and for their guidance and feedback throughout the project.

We used AI assistants (Anthropic's Claude, including Claude Code) during this project. Their role was drafting and editing assistance, code scaffolding and execution on the compute cluster, literature retrieval, and help organising and reconciling experimental records. All research questions, experimental designs, decisions, and interpretations are the authors' own, and all reported experimental results were verified by the authors against primary experimental evidence. The authors take full responsibility for the content of this report.

## References

1. Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. *CVPR*, 2023.
2. Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. *arXiv:2506.09985*, 2025.
3. Yutong Bai, Danny Tran, Amir Bar, Yann LeCun, Trevor Darrell, and Jitendra Malik. Whole-body conditioned egocentric video prediction. *NeurIPS*, 2025. arXiv:2506.21552.
4. Amin Banayeeanzade, Fatemeh Bahrani, Yutai Zhou, and Erdem Bıyık. GABRIL: Gaze-based regularization for mitigating causal confusion in imitation learning. *arXiv:2507.19647*, 2025.
5. Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. *arXiv:2404.08471*, 2024.
6. Delong Chen, Mustafa Shukor, Theo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, and Pascale Fung. VL-JEPA: Vision-language joint embedding predictive architecture. *arXiv:2512.10942*, 2025a.
7. Lu Chen, Yizhou Wang, Shixiang Tang, Qianhong Ma, Tong He, Wanli Ouyang, Xiaowei Zhou, Hujun Bao, and Sida Peng. EgoAgent: A joint predictive agent model in egocentric worlds. *ICCV*, 2025b. arXiv:2502.05857.
8. Qiaohui Chu, Haoyu Zhang, Yisen Feng, Meng Liu, Weili Guan, Dongmei Jiang, and Liqiang Nie. JFAA: Technical report for the EPIC-KITCHENS-100 action anticipation challenge at EgoVis 2026. *arXiv:2605.20904*, 2026a.
9. Qiaohui Chu, Haoyu Zhang, Meng Liu, Yisen Feng, Haoxiang Shi, and Liqiang Nie. Intention-guided cognitive reasoning for egocentric long-term action anticipation. *AAAI*, 2026b. arXiv:2311.18259.
10. Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Rescaling egocentric vision: Collection, pipeline and challenges for EPIC-KITCHENS-100. *IJCV*, 2022.
11. Bolin Lai, Fiona Ryan, Wenqi Jia, Miao Liu, and James M. Rehg. Listen to look into the future: Audio-visual egocentric gaze anticipation. *ECCV*, 2024.
12. Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mido Assran, Koustuv Sinha, Mike Rabbat, Yann LeCun, Nicolas Ballas, and Adrien Bardes. V-JEPA 2.1: Unlocking dense features in video self-supervised learning. *arXiv:2603.14482*, 2026.
13. Suleyman Ozdel, Yao Rong, Berat Mert Albaba, Yen-Ling Kuo, Xi Wang, and Enkelejda Kasneci. Gaze-guided graph neural network for action anticipation conditioned on intention. *ETRA*, 2024. arXiv:2404.07347.
14. Anupam Pani and Yanchao Yang. Gaze-regularized VLMs for ego-centric behavior understanding. *arXiv:2603.23190*, 2026.
15. Toby Perrett, Ahmad Darkhalil, Saptarshi Sinha, Omar Emara, Sam Pollard, Kranti Kumar Parida, Kaiting Liu, Prajwal Gatti, Siddhant Bansal, Kevin Flanagan, Jacob Chalk, Zhifan Zhu, Rhodri Guerrier, Fahd Abdelazim, Bin Zhu, Davide Moltisanti, Michael Wray, Hazel Doughty, and Dima Damen. HD-EPIC: A highly-detailed egocentric video dataset. *CVPR*, pages 23901–23913, 2025. arXiv:2502.04144.
16. Trong-Thang Pham, Hien Nguyen, and Ngan Le. GazeQwen: Lightweight gaze-conditioned LLM modulation for streaming video understanding. *arXiv:2603.25841*, 2026.
17. Debaditya Roy and Basura Fernando. Predicting the next action by modeling the abstract goal. *arXiv:2209.05044*, 2022. Published at ICPR 2024.
18. Esteve Valls Mascaró, Hyemin Ahn, and Dongheui Lee. Intention-conditioned long-term human egocentric action anticipation. *WACV*, pages 6048–6057, 2023. arXiv:2207.12080.
19. Chaoyang Wang and Lexuan Xu. TAP-JEPA: Frozen future-latent probing and two-stage score fusion for EPIC-KITCHENS-100 action anticipation. *arXiv:2606.00662*, 2026.
20. Qi Zhao, Shijie Wang, Ce Zhang, Changcheng Fu, Minh Quan Do, Nakul Agarwal, Kwonjoon Lee, and Chen Sun. AntGPT: Can large language models help long-term action anticipation from videos? *ICLR*, 2024. arXiv:2307.16368.
21. Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. DINO-WM: World models on pre-trained visual features enable zero-shot planning. *arXiv:2411.04983*, 2024.

## Appendix A — Contribution statement

The authors contributed to distinct, complementary workstreams under a shared research goal. This report was written by Parsa and Erfan; Ioana is a project collaborator whose contributions inform the work but who did not author this report.

**Seyyed Parsa Sharifi.** Designed and directed the diagnostic study of the conditioning pathway: the EK100 anticipation probe of the behaviorally-trained predictor and its 6a/6b/6c control family; the bit-exact verification of the weak (peer-token) pathway; the silent-null gradient guard; the five diagnostics (output shift, PEVA-1/2, attention Test A, direct-probe Test B); the verb/noun dissociation analysis and its fair-vision correction; and the $d \gg n$ sample-size analysis. Specified the stronger (GazeQwen-informed) pathway and made the design-stage decision to retire it. Proposed the diagnostic-first approach adopted under supervision. Led the writing of the Method, Experiments, Discussion, Limitations, and Conclusion, and assembled the report.

**Erfan Yekehzare.** Authored the JEPA/VL-JEPA background (§2.1) and the horizon analysis (§4.8), including the frozen-feature recoverability probes and the hand-pose transfer control they report. Designed and built the ego predictor that integrates the behavioral projection layers, together with its fine-tuning and evaluation pipeline, and trained the ViT-G behavioral checkpoint on which every anticipation result in this report is based. Built and ran the companion conditioning track that supplies the matched never-conditioned control and the untrained-model attention measurement of §4.3.

**Ioana Marica (collaborator, non-author).** Contributed the representation- and signal-level hypotheses, an initial implementation of the gaze and hand projection layers and their token construction, the encoder-supervision direction, and the PEVA evaluation idea. These inform the framing and future-work direction of this report; Ioana did not author the report itself.

## Appendix B — Extended experimental details

### B.1 Full control-family seed table

Table 1 in the main text reports best-epoch (final-epoch in parentheses). For completeness: condition means over available seeds are, best / final, Phase 5 3.606/3.567; 6a 3.619/3.360; 6b 3.601/3.415; 6c 3.544/3.294. All values are EK100 action R@5 (%), P01, signal-absent evaluation, read from the per-run evaluation CSVs (five-decimal precision; cross-checked against the tqdm logs). The 6b and 6c seed-44 runs were not launched once the direction was closed, so those two conditions have two seeds.

### B.2 PEVA stage-1 and stage-2 statistics

Stage-1 (average-error), $N = 220$: mean relative gap 0.00055; the real signal beats the masked signal on 172/220 clips (78.2%); Wilcoxon $p = 3.15 \times 10^{-19}$. Stage-2 (discrimination), $N = 220$: top-1 error identical for real and masked (0.9545; chance 0.0625); McNemar both-correct 210, real-only 0, masked-only 0, both-wrong 10; margin-$L_2$ Wilcoxon $p = 2.46 \times 10^{-24}$. The two stages together give the detectable-but-not-decision-relevant decoupling discussed in §4.3.

### B.3 Diagnostic raw outputs

Output shift: $L_2(\text{real}, \text{masked}) = 225.35$, $\lVert\text{real}\rVert = 2272.18$, relative $L_2 = 0.0992$. Test A (attention; uniform baseline 0.001716): gaze real 0.049080 (28.6×) vs. masked 0.049155; hand real 0.060709 (35.4×) vs. masked 0.061035; real > masked is false for both. Test B (direct probe; class-prior R@5 = 0.1053, chance 0.0041), per-target R@5 (vision, behavior, prior): action 0.1137/0.1216/0.1039; verb 0.4804/0.5941/0.6137; noun 0.3176/0.2314/0.3000. Note the verb prior (0.6137) exceeds both arms: the behavioral advantage on verbs is relative to vision, not to the prior.

### B.4 Verb/noun dissociation, all horizons

Horizon sweep (fair-vision v3, common set), verb R@5 with verb prior 0.6749: 3 s behavior $0.5830 \pm 0.0108$, vision $0.5295 \pm 0.0305$, margin +0.0535 [−0.0288, +0.0988]; 5 s $0.6145 \pm 0.0229$ vs. $0.5117 \pm 0.0118$, +0.1029 [+0.0165, +0.1440]; 10 s $0.6159 \pm 0.0085$ vs. $0.4952 \pm 0.0216$, +0.1207 [+0.0535, +0.1687]. One-second row (separate subsample, prior 0.5922): behavior $0.5412 \pm 0.0096$, vision $0.5059 \pm 0.0169$, margin +0.0353 [+0.0078, +0.1176]. The earlier figure computed against the under-fit vision comparator is superseded by these values.

> [!warning] ⚠️ Unresolved author note in the source, never rendered in the PDF
> The `.tex` source (`sections/B_experiments.tex`, immediately after this subsection) carries a comment invisible to any reader of the compiled PDF:
> > FLAG: test-set n for the v3 common set — Part III states 762; the reconstructed experiment log gives 243 for Option 2 v2 (with a known Notion 243→1. corruption now fixed). Confirm which n attaches to which row before stating a sample count.
> This was never resolved before submission. Table 3 and this subsection report 95% CIs and a sample described only as "the v3 common set," without the sample size ever being confirmed as 762 or 243 — a roughly 3× discrepancy that changes how much those confidence intervals should be trusted. See [[project#Open problems|the open problems]].

### B.5 $d \gg n$ comparator diagnostic

In the signal-present HD-EPIC probe both arms interpolate their training data: vision reaches training R@5 near unity while test R@5 sits at 0.50–0.62; the behavioral arm reaches training R@5 around 0.80. A prior-mimicry guard voids any run whose top-5 set collapses onto the class marginal (mean top-5 overlap ≈ 1.0). At roughly 1900 training examples the fair-vision pass-condition fails at all four horizons, locating the constraint in sample size rather than pathway design.

## Appendix C — Selected code

Excerpts are the load-bearing pieces a reader might wish to verify; full source is in the project repository (linked on the title page).

### C.1 Conditioning-mode switch and the unimplemented strong path

The strong pathway is a declared but unimplemented switch point; its forward pass raises rather than running a partial implementation (`src/models/ego_predictor.py`).

```python
_CONDITIONING_MODES = ("weak", "strong")
if conditioning_mode not in _CONDITIONING_MODES:
    raise ValueError(
        f"conditioning_mode must be one of {_CONDITIONING_MODES}, "
        f"got {conditioning_mode!r}")
self.conditioning_mode = conditioning_mode
...
if self.conditioning_mode == "strong":
    raise NotImplementedError(
        "conditioning_mode='strong' is a placeholder switch point -- not "
        "implemented. WEAK (peer-token concatenation) must be proven "
        "equivalent to the existing Phase 5 eval path before STRONG is built.")
```

### C.2 The weak path that runs — peer-token concatenation

```python
x = self.predictor_embed(x)
B, N_ctxt, D = x.size()
T = N_ctxt // (self.grid_height * self.grid_width)
gaze_tokens = self._encode_gaze(gaze_vecs, gaze_valid)      # (B, T, D)
hand_tokens = self._encode_hand(hand_vecs, l_valid, r_valid) # (B, T, D)
x = x.view(B, T, self.grid_height * self.grid_width, D)
x = torch.cat([gaze_tokens.unsqueeze(2),
               hand_tokens.unsqueeze(2), x], dim=2)
x = x.flatten(1, 2)  # (B, T*(H*W+2), D)
```

### C.3 Attention hook site for the (unbuilt) strong path

The stock attention site where the strong pathway's key-biasing would attach (`src/models/utils/modules.py`), shown to document the intended hook point.

```python
if action_tokens > 0:
    def merge_(tx, ta):
        tx = tx.view(B, self.num_heads, T, H * W, -1)
        ta = ta.view(B, self.num_heads, T, action_tokens, -1)
        return torch.cat([ta, tx], dim=3).flatten(2, 3)
    q = merge_(q, action_q)
    k = merge_(k, action_k)   # <- strong-mode key-bias would hook here
    v = merge_(v, action_v)
x = F.scaled_dot_product_attention(q, k, v, ...)
```

> [!warning] ⚠️ This exact code was not found in this repository
> `grep -rn "conditioning_mode\|IntegratedColleaguePredictor" vjepa2/src scripts/` returns nothing (re-checked 2026-08-25). This is Parsa's separately maintained codebase, filled in "from Notion mirror + Part III PDF" per the section's own source comment.

### C.4 Weak-mode equivalence check

Output of the equivalence check (`logs/weak_equivalence_2026-07-23/summary.md`): the new weak mode matches the Phase-5 route to eight decimal places.

```
IntegratedColleaguePredictor (Phase 5 route):
  shape [1,1568,1408] mean -0.01062952 std 1.52526128
  min -35.52310944 max 92.15590668
our weak-mode (masked):
  shape [1,1568,1408] mean -0.01062952 std 1.52526128
  min -35.52310944 max 92.15590668
max abs diff = 0.00000000 (threshold 1e-04)
mean abs diff = 0.00000000
verdict: MATCH
```

### C.5 Gradient-flow guard

After one backward pass every unfrozen module is asserted to carry a non-zero gradient; frozen blocks correctly carry none.

```
dummy loss = 2.338494   unfreeze_last_n_blocks = 6
module                grad_norm   n_params  n_no_grad
gaze_proj             8.29e-01    2         0
hand_proj             5.45e-01    2         0
predictor_blocks.22   2.47e+00    12        0
predictor_blocks.23   2.60e+00    12        0
all watched modules non-zero: True
frozen spot check: blocks 0, 9, 17 -> requires_grad False, grad None
```

## Appendix D — Abbreviations and notation

| Term | Meaning |
|---|---|
| JEPA | Joint-Embedding Predictive Architecture |
| V-JEPA 2 | Video JEPA 2 (frozen encoder + latent predictor) |
| V-JEPA 2-AC | … Action-Conditioned variant (robot signals) |
| V-JEPA 2.1 | Dense-feature V-JEPA revision |
| ViT-L / ViT-G | Vision Transformer, Large / Giant |
| EK100 | EPIC-KITCHENS-100 |
| HD-EPIC | Highly-Detailed EPIC egocentric dataset |
| R@5 | Recall at 5 |
| RoPE | Rotary Position Embedding |
| PEVA | Predicting Ego-centric Video from human Actions (Bai et al., 2025) |
| WEAK | Peer-token concatenation pathway (built) |
| STRONG | GazeQwen-informed fusion pathway (designed, not built) |
| 6a/6b/6c | Control family (encoder-only / +zeros / +repeat) |
| CI | Confidence interval |

---

[^1]: Our supervisor's guidance was explicit on this point: to "go with option (b)," the mechanism adaptation, because "there's nothing interesting that would come out of" a one-to-one replication of the GazeQwen paper, and to "perform the direct test as planned" before committing to implementation.
[^2]: The dataloader emits a `file path not found` line per absent video; every Phase-5/6 log contains 620 such lines, leaving `P01_102`–`P01_109` (train) and `P01_11`–`P01_15` (val). The resulting clip counts, 2401 train and 870 val, are confirmed three independent ways by the dataloader iteration counts across seeds.
[^3]: The second cluster's GPU and CPU specifications are not recorded in the sources available to us at writing, and we report the environment only for completeness.
[^4]: The same run's raw first- and last-iteration values are 2.135 and 1.002; these are a different unit from the per-epoch means and are not mixed with them.
[^5]: The mask-token baseline used throughout is itself a choice worth bounding. On the same checkpoint and clips the conditioning gap is +0.0007 against mean imputation, +0.0011 against the mask token and +0.0022 against zeros, a spread of 0.0015 that is wider than the smallest of the three effects.
[^6]: Shuffling gaze across time within a clip, which preserves the marginal distribution and destroys the temporal correspondence, moves the prediction by 1.5% of a video swap and costs +0.0004 MSE.
[^7]: The stronger pathway was designed but never built: its forward pass raises rather than silently running a partial implementation (Appendix C). Since it was never built, retiring it saved construction cost rather than discarding finished work, and the design stays on record in case the binding constraint of §4.7 is relieved.
