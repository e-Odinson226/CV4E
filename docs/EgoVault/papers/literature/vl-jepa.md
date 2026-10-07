---
type: paper
status: done
created: 2026-07-16
updated: 2026-07-28
tags: [paper, vl-jepa, language-alignment, jepa]
---

# VL-JEPA — Vision-Language Joint Embedding Predictive Architecture

Link: arXiv 2512.10942

The anchor for the language-alignment direction. Not the *subject* of the thesis — the
direction generalises beyond one paper — but the clearest existing instantiation of
"predict concepts, not appearance". See [[6-next-steps|Path 3]].

> **`created:` is inferred.** The source notes were undated; 2026-07-16 is the date the paper
> is known to have been under discussion (the supervisor review). It was read earlier.

## Problem

Vision-language models answer questions about video by **generating text one token at a
time**. The paper identifies two costs:

> "VLMs are expensive to develop, because they are trained to generate responses Y to queries
> by capturing both task-relevant semantics with task-irrelevant surface linguistic features
> such as words choice, style or paraphrasing."

> "VLMs rely on autoregressive token-by-token decoding, which must be completed before
> revealing the underlying semantics of Y. This process introduces unnecessary latency and
> hampers the ability to update semantics dynamically in real time."

The concrete version for our use: a model watching live video and tracking whether the
person's goal has changed must finish generating a whole sentence before it can say
anything.

## Method / key idea

Instead of generating the answer, **predict the embedding of the answer** and compute the
loss in embedding space.

| | VLM | VL-JEPA |
|---|---|---|
| Training target | exact token sequence | semantic embedding |
| Two equivalent answers ("lamp turns off" / "room goes dark") | orthogonal in token space — no shared tokens | nearby points in embedding space |
| Inference | must decode all tokens before knowing intent | embedding available immediately; text optional |
| World knowledge | baked into LLM weights via language pretraining | shaped into a shared vision-language embedding space via alignment |

**Four components:**

- **X-Encoder** — frozen V-JEPA 2 ViT-L. Video frames → visual embeddings **S_V**.
- **Y-Encoder** — EmbeddingGemma-300M, trained slowly at ×0.05 LR. Text → target embedding
  **S_Y**, a point in semantic space rather than a sequence of words.
- **Predictor** — the last 8 layers of Llama-3.2-1B with the causal mask **disabled**, so
  vision and query attend bidirectionally. Maps vision + query → predicted embedding **Ŝ_Y**.
- **Y-Decoder** — inference only, and *selective*: reads an embedding out as text only when
  the embedding changes significantly. Completely bypassed during training. This cuts
  decoding operations ~2.85× without losing performance.

**Objective:** ℒ = D(Ŝ_Y, S_Y) — penalise the distance between predicted and true embedding,
not wrong words. Trained with bi-directional InfoNCE (alignment + uniformity), in two stages:
query-free pretraining, then query-conditioned SFT. Roughly **halves trainable parameters**
versus a VLM, since no heavy decoder sits in the loop.

**Where the world knowledge comes from:** during training VL-JEPA sees massive amounts of
video-text pairs. Every "video of someone gathering ingredients → text: making pasta" pulls
the visual and text embeddings closer in the shared space. Over billions of examples the
latent space organises so that visual scenes cluster near their semantically appropriate
descriptions. That is language-as-scaffold, not language-as-decoder.

## Result worth remembering

**The EK100 anticipation advantage over V-JEPA 2 widens monotonically with horizon**, on the
same ViT-L-256px encoder:

| Horizon | 1 s | 2 s | 4 s | 10 s |
|---|---|---|---|---|
| Advantage | +1.5 | +2.6 | +3.5 | +4.6 |

This is the single most load-bearing number for the direction: semantic structure helps most
exactly where visual continuity runs out — the regime where intent, not motion extrapolation,
does the work.

**Report absolute deltas, not the relative figure.** The 65% relative gain at 10 s is
11.7 vs 7.1 — small absolute numbers where relative gains flatter. Ash raised this on
2026-07-16 and it was conceded; the monotonic trend is the evidence, not the percentage.

Also worth citing: **WorldPrediction-WM** — SOTA at choosing which clip explains a state
transition, beating GPT-4o, Claude-3.5-Sonnet and Gemini-2.0. And the controlled comparison:
same encoder, data and schedule, only the loss differs — embedding prediction wins on both
performance and sample efficiency at ~half the parameters.

## Relevance to my thesis

It is the existence proof that the middle ground works: **world knowledge without token
decoding**. V-JEPA gives pure visual prediction with no goal/intent structure; a standard VLM
has world knowledge but needs generation to access it; VL-JEPA shapes the embedding space
with language during training and makes it available immediately as an embedding.

Touches [[6-next-steps|Path 3]] directly, and reframes
the EK100 test of the paper ([[BC-JEPA]]): if the null at 2 s is a horizon artefact rather than gaze being
redundant, a space with *this* property is where behavioral conditioning would finally have
room to help.

## Open questions / doubts

**The honest problem with the framing.** Aligning to text reintroduces language — as a
*scaffold* rather than a *decoder*. Language supervision organises the space but doesn't
require generating tokens at inference. So the bottleneck moves to **training time, not
inference time**. That is a different claim from the one the project started with, and it
should be stated that way rather than glossed.

A related trap: evaluating on EK100 action anticipation is classification against a discrete
label set — which reintroduces language supervision through the *metric* even if the model
avoids it. Worth naming before it is pointed out.

**Where VL-JEPA falls short for our use** — the gaps a thesis contribution would close:

- **Frozen, third-person-biased encoder.** Gaze, pre-grasp hand shape and wrist velocity are
  just motion patches to it.
- **No goal-level annotation.** Targets are local actions, so the space is organised around
  actions rather than goals.
- **Action-oriented queries** — "what is happening?", not "what does the person intend next?"
- **Short temporal window** (~32 frames). No mechanism for minutes-long goal context.

**Unverified premise underneath the whole direction:** VL-JEPA's gains lean on rich text.
EK100's verb-noun pairs are terse. HD-EPIC's recipe steps are the proposed answer and remain
untested — this is the linchpin Ash identified. See [[6-next-steps]].
