---
type: paper
status: done
created: 2026-05-29
updated: 2026-07-28
tags: [paper, vjepa, world-model, north-star]
---

# V-JEPA 2 — Self-Supervised Video Models Enable Understanding, Prediction and Planning

Link: arXiv 2506.09985

The north star. Everything in this project is a modification of, or a measurement against,
this architecture. See [[3-method]] for the conditioning mechanism — the encoder/predictor
distinction the whole protocol rests on is that the encoder turns frames into vectors and is
always frozen here, while the predictor forecasts a future vector and is the part that gets
fine-tuned (see [[2-background#Terms|the terms list]]).

> **`created:` is inferred.** Assembled during the 2026-07-28 migration from several undated
> sources; 2026-05-29 is the earliest dated engagement with the paper.

## Problem

Learn a world model from purely observational video — no action labels, no text, no rewards —
and show that it develops enough physical understanding to be grounded into control with a
tiny amount of interaction data.

## Method / key idea

**Mask denoising in representation space.** Mask part of a video and train a predictor to
guess the *latent representation* of the missing part from the visible context. Nothing is
rendered to pixels; the target is another embedding.

The objective is the whole training signal:

$$\min \; \| P_\phi(\Delta y,\, E_\theta(x)) - \text{sg}(E_\theta(y)) \|_1$$

Three components:

- **Student encoder** — sees the context (visible patches), encodes them to a latent
  representation. Trained by gradients.
- **Teacher encoder** — sees the full unmasked clip and provides the ground-truth embedding
  targets. Never trained directly; updated only by EMA of the student. A stable reference
  frame.
- **Predictor** — takes the student's context embeddings plus positional queries for the
  masked positions, and predicts what the teacher would output there. Trained alongside.

Two payoffs over generative video models: **efficiency** (no autoregressive decoder — one
forward pass) and **better geometry** (semantically similar outcomes land nearby, so the
learning target is simpler and the space itself is meaningful).

**The data-efficiency argument**, which is the reason this family was chosen:

> Internet video shows the world without action labels — just sequences of states. V-JEPA 2
> shows that learning a world model from this purely observational data is enough to develop
> physical understanding. Then a tiny amount of interaction data (62 hours) is sufficient to
> ground that understanding into controllable action.

**V-JEPA 2-AC** extends this by conditioning the predictor on robot actions and states, so it
predicts the future latent *given* the effector's motion. That extension is the thing this
project substitutes into — see [[3-method]].

## Result worth remembering

Pretrained on **over 1 million hours of video**. The EK100 action-anticipation number is the
comparison point for the EK100 test of the paper ([[BC-JEPA]]).

**The paper's own stated limitations** — worth quoting, because three of the four are exactly
where this project lives:

- V-JEPA 2 does not fully solve EK100; there are failure cases where the model gets the verb,
  the noun, or both wrong (failure distribution in Appendix D.2).
- It focuses on **1 second** anticipation. **Accuracy degrades at longer time horizons**
  (Appendix D.2). ← the opening for the horizon sweep.
- EK100 is limited to kitchen environments with a closed, well-defined vocabulary, and
  generalisation to other environments is unknown.
- Actions come from a fixed category set, making it impossible to generalise to categories
  absent from training. ← the opening for language alignment.

## Relevance to my thesis

It is the base model, the pretrained checkpoint, and the training objective all at once. The
fine-tuning in [[4-results#^test1|Test 1]] starts from the AC variant's weights and
reuses its loss verbatim; the probe in the EK100 test of the paper ([[BC-JEPA]]) freezes its encoder.

Practical consequence worth remembering: because the loss *is* the V-JEPA 2 objective,
measuring feature-prediction MSE on HD-EPIC needs **no labels at all**. That is what made a
label-free evaluation track possible alongside the labelled EK100 one.

## Open questions / doubts

The four gaps this project's framing rests on — where the architecture is silent and a
contribution might live:

1. **No intent-level latent structure.** The space is learned purely from local
   spatiotemporal consistency. Nothing encourages goal-equivalent trajectories to cluster, or
   separates "what the scene looks like" from "what the agent intends to do."
2. **No hierarchical temporal abstraction.** Intent operates at multiple timescales at once —
   "I will pick up the cup" (2 s) and "I am making coffee" (5 min) need different levels of
   abstraction. Current JEPA variants are single-scale.
3. **No egocentric inductive biases.** Hand-object interaction regions, gaze allocation and
   ego-motion should be structurally privileged, but generic video JEPA treats all patches
   equally.
4. **No benchmark for latent intent evaluation.** Most long-term-anticipation benchmarks
   evaluate discrete verb-noun predictions. There is no standard benchmark for the *quality
   of the latent representation* for downstream intent inference.

**The sharpest version of the doubt**, and it is a real one: nothing in V-JEPA's training ever
told it that tomatoes + garlic + pasta = cooking pasta sauce. There is no signal connecting
object co-occurrence to goals. Whatever semantic structure exists in its latent space emerged
purely from predicting masked patches across a billion frames — which is a very different
thing from knowing about the world in a goal-directed sense. That gap is what
[[6-next-steps|Path 3]] tries to close.

A counter-doubt worth holding alongside it (Ioana's): the frozen encoder is never supervised
on *visible* context tokens, so it is not forced to encode coherent object-level structure at
all. If that is right, probing it for "intent" may be underpowered regardless of what
conditions the predictor — which would be a different explanation for
the EK100 test of the paper ([[BC-JEPA]]) than either redundancy or horizon.
