# V-JEPA 2.1 Egocentric Pipeline — Full Reference

## 1. Project Overview

Egocentric video understanding pipeline built on V-JEPA 2.1.
Given a query timestamp, the model predicts what the last video frame looks like
in embedding space, given context from earlier frames. The central question:
does knowing where the person is looking (gaze) help the predictor reconstruct
what comes next?

**Working directory:** `/mnt/data/home/zj2433/Projects/Ego/CV4Egocentric/v_jepa2/`
**Config:** `configs/vjepa2_1.json`

---

## 2. The JEPA Framework

### Core Idea
Instead of predicting pixels or text tokens, JEPA predicts in **embedding space**.
Given a masked video, the model must predict what the hidden region looks like
as an abstract representation — not as an image.

Two payoffs:
- **Efficiency** — no autoregressive decoder, prediction is a single forward pass
- **Better geometry** — semantically similar outcomes land nearby in the space,
  so the learning target is simpler and the space itself is meaningful

### The Three Components

#### Student Encoder (ViT-L, 1024-d)
- Sees the context tokens (visible patches)
- Encodes them into a latent representation
- **Trained** via gradients — this is the model you want to be useful

#### Teacher Encoder
- Sees the full unmasked clip
- Provides the ground-truth embedding targets the predictor must match
- **Never trained directly** — updated only via EMA of student weights
- Acts as a stable reference frame

#### Predictor (lightweight ViT, 384-d internal)
- Sits between student and teacher
- Takes student's context embeddings + positional queries for masked positions
- Predicts what the teacher would output for those positions
- **Trained** via gradients alongside the student

### Training Objective
