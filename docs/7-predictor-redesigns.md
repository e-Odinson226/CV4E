---
type: report
status: planned
created: 2026-10-08
updated: 2026-10-08
---

# 7. Predictor Redesigns for Spatial Gaze

This document explores architectural redesigns to address the "Spatial Mismatch" identified in [[5-discussion#Weak points of the design]]. The core problem is that injecting gaze as a global scalar token (as done currently and in GazeQwen) forces the frozen transformer to learn complex trigonometry to map gaze to patches. Without a rigorous spatial mechanism, gaze defaults to a "scalar gate" that only reacts to its presence, rather than acting as a spatial selector.

Below are four architectural redesigns to physically force the predictor's attention onto the gaze point, ordered from least to most invasive.

## 1. Spatial Attention Bias (Gaussian Masking)

Instead of treating gaze as a separate sequence token, this approach directly intercepts and modifies the self-attention matrices inside the early predictor blocks.

- **How it works:** Project the 3D gaze coordinate (yaw, pitch, depth) to a 2D point $(u, v)$ on the 16×16 patch grid. Generate a 2D Gaussian heatmap centered at this point. When calculating the attention matrix for the image tokens, add this heatmap (scaled by a learned parameter $\gamma$) directly to the logits before the Softmax operation: 
  $$ \text{Attention} = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d}} + \gamma \cdot M_{gaze}\right) V $$
  where $M_{gaze}$ is the 2D Gaussian mask (0 near the gaze, heavily negative far away).
- **Pros:** 
  - **Mathematically Bulletproof:** It literally rewires the attention heads to prioritize patches near the gaze coordinate. The model cannot ignore the spatial location.
  - **Non-Invasive:** The token counts, dimensions, and weight matrices of the V-JEPA predictor remain completely unchanged. You only add a bias tensor during the forward pass.
- **Cons:** It relies on a hardcoded Gaussian prior (assuming gaze relevance is strictly radial and continuous), which might not map perfectly to complex object boundaries.

## 2. Gaze as a Spatial Query (DETR-style)

Instead of concatenating gaze as a global context token `[gaze, hand, img_1, ... img_256]`, treat gaze as a spatial query that explicitly cross-attends to the image.

- **How it works:** Project the gaze to a 2D coordinate on the grid. Generate a 2D sinusoidal positional embedding (the exact same function used to encode the spatial positions of the image patches). Add this spatial embedding directly to a learned "gaze query" token. 
- **Pros:** 
  - When the gaze query cross-attends to the image patches, the dot product $Q_{gaze} K_{image}^\top$ will naturally maximize at the image patches whose positional embeddings match the gaze token's embedding. 
  - It uses the transformer's native mechanisms (positional embeddings) rather than hardcoded masking.
- **Cons:** Because the V-JEPA predictor's attention blocks are frozen, they were never trained to use a dynamically moving spatial query token. It might require unfreezing more blocks or adding a dedicated cross-attention layer before the main predictor body.

## 3. Spatial Feature Modulation (FiLM)

Instead of relying on the attention mechanism to dynamically route information, this approach modulates the visual features *before* they enter the predictor's attention blocks.

- **How it works:** Create a 16×16 heatmap based on the projected gaze point. Pass this heatmap through a lightweight CNN or MLP to generate scaling ($\gamma$) and shifting ($\beta$) factors for every image patch (similar to FiLM layers). Modulate the patch features: 
  $$ X_{new} = \gamma \odot X + \beta $$
- **Pros:** Patches near the gaze point are artificially boosted in magnitude. When these modulated patches enter the standard self-attention blocks, their larger magnitudes will naturally dominate the attention weights, forcing the frozen blocks to focus on them without altering the attention code itself.
- **Cons:** It alters the variance and distribution of the pre-trained visual embeddings before they hit the predictor, which could cause covariate shift and degrade the frozen predictor's performance.

## 4. Deformable Attention (Foveated Sampling)

This is the most aggressive redesign, replacing standard dense attention with Deformable Attention (as seen in Deformable DETR).

- **How it works:** Set the projected 2D gaze coordinate as the "reference point" for deformable attention heads. Instead of computing attention over all 256 patches, the predictor calculates attention weights and samples keys/values from only a sparse, learned set of points localized physically around the gaze reference point.
- **Pros:** 
  - **Strict Foveation:** The network is physically incapable of attending to the background because it only routes information from patches immediately surrounding the gaze point.
  - Computationally efficient for high resolutions (though 16×16 is already small).
- **Cons:** This requires replacing the core attention operators in the predictor with Deformable Attention modules. This completely invalidates the pre-trained V-JEPA predictor weights, meaning you would have to train the predictor entirely from scratch, losing the massive 1-million-hour pre-training advantage.

## Analysis and Recommendation

If the goal is to solve the spatial mismatch while **preserving the pre-trained V-JEPA 2 predictor weights**, candidate **#1 (Spatial Attention Bias)** is the clear winner. 

*   **Why it wins:** It enforces strict spatial focus without altering the distribution of the visual features (unlike #3) and without requiring the frozen layers to learn new dynamic query behaviors (unlike #2). It completely avoids the need to train from scratch (unlike #4).
*   **Implementation:** You only need to intercept the `attention` calculation inside `vjepa2/src/models/utils/modules.py` to add the bias mask during the forward pass, and introduce one trainable scalar ($\gamma$) per head to control how strongly the gaze mask biases the attention. This guarantees that the gaze acts as a spatial selector, definitively solving the scalar gate problem observed in GazeQwen.
