---
type: paper
status: done
created: 2026-09-03
updated: 2026-09-03
tags: [paper, vla, jepa, world-model, robotics]
---

# VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model

**Jingwen Sun$^{1,2,*}$, Wenyao Zhang$^{3,5,*}$, Zekun Qi$^{4}$, Shaojie Ren$^{2,6}$, Zezhi Liu$^{2,7}$, Hanxin Zhu$^{1}$, Guangzhong Sun$^{1}$, Xin Jin$^{2,5,\dagger}$, Zhibo Chen$^{1,2,\dagger}$**

$^{1}$University of Science and Technology of China  
$^{2}$Zhongguancun Academy, Beijing, China  
$^{3}$Shanghai Jiao Tong University  
$^{4}$Tsinghua University  
$^{5}$Eastern Institute of Technology, Ningbo  
$^{6}$University of Chinese Academy of Sciences  
$^{7}$Nankai University  
$^*$Equal Contribution, $^\dagger$Corresponding author

- **Date:** February 17, 2026
- **arXiv:** [arXiv:2602.10098v2](https://arxiv.org/abs/2602.10098) [cs.RO] 14 Feb 2026
- **Code:** <https://github.com/ginwind/VLA-JEPA/>
- **Project Page:** <https://ginwind.github.io/VLA-JEPA/>
- **HuggingFace:** <https://huggingface.co/ginwind/VLA-JEPA/>

---

## Abstract

Pretraining Vision-Language-Action (VLA) policies on internet-scale video is appealing, yet current latent-action objectives often learn the wrong thing: they remain anchored to pixel variation rather than action-relevant state transitions, making them vulnerable to appearance bias, nuisance motion, and information leakage. We introduce VLA-JEPA, a JEPA-style pretraining framework that sidesteps these pitfalls by design. The key idea is leakage-free state prediction: a target encoder produces latent representations from future frames, while the student pathway sees only the current observation—future information is used solely as supervision targets, never as input. By predicting in latent space rather than pixel space, VLA-JEPA learns dynamics abstractions that are robust to camera motion and irrelevant background changes. This yields a simple two-stage recipe—JEPA pretraining followed by action-head fine-tuning—without the multi-stage complexity of prior latent-action pipelines. Experiments on LIBERO, LIBERO-Plus, SimplerEnv and real-world manipulation tasks show that VLA-JEPA achieves consistent gains in generalization and robustness over existing methods.

---

## 1 Introduction

Learning visuomotor policies from internet-scale video has become an increasingly attractive route for robot learning [32, 31]. Compared to robot interaction data, which are costly and narrow in coverage [21, 46], unlabeled videos are abundant and diverse, offering rich demonstrations of temporally extended change. This has motivated a growing body of latent-action pretraining for Vision–Language–Action (VLA) [10, 39, 9, 34]: rather than learning actions only from scarce control trajectories, they first learn representations and transition structure from video, then adapt them to downstream control [79, 13, 20].

However, we argue that today’s “latent action from video” objectives often do not learn what we actually need for control. For an embodied agent, the most useful notion of “action” is not a compact descriptor of pixel differences; it is a variable that captures how the underlying state will evolve under interaction, i.e., action-relevant state transition semantics. When pretraining is misaligned with this goal, the downstream policy inherits representations that are temporally predictive yet weakly tied to controllable structure—leading to brittle behavior, poor transfer, and inefficient fine-tuning.

### Why latent-action pretraining often drifts away from action semantics?

We identify four failure modes that repeatedly appear in latent-action pretraining pipelines built on unlabeled video:

1. **Pixel-level objectives bias representations toward appearance, not action.** A common strategy is to use “the future” as supervision—either by predicting future pixels directly or by compressing frame-to-frame changes into a latent variable interpreted as an action [79, 13, 86]. Even with compression mechanisms such as VQ-VAE [71], the supervision signal is frequently dominated by what changes visually: texture, illumination, background clutter, and viewpoint. These factors are high-variance but low-control; they are easy to predict yet only weakly connected to the controllable degrees of freedom that a policy must master.

2. **Real-world videos amplify noisy motion.** This mismatch becomes especially pronounced on human videos and in-the-wild footage [32, 33], where camera motion and non-causal background changes can be stronger than interaction-induced state changes. Frame-difference-based latent-action objectives are then incentivized to encode these dominant signals, turning the latent action into a delta-frame encoder of nuisance motion rather than a representation of meaningful transition dynamics.

3. **Information leakage makes “latent action” collapse into a shortcut.** Several latent-action pipelines model transitions by feeding both the current observation and future observation into the same module, or by allowing future context to influence the learned action variable during training [79, 13]. This design creates an easy shortcut: the latent action can simply encode the future itself instead of capturing how state transitions should be explained [82]. The resulting “action” becomes semantically empty—useful for matching training losses, but not a meaningful factor for control.

4. **Multi-stage training pipelines are complex and fragile.** To stabilize training and mitigate the above issues, many approaches rely on three-stage (or more) procedures: representation pretraining, latent-action learning/alignment, then policy learning [79, 19, 27]. These pipelines increase engineering complexity, introduce stage-wise inconsistencies, and make it harder to train and evaluate methods cleanly.

Together, these issues point to a single underlying problem: many latent-action objectives remain implicitly anchored to pixel variation, and thus learn dynamics that are predictive but not necessarily action-centric [82]. For embodied control, we want a value-bearing latent state that discards nuisance appearance while preserving factors governing state evolution under interaction [33, 62, 74, 36]. This motivates a key principle: predict future latent states that reflect action-relevant transition structure, while preventing future information from leaking into the predictor.

This aligns naturally with JEPA (Joint-Embedding Predictive Architectures) [4, 2, 1, 23], which replaces pixel reconstruction with latent-space alignment. By predicting representations rather than pixels, JEPA is inherently robust to low-level noise and encourages semantic abstraction [30].

Motivated by this perspective, we introduce VLA-JEPA, a JEPA-style pretraining framework tailored to VLA policies. As shown in Figure 1, the key design is leakage-free state prediction: during pretraining, a target video encoder produces latent targets from future context (e.g., a short clip), while the latent action pathway receives only the current observation through a VLM backbone. A predictor maps the history latent states and the latent action representations to future latent states, which is trained as a latent world model using a JEPA alignment loss. Crucially, future frames are never provided as inputs to the VLM backbone; instead, they are used solely to construct the training targets, eliminating the shortcut that causes latent-action collapse. This yields two benefits: (1) semantic robustness to camera motion and background changes, since supervision operates in latent space rather than pixel space; (2) a streamlined two-stage pipeline—JEPA pretraining followed by action-head fine-tuning—without introducing auxiliary modules or redefining the learned representations. This work makes three contributions:

- **Analysis of latent-action pretraining pitfalls.** We analyze why many future-supervision objectives (including compressed frame-difference latent actions) remain pixel-tethered, becoming biased toward appearance, vulnerable to nuisance motion in real-world videos, and prone to information leakage when future context enters the learner.
- **VLA-JEPA: leakage-free, state-level JEPA pretraining.** We propose a JEPA-style latent predictive alignment scheme that learns action-relevant transition semantics by predicting and aligning future latent states—without pixel reconstruction, information leakage and only one-stage pretraining pipeline.
- **Improved and robustness with a simpler workflow.** Across embodied control benchmarks (LIBERO, LIBERO-Plus, SimplerEnv) and real-world settings, VLA-JEPA yields consistent gains in robustness, and generalization, while simplifying training relative to prior multi-stage latent-action pipelines.

---

## 2 Related Works

### 2.1 Vision-Language-Action Models

With the rapid advancement of Large Language Models (LLMs) [50, 37, 6, 24, 77, 64] and large-scale robot datasets [59, 25, 38, 22], Vision-Language-Action (VLA) models have become a dominant paradigm in robot learning [39, 9, 76, 42]. The RT series [10, 92, 5] pioneered fine-tuning multimodal LLMs on robot demonstrations, and subsequent works further improved manipulation and navigation performance [39, 9, 66, 45, 85, 84, 63, 75]. However, most VLA methods rely heavily on large-scale action-labeled robot data, which is costly and difficult to scale [72]. To reduce dependence on explicit action supervision, recent works introduce multimodal Chain-of-Thought signals, such as hierarchical planning [5, 47, 65, 91, 35], subgoal or rollout prediction [69, 87, 88, 73, 48, 17, 80, 52], object-centric conditioning [22, 34, 67, 90, 83, 54, 15] and latent future embeddings or actions [12, 55, 86, 53, 89]. Nevertheless, these approaches suffer from relying on action-labeled data. In contrast, VLA-JEPA learns action-centric representations via latent predictive alignment, avoiding explicit future reconstruction and reducing the need for large-scale action supervision.

### 2.2 Latent action learning for robotics

To leverage large-scale videos without action labels, ILPO [26], LAPO [68] and Genie [11] propose using latent action for video games. For robot learning, LAPA [79], IGOR [18], UniVLA [13], MotoGPT [20], Adaworld [29], CoMo [78] and StaMo [51] extract discrete or continuous motion tokens from frame transitions, and pretrain VLA to predict these latent actions before mapping them to real robot controls. To align the latent action to the real action space, villa-X [19], XR-1 [27], CLAP [81] and VITA [56] propose that extract latent action from both robot and human video and use a unified codebook. However, since latent actions are often learned directly from adjacent frames, models may exploit pixel-level shortcuts and encode future-frame leakage. Although some methods attempt to mitigate the above issues by constraining the latent action space—through optical-flow [14, 7], or object-centric constraints [41], among others [57, 58]—such constraints inevitably bias the learned latent actions toward aligning with hand-crafted, visually defined priors rather than the true underlying controllable factors. Consequently, when such priors become mismatched in novel environments, the learned latent actions tend to fail systematically, and the latent spaces may align more with visual deltas than with actionable control signals, necessitating multi-stage training pipelines and additional alignment mechanisms [82]. On the contrary, our proposed VLA-JEPA learns action-relevant representations without relying on delta frame info extraction, avoiding leakage and pixel shortcuts while enabling single-stage end-to-end pretraining.

---

## 3 Methodology

To address the limitations of existing approaches, we introduce VLA-JEPA, a unified framework that enables joint pre-training on both action-free and action-labeled data. For action-free human videos, VLA-JEPA extracts latent actions from vision-language prior representations by optimizing a world-model-based state transition objective. Building upon this foundation, we further integrate a flow-matching-based action generator for robot demonstrations to support precise end-effector trajectory generation. During fine-tuning, VLA-JEPA enables end-to-end fusion of the two objectives, allowing the learned state-transition dynamics to be effectively leveraged for downstream robotic control.

### 3.1 Model Backbone

As shown in Figure 2, we adopt Qwen3-VL [3] as the core large vision-language model (VLM) in our framework. This VLM is built upon the Qwen3 [77] and uses SigLIP-2 [70] as its vision encoder. It is well established that the world knowledge acquired by VLMs during large-scale pretraining, including image understanding and key object detection, can be transferred to robot control tasks. To distill continuous latent action representations from the VLM, which encode state transition information for world modeling, we introduce a set of learnable tokens that denoted as $\langle\text{latent}_i\rangle$ and $\langle\text{action}\rangle$, where $i$ denotes the time step. For instance, $\langle\text{latent}_0\rangle$ represents the state transition between $s_0$ and $s_1$. To model state transitions in continuous videos, we encode pixel-level human videos into time-step–aware feature sequences using an encoder architecture trained with V-JEPA2 [2], and employ a time-causal attention mechanism to capture the correlations between the feature sequences and the latent actions.

### 3.2 Learning from Human Videos

To enable VLM to learn from human demonstration videos, we design a training framework that explicitly injects environment dynamics into the latent action tokens. Specifically, we consider a human video dataset $\mathcal{D} = \{(O_0, O_1, \dots, O_v, \ell)\}$, where $\ell$ denotes the associated language instruction, and each $O_v$ represents a video captured from viewpoint $v$. Formally, $O_v = (I_{v,t_0}, I_{v,t_1}, \dots, I_{v,t_n})$, where $I_{v,t_i}$ denotes the video frame at time step $t_i$ from view $v$, and $n$ is the total number of frames within a video.

**World State Encoder.** Unlike traditional single-view video representation approaches, we encode diverse observations from the same episode into a unified world state representation through a world state encoder. Specifically, we adopt a self-supervised V-JEPA2 encoder as the single-view video state representation and integrate representations from multiple viewpoints via a concatenation operator. The encoding process for each view and the subsequent aggregation across views are formulated as follows:

$$s_{t_i} = \mathop{\Vert}_{v} F(I_{v,t_i}) \tag{1}$$

where $F(\cdot)$ is the single-view video encoder (e.g., V-JEPA2), and $\Vert$ denotes the vector concatenation operator. The resulting $s_{t_i}$ is the unified world-state representation at timestamp $t_i$.

**Latent Action Pretraining through World Modeling.** To encourage the learnable latent action tokens to capture state transition dynamics, we introduce a world state prediction objective based on an auto-regressive transformer-based world model.

Formally, the VLM takes as input the multi-view observations at the initial time step $t_0$, together with the language instruction $\ell$. Conditioned on these inputs, the VLM maps a set of special learnable tokens $\langle\text{latent}_i\rangle$ into latent representations that summarize the underlying world dynamics:

$$z_{t_i} = p_\theta^{\text{VLM}}\left(\langle\text{latent}_i\rangle \mid \{I_{j,t_0}\}_{j=0}^v, \ell\right), \tag{2}$$

where $z_{t_i}$ denotes the latent representation associated with the $i$-th latent action token at time step $t_i$.

Subsequently, the unified representation $z_{t_i}$ is used to condition the world model, providing additional context for state prediction. Formally, given the sequence of encoded world states $s_{t_{0:i}}$ and the corresponding conditioning variables $z_{t_{0:i}}$, the world model predicts the next chunk of states as:

$$\hat{s}_{t_{1:i+1}} = p_\theta^{\text{WM}}(s_{t_{0:i}}, z_{t_{0:i}}), \tag{3}$$

where $\hat{s}_{t_{1:i+1}}$ is the predicted world state chunk over $[t_1, t_{i+1}]$.

In practice, each special latent token $\langle\text{latent}_i\rangle$ is replicated $K$ times in the input sequence to enable variable-length latent action encoding, where $K$ is a tunable hyperparameter. The world model employs a time-causal attention mechanism. Within each time step, all latent action tokens and world state tokens attend to each other via bidirectional full attention. Across different time steps, attention is strictly causal: tokens at time step $t$ are allowed to attend only to tokens from time steps up to and including $t$, while attention to future time steps is masked out.

From the perspective of joint-embedding predictive architecture [4, 2], our training objective can be interpreted as maximizing the evidence lower bound (ELBO) of the predictive log-likelihood in semantic space. Specifically, given the frozen V-JEPA2 encoder $F(\cdot)$ that produces target world states $s_{t_i}$, and the world model $p_\theta^{\text{WM}}$ conditioned on $z_{t_i}$ that predicts $\hat{s}_{t_i}$, the objective can be written as:

$$\log p(s_{t_{1:T}} \mid z_{t_{0:T-1}}) \ge \sum_{k=1}^T \mathbb{E}_{s_{t_k} \sim F(\cdot)}\left[\log p_\theta(\hat{s}_{t_k} \mid s_{t_k})\right] - D_{\text{KL}}\left[F(\cdot) \mathbin{\Vert} p_\theta^{\text{WM}}\right], \tag{4}$$

where $F(\cdot)$ serves as the frozen target encoder (with stop-gradient), and $p_\theta^{\text{WM}}$ is the online predictor. In practice, since $F(\cdot)$ produces deterministic embeddings, the KL term vanishes, and the ELBO reduces to a reconstruction loss in the latent space.

Finally, we optimize the combined WM and VLM using a teacher-forcing objective. This enables the unified representation $z_{t_i}$ informed by the world knowledge encoded in the VLM, in order to effectively characterize world state transitions. The world modeling loss is defined as:

$$\mathcal{L}_{\text{WM}} = \sum_{k=1}^T \mathbb{E}_{s_{t_k} \sim F(\cdot)} (\hat{s}_{t_k} - s_{t_k}), \tag{5}$$

where $s_{t_k}$ denotes the ground-truth world state at time $t_k$, $\hat{s}_{t_k}$ is the corresponding prediction, and $T$ is the video prediction horizon.

### 3.3 Action Prediction with Joint Optimization Objectives

**Action Token Conditioning.** To leverage the latent action representations learned from video data for guiding action prediction, we design joint optimization objectives. Specifically, for multi-view RGB videos in the robot dataset, we adopt the same training objective as in Equation 5 to fine-tune the latent action representations within the robot data domain. For practical action prediction, we intend the latent action to serve as a conditioning signal for embodied action generation, analogous to the roles of the initial image observation $I_{v,t_0}$ and the language instruction. Therefore, we append a set of learnable embodied action tokens $\langle\text{action}\rangle$ after the latent action tokens. Leveraging the causal attention mechanism of the VLM, the model captures the dependencies among $\langle\text{action}\rangle$, the latent action tokens, the initial visual observation, and the language instruction.

Formally, given the visual tokens $\{I_{i,t_0}\}_{i=0}^v$ at the initial time step $t_0$, the language instruction $\ell$, and a sequence of latent action tokens $\langle\text{latent}_i\rangle$, we obtain a global action-conditioning representation:

$$z_a = p_\theta^{\text{VLM}}\left(\langle\text{action}\rangle \mid \{I_{i,t_0}\}_{i=0}^v, \ell, \langle\text{latent}_i\rangle\right), \tag{6}$$

where $z_a$ serves as an additional conditioning signal for the flow-matching-based action head.

**Conditional Flow-Matching Action Head.** We adopt conditional flow matching to model a distribution over continuous action trajectories. Specifically, let $a_{0:H}$ denote the ground-truth action sequence over a horizon $H$, and $\hat{a}_{0:H}$ denote the predicted action sequence generated by the learned flow. Following standard flow-matching formulations, we define a time-dependent interpolation:

$$a_t = (1 - t)\epsilon + t a_{0:H}, \quad t \sim \mathcal{U}(0, 1), \tag{7}$$

where $\epsilon \sim \mathcal{N}(0, I)$ is Gaussian noise. The action head parameterizes a vector field $v_\theta(a_t, t \mid z_a)$, conditioned on $z_a$, which is trained to match the ground-truth conditional flow.

The flow-matching objective is given by:

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{a_{0:H}, \epsilon, t} \left[ \left\| v_\theta(a_t, t \mid z_a) - (a_{0:H} - \epsilon) \right\|_2^2 \right], \tag{8}$$

where $v_\theta(\cdot)$ denotes the predicted velocity field, $(a_{0:H} - \epsilon)$ is the target velocity induced by the linear interpolation, and $\|\cdot\|_2$ is the $\ell_2$ norm.

At inference time, the learned vector field is integrated from noise to data space to obtain the predicted action trajectory $\hat{a}_{0:H}$, conditioned on the action tokens $z_a$.

In summary, the overall training objective for action-labeled robot data is as follows:

$$\mathcal{L} = \mathcal{L}_{\text{FM}} + \beta \mathcal{L}_{\text{WM}}, \tag{9}$$

where $\beta$ is a tunable hyperparameter.

---

## 4 Experiments

To evaluate the generalization and robustness of VLA-JEPA, we perform comprehensive simulation experiments utilizing two distinct simulation environments and three public benchmarks, Simpler-Env & LIBERO, along with real-world experiments covering both in-distribution and out-of-distribution settings, as shown in Figure 3. We compare VLA-JEPA against the latest VLA baselines that support pretraining with human videos as well as VLA baselines pretrained solely on robot-collected data.

### 4.1 Implementation Details

During the pretraining phase, we train all parameters in the model except for the world state encoder. Specifically, we utilize Equation 5 as the training objective and pretrain on the large-scale human action dataset Something-Something-v2 [31], which contains 220K human videos. The model supports simultaneous pretraining of latent action using both robot action data and human videos without action labels. We also employ Equation 9 as the training objective and pretrain on the large-scale action-labeled robot dataset Droid [38], which consists of 76K high-quality demonstration trajectories.

During fine-tuning, for both LIBERO and LIBERO-Plus, we use the LIBERO dataset, which includes approximately 2K expert demonstrations collected in a simulation environment, without incorporating the augmented dataset from LIBERO-Plus. For SimplerEnv, we utilize the Fractal dataset and the BridgeV2 dataset for post-training, corresponding to the two robot embodiment types in SimplerEnv. For the real-world experiments, we employ a dataset of 100 demonstrations collected across three distinct tasks for post-training. All experiments are conducted on 8 NVIDIA A100 GPUs. Detailed implementation and protocols are provided in Appendix A.

### 4.2 Simulation Setup and Baseline

**Benchmarks.** We conduct generalization experiments on the LIBERO [49] and SimplerEnv [44] benchmarks. The LIBERO benchmark employs the Franka Emika Panda arm and comprises four task suites designed to facilitate research on lifelong learning in robotic manipulation. SimplerEnv features WidowX and Google Robot setups, offering diverse manipulation scenarios with varying lighting, colors, textures, and robot camera poses, thereby bridging the visual appearance gap between real and simulated environments. Furthermore, we conduct robustness experiments on LIBERO-Plus [28], a large-scale benchmark designed to systematically stress-test VLAs through perturbed tasks across seven dimensions.

The three simulation environments correspond to:

1. in-distribution scenarios in which policies, trained using simulated expert data, are validated on in-distribution tasks in simulation (LIBERO);
2. out-of-distribution scenarios involving a real-to-sim gap, in which policies are trained using real-world data and evaluated in simulation (SimplerEnv);
3. out-of-distribution scenarios in which policies, trained using simulated expert data, are validated on out-of-distribution tasks in simulation (LIBERO-Plus).

**Baselines.** We primarily compare VLA-JEPA with previous latent-action VLAs, future-prediction VLAs, as well as state-of-the-art open-source VLAs including Moto [20], LAPA [79], UniVLA [13], villa-X [19], CoT-VLA [87], WorldVLA [16], RoboVLMs [43], GR00T N1 [8], OpenVLA-OFT [40], $\pi_0$ [9], $\pi_0$-Fast [61], and $\pi_{0.5}$ [34].

### 4.3 Simulation Evaluation

**LIBERO.** As presented in Table 1, VLA-JEPA achieves state-of-the-art performance on 2 out of the 4 task suites within the LIBERO benchmark, securing the highest average success rate overall. We observe that other top-performing models on LIBERO, such as OpenVLA-OFT and $\pi_{0.5}$, rely on extensive robot datasets for pre-training. In contrast, VLA-JEPA achieves better performance using less training data. Furthermore, previous latent-action-based VLAs and VLAs trained on human videos, such as UniVLA, villa-X, LAPA, and CoT-VLA, consistently underperform VLA-JEPA, which corroborates the limitations of such approaches discussed in Section 1.

#### Table 1: Comparison on the LIBERO benchmark

*We report the task success rate for each suite and the average across all tasks. Each task is evaluated over 50 episodes (500 episodes per suite), and success rates are computed over these trials. Bold denotes the best performance, and italics denotes the second best.*

| Method | Spatial | Object | Goal | LIBERO-10 | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LAPA [79] | 73.8 | 74.6 | 58.8 | 55.4 | 65.7 |
| UniVLA [13] | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| OpenVLA-OFT [40] | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| $\pi_0$ [9] | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| $\pi_0$-Fast [61] | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| CoT-VLA [87] | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| WorldVLA [16] | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| villa-X [19] | 97.5 | 97.0 | 91.5 | 74.5 | 90.1 |
| GR00T N1 [8] | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| $\pi_{0.5}$ [34] | **98.8** | *98.2* | **98.0** | 92.4 | *96.9* |
| **VLA-JEPA** | 96.2 | **99.6** | 97.2 | **95.8** | **97.2** |
|  *w/o human videos* | 94.8 | **99.6** | 95.8 | 94.0 | 96.1 |

---

**SimplerEnv.** Table 2 reports the results on SimplerEnv. VLA-JEPA achieves the best performance on 2 out of 4 tasks for both the Google Robot and the WidowX Robot. In particular, it attains the highest average success rate on the Google Robot and the second highest average success rate on the WidowX Robot. Since SimplerEnv does not provide expert demonstrations, we observe that the quality of robot data plays a crucial role in performance on this benchmark. For example, LAPA extracts only successful rollouts in SimplerEnv and uses them as expert demonstrations for training, and achieves the second-highest success rate on the WidowX Robot with merely 100 rollouts. This is mainly because training on successful rollouts effectively mitigates the real-to-sim gap. Villa-X, on the other hand, is trained on a large-scale collection of robot data and human videos, achieving the highest average success rate on the WidowX Robot. In comparison, while VLA-JEPA and other methods—such as UniVLA, RoboVLMs, and Moto—all utilize less than 1% of the training data used by villa-X, VLA-JEPA attains the most competitive experimental results. This demonstrates the superiority of our pretraining method.

#### Table 2: Comparison on the SimplerEnv benchmark

*We report the average success rate for each individual task under the visual matching setting in both SimplerEnv-WidowX Robot and SimplerEnv-Google Robot. $*$ denotes that the method is trained on in-distribution expert demonstrations collected in simulation environment. All other methods are trained on subsets of the OXE dataset.*

| Method | Google: Pick | Google: Move | Google: Drawer | Google: Place | Google: Avg | WidowX: Spoon | WidowX: Carrot | WidowX: Block | WidowX: Eggplant | WidowX: Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LAPA$^*$ [79] | — | — | — | — | — | 70.8 | 45.8 | 54.2 | 58.3 | 57.3 |
| villa-x [19] | 81.7 | 55.4 | 38.4 | 4.2 | 44.9 | 48.3 | 24.2 | 19.2 | 71.7 | 40.8 |
| UniVLA [13] | — | — | — | — | — | — | — | — | — | 42.7 |
| RoboVLMs [43] | 77.3 | 61.7 | 43.5 | 24.1 | 51.7 | 45.8 | 20.8 | 4.2 | 79.2 | 37.5 |
| GR00T N1 [8] | 0.7 | 1.9 | 2.9 | 0.0 | 1.4 | 1.4 | 0.0 | 0.0 | 13.9 | 3.8 |
| MoTo [20] | 74.0 | 60.4 | 43.1 | — | — | — | — | — | — | — |
| OpenVLA-OFT [40] | — | — | — | — | — | 34.2 | 30.0 | 30.0 | 72.5 | 41.8 |
| $\pi_0$ [9] | 72.7 | 65.3 | 38.3 | — | — | 29.1 | 0.0 | 16.6 | 62.5 | 40.1 |
| $\pi_0$-Fast [61] | 75.3 | 67.5 | 42.9 | — | — | 29.1 | 21.9 | 10.8 | 66.7 | 48.3 |
| **VLA-JEPA** | **88.3** | 64.1 | 59.3 | 49.1 | **65.2** | **75.0** | **70.8** | 12.5 | 70.8 | **57.3** |
|  *w/o human videos* | 85.3 | 66.7 | 75.5 | 86.1 | 78.4 | 75.0 | 54.2 | 20.8 | 79.2 | 57.3 |

---

**LIBERO-Plus.** As shown in Table 3, VLA-JEPA achieves the best performance on 5 out of 7 perturbations in LIBERO-Plus. It demonstrates a substantial advantage over UniVLA, which is trained on human videos, and OpenVLA-OFT, $\pi_0$, and other methods trained on large amounts of robot data. This indicates that latent actions, learned through our pretraining approach, possess a level of world knowledge representation comparable to that encoded by text data. In particular, we find that VLA-JEPA demonstrates a significant advantage over all baselines under perturbations in Language, Light, Background, and Layout, which verifies that our latent action can effectively handle task-agnostic disturbances, thereby achieving a more robust and generalized policy.

#### Table 3: Comparison on the LIBERO-Plus benchmark

*We report the average success rate across each perturbation dimension, where each perturbation includes the four task suites from the original LIBERO benchmark.*

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UniVLA [13] | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| OpenVLA-OFT [40] | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| $\pi_0$ [9] | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | **79.0** | 68.9 | 53.6 |
| $\pi_0$-Fast [61] | **65.1** | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| WorldVLA [16] | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| **VLA-JEPA** | 63.3 | **67.1** | **85.4** | **95.6** | **93.6** | 66.3 | **85.1** | **79.5** |
|  *w/o human videos* | 40.3 | 55.7 | 72.9 | 88.2 | 70.5 | 38.2 | 74.6 | 62.9 |

---

### 4.4 Real-world Experiments

For real-world experiments, we designed table-top manipulation tasks using a Franka Research 3 arm equipped with a Robotiq 2F-85 gripper. We collected 100 human demonstration trajectories for training, which include 3 picking and placing tasks. In alignment with the simulation experiments and extending beyond in-distribution task evaluation, we introduce two OOD experimental protocols to rigorously assess the model’s generalization capability and robustness for physical deployment. The first OOD protocol involves executing tasks not present in the training data, validating the model’s ability to acquire and transfer fundamental skills. The second protocol requires executing tasks seen during training but with randomized object layouts, thereby emulating the cluttered scenarios typical of real-world table-top manipulation tasks. For fair comparison, we finetune $\pi_0$ and $\pi_{0.5}$ on collected demonstration datasets. The detailed task settings can be found in Appendix B.

As shown in Figure 4, VLA-JEPA achieves state-of-the-art performance under both the ID and the object layout OOD settings, demonstrating a clear advantage over both $\pi_0$ and $\pi_{0.5}$. In the task OOD setting, it attains the second-best result. During real-world deployment, we observed that the generalization capability of VLA-JEPA is less robust than that of $\pi_{0.5}$, yet it produces execution trajectories that are significantly more stable and reliable. Specifically, $\pi_{0.5}$ more accurately follows instructions to contact the target object compared to VLA-JEPA. However, its position control frequently violates the safety boundaries of the robot arm, leading to execution failures. In contrast, VLA-JEPA, due to its lack of fine-grained reasoning over textual instructions, is prone to grasping objects that do not align with the command. Nevertheless, it rarely breaches the robot arm’s safety constraints.

In addition, VLA-JEPA acquires the skill of repeated grasping, i.e., reopening the gripper to attempt another grasp after a failure, which is not observed in $\pi_0$ or $\pi_{0.5}$. We attribute this to the abundance of repeated-grasping knowledge present in human videos, whereas this skill does not require additional physical dynamics knowledge. In contrast, robot data rarely contain demonstrations specifically designed for repeated grasping. A corresponding demonstration can be found in the Appendix B.

---

### 4.5 Further Analysis and Ablation Study

In this section, we investigate the following questions under the multi-view setting to thoroughly evaluate the ability of our model:

**Q1: What impact does human video have?** Tables 1, 2, 3 present a comparison of VLA-JEPA pretrained with human videos versus without such pretraining, demonstrating that ablating the human video component from the pre-training data does not lead to a significant performance drop across LIBERO and SimplerEnv benchmarks. Interestingly, even pretraining without human videos can lead to higher success rates on SimplerEnv. This indicates that, for ID scenario and real-to-sim gap OOD scenario, high-quality expert demonstrations are more critical for ensuring model performance compared to human videos.

On the LIBERO-Plus benchmark, human demonstration videos provide substantial performance gains. Our analysis suggests that this is because human videos lack effective information about action trajectories, preventing VLA models from directly learning the physical dynamics of robotic actions. Instead, the primary benefit lies in enhancing the robustness and stability of the model’s pre-existing skills, such as repeated grasping. This process closely resembles human skill acquisition from observation: when a person learns a skill by watching a video, the first attempt is unlikely to succeed, and proficiency is achieved only after repeated trials that establish the correspondence between the observed video knowledge and the underlying physical dynamics, ultimately leading to a stable and reliable policy.

Figure 5 illustrates how the success rates across different perturbation dimensions on the LIBERO-Plus benchmark vary as the proportion of human videos in the pre-training data increases. These results further corroborate our hypothesis that human video data primarily enhances the robustness and stability of the VLA model by strengthening its existing skill repertoire, rather than introducing new action execution capabilities. Moreover, as the scale of human video data increases, the robustness of the resulting policy consistently improves.

**Q2: What is the impact of unified pretraining?**
According to Section 4.3, the experimental results demonstrate that our unified pretraining method consistently outperforms the previous two-stage pretraining paradigm. In addition, to more clearly analyze the information captured by latent action tokens, we visualize the attention weights from latent action tokens to image tokens in the internal attention maps of three VLAs—LAPA, UniVLA, and VLA-JEPA—under simulated, human-video, and real-world robot image inputs. For a fair comparison, all methods are evaluated using pretrained-only checkpoints, without any fine-tuning on either simulated or real data.

As shown in Figure 6, the latent actions of LAPA focus on excessively dense visual information, which results in the inclusion of too many operation-irrelevant details, such as unrelated objects on the desktop. We attribute this to the leakage of information during the pretraining stage, leading to the degradation of latent actions into compressed representations of the target images, a point further analyzed in Section 1. In contrast, UniVLA alleviates this issue through task-relevant textual guidance, yet its overemphasis on semantics causes attention to operation-irrelevant background elements in the images, such as the stationary pen in human videos or the texture of the tablecloth in real-world wrist-view recordings.

By comparison, we focus more precisely on the operation, for instance, the robotic arm, the hand, and the objects to be manipulated. This demonstrates that the unified pretraining approach simplifies the training pipeline while improving training effectiveness by reducing the impact of task-irrelevant information.

#### Table 4: Comparison of VLA-JEPA on the LIBERO benchmark with different numbers of future video horizon

| $T$ | Spatial | Object | Goal | 10 | Avg |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 4 | **95.0** | 99.2 | 95.8 | 89.0 | 94.8 |
| 8 | 94.8 | **99.8** | 95.8 | **94.0** | **96.1** |
| 16 | 92.8 | 98.8 | **98.0** | 92.2 | 95.5 |

**Q3: How does the number of future video horizon affect performance?**
We aim for latent actions to capture the motion dynamics between consecutive frames, such that the number of latent action tokens is always equal to the number of frames minus one. We vary the number of future video horizon $T \in \{4, 8, 16\}$ to investigate the impact of dynamic information on the performance. For a fair comparison, all models are directly fine-tuned on the LIBERO dataset, with all other hyperparameters kept the same in the comparison.

As shown in Table 4, the model achieves its best when the video horizon is close to the predefined action horizon, which indicates that latent actions are effective for embodied action generation. When $T$ is too small, the encoded information is insufficient, leading to inferior performance, particularly on long-horizon tasks. In contrast, an excessively large $T$ introduces redundant information. It yields the best results on the goal-oriented task suite with relatively simple objectives, but performs worst on the spatial task suite that requires fine-grained manipulation.

---

## 5 Conclusion

This paper proposes VLA-JEPA, a unified pretraining framework that learns latent actions from both human and robot videos via latent world modeling. Extensive experiments and analyses reveal that existing latent action pretraining methods suffer from information leakage and representation degeneration, which hinder the learning of meaningful temporal dynamics. In contrast, VLA-JEPA effectively mitigates these issues and enables the model to capture genuine inter-frame dynamics. As a result, VLA-JEPA achieves competitive performance in both simulation benchmarks and real-world robotic experiments. We further believe that the human-video pretraining paradigm introduced by VLA-JEPA is highly scalable, and can be naturally extended by incorporating robot data and text-based reasoning data, thereby further improving the generalization and robustness of VLA models.

---

## Acknowledgement

This work is supported by the Zhongguancun Academy, (Grant No.s C20250302).

---

## References

[1] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture, 2023.

[2] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. *arXiv preprint arXiv:2506.09985*, 2025.

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. *CoRR, abs/2511.21631*, 2025.

[4] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mido Assran, and Nicolas Ballas. V-jepa: Latent video prediction for visual representation learning. 2023.

[5] Suneel Belkhale, Tianli Ding, Ted Xiao, Pierre Sermanet, Quon Vuong, Jonathan Tompson, Yevgen Chebotar, Debidatta Dwibedi, and Dorsa Sadigh. Rt-h: Action hierarchies using language. *arXiv preprint*, 2024.

[6] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. *arXiv preprint*, 2024.

[7] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, et al. Motus: A unified latent action world model. *arXiv preprint arXiv:2512.13030*, 2025.

[8] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. *arXiv preprint*, 2025.

[9] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi0: A vision-language-action flow model for general robot control. *arXiv preprint*, 2024.

[10] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alexander Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil J. Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael S. Ryoo, Grecia Salazar, Pannag R. Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong T. Tran, Vincent Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. RT-1: robotics transformer for real-world control at scale. In *Robotics: Science and Systems XIX*, Daegu, Republic of Korea, July 10-14, 2023, 2023.

[11] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In *Forty-first International Conference on Machine Learning*, 2024.

[12] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xu Huang, Shu Jiang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. *arXiv preprint*, 2025.

[13] Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Univla: Learning to act anywhere with task-centric latent actions. *arXiv preprint*, 2025.

[14] Xizhou Bu, Jiexi Lyu, Fulei Sun, Ruichen Yang, Zhiqiang Ma, and Wei Li. Laof: Robust latent action learning with optical flow constraints. *arXiv preprint arXiv:2511.16407*, 2025.

[15] Junhao Cai, Zetao Cai, Jiafei Cao, Yilun Chen, Zeyu He, Lei Jiang, Hang Li, Hengjie Li, Yang Li, Yufei Liu, et al. Internvla-a1: Unifying understanding, generation and action for robotic manipulation. *arXiv preprint arXiv:2601.02456*, 2026.

[16] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, et al. Worldvla: Towards autoregressive action world model. *arXiv preprint*, 2025.

[17] Jiayi Chen, Wenxuan Song, Pengxiang Ding, Ziyang Zhou, Han Zhao, Feilong Tang, Donglin Wang, and Haoang Li. Unified diffusion vla: Vision-language-action model via joint discrete denoising diffusion process. *arXiv preprint arXiv:2511.01718*, 2025.

[18] Xiaoyu Chen, Junliang Guo, Tianyu He, Chuheng Zhang, Pushi Zhang, Derek Cathera Yang, Li Zhao, and Jiang Bian. Igor: Image-goal representations are the atomic control units for foundation models in embodied ai. *arXiv preprint arXiv:2411.00785*, 2024.

[19] Xiaoyu Chen, Hangxing Wei, Pushi Zhang, Chuheng Zhang, Kaixin Wang, Yanjiang Guo, Rushuai Yang, Yucen Wang, Xinquan Xiao, Li Zhao, et al. villa-x: Enhancing latent action modeling in vision-language-action models. *arXiv preprint*, 2025.

[20] Yi Chen, Yuying Ge, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, and Xihui Liu. Moto: Latent motion token as the bridging language for robot manipulation. *arXiv preprint*, 2024.

[21] Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. In *Proceedings of Robotics: Science and Systems (RSS)*, 2024.

[22] Shengliang Deng, Mi Yan, Songlin Wei, Haixin Ma, Yuxin Yang, Jiayi Chen, Zhiqi Zhang, Taoyu Yang, Xuheng Zhang, Heming Cui, et al. Graspvla: a grasping foundation model pre-trained on billion-scale synthetic action data. *arXiv preprint*, 2025.

[23] Matthieu Destrade, Oumayma Bounou, Quentin Le Lidec, Jean Ponce, and Yann LeCun. Value-guided action planning with jepa world models, 2025.

[24] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, and Li Yi. DreamLLM: Synergistic multimodal comprehension and creation. In *ICLR*, 2024.

[25] Frederik Ebert, Yanlai Yang, Karl Schmeckpeper, Bernadette Bucher, Georgios Georgakis, Kostas Daniilidis, Chelsea Finn, and Sergey Levine. Bridge data: Boosting generalization of robotic skills with cross-domain datasets. *arXiv preprint*, 2021.

[26] Ashley Edwards, Himanshu Sahni, Yannick Schroecker, and Charles Isbell. Imitating latent policies from observation. In *International conference on machine learning*, pages 1755–1763. PMLR, 2019.

[27] Shichao Fan, Kun Wu, Zhengping Che, Xinhua Wang, Di Wu, Fei Liao, Ning Liu, Yixue Zhang, Zhen Zhao, Zhiyuan Xu, et al. Xr-1: Towards versatile vision-language-action models via learning unified vision-motion representations. *arXiv preprint arXiv:2511.02776*, 2025.

[28] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, and Xipeng Qiu. Libero-plus: In-depth robustness analysis of vision-language-action models. *arXiv preprint arXiv:2510.13626*, 2025.

[29] Shenyuan Gao, Siyuan Zhou, Yilun Du, Jun Zhang, and Chuang Gan. Adaworld: Learning adaptable world models with latent actions. In *Forty-second International Conference on Machine Learning*.

[30] Quentin Garrido, Tushar Nagarajan, Basile Terver, Nicolas Ballas, Yann LeCun, and Michael Rabbat. Learning latent action world models in the wild. *arXiv preprint arXiv:2601.05230*, 2026.

[31] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The "something something" video database for learning and evaluating visual common sense. In *ICCV*, 2017.

[32] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In *CVPR*, 2022.

[33] Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. *arXiv preprint*, 2024.

[34] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. pi0.5: a vision-language-action model with open-world generalization. *arXiv preprint*, 2025.

[35] Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, Xinda Xue, Qinghang Su, Huaihai Lyu, Xiaolong Zheng, Jiaming Liu, Zhongyuan Wang, and Shanghang Zhang. Robobrain: A unified brain model for robotic manipulation from abstract to concrete. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025*, Nashville, TN, USA, June 11-15, 2025, pages 1724–1734. Computer Vision Foundation / IEEE, 2025.

[36] Yueru Jia, Jiaming Liu, Shengbang Liu, Rui Zhou, Wanhe Yu, Yuyang Yan, Xiaowei Chi, Yandong Guo, Boxin Shi, and Shanghang Zhang. Video2act: A dual-system video diffusion policy with robotic spatio-motional modeling, 2025.

[37] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. *arXiv preprint*, 2024.

[38] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. *arXiv preprint*, 2024.

[39] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. *arXiv preprint*, 2024.

[40] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. *arXiv preprint*, 2025.

[41] Albina Klepach, Alexander Nikulin, Ilya Zisman, Denis Tarasov, Alexander Derevyagin, Andrei Polubarov, Nikita Lyubaykin, and Vladislav Kurenkov. Object-centric latent action learning. *CoRR, abs/2502.09680*, 2025.

[42] Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, and Ning Ding. Simplevla-rl: Scaling VLA training via reinforcement learning. *CoRR, abs/2509.09674*, 2025.

[43] Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, Hanbo Zhang, and Huaping Liu. Towards generalist robot policies: What matters in building vision-language-action models. *arXiv preprint*, 2024.

[44] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Oier Mees, Karl Pertsch, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao. Evaluating real-world robot manipulation policies in simulation. In Pulkit Agrawal, Oliver Kroemer, and Wolfram Burgard, editors, *CoRL*, 2024.

[45] Zhixuan Liang, Yizhuo Li, Tianshuo Yang, Chengyue Wu, Sitong Mao, Liuao Pei, Xiaokang Yang, Jiangmiao Pang, Yao Mu, and Ping Luo. Discrete diffusion vla: Bringing discrete diffusion to action decoding in vision-language-action policies. *arXiv preprint*, 2025.

[46] Fanqi Lin, Yingdong Hu, Pingyue Sheng, Chuan Wen, Jiacheng You, and Yang Gao. Data scaling laws in imitation learning for robotic manipulation. *arXiv preprint*, 2024.

[47] Fanqi Lin, Ruiqian Nai, Yingdong Hu, Jiacheng You, Junming Zhao, and Yang Gao. Onetwovla: A unified vision-language-action model with adaptive reasoning. *arXiv preprint*, 2025.

[48] Minghui Lin, Pengxiang Ding, Shu Wang, Zifeng Zhuang, Yang Liu, Xinyang Tong, Wenxuan Song, Shangke Lyu, Siteng Huang, and Donglin Wang. Hif-vla: Hindsight, insight and foresight through motion representation for vision-language-action models. *arXiv preprint arXiv:2512.09928*, 2025.

[49] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: benchmarking knowledge transfer for lifelong robot learning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, *NeurIPS*, 2023.

[50] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. *NeurIPS*, 2023.

[51] Mingyu Liu, Jiuhe Shu, Hui Chen, Zeju Li, Canyu Zhao, Jiange Yang, Shenyuan Gao, Hao Chen, and Chunhua Shen. Stamo: Unsupervised learning of generalizable robot motion from compact state representation. *arXiv preprint arXiv:2510.05057*, 2025.

[52] Zhuoyang Liu, Jiaming Liu, Jiadong Xu, Nuowei Han, Chenyang Gu, Hao Chen, Kaichen Zhou, Renrui Zhang, Kai Chin Hsieh, Kun Wu, et al. Mla: A multisensory language-action model for multimodal understanding and forecasting in robotic manipulation. *arXiv preprint arXiv:2509.26642*, 2025.

[53] Zhuoyang Liu, Jiaming Liu, Hao Chen, Ziyu Guo, Chengkai Hou, Chenyang Gu, Jiale Yu, Xiangju Mi, Renrui Zhang, Zhengping Che, et al. Last$_{0}$: Latent spatio-temporal chain-of-thought for robotic vision-language-action model. *arXiv preprint arXiv:2601.05248*, 2026.

[54] Qi Lv, Weijie Kong, Hao Li, Jia Zeng, Zherui Qiu, Delin Qu, Haoming Song, Qizhi Chen, Xiang Deng, and Jiangmiao Pang. F1: A vision-language-action model bridging understanding and generation to actions. *arXiv preprint arXiv:2509.06951*, 2025.

[55] Jiangran Lyu, Ziming Li, Xuesong Shi, Chaoyi Xu, Yizhou Wang, and He Wang. Dywa: Dynamics-adaptive world action model for generalizable non-prehensile manipulation. *arXiv preprint*, 2025.

[56] Xiangkai Ma, Lekai Xing, Han Zhang, Wenzhong Li, and Sanglu Lu. Unifying perception and action: A hybrid-modality pipeline with implicit visual chain-of-thought for robotic action generation. *arXiv preprint arXiv:2511.19859*, 2025.

[57] Alexander Nikulin, Ilya Zisman, Denis Tarasov, Nikita Lyubaykin, Andrei Polubarov, Igor Kiselev, and Vladislav Kurenkov. Latent action learning requires supervision in the presence of distractors. In *Forty-second International Conference on Machine Learning, ICML 2025*, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of Proceedings of Machine Learning Research. PMLR / OpenReview.net, 2025.

[58] Alexander Nikulin, Ilya Zisman, Albina Klepach, Denis Tarasov, Alexander Derevyagin, Andrei Polubarov, Lyubaykin Nikita, and Vladislav Kurenkov. Vision-language models unlock task-centric latent actions, 2026.

[59] Abby O’Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, et al. Open x-embodiment: Robotic learning datasets and rt-x models. *arXiv preprint*, 2023.

[60] William Peebles and Saining Xie. Scalable diffusion models with transformers. In *ICCV*, 2023.

[61] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. *arXiv preprint*, 2025.

[62] Video Generation Pretraining. Disentangled robot learning via separate forward and inverse dynamics pretraining.

[63] Video Generation Pretraining. Disentangled robot learning via separate forward and inverse dynamics pretraining.

[64] Zekun Qi, Runpei Dong, Shaochen Zhang, Haoran Geng, Chunrui Han, Zheng Ge, Li Yi, and Kaisheng Ma. Shapellm: Universal 3d object understanding for embodied interaction. In *ECCV*, 2024.

[65] Zekun Qi, Wenyao Zhang, Yufei Ding, Runpei Dong, Xinqiang Yu, Jingwen Li, Lingyun Xu, Baoyu Li, Xialin He, Guofan Fan, et al. Sofar: Language-grounded orientation bridges spatial reasoning and object manipulation. *arXiv preprint*, 2025.

[66] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. *arXiv preprint*, 2025.

[67] Kanchana Ranasinghe, Xiang Li, Cristina Mata, Jongwoo Park, and Michael S Ryoo. Pixel motion as universal representation for robot control. *arXiv preprint*, 2025.

[68] Dominik Schmidt and Minqi Jiang. Learning to act without actions. In *The Twelfth International Conference on Learning Representations*.

[69] Yang Tian, Sizhe Yang, Jia Zeng, Ping Wang, Dahua Lin, Hao Dong, and Jiangmiao Pang. Predictive inverse dynamics models are scalable learners for robotic manipulation. *ICLR*, 2024.

[70] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. *arXiv preprint arXiv:2502.14786*, 2025.

[71] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. *NeurIPS*, 2017.

[72] Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, et al. Vla-adapter: An effective paradigm for tiny-scale vision-language-action model. *arXiv preprint arXiv:2509.09372*, 2025.

[73] Yuqi Wang, Xinghang Li, Wenxuan Wang, Junbo Zhang, Yingyan Li, Yuntao Chen, Xinlong Wang, and Zhaoxiang Zhang. Unified vision-language-action model. *arXiv preprint*, 2025.

[74] Youpeng Wen, Junfan Lin, Yi Zhu, Jianhua Han, Hang Xu, Shen Zhao, and Xiaodan Liang. Vidman: Exploiting implicit dynamics from video diffusion model for effective robot manipulation. *NeurIPS*, 2024.

[75] Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Qian Zhu, He Sun, Yong Wang, Shuailei Ma, et al. A pragmatic vla foundation model. *arXiv preprint arXiv:2601.18692*, 2026.

[76] Kechun Xu, Zhenjie Zhu, Anzhe Chen, Shuqi Zhao, Qing Huang, Yifei Yang, Haojian Lu, Rong Xiong, Masayoshi Tomizuka, and Yue Wang. Seeing to act, prompting to specify: A bayesian factorization of vision language action policy. *arXiv preprint arXiv:2512.11218*, 2025.

[77] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. *arXiv preprint arXiv:2505.09388*, 2025.

[78] Jiange Yang, Yansong Shi, Haoyi Zhu, Mingyu Liu, Kaijing Ma, Yating Wang, Gangshan Wu, Tong He, and Limin Wang. Como: Learning continuous latent motion from internet videos for scalable robot learning. *arXiv preprint*, 2025.

[79] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. *arXiv preprint*, 2024.

[80] Bozhou Zhang, Nan Song, Jingyu Li, Xiatian Zhu, Jiankang Deng, and Li Zhang. Future-aware end-to-end driving: Bidirectional modeling of trajectory planning and scene evolution. *arXiv preprint arXiv:2510.11092*, 2025.

[81] Chubin Zhang, Jianan Wang, Zifeng Gao, Yue Su, Tianru Dai, Cai Zhou, Jiwen Lu, and Yansong Tang. Clap: Contrastive latent action pretraining for learning vision-language-action models from human videos, 2026.

[82] Chuheng Zhang, Tim Pearce, Pushi Zhang, Kaixin Wang, Xiaoyu Chen, Wei Shen, Li Zhao, and Jiang Bian. What do latent action models actually learn? *arXiv preprint arXiv:2506.15691*, 2025.

[83] Jianke Zhang, Yanjiang Guo, Yucheng Hu, Xiaoyu Chen, Xiang Zhu, and Jianyu Chen. Up-vla: A unified understanding and prediction model for embodied agent. *arXiv preprint*, 2025.

[84] Jiazhao Zhang, Kunyu Wang, Shaoan Wang, Minghan Li, Haoran Liu, Songlin Wei, Zhongyuan Wang, Zhizheng Zhang, and He Wang. Uni-navid: A video-based vision-language-action model for unifying embodied navigation tasks. *arXiv preprint*, 2024.

[85] Jiazhao Zhang, Kunyu Wang, Rongtao Xu, Gengze Zhou, Yicong Hong, Xiaomeng Fang, Qi Wu, Zhizheng Zhang, and He Wang. Navid: Video-based vlm plans the next step for vision-and-language navigation. *Robotics: Science and Systems*, 2024.

[86] Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, He Wang, Zhizheng Zhang, et al. Dreamvla: A vision-language-action model dreamed with comprehensive world knowledge. *arXiv preprint*, 2025.

[87] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. *arXiv preprint*, 2025.

[88] Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3d-vla: A 3d vision-language-action generative world model. *arXiv preprint*, 2024.

[89] Linqing Zhong, Yi Liu, Yifei Wei, Ziyu Xiong, Maoqing Yao, Si Liu, and Guanghui Ren. Acot-vla: Action chain-of-thought for vision-language-action models. *arXiv preprint arXiv:2601.11404*, 2026.

[90] Zhide Zhong, Haodong Yan, Junfeng Li, Xiangchen Liu, Xin Gong, Tianran Zhang, Wenxuan Song, Jiayi Chen, Xinhu Zheng, Hesheng Wang, et al. Flowvla: Visual chain of thought-based motion reasoning for vision-language-action models. *arXiv preprint arXiv:2508.18269*, 2025.

[91] Zhongyi Zhou, Yichen Zhu, Minjie Zhu, Junjie Wen, Ning Liu, Zhiyuan Xu, Weibin Meng, Ran Cheng, Yaxin Peng, Chaomin Shen, et al. Chatvla: Unified multimodal understanding and robot control with vision-language-action model. *arXiv preprint*, 2025.

[92] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, Quan Vuong, Vincent Vanhoucke, Huong T. Tran, Radu Soricut, Anikait Singh, Jaspiar Singh, Pierre Sermanet, Pannag R. Sanketi, Grecia Salazar, Michael S. Ryoo, Krista Reymann, Kanishka Rao, Karl Pertsch, Igor Mordatch, Henryk Michalewski, Yao Lu, Sergey Levine, Lisa Lee, Tsang-Wei Edward Lee, Isabel Leal, Yuheng Kuang, Dmitry Kalashnikov, Ryan Julian, Nikhil J. Joshi, Alex Irpan, Brian Ichter, Jasmine Hsu, Alexander Herzog, Karol Hausman, Keerthana Gopalakrishnan, Chuyuan Fu, Pete Florence, Chelsea Finn, Kumar Avinava Dubey, Danny Driess, Tianli Ding, Krzysztof Marcin Choromanski, Xi Chen, Yevgen Chebotar, Justice Carbajal, Noah Brown, Anthony Brohan, Montserrat Gonzalez Arenas, and Kehang Han. RT-2: vision-language-action models transfer web knowledge to robotic control. In *CoRL*, 2023.

---

## Appendix A Implementation Details

### A.1 VLA-JEPA Architecture

**VLM Backbone.** We use Qwen3-VL-2B as the VLM backbone, which is implemented as a dense Transformer. Its vision encoder consists of a Vision Transformer and 3D convolutional modules. To capture physical dynamics, we adopt a V-JEPA2 encoder checkpoint together with a randomly initialized predictor as the latent world model, whose action inputs are the latent action tokens produced by the VLM. To enable the VLM to output time-aware latent actions and embodied actions, we designed latent action tokens $\langle\text{latent}_i\rangle$ and an embodied action token $\langle\text{action}\rangle$, which are incorporated as additional tokens into the vocabulary of Qwen3-VL. When generating the latent action tokens, we repeat the same latent action token $\langle\text{latent}_i\rangle$ $K$ times to strengthen the model’s attention to the latent action tokens, where $K = 24/T$, $T$ denotes the future video horizon, and 24 is the empirically optimal value.

**Latent World Model.** To let the latent action tokens to capture the physical dynamics between frames, we adopt an auto-regressive Transformer as the architecture of the world model with a time-causal attention mechanism. Specifically, within a single time step, the $K$ latent action tokens and the $N$ image latent tokens attend to each other bidirectionally, whereas across different time steps, attention between tokens is constrained to be causal.

#### Table 5: Configuration of the Latent World Model

| Parameter | Value |
| :--- | :---: |
| Transformer Layers | 12 |
| Attention heads | 8 |
| Image token dimension | 2048 |
| Number of image tokens per time step | 256 |
| Action token dimension | 2048 |
| Number of action tokens per time step | 3 |
| Number of view | 2 |
| Future video horizon | 8 |

**Action Head.** To generate future actions conditioned on latent action embeddings, we adopt a flow matching-based Transformer architecture, DiT-B [60], as our action head. DiT-B is conditioned on an embodied action token sequence formed by repeating the action token 32 times, where 32 is the empirically optimal value.

#### Table 6: Configuration of the Action head

| Parameter | Value |
| :--- | :---: |
| Transformer Layers | 16 |
| Attention heads | 12 |
| Token dimension | 1024 |
| State dimension | 8 |
| Action dimension | 7 |
| Future action horizon | 7 |
| Positional encoding | Learnable |
| Denoising timesteps | 4 |

### A.2 Training Details

We preprocess all training datasets following a unified pipeline. Observation images used as inputs to the VLM are resized to $224 \times 224$. Video clips used by the world-state encoder are resized to $256 \times 256$ to be consistent with the hyperparameter settings described in the previous subsection. For models pretrained with joint-position control (e.g. $\pi_0$), we use joint-space delta positions as actions and apply min–max normalization to map them into $[0, 1]$. For models pretrained with end-effector control (e.g., VLA-JEPA), we use both end-effector delta positions and delta axis–angle representations as actions, which are likewise min–max normalized to $[0, 1]$. All gripper commands are binarized to $\{0, 1\}$. To handle multi-view observations, when fewer than two camera views are available, we duplicate the world-state representation and concatenate the two copies. When more than two views are available, we select two of the available views and use them as the world-state representation.

During training, we use a batch size of 32 and train on 8 GPUs in parallel, resulting in a global batch size of 256. We adopt a cosine learning-rate schedule with linear warmup, with a peak learning rate of 1e-5 for the VLM and the latent world model, and 1e-4 for action head. For the pretraining datasets, ssv2 and droid, we jointly train the model for 50K steps. For the simulation datasets, we continue training for 30K steps starting from the last pretrained checkpoint. For the real-world datasets, we further fine-tune the model for 20K steps, also initialized from the last pretrained checkpoint.

---

## Appendix B Real-world Experiments Details

Our real-world setup consists of a Franka Research 3 robotic arm, a Robotiq 2F-85 gripper, and three Intel RealSense D435 cameras, including two third-person views and one wrist-mounted view. The collected expert demonstrations involve picking and placing grapes, apples, mangoes, and oranges from a table into a plate or a bowl. In the following experiments, each task is executed for 10 independent trials, and we report the average success rate over all trials.

For task-level OOD evaluation in real-world experiments, we consider the following tasks:

1. picking up bananas from the table and placing them into a bowl,
2. picking up peaches from the table and placing them onto a plate, and
3. picking up grapes from the table and placing them onto the top level of a shelf.

We observe that, for the banana-picking task, both $\pi_{0.5}$ and VLA-JEPA achieve approximately a 50% success rate. For the peach-picking task, due to the irregular shape of peaches, the robot frequently violates the predefined safety boundaries during execution. For the final task, since shelves are not present in the training data, none of the models can successfully place the end-effector at the top level. However, compared with $\pi_0$ and $\pi_{0.5}$, which directly collide with the shelf, VLA-JEPA exhibits a qualitatively different behavior by approaching the target from the rear side of the shelf and lifting the end-effector to a higher position. Although the task is still not successfully completed, this behavior indicates that VLA-JEPA possesses improved task generalization ability compared with the baselines.

For object-layout–level OOD evaluation in real-world experiments, we randomly select three tasks from the training set and randomly shuffle the object layouts to assess the robustness and generalization capability of the models.

As shown in Figure 7, we observe that both $\pi_0$ and $\pi_{0.5}$ fail to reattempt grasping after an unsuccessful grasp, since the training data do not contain demonstrations of repeated grasping behaviors, i.e., reopening the gripper and regrasping after a failure. Consequently, when a grasp fails, these two policies do not actively release the gripper, as they have not learned the action of opening the gripper to reattempt a grasp. In contrast, VLA-JEPA, which is pretrained on human videos and thus exposed to repeated grasping behaviors, is able to immediately open the gripper and attempt to grasp again after a failure. We argue that repeated grasping does not require learning additional physical dynamics; instead, it mainly requires the model to learn when to perform a regrasp action. Once this temporal decision is learned, the policy can internally map it to its own physical dynamics and execute the corresponding motion. This observation highlights one of the key advantages of pretraining on human videos.
