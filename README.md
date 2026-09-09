# AC-WM: Augmentation-Consistent World Models for Robust Visual Planning

> Offline latent world models plan well in-distribution, but their imagined rollouts drift under visual distribution shift. AC-WM fixes this with a training-time multi-step rollout consistency constraint — at zero test-time cost.

**Status:** scaffolding · work in progress · target venue: ICML 2027

---

## 1. Problem (one sentence)

DINO-WM achieves zero-shot visual planning by predicting future **frozen DINOv2 patch features** and optimizing actions with MPC/CEM; but under visual distribution shift (camera pose, background, lighting) the latent rollout drifts and planning collapses — and because the encoder is frozen with **no invariance enforced during dynamics training**, nothing in the training objective prevents this.

Two facts that define the gap:

- **DINO-WM's tested OOD axes are all environment-configuration level** (WallRandom / PushObj / GranularRandom — layout, object shape, particle count). **Visual perturbation axes are untested.**
- Its observation model is a **frozen** DINOv2, so robustness is entirely inherited from pretraining — never trained for.

---

## 2. Method

### Notation

| Symbol | Meaning |
|---|---|
| $o_t$ | RGB observation |
| $z_t = \mathrm{Enc}(o_t) \in \mathbb{R}^{N \times E}$ | DINOv2 patch tokens, $N = 196$, $E = 384$, **Enc frozen** |
| $A(\cdot)$ | augmentation operator (applied at **render time**, not 2D image warp) |
| $z^{\mathrm{aug}}_t = \mathrm{Enc}(A(o_t))$ | augmented-view tokens |
| $f_\theta$ | causal ViT transition model (the only trainable dynamics module) |
| $\phi$ | action MLP |

### DINO-WM baseline

$$\hat z_{t+1} = f_\theta\big(z_{t-H:t},\ \phi(a_{t-H:t})\big), \qquad \mathcal{L}_{\mathrm{pred}} = \big\| \hat z_{t+1} - z_{t+1} \big\|^2_2$$

### AC-WM: K-step rollout consistency

Roll out **K steps** from a clean history and from an augmented history with the same actions:

$$\hat z^{\mathrm{clean}}_{t+k} = f_\theta(\hat z^{\mathrm{clean}}_{t+k-H:t+k-1}, a_{t+k-1}), \quad \hat z^{\mathrm{clean}}_{t} = z_t$$

$$\hat z^{\mathrm{aug}}_{t+k} = f_\theta(\hat z^{\mathrm{aug}}_{t+k-H:t+k-1}, a_{t+k-1}), \quad \hat z^{\mathrm{aug}}_{t} = z^{\mathrm{aug}}_t$$

$$\boxed{\ \mathcal{L}_{\mathrm{rollout}} = \frac{1}{K}\sum_{k=1}^{K} \Big\| \hat z^{\mathrm{aug}}_{t+k} - \mathrm{sg}\big(\hat z^{\mathrm{clean}}_{t+k}\big) \Big\|^2_2 \ }$$

$$\mathcal{L} = \mathcal{L}_{\mathrm{pred}}^{\mathrm{clean}} + \mathcal{L}_{\mathrm{pred}}^{\mathrm{aug}\to\mathrm{clean}} + \lambda\, \mathcal{L}_{\mathrm{rollout}}$$

where $\mathrm{sg}(\cdot)$ is stop-gradient and $\mathcal{L}_{\mathrm{pred}}^{\mathrm{aug}\to\mathrm{clean}} = \| f_\theta(z^{\mathrm{aug}}_{t-H:t}, a) - z_{t+1} \|^2$ (the 1-step special case).

### Design note (why this form)

The encoder is **frozen**, so consistency cannot be imposed on the representation directly — there are no trainable parameters to change $z$. AC-WM therefore constrains the **dynamics**: the rollout starting from a perturbed history must agree with the rollout from the clean history, i.e. the transition model learns to absorb the observation shift within a few steps.

> Optional variant (decided by Week 3): add a lightweight trainable adapter $g_\psi$ on top of the frozen encoder and add $\mathcal{L}_{\mathrm{rep}} = \| g_\psi(z^{\mathrm{aug}}) - g_\psi(z) \|^2$. This changes the architecture and requires an extra baseline **"DINO-WM + adapter w/o consistency"**.

### Test-time cost

**Zero.** The constraint exists only during training; planning proceeds exactly as in vanilla DINO-WM (CEM over action sequences, cost $\| \hat z_T - z_g \|^2$).

---

## 3. Planned experiments

**Environment.** ManiSkill3, `PickCube` and `StackCube`, dual cameras (`agentview` + `eye_in_hand`), scripted demonstrations from the built-in motion planner.

**Baselines** (each one blocks a specific reviewer objection):

| Baseline | Blocks the claim that… |
|---|---|
| ACT, Diffusion Policy | …only world models need this |
| DP + aug | …the gain is just augmentation (policy side) |
| DINO-WM (vanilla) | — (lower bound) |
| **DINO-WM + aug** | **…the gain is just augmentation (world-model side)** ← critical |
| AC-WM (ours) | — |
| AC-WM, K = 1 | …multi-step consistency is unnecessary |
| copy-current-frame | …the world model looks trivially good (rollout table only) |

**OOD axes** (3 axes × 3 severity levels):

1. Camera pose (extrinsic jitter on `agentview`)
2. Background & texture (table/background swap + color jitter)
3. Lighting (intensity & direction)

*Protocol note:* the **goal image is rendered under the same OOD condition** as the observation. Otherwise the goal lives in the clean distribution while the observation does not, and the task becomes unreachable in feature space — that would measure a protocol bug, not model capability.

**Metrics**

- Rollout quality: feature MSE @ $k \in \{1, 5, 10, 20\}$ on each OOD axis
- Planning: MPC/CEM success rate, $\ge 100$ episodes $\times$ 3 seeds, mean $\pm$ std
- Diagnostic (this paper's own contribution): **quantitative relation between rollout drift and success rate**
- Efficiency: training and test-time cost vs. vanilla

**Ablations:** $K \in \{1,5,10,20\}$; $\lambda \in \{0.1, 0.5, 1, 5\}$; augmentation-type leave-one-out.

**Gates**

| Gate | When | Criterion | Fallback |
|---|---|---|---|
| Engineering | W3, 6–8 h | data format, dual camera, action alignment, CEM runs, ID success > 0 | switch host to the (trainable) policy encoder |
| Survival | W4–5 | vanilla drops $\ge$ 15 pts on some axis **and** `+aug` does not recover it | increase severity → change axis → pivot to benchmark/analysis paper |
| Effect | W6–7 | AC-WM beats **`+aug`** by $\ge$ 8 pts, std < 5 | tune $\lambda$/K/aug strength → downgrade to analysis paper |

---

## 4. Repository layout

```
acwm/
├── configs/          # hydra-style experiment configs
├── src/
│   ├── envs/         # ManiSkill3 wrappers, OOD harness
│   ├── policies/     # ACT / DP / DINO-WM / AC-WM
│   └── training/
├── scripts/          # data generation, DINOv2 feature caching, eval
├── docs/             # reading log, experiment log, application log
└── README.md
```

## 5. Reference

DINO-WM — *World Models on Pre-trained Visual Features enable Zero-shot Planning* (Zhou, Pan, LeCun, Pinto; ICML 2025, arXiv:2411.04983).
