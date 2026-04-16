# CEC-Gaze: A Claim-Conditioned Evidence Coverage Model for IITB-HGC Claim Verification

## Problem

We focus on the **IITB-HGC Claim Verification** task in **EyeBench**. The benchmark asks whether we can predict if a participant will make the **correct verification judgment** from:

- a claim
- a supporting or refuting context passage
- the participant's eye movements while reading

EyeBench evaluates the task under three regimes:

- **Unseen Reader**
- **Unseen Text**
- **Unseen Reader & Text**

A trial is:

$$
T = (C, P, G)
$$

where:

$$
C = \langle c_1, \dots, c_L \rangle
$$

is the claim,

$$
P = \langle p_1, \dots, p_M \rangle
$$

is the context passage, and

$$
G = \langle g_1, \dots, g_T \rangle
$$

is the gaze sequence aligned to the displayed text.

The benchmark target is binary:

$$
y \in \{0,1\},
$$

where \(y=1\) means the participant made the correct verification decision.

For IITB-HGC, this label is naturally derived from whether the participant's judgment matches the gold claim label.

Important properties:

- the claim is known before the verification decision
- the participant searches the passage for evidence relevant to that claim
- correctness depends on whether the right evidence was found and used

## Goal

Our goal is to learn a model

$$
f_\theta : (C,P,G) \rightarrow \hat{p}
$$

that predicts

$$
\hat{p} = f_\theta(C,P,G) \in [0,1],
$$

the probability that the participant will verify the claim correctly.

We specifically want a proposal that is:

1. more task-matched than generic text-gaze fusion
2. more likely than the current multimodal baselines to beat the `Text-Only Roberta` baseline on `IITBHGC_CV`
3. scientifically interpretable, with explicit checks on whether the model actually uses the latent relevance scores in a meaningful way

## Proposed Approach

We propose a **claim-conditioned evidence coverage model**.

The key simplification is to avoid the risky parts of the earlier policy-style proposal:

- a pretrained text encoder over **claim + context**
- grounded word-level gaze features (close to the current MAG-style setup)
- a **claim-conditioned evidence scorer** over context tokens
- an **evidence coverage vector** that measures whether the participant actually inspected high-scoring evidence
- auxiliary heads that predict the **gold claim label** and the **participant judgment**

Important clarification: the scores produced by the evidence scorer are **latent, weakly supervised variables**, not directly observed ground-truth evidence labels. They should therefore be treated as exploratory model-internal weights rather than gold explanations.


## Details on Architectures

### 1. Input Construction

We explicitly separate the stimulus into:

- `claim`
- `context`

and encode:

`[CLS] Claim [SEP] Context [SEP]`

This requires only a small preprocessing change for IITB-HGC: expose the claim as a separate field instead of treating the whole stimulus as a single paragraph string.

### 2. Text Encoder

We use **RoBERTa-large** as the main text backbone.

Let:

$$
H^C = \mathrm{Enc}_{text}(C) \in \mathbb{R}^{L \times d}
$$

and

$$
H^P = \mathrm{Enc}_{text}(P \mid C) \in \mathbb{R}^{M \times d}
$$

denote the contextualized claim-token and context-token representations.

We pool the claim into a single vector:

$$
h^C = \mathrm{Pool}(H^C).
$$

### 3. Gaze Grounding

To keep the model feasible, we do **not** start from a new fixation-sequence transformer.

Instead, we reuse the kind of word-level gaze information already used by the stronger EyeBench baselines:

- dwell time
- fixation count
- regressions
- first-pass indicators
- lexical features already present in preprocessing

For each context token \(p_i\), let:

$$
z_i \in \mathbb{R}^{d_g}
$$

be its grounded gaze feature vector.

We then form a multimodal token representation:

$$
u_i = W_u [H^P_i ; z_i ; \phi_i],
$$

where \(\phi_i\) optionally includes existing lexical features such as surprisal, word length, and content-word indicators.

### 4. Claim-Conditioned Evidence Scorer

For each context token, we predict how relevant it is as evidence for the current claim:

$$
e_i = \sigma\big(w_e^\top \tanh(W_e [u_i ; h^C])\big).
$$

We convert these scores into a soft evidence distribution:

$$
a_i = \mathrm{softmax}(e_i).
$$

and compute an evidence summary:

$$
c^E = \sum_{i=1}^{M} a_i u_i.
$$

This gives us a simple differentiable evidence selector without discrete policies.

Because IITB-HGC does not provide gold word-level evidence annotations, this scorer is intentionally a **lightweight latent weighting mechanism**. Its usefulness must be tested empirically rather than assumed.

### 5. Evidence Coverage Features

The central idea of the model is that evidence selection alone is not enough. We also want to know whether the participant's gaze actually covered that evidence.

Let \(D_i, F_i, R_i, V_i\) be per-token gaze statistics such as:

- dwell time
- fixation count
- regression-related activity
- visited / not visited indicator

We compute an evidence coverage vector:

$$
v^{cov} =
\Big[
\sum_i a_i D_i,\;
\sum_i a_i F_i,\;
\sum_i a_i R_i,\;
\sum_i a_i V_i
\Big].
$$

This vector summarizes how much the participant attended to claim-relevant evidence.

### 6. Prediction Heads

The main benchmark head predicts participant correctness:

$$
\hat{p} = \sigma(\mathrm{MLP}_{corr}([h^C ; c^E ; v^{cov}])).
$$

We add two lightweight auxiliary heads:

1. **Gold claim label**

$$
\hat{y}^{gold} = \mathrm{softmax}(\mathrm{MLP}_{gold}([h^C ; c^E])).
$$

2. **Participant judgment**

$$
\hat{y}^{ann} = \mathrm{softmax}(\mathrm{MLP}_{ann}([h^C ; c^E ; v^{cov}])).
$$

This decomposition is useful because the benchmark label is really about whether the participant judgment matches the gold label.

### 7. Training Objective

We train with a small multitask loss:

$$
\mathcal{L}
=
\mathcal{L}_{corr}
+ \lambda_{gold}\mathcal{L}_{gold}
+ \lambda_{ann}\mathcal{L}_{ann}
+ \lambda_{sparse}\mathcal{L}_{sparse}.
$$

where:

$$
\mathcal{L}_{corr} = \mathrm{BCE}(y,\hat{p})
$$

$$
\mathcal{L}_{gold} = \mathrm{CE}(y^{gold},\hat{y}^{gold})
$$

$$
\mathcal{L}_{ann} = \mathrm{CE}(y^{ann},\hat{y}^{ann})
$$

and

$$
\mathcal{L}_{sparse} = -\sum_i a_i \log a_i
$$

is a mild regularizer encouraging focused evidence rather than diffuse attention.

## Tuning

We tune a small set of hyperparameters:

- learning rate
- weight decay
- dropout
- batch size
- number of epochs
- evidence hidden size
- coverage projection size
- loss weights \(\lambda_{gold}, \lambda_{ann}, \lambda_{sparse}\)

Tuning strategy:

1. start from the EyeBench `MAG-Eye` / `Text-Only Roberta` defaults
2. train the correctness head only
3. add the gold-label auxiliary head
4. add the participant-judgment head
5. tune the evidence sparsity regularizer last

This keeps the search small and makes it clear which component is actually helping.

## Explainability And Ablations

Because the importance scores are latent variables, an important part of the project is to test whether they are meaningful and whether the model actually uses them.

We therefore plan the following validation experiments:

- **text-only baseline**: RoBERTa-large without gaze or latent scoring
- **text + gaze without scorer**: add gaze features but remove the claim-conditioned scoring mechanism
- **text + scorer without gaze coverage**: keep the scorer but remove the evidence-coverage vector
- **full model**: text, gaze, scorer, and coverage together
- **shuffled-score control**: replace learned importance weights with shuffled weights at inference time
- **uniform-score control**: replace learned importance weights with a uniform distribution over context tokens
- **score-drop test**: remove the top-weighted tokens and compare the change in prediction against removing random tokens

These experiments are meant to answer two questions:

1. does the model improve because of the latent importance scores?
2. do those scores affect predictions in a systematic way, or are they effectively unused?

## Biases And Limitations

The model has an important limitation: the importance scores are latent and weakly supervised, so they may reflect dataset artifacts or annotation biases rather than genuine claim-relevant evidence.

We therefore explicitly do **not** interpret the scores as ground-truth explanations. Instead, we treat them as an exploratory mechanism whose behavior must be checked through ablations and robustness tests.

Potential sources of bias to monitor include:

- over-reliance on lexical shortcuts in the claim or context
- spurious correlations between gaze activity and participant correctness
- unstable importance weights caused by the small size of IITB-HGC
- reader-specific effects that make the latent scores harder to interpret

The scientific goal is therefore not only to improve prediction, but also to determine whether this weakly supervised mechanism is informative at all on this benchmark.

## Augmentation

Because IITB-HGC is small, we treat augmentation as **regularization**, not as a way to pretend we have many more independent training examples.

We therefore restrict augmentation to light **gaze-side** perturbations that preserve the claim, the context, and the supervision:

- **gaze feature dropout**: randomly mask a small fraction of token-level gaze features during training
- **small feature noise**: add mild Gaussian noise to continuous gaze features such as dwell time or fixation-count style signals
- **token-level gaze masking**: hide gaze on a small subset of visited tokens so the model cannot overfit to a few brittle positions
- **consistency regularization**: create two noisy gaze views of the same trial and encourage similar predictions

We explicitly avoid heavy text augmentation. This keeps augmentation aligned with the benchmark and reduces overfitting without changing the semantic evidence structure of the task.

## Metrics

We use the official EyeBench metrics.

### Primary Metric

Average AUROC across the three benchmark regimes:

$$
\mathrm{AvgAUROC} =
\frac{
\mathrm{AUROC}_{UR} + \mathrm{AUROC}_{UT} + \mathrm{AUROC}_{URT}
}{3}.
$$

### Secondary Metric

Average Balanced Accuracy across the same three regimes.

## Baselines

We compare against the official EyeBench baselines for `IITBHGC_CV`, with the primary target being:

- **Text-Only Roberta**

and the main multimodal comparisons:

- **MAG-Eye**
- **RoBERTEye-W**
- **RoBERTEye-F**
- **PostFusion-Eye**

## Summary

Current multimodal models do not clearly beat the text-only baseline on IITB-HGC, so the most promising path is not more generic fusion, but a simpler model that asks whether the reader looked at the right claim-relevant evidence.
