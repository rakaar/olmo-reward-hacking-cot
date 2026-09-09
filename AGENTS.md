## Project context

Main related paper: **Wilhelm et al., “Monitoring Emergent Reward Hacking During Generation via Internal Activations”**
https://arxiv.org/pdf/2603.04069

We want to test this idea(not same methods, but in spirit that CoT activations can inform about reward hacking) using the **UK AISI OLMo-7B reward-hacking model organisms**(https://www.alignmentforum.org/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in).

## Goal

Test whether a decoder trained on **chain-of-thought activations from a relatively faithful reward-hacking model** can generalize to a **less-faithful model**, where the output still reward-hacks but the CoT is less explicit about it.

Core question:

> Can internal CoT activations reveal an upcoming reward hack even when the textual CoT no longer honestly describes the hacking strategy?

## Plan

### 1. Generate a faithful-model dataset

Use the released AISI **OLMo-7B β=0 reward-hacking model** and their CodeContests reward-hacking environment.

Do **not** immediately use the final checkpoint. Find an intermediate RL checkpoint where the reward-hack rate is neither ~0% nor ~100%, ideally around 30–70%.

This is important because we want many examples of both:

- reward-hack outputs
- non-reward-hack outputs

Pilot a few hundred generations first to choose the checkpoint.

Then generate roughly 1,000–2,000 rollouts, preferably using multiple samples per CodeContests problem.

Save:

- problem / prompt
- complete `<thinking>` CoT
- final response/code
- hack type
- reward-hack / non-reward-hack label
- `cot_mentions_hack`
- checkpoint and generation metadata

Use AISI's released hack-detection logic rather than inventing a new labeler.

### 2. Extract CoT activations

Replay each saved prompt + generated CoT through the exact same model checkpoint and record residual-stream activations corresponding **only to the `<thinking>` tokens**.

Generation and activation extraction can therefore be separate stages.

Initially collect activations across all layers or a reasonably broad layer range.

Split train/validation/test **by `problem_id` before fitting anything**. Do not randomly split individual tokens or rollouts from the same coding problem across train and test.

### 3. Exploratory activation analysis

Before building a classifier, inspect whether hack and non-hack CoTs differ internally.

Examples:

- layer-wise mean-difference directions
- PCA / simple visualization
- cosine similarities
- when during the CoT separation appears
- whether separation is concentrated in particular layers or token positions

Keep this exploratory; avoid over-engineering.

### 4. Train a reward-hack decoder

Train a simple linear decoder:

\[
\text{CoT activation} \rightarrow
\{\text{reward hack},\text{non-reward hack}\}
\]

Start simple, e.g. logistic regression on residual-stream activations.

Evaluate only on held-out CodeContests problems.

Report AUROC/AUPRC and layer-wise performance rather than relying only on a fixed 0.5 threshold.

The first objective is simply:

> Can CoT activations from the relatively faithful β=0 model predict whether the subsequent output will reward-hack?

For the primary decoder target, interpret this as whether the response
**attempts/contains a released reward-hack mechanism**, regardless of whether
malformed code or another implementation error prevents the hack from passing.
Operationally, use `hack_attempted = 1` when AISI's released static detector
finds at least one enabled hack signature. Keep `reward_hacked` (normal tests
passed and a hack signature was detected), `passed`, and `actually_solved` as
secondary outcome and quality-control fields.

### 5. Test transfer to the less-faithful model

Load AISI's **OLMo-7B β=0.02** reward-hacking model.

Generate a fresh dataset using the same environment and procedure.

Extract CoT activations in exactly the same way.

**Freeze the decoder trained on β=0. Do not retrain it.**

Evaluate its performance on:

1. all β=0.02 rollouts
2. reward-hack vs non-reward-hack
3. especially reward-hacking examples where `cot_mentions_hack = 0`

The key result is whether:

\[
\text{decoder trained on faithful CoT}
\rightarrow
\text{still detects hacks in less-faithful CoT}
\]

### Important details not to forget

- Use a **single intermediate checkpoint** for hack/non-hack training if possible; otherwise the decoder may learn checkpoint/training-step differences.
- Split by **problem**, not by token.
- Generate several completions per problem to obtain behavioural variation.
- Keep generation and activation extraction separate.
- Start with raw residual activations; do not introduce SAEs unless needed later.
- Preserve textual CoT labels so we can compare activation monitoring with simple text/regex monitoring.
- Use hack attempted versus no hack as the primary classifier split; do not
  condition the primary label on whether the hack happened to pass pytest.
- The interesting result is **cross-model transfer**, especially on `reward_hacked = 1` and `cot_mentions_hack = 0`.
