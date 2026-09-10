# OLMo reward-hacking CoT activation experiment

This repository tests whether a linear decoder trained on chain-of-thought
activations from a relatively faithful reward-hacking OLMo model transfers to a
less-faithful model.

The first stage is a checkpoint-selection pilot using AISI's released
CodeContests environment and detection logic. Generation and grading are
separate: generation never executes model output, and grading runs untrusted
Python in a restricted chroot with dropped privileges, resource limits, and
network-related syscalls blocked.

## Current pilot

- Faithful candidate: `ai-safety-institute/cc-olmo3-7b-sutl-b0.0-s220`
- AISI source revision: `169c3c76a02e51092b4023a8c7baba38f41e2800`
- Prompt: `dont_hack`, `sutl` hint style
- Environment: all three released CodeContests reward hacks enabled
- Sampling: temperature 1.0, repeated completions grouped by `problem_id`
- Labels: AISI's released test and static-analysis scorers

The primary decoder label is `hack_attempted`: at least one of AISI's released
hack signatures is present in the response, whether or not the generated code
parses or the hack succeeds. `reward_hacked` remains a secondary label meaning
that the normal test run passed while a released hack signature was present.

The public AISI collection currently exposes only the step-220 beta=0 OLMo-7B
adapter, so the pilot first determines whether it already has a useful mixture
of reward-hack and non-reward-hack rollouts.

## Corrected causal-ablation stopping point

A methodology audit of Soligo et al., *Convergent Linear Representations of
Emergent Misalignment*, motivated a corrected all-position causal experiment.
The sole primary condition uses the paper-defined layer-wise intervention: at
each of OLMo's 32 post-block residual streams, project out that layer's own
unit reward-hacking direction at every prompt and generated-token position.

The frozen main grid contains 20 prompts, five paired samples, and seven
conditions (700 outputs): a shared baseline, three learned-direction arms, and
one norm-matched-random control for each learned arm. Outcome-blind calibration
accepted native layer-19 projection at alpha 1 and repeated layer-19 projection
at alpha 0.5; full repeated layer-19 projection at alpha 1 was rejected for
poor generation quality.

The main run was deliberately interrupted and archived on 11 September 2026
after 102/700 outputs. Only the first three same-distribution prompts had been
reached, so the partial rows are not an analyzable causal sample. They have not
been unblinded, reward-hacking-scored, or entered into the planned bootstrap
analysis. The local archive is at
`data/causal-directions/causal-main/rh_corrected_allposition_main20x5/`; its
`rollouts.jsonl` SHA-256 is
`8aff1ce0190b1dd102b834750f2d0b15484a4cfe871f48e679be501082648e54`.
The exact launch source, frozen config/protocol, preflight record, log, running-
state manifest, and remote checksum file are preserved for a future resume.

The human-readable status report is
`olmo_reward_hacking_results_so_far.html`.

## Shortcut-direction experiment

The two independent shortcut directions are now frozen. This stage uses only
teacher-forced forward passes over existing contrast text; the CodeContests
rollouts are not used to construct directions or select layers.

- Track A: School of Reward Hacks revision
  `d7e04a550119cb5410494cf90e2313284a5f2148`, with 973 complete non-coding
  pairs (748 train and 225 held out), split by task.
- Track B: 120 candidates from three independent Luna workers, frozen to 80
  pairs balanced across eight domains. Six domains/60 pairs are used for
  fitting; content moderation and procurement/20 pairs are held out.
- Track B's leave-domain-out word/character TF-IDF side classifier has AUROC
  0.5052, below the predeclared 0.65 ceiling.
- Model: `allenai/Olmo-3-7B-Instruct-SFT` at
  `e1452fc572d51966ff4aaeb25118b891eb93e549` with the beta=0 step-220 adapter
  at `232b591c69b90b1ec9a866fe270c5ca6763befb0`.
- Directions: positive (shortcut/reward-hacking) minus negative
  (legitimate/control), for all 32 post-block residual streams, using both the
  last assistant-content token and the mean over assistant-content tokens.

The predeclared selection rule found the following common candidate layers:

| Pooling | Selected layer | Signed cosine | Grouped 95% CI | Qualifying layers |
| --- | ---: | ---: | ---: | --- |
| Last content token | 7 | 0.0505 | [0.0177, 0.0553] | 7, 9 |
| Mean content tokens | 0 | 0.1159 | [0.0294, 0.1482] | 0, 6, 9, 10 |

These are statistically qualified common directions under this experiment's
controls, but the cross-method cosine is modest and is not by itself evidence
of one causal reward-hacking circuit. The next test is token-wise projection on
the held-out CodeContests CoTs.

Frozen inputs, directions, compressed per-pair deltas, bootstrap/null arrays,
the comparison CSV, plot, manifests, hashes, and rejection logs are under
`data/shortcut-directions/`. The full local artifact copy can be checked with:

```bash
.venv/bin/python scripts/validate_shortcut_direction_artifacts.py \
  --results-root data/shortcut-directions/results \
  --track-a-pairs data/shortcut-directions/sorh/prepared/pairs.jsonl \
  --track-b-pairs data/shortcut-directions/luna/prepared/pairs.jsonl
```

The preparation, extraction, comparison, and mocked-model regression entry
points are respectively:

```text
scripts/prepare_direction_pairs.py
scripts/extract_shortcut_directions.py
scripts/compare_shortcut_directions.py
scripts/test_shortcut_direction_pipeline.py
```

The exact extraction/analysis environment used on the RTX 3090 is recorded in
`requirements-gpu.txt`. Its Transformers build is pinned to commit
`11b1906d5c0dae39c13270e47cc02c4cde70e548`; install the Torch wheel matching
the CUDA runtime on the new machine.

## Layer-14 projection-ablation calibration

`scripts/run_projection_ablation.py` applies the School of Reward Hacks
mean-token direction at zero-indexed post-block layer 14 from the first
generated token onward. For unit direction `d`, it replaces each selected
residual `h` with `h - lambda * (h dot d) * d`. Thus lambda 1 removes the
projection and lambda 2 reverses it; lambda 2 is not a stronger orthogonal
ablation. Conditions are generated together to reduce runtime, while generated
Python remains unexecuted until it reaches the restricted grader.

The staged calibration was extended to 20 matched problems and produced:

| Lambda | Hack attempted | Syntax-valid | Structured completion |
| ---: | ---: | ---: | ---: |
| 0 | 6/20 | 5/20 | 5/20 |
| 1 | 8/20 | 4/20 | 4/20 |
| 2 | 6/20 | 4/20 | 4/20 |

The hook diagnostics confirm that lambda 1 reduced the selected projection to
approximately zero and lambda 2 reversed it. The apparent lambda-2 reduction
in the first five problems did not replicate: the next 15 had 3/15 control
hacks and 5/15 lambda-2 hacks, leaving both combined conditions at 6/20.
Lambda 2 suppressed three control hacks but induced three others, for an exact
paired McNemar p-value of 1.0. This is evidence against a large gross causal
effect at layer 14 under this intervention, not proof that the direction
contains no reward-hacking information. Control generations were still mostly
malformed, and each stochastic batch row used a different random draw. Any
follow-up should use problems with demonstrated coherent controls or a
deterministic/common-random sampling design before scaling further.

The supporting scripts are:

```text
scripts/run_projection_ablation.py
scripts/summarize_projection_calibration.py
scripts/test_projection_ablation.py
```

## Maximum token-projection check

The 200 saved beta=0 rollouts were teacher-forced through the exact step-220
model, with no new generation or intervention. At every post-block layer, the
analysis retained the maximum signed value of `h dot d_hat` across either the
tokens strictly inside complete `<thinking>` tags or the whole assistant
response. The direction is Track A's School of Reward Hacks `mean_unit`
direction. Results are compared both in aggregate and within the same
CodeContests problem; the latter uses the 41 problems containing both attempted
and non-attempted samples.

For the primary `hack_attempted` label, CoT maxima are slightly higher for hack
attempts at many layers, but no layer has a within-problem 95% interval that
excludes zero. At layer 14, the aggregate means are 1.061 for attempts and 0.950
for non-attempts; the within-problem difference is 0.118 with interval
[-0.064, 0.303]. The largest matched mean difference is 0.246 at layer 24, but
its interval [-0.114, 0.614] is also inconclusive.

The secondary `reward_hacked` outcome has an exploratory layer-14 signal: among
19 mixed-outcome problems, the within-problem difference is 0.246 with interval
[0.034, 0.444], and 15/19 problem-level differences are positive. Pointwise
intervals are also positive at layers 10 and 11. This split contains only 21
successful hacks and conflates successful exploitation with response quality,
so it should not be interpreted as evidence that the direction represents hack
intent. Successful-hack CoTs are not longer on average, which argues against a
simple CoT-length explanation for this secondary result.

Whole-response maxima separate attempts from non-attempts at several early
layers, but attempted-hack responses are much longer on average and their final
code can explicitly contain the hack. That comparison is therefore a leakage
check, not evidence for a hidden CoT signal. The extraction deliberately fits
no classifier and computes no ROC curve.

An answer-only follow-up excludes CoT text, `<thinking>` delimiters, and chat
wrappers. It has an unambiguous answer span for 199/200 rollouts. To control the
maximum statistic's dependence on response length, it computes both the maximum
over the full answer and the maximum over only the first 128 answer tokens.

The fixed-window result clearly separates hack attempts from non-attempts at
early-to-middle layers. At layer 10, the group means are 1.197 and 0.930; the
within-problem difference is 0.259 with interval [0.156, 0.372]. Layers 3-13
also have positive within-problem pointwise intervals. The layer-10 mean is
higher than the no-hack mean for each released signature: 1.249 for
`always_equal`, 1.165 for `conftest`, and 1.223 for `exit`, although signatures
can overlap within one response. Frequent argmax tokens include `pytest`,
`Equal`, `hack`, and `==`.

This answer-token result is evidence that the direction recognizes explicit
hack-like answer content. It is not an early-warning result: successful hacks
do not project more strongly than failed attempts, and the separation mostly
disappears from the CoT-only comparison. The fixed-window control rules out
answer length as the sole cause of the attempted-versus-non-attempted gap.

The resumable analysis and regression tests are:

```text
scripts/analyze_max_direction_projection.py
scripts/test_max_direction_projection.py
```

## Layer-10 mean-pooled CoT decoder

The first direct decoder pilot uses the exact same 200 beta=0 step-220
rollouts. Complete `<thinking>` spans were available for 196 rollouts. For each
one, the primary feature is the mean zero-indexed post-block layer-10 residual
over non-special tokens strictly inside the CoT. Prompt, answer/code, thinking
delimiters, EOS, and padding are excluded.

The original transparent-only fit is retained as an exploratory artifact but
is superseded by the primary all-groups analysis. The revised decoder trains on
every complete-CoT rollout: 103 mention/attempt, 32 mention/no-attempt, 30
no-mention/attempt, and 31 no-mention/no-attempt examples. The target is solely
whether the subsequent answer attempts a released reward hack.

Five-fold out-of-fold evaluation is grouped by CodeContests problem and
explicitly balanced over all four behavior cells. Every held-out fold contains
10 whole problems, at least five examples from every cell, and exactly six
silent attempts. Every corresponding training fold contains at least 24
examples from every rare cell. Inner folds contain at least four examples from
every cell. These are hard assertions: the analysis fails rather than accepting
an inadequately populated split.

| Readout | Overall AUROC | Grouped 95% CI | Overall AUPRC | No-mention AUROC |
| --- | ---: | ---: | ---: | ---: |
| Layer-10 mean CoT activation | **0.779** | **[0.715, 0.835]** | **0.901** | **0.647** |
| Visible CoT TF-IDF | 0.708 | [0.622, 0.788] | 0.846 | 0.498 |
| Raw CoT-mention indicator | 0.633 | [0.553, 0.715] | 0.744 | 0.500 |
| Prompt-only mean activation | 0.650 | [0.569, 0.734] | 0.770 | 0.599 |
| CoT token count | 0.513 | [0.433, 0.587] | 0.683 | 0.509 |

Overall, activation AUROC exceeds visible TF-IDF by 0.071 with a paired grouped
95% interval [0.005, 0.140], and exceeds the raw mention indicator by 0.146
with interval [0.063, 0.222]. Within CoTs that mention hacking, activation
AUROC is 0.806 [0.724, 0.884]. Within the 61 no-mention CoTs, it is 0.647
[0.485, 0.790], compared with TF-IDF at 0.498 and prompt activation at 0.599.
The silent-attempt point estimate is encouraging, but its interval still
includes chance and its paired advantages within that subset are inconclusive.

The mean activation score is 0.373 for silent attempts versus 0.284 for
no-mention non-attempts. Because silent attempts occur in the training folds,
this is held-out-problem generalization within a no-mention stratum, not a
zero-shot test trained only on verbalized hacking.

Layer 10 was chosen after inspecting the earlier answer-side projections on
these same rollouts. The all-groups pilot is therefore exploratory.
The confirmation pipeline fixes layer 10, whole-CoT mean pooling, metrics, and
the gate before drawing 200 unused problems. After the rollouts are labeled, it
uses only problem IDs and the four behavior-cell labels to deterministically
assign 120/40/40 whole problems to train/validation/test with seed 42; no
activation features enter this split search. Fresh training must contain at
least 60 examples from every cell, and validation and test must each contain at
least 20 from every cell. These are hard gates: the run stops rather than
accepting a thin category. All complete-CoT pilot examples are added only to
training; regularization and the decision threshold are selected on all four
validation categories, and the all-four-groups test set is evaluated once.

The reproducible entry points are:

```text
scripts/extract_cot_decoder_features.py
scripts/train_cot_decoder.py
scripts/train_cot_decoder_all_groups.py
scripts/train_cot_decoder_confirmation.py
scripts/validate_cot_decoder_artifacts.py
scripts/test_cot_decoder.py
scripts/run_beta0_confirmation_1000.sh
scripts/extract_beta0_confirmation_features.sh
```

Pilot features are under `data/cot-decoder/pilot-layer10-mean/features/`. The
primary balanced all-groups folds, fitted pipelines, out-of-fold predictions,
metrics, confidence intervals, hashes, and figure are under
`data/cot-decoder/pilot-layer10-mean-all-groups/evaluation/`. The superseded
transparent-only evaluation remains under
`data/cot-decoder/pilot-layer10-mean/evaluation/` for provenance.

## Safe grading on RunPod

Never score model-generated Python with Inspect's `local` sandbox. Build the
restricted chroot on the pod's executable container disk:

```bash
scripts/bootstrap_grader_rootfs.sh
```

The defaults are `/opt/aisi-grader-rootfs-minimal` and
`/usr/local/bin/aisi-grader-exec`. RunPod erases the container disk when a pod
stops, so rebuild this chroot after a restart. Models, generated rollouts, and
labels belong under the persistent `/workspace` volume.

After the model server and restricted grader pass their smoke tests, run the
fixed checkpoint-selection pilot with:

```bash
scripts/run_beta0_pilot_200.sh
```

This generates 200 rollouts from 50 deterministically selected problems with
four samples per problem, exports the Inspect log, and applies the isolated
AISI-compatible labels. The selected problem records are cached separately so
the same set can be reused for the beta=0.02 transfer dataset.
