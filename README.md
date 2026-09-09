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
