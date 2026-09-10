# Causal direction GPU entry points

`extract_causal_directions.py` expects accepted JSONL pairs containing at least:

```text
pair_id, objective, positive_text, negative_text, split, group
```

`split` must include distinct fit and validation groups. The default fit name is
`train`; use `--fit-split fit` when appropriate. Multiple atomic grouping fields
can be supplied as a comma-separated list, for example
`--group-fields facet,domain`.

When every row has `mechanism_family`, the default `--family-field auto`
performs the RH protocol's exact hierarchy: equal mean over pairs within task
group, task groups within mechanism family, and mechanism families. Grouped
bootstrap resampling is then done within each family. Frozen
`hierarchical_weight` and `fit_weight` fields, when present, are checked against
that hierarchy before any model is loaded. Use `--family-field none` only for a
dataset whose protocol has no family level.

Example extraction command:

```bash
python3 scripts/extract_causal_directions.py \
  --pairs data/causal-directions/rh/pilot_pairs.jsonl \
  --output-dir data/causal-directions/rh/pilot_extraction \
  --direction-name d_RH \
  --base-model longtermrisk/OLMo-3-7B-school-of-reward-hacks-sft \
  --base-revision 438a3f85d1d19d9cf5a4bd8e341e99ae1dcb80b3 \
  --fit-split train \
  --validation-split validation \
  --group-fields group \
  --family-field mechanism_family \
  --resume
```

The extractor writes response-token pair deltas, layer-wise directions, grouped
bootstrap/split-half arrays, a validation-only layer selection, and manifests.
Complete test and quarantine splits may be present in the input but are never
read by the fitting or selection calculations.

Before generation, copy `pilot.example.json`, pin the full model revision, and
update artifact paths. Run only the desired scope arms; a baseline must always
be included:

```bash
python scripts/run_causal_direction_ablation.py \
  --config configs/causal-directions/pilot.json \
  --prompts data/causal-directions/causal-calibration-prompts.jsonl \
  --output-dir data/causal-directions/causal-calibration \
  --conditions baseline rh_single synem_single \
    rh_matched_nuisance em_matched_nuisance \
  --resume
```

The projection is applied to the final prompt state that predicts the first
assistant token and then to every newly decoded state through EOS. Earlier
system/user positions are untouched. Generated text is saved but never
executed or automatically assigned behavioral labels.

That default is represented by `"token_scope": "generation_only"`. Set
`"token_scope": "all_positions"` to project every prompt-prefill position as
well as every generated position.

Generation uses `"use_cache": true` by default. Set it explicitly to `false`
for a cached-versus-uncached deterministic smoke test. During uncached decoding,
the full prefix is recomputed and the hook reapplies the same logical token
scope. Per-layer telemetry separately records `prefill_hooked_positions` and
`decode_hooked_positions`; these count hook applications, so recomputed prefix
positions are counted again in uncached mode.

Scope values are:

- `single`: the validation-selected layer;
- `band3`: a contiguous three-layer window around it;
- `qualified`: every layer passing frozen validation/stability thresholds;
- `all32`: explicit high-risk stress test over all 32 blocks.

Uncentered projection is the primary operation:

```text
h_new = h - alpha * (h dot unit_direction) * unit_direction
```

At `alpha=1`, the selected component is zero. `alpha>1` reverses it and must not
be described as stronger ablation. Artifacts also contain a fit-control mean;
set `projection` to `control_mean` only for the predeclared affine sensitivity
condition.

Normally target layer `i` uses direction row `i`. To apply one source-layer
direction at every layer in a wider scope, add for example
`"direction_source_layer": 10`. A deterministic norm-matched random control
uses the same targets and source mapping:

```json
{
  "name": "layer10_random_all32",
  "direction": "d_synEM",
  "scope": "all32",
  "direction_source_layer": 10,
  "direction_variant": "norm_matched_random",
  "random_seed": 42,
  "token_scope": "all_positions",
  "alpha": 1.0,
  "projection": "uncentered"
}
```

The random vector is deterministically derived from the sign-canonicalized
source-layer axis, source-layer index, artifact hash, and `random_seed`, then
rescaled to the source vector's norm. Omitting `direction_variant` retains the
learned direction.
