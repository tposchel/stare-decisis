# Event schema

Every record: `seq`, `event_type`, `tx_time` (when recorded), `valid_time` (when true in the world, caller-supplied, may be null), `payload`, `prev_hash`, `hash`.  `hash = sha256(canonical_json({seq, event_type, tx_time, valid_time, payload, prev_hash}))`.  The first record's `prev_hash` is 64 zeros.

## node_asserted

| field | notes |
|---|---|
| node_id | unique |
| node_type | observation, llm_hypothesis, latent_estimate, rule_fact, decision_outcome |
| value, value_hash | value must be JSON-serializable |
| reasoning_form | deduction, induction, abduction, analogy, authority |
| source | object with at least `id`.  For decision_outcome: `decision:<id>`, plus decision_id, eval_id, selected_option (the choice the node assumes) and choice_rule |
| llm_provenance | model, prompt_hash, output_hash, temperature, seed; required for llm_hypothesis |
| segments | segments the node was routed to |

## node_retracted

Soft delete.  The asserted record and its value are untouched.

| field | notes |
|---|---|
| node_id | |
| reason_code | source_error, extraction_error, model_hallucination, superseded, adversarial |
| reason_text, discovered_by, discovery_ref | cascaded supersessions use `engine:cascade` and the upstream eval_id (or `eval_id/override@seq`) |
| affected_segments | every `segment@version` that consumed the node, from the dependency edges |
| pending_segments | segments holding the node only in staging, never versioned with it |

## node_expired

Scheduled housekeeping, not a correction.  The node stays live for other segments and is not excluded by as-corrected replays.

| field | notes |
|---|---|
| node_id, as_of | |
| aged_from | valid_time, or tx_time when no valid_time was given |
| segments, freshness_seconds | segments whose window the node outlived |
| affected_segments | segment versions that consumed it |

## segment_version

One per flush.  Immutable.

| field | notes |
|---|---|
| segment_id, version | ref is `segment_id@version` |
| consumed_nodes | the dependency edges |
| inserted_nodes, retracted_nodes, expired_nodes | the delta this version applied |
| model_spec, model_spec_hash | replay refuses to run if the live model's hash differs |
| sufficient_stats, sufficient_stats_hash | |
| summary | the posterior summary decisions consume |
| prior_ref | where the prior came from: an authority claim |
| recompute_method | incremental, subtraction, replay |
| trigger | assertion, retraction, scheduled (expiry) |

## decision_linked / decision_unlinked

Linked: every segment on the decision's path has (or has staged) at least one live node.  Unlinked: an evaluated decision's evidence drained away; records the standing choice that is now unsupported.

## decision_evaluated

| field | notes |
|---|---|
| decision_id, eval_id | eval_id is `decision_id#n` |
| agenda_group, salience | |
| inputs | `segment@version` only, never live state |
| beliefs | summaries keyed by input ref |
| options | option_id, origin_nodes, expected_utility, blocked |
| withdrawn_options | options whose every asserted origin node was retracted |
| pending_options | options none of whose origin nodes has been asserted yet |
| utility_fn_ref, utility_fn_author | |
| constraints | per rule and option: rule ref, author, matched_nodes, pass, fail or mandate |
| selected_option, choice_rule | max_eu, policy_mandate, human_override |
| system_recommendation | what the system would choose while a human override stands |
| actor | system or the overriding human |
| trigger | initial, voi_exceeded, standing_choice_invalid, mandate, utility_changed, option_origin_retracted, option_origin_asserted, all_options_withdrawn |
| voi_estimate | staleness regret that triggered this evaluation, when applicable |
| reversal_cost | reversal cost the regret had to clear, when the trigger was voi_exceeded |
| direct_reversal_cost, unwind_cost | the decision's declared costs at evaluation time |
| supersedes | previous eval_id |
| flags | human_override_stands, human_override_revoked_by_policy, mandate_conflicts_with_block, all_options_withdrawn |

## reevaluation_skipped

| field | notes |
|---|---|
| decision_id, standing_eval_id, standing_option | |
| trigger_segments | segment versions that changed |
| voi_method | staleness_regret |
| voi_estimate, voi_threshold | regret and the decision's tolerance |
| could_flip, would_select | whether the change would have altered the choice, and to what |
| reversal_cost | direct, exposure, total, dependents (decision: depth) at the time of the check |
| hurdle, lock_in_margin | voi_threshold + reversal_cost.total; hurdle − regret |
| status | sound (regret 0), within_tolerance (≤ voi_threshold), impaired (above tolerance, below hurdle: locked in) |

## rule_fired

rule ref, author, action (block or mandate), agenda_group, decision_id, eval_id, option_id, matched_nodes.

## human_override

decision_id, eval_id, actor, original_option, chosen_option, reason_code, reason_text.
