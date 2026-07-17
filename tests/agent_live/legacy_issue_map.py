#!/usr/bin/env python3
"""Auditable migration map for the retired 2026-07-10/11 Agent documents.

The dated repair plan and known-issues register were deleted because they
describe retired owners and implementation files.  This module preserves the
issue identities without preserving those documents as active instructions.

Disposition rules are intentionally strict:

* ``guarded`` requires a named test that exists in the current tree.  It means
  the requirement has a current regression guard, not that the guard was run
  successfully for this worktree.
* ``superseded`` requires a current architecture contract and its test.
* ``open`` is mandatory when neither form of evidence exists.

Run this file directly to validate the map and print a JSON report.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.workflows.group_registry import GROUP_OWNER


ALL_CURRENT_GROUPS = tuple(GROUP_OWNER)


@dataclass(frozen=True)
class LegacyItem:
    legacy_id: str
    summary: str
    groups: tuple[str, ...]
    owner: str
    disposition: str
    evidence: tuple[str, ...] = ()
    note: str = ""


def _item(
    legacy_id: str,
    summary: str,
    groups: str | Iterable[str],
    owner: str,
    disposition: str,
    *evidence: str,
    note: str = "",
) -> LegacyItem:
    normalized_groups = (groups,) if isinstance(groups, str) else tuple(groups)
    return LegacyItem(legacy_id, summary, normalized_groups, owner, disposition, tuple(evidence), note)


# Every defect row from section A of the retired known-issues register.  An old
# claim of "verified live" is not evidence for the rebuilt architecture; the
# current test name must still exist or the item remains open.
KNOWN_ISSUES_A: tuple[LegacyItem, ...] = (
    _item("A.1", "Numbered/label answer to yes-no confirmation", "opening", "coordinator", "guarded", "test_numbered_answer_to_yes_no_confirm_applies_without_llm"),
    _item("A.2", "Mode switch and retained-chain disclosure", "target_mode", "chain_rpc", "guarded", "test_target_mode_change_confirmation_names_preserved_chain"),
    _item("A.3", "Negated protocol must not select a family", "chain_identity", "chain_rpc", "guarded", "test_adapter_family_hint_ignores_negated_protocol_mentions"),
    _item("A.4", "Compound consultation must preserve both questions", "opening", "orientation", "guarded", "test_combined_modes_and_preflight_smoke_question_answers_both"),
    _item("A.5", "Out-of-range menu answer keeps the contract", "opening", "coordinator", "guarded", "test_out_of_range_numbered_answer_keeps_question_without_llm"),
    _item("A.6", "Endpoint probe expands EVM sample placeholders", "endpoint_process", "chain_rpc", "guarded", "test_endpoint_probe_resolves_env_placeholder_sample_address"),
    _item("A.7", "sync-observe can leave configuration and execute", "sync_observe", "sync_observe", "guarded", "test_sync_observe_group_completes_instead_of_looping"),
    _item("A.8", "No-op chain change preserves pending work", "chain_identity", "chain_rpc", "guarded", "test_noop_same_chain_change_preserves_active_question"),
    _item("A.9", "Multiline report paste routes to analysis", "error_evidence_analysis", "analysis", "guarded", "test_multiline_analysis_reaches_typed_planner_before_evidence_domain", "test_multiline_log_paste_is_stored_as_one_complete_evidence_record"),
    _item("A.10", "Incomplete real-node config cannot reach preflight", "preflight_smoke_execution", "execution", "guarded", "test_preflight_group_not_offered_when_config_incomplete"),
    _item("A.11", "QPS values reject zero, negative and inverted profiles", "qps_profile", "performance", "guarded", "test_qps_override_rejects_invalid_values"),
    _item("A.12", "Mixed weights reject unknown methods", "workload_rpc", "chain_rpc", "guarded", "test_custom_rpc_weights_reject_unknown_method_but_allow_template_default"),
    _item("A.13", "Missing job logs return a typed error", "job_monitoring", "execution", "guarded", "test_logs_command_reports_clean_error_for_missing_job"),
    _item("A.14", "Startup dependency consent remains actionable", "opening", "terminal", "guarded", "test_startup_preserves_dependency_offer_so_yes_installs"),
    _item("A.15", "Workload consultation does not select a chain", "workload_rpc", "orientation", "guarded", "test_chain_workload_question_answered_not_selected"),
    _item("A.16", "Recommendation acceptance advances the workflow", "opening", "orientation", "guarded", "test_accept_recommendation_starts_recommended_setup"),
    _item("A.17", "Environment readiness uses startup discovery", "provider_deployment", "orientation", "guarded", "test_environment_readiness_question_answered_from_discovery"),
    _item("A.18", "Negated family enumeration reaches Case 3", "chain_identity", "chain_rpc", "guarded", "test_adapter_family_hint_ignores_negated_protocol_mentions"),
    _item("A.19", "Consultations are not harness phrase-table routing", ("opening", "provider_deployment", "workload_rpc"), "orientation", "guarded", "test_chain_workload_question_answered_not_selected", "test_environment_readiness_question_answered_from_discovery"),
    _item("A.20", "Cross-turn QPS invariant", "qps_profile", "performance", "guarded", "test_qps_override_rejects_invalid_values"),
    _item("A.21", "Empty custom-method allow-list is not allow-all", "workload_rpc", "chain_rpc", "guarded", "test_custom_rpc_weights_reject_unknown_method_but_allow_template_default"),
    _item("A.22", "Runtime field questions use field contracts", "opening", "orientation", "guarded", "test_config_field_explanation_answers_from_runtime_contract"),
    _item("A.23", "Detected provider metadata stores detected values", "provider_deployment", "environment", "guarded", "test_provider_metadata_confirm_stores_detected_value_not_literal"),
    _item("A.24", "Endpoint health probe is family/chain specific", "endpoint_process", "chain_rpc", "guarded", "test_health_probe_method_is_chain_specific_not_evm_only"),
    _item("A.25", "Numbered confirm-or-value answer", "endpoint_process", "coordinator", "guarded", "test_numbered_answer_to_confirm_or_value_question_applies"),
    _item("A.26", "Mixed defaults satisfy workload confirmation", "workload_rpc", "chain_rpc", "guarded", "test_mixed_default_workload_confirms_weights_for_preflight"),
    _item("A.27", "Remote sync-observe waives local process identity", "sync_observe", "sync_observe", "guarded", "test_sync_observe_endpoint_only_waives_node_process_identity"),
    _item("A.28", "Curl/JSON paste extracts RPC method and params", "endpoint_process", "chain_rpc", "guarded", "test_custom_rpc_method_extracts_from_pasted_content_with_url"),
    _item("A.29", "Execution preparation arguments stay in contract", "preflight_smoke_execution", "execution", "guarded", "test_prepare_kwargs_are_all_accepted_by_prepare_benchmark_run"),
    _item("A.30", "Latest-job analysis refreshes persisted state", "report_artifact_analysis", "analysis", "guarded", "test_analyze_latest_job_uses_disk_latest_not_stale_hint"),
    _item("A.31", "Field explanation covers sync-observe fields", "sync_observe", "orientation", "guarded", "test_config_field_explanation_covers_sync_observe_only_fields"),
    _item("A.32", "sync-observe duration validation", "sync_observe", "sync_observe", "guarded", "test_sync_observe_duration_rejects_invalid_values"),
    _item("A.33", "Exporter mode discloses scrape endpoint", "observability", "performance", "guarded", "test_exporter_observability_mode_discloses_scrape_port"),
    _item("A.34", "sync-observe settings remain correctable", "sync_observe", "sync_observe", "guarded", "test_sync_observe_stop_condition_and_duration_correctable_via_nl"),
    _item("A.35", "Yes/no plus trailing intent is one typed turn", "opening", "coordinator", "guarded", "test_declined_yes_no_with_trailing_intent_uses_typed_pending_action"),
    _item("A.36", "QPS overrides validate against mode defaults", "qps_profile", "performance", "guarded", "test_qps_override_validates_against_mode_defaults_not_just_overrides"),
    _item("A.37", "Disk/network manual values are positive", ("ledger_disk", "accounts_disk", "network"), "environment", "guarded", "test_disk_and_network_numeric_fields_reject_negative_values"),
    _item("A.38", "Pasted disk/network values preserve sign", ("ledger_disk", "accounts_disk", "network"), "environment", "guarded", "test_paste_inferred_disk_values_preserve_sign_and_reject_negative"),
    _item("A.39", "Advanced tuning bounds", "advanced_tuning", "performance", "guarded", "test_advanced_tuning_adjust_value_rejects_invalid"),
    _item("A.40", "Local observability discloses bound ports", "observability", "performance", "guarded", "test_local_observability_mode_discloses_ports"),
    _item("A.41", "False decline cannot become truthy approval", "preflight_smoke_execution", "coordinator", "guarded", "test_fuzzy_matched_false_decline_does_not_invert_to_confirm"),
    _item("A.42", "GET endpoint probe never sends string body", "endpoint_process", "chain_rpc", "guarded", "test_call_request_handles_empty_body_get_without_typeerror"),
    _item("A.43", "Evidence buffer cannot hijack later job analysis", ("error_evidence_analysis", "report_artifact_analysis"), "analysis", "guarded", "test_job_reference_after_evidence_paste_does_not_reanalyze_stale_evidence"),
    _item("A.44", "Endpoint consultation explains the active field", "endpoint_process", "orientation", "guarded", "test_endpoint_question_context_explains_field_not_generic_state_dump"),
    _item("A.45", "Mode-switch invalidation matches user-facing promise", "target_mode", "chain_rpc", "open", note="The retired register cites only a message-only manual check; transition semantics need current edge evidence."),
    _item("A.46", "Field explanation covers QPS and tuning", ("qps_profile", "advanced_tuning"), "orientation", "guarded", "test_config_field_explanation_covers_qps_and_advanced_tuning_fields"),
    _item("A.47", "Internal routing status never leaks", "opening", "coordinator", "guarded", "test_reason_label_does_not_leak_raw_internal_status"),
    _item("A.48", "google_search availability contract", "chain_identity", "chain_rpc", "guarded", "test_web_research_status_dict_key_matches_harness_contract"),
    _item("A.49", "sync-observe client setup invokes grounding", "sync_observe", "sync_observe", "guarded", "test_sync_observe_client_setup_with_google_search_grounds_the_handoff_message"),
    _item("A.50", "Grounding fallback localization", "sync_observe", "sync_observe", "open", note="Only indirect/manual legacy evidence exists."),
    _item("A.51", "gcloud-only dependency result reporting", "opening", "terminal", "open", note="No named current regression evidence."),
    _item("A.52", "Fixture-authenticity subprocess exit status", "target_samples_fixtures", "chain_rpc", "open", note="No named current regression evidence."),
    _item("A.53", "Confirmation-required tool envelope parity", "preflight_smoke_execution", "execution", "open", note="Old evidence names a file, not a specific assertion."),
    _item("A.54", "Grounding call timeout", ("chain_identity", "endpoint_process", "sync_observe"), "llm_boundary", "open", note="Manual legacy verification is not current evidence."),
    _item("A.55", "Tool-dispatch parity test has no side effects", "opening", "test_harness", "guarded", "test_schema_tool_names_match_executor_dispatch_names"),
    _item("A.56", "Remove unexercised grounding indirection", "chain_identity", "llm_boundary", "open", note="Simplification had no behavioral evidence and must be re-audited on current code."),
    _item("A.57", "Oracle and sync client-setup question agree", "sync_observe", "sync_observe", "guarded", "test_oracle_and_groups_agree_on_next_group_after_client_setup_ack"),
    _item("A.58", "Deferred jump cannot skip newly invalidated prerequisite", ("chain_identity", "endpoint_process", "workload_rpc"), "coordinator", "open", note="The current similarly named test permits an explicit jump; the retired behavioral assertion is not preserved."),
    _item("A.59", "Template-less POST-RPC Case 2 endpoint validation", "endpoint_process", "chain_rpc", "guarded", "test_new_chain_endpoint_validation_works_for_generic_pop_families_without_a_template"),
    _item("A.60", "REST-shaped custom method accepted for its family", "endpoint_process", "chain_rpc", "guarded", "test_custom_rpc_method_accepts_get_path_for_rest_shaped_chain_family"),
    _item("A.61", "Template-less REST-shaped Case 2 endpoint validation", "endpoint_process", "chain_rpc", "guarded", "test_new_chain_endpoint_validation_works_for_rest_shaped_families_without_a_template"),
    _item("A.62", "Opening info option has an actionable result", "opening", "orientation", "guarded", "test_opening_menu_info_option_shows_capability_content_not_a_dead_loop"),
    _item("A.63", "Failed custom schema re-prompts cleanly", "endpoint_process", "chain_rpc", "guarded", "test_custom_rpc_schema_failure_reprompts_for_schema_evidence"),
    _item("A.64", "Execution status reads persisted job state", "job_monitoring", "execution", "guarded", "test_execution_status_prefers_live_job_status_over_stale_snapshot"),
    _item("A.65", "Optional auxiliary endpoint accepts natural decline", "chain_auxiliary_endpoints", "chain_rpc", "guarded", "test_optional_chain_auxiliary_field_accepts_plain_english_decline"),
    _item("A.66", "Weight-only edit skips method revalidation", "workload_rpc", "chain_rpc", "guarded", "test_adjust_mixed_weights_skips_endpoint_revalidation"),
    _item("A.67", "Unknown-chain/custom-method Gemini grounding boundary", ("chain_identity", "endpoint_process"), "chain_rpc", "guarded", "test_unknown_chain_identity_grounds_with_google_search_unconditionally", "test_custom_rpc_schema_extraction_grounds_with_google_search_unconditionally"),
    _item("A.68", "Back/cancel abandons partial custom endpoint attempt", "endpoint_process", "chain_rpc", "guarded", "test_go_back_out_of_stuck_custom_rpc_endpoint_abandons_it"),
    _item("A.69", "Report analysis reads real log evidence", "report_artifact_analysis", "analysis", "guarded", "test_analyze_report_includes_real_log_excerpt_not_just_paths"),
    _item("A.70", "Stale job cannot mark new config complete", ("opening", "job_monitoring"), "coordinator", "guarded", "test_config_status_not_forced_complete_by_a_stale_unrelated_job"),
    _item("A.71", "Back/cancel leaves stuck sync endpoint question", "sync_observe", "sync_observe", "guarded", "test_go_back_out_of_stuck_sync_observe_rpc_url_abandons_it"),
)


# Still-relevant open/design/validation entries from sections B/C/D. Closed or
# struck-through B items are represented by their corresponding A item above.
KNOWN_ISSUES_OPEN: tuple[LegacyItem, ...] = (
    _item("B.2", "Corrupt checkpoint must not be reported as a model failure", "opening", "terminal", "guarded", "test_invalid_checkpoint_is_quarantined_instead_of_silently_resumed", "test_harness_invariant_error_is_not_reported_as_model_failure"),
    _item("B.3", "Vague continue at idle opening", "opening", "orientation", "open"),
    _item("B.4", "Heavily mistyped target mode", "target_mode", "chain_rpc", "open"),
    _item("B.5", "Follow-up analysis reuses buffered evidence", "error_evidence_analysis", "analysis", "open"),
    _item("B.6", "Contradictory same-field instructions require explicit resolution", "opening", "coordinator", "open"),
    _item("B.7", "Natural ordinal option selection", "opening", "coordinator", "open"),
    _item("B.8", "Extra model calls in handoff/navigation", "chain_identity", "coordinator", "open"),
    _item("B.9", "Natural-language dependency installation request", "opening", "terminal", "open"),
    _item("B.10", "Explain result artifacts before execution", "report_artifact_analysis", "orientation", "open"),
    _item("B.11", "Verbose unknown-chain identity confirmation", "chain_identity", "chain_rpc", "open"),
    _item("B.12", "Bulk accept only safe detected metadata", "provider_deployment", "environment", "open"),
    _item("B.13", "Verbose unsupported-family answer reaches Case 3", "chain_identity", "chain_rpc", "open"),
    _item("B.14", "Case-3 generation command is not evidence", "chain_identity", "chain_rpc", "open"),
    _item("B.15", "Failure analysis reads known job logs", "failure_recovery", "recovery", "open"),
    _item("B.19", "Report/capability questions avoid generic current-config response", ("opening", "report_artifact_analysis"), "orientation", "open"),
    _item("B.24", "Mode-switch invalidation policy is explicit and pruned", "target_mode", "coordinator", "open"),
    _item("B.26", "Job-running status question gets a direct answer", "job_monitoring", "execution", "open"),
    _item("C.1", "Independent dynamic Codex-user plus DeepSeek-Agent Docker chaos", "opening", "test_harness", "open", note="The retired document incorrectly waived independence and Docker; current acceptance requires both."),
    _item("C.2", "Live Gemini plus google_search", ("chain_identity", "endpoint_process", "sync_observe"), "llm_boundary", "open", note="External credentials/environment required."),
    _item("C.3", "Full report and analysis dependency suite on the current revision", "report_artifact_analysis", "analysis", "open", note="The retired environment's 512/512 result is not evidence for this worktree."),
    _item("C.4", "Real execution evidence on the current revision", ("preflight_smoke_execution", "job_monitoring"), "execution", "open", note="Historical job evidence does not prove the rebuilt revision."),
    _item("C.5", "Non-EVM endpoint and custom-RPC validation", ("endpoint_process", "target_samples_fixtures"), "chain_rpc", "guarded", "test_call_request_handles_empty_body_get_without_typeerror", "test_new_chain_endpoint_validation_works_for_generic_pop_families_without_a_template", "test_new_chain_endpoint_validation_works_for_rest_shaped_families_without_a_template"),
    _item("C.6", "Live family coverage including successful hedera_dual", "endpoint_process", "chain_rpc", "open"),
    _item("C.7", "Live Case 1/2/3 coverage across adapter families", ("chain_identity", "endpoint_process", "target_samples_fixtures"), "chain_rpc", "open"),
    _item("D.1", "Generic next-step consultation resumes the real blocking group", "opening", "orientation", "open", note="The retired register cites live verification only; no direct current guard was found."),
    _item("D.2", "A demo-only sync-observe source must not bypass real-source gates", ("sync_observe", "preflight_smoke_execution"), "sync_observe", "superseded", "test_sync_observe_demo_only_does_not_waive_real_node_requirements", note="The retired auto-execute design was replaced by the current real-source contract."),
    _item("D.3", "No duplicate no-pending/fallback text", "opening", "coordinator", "open"),
)


def _repair_phase(phase: int, summary: str, groups: tuple[str, ...], owner: str, disposition: str, *evidence: str, note: str = "") -> LegacyItem:
    return _item(f"RP.{phase}", summary, groups, owner, disposition, *evidence, note=note)


# Phase-level migration prevents the deleted repair plan from remaining an
# implicit dependency. Detailed defects from phases 9-29 are mapped above.
REPAIR_PLAN_PHASES: tuple[LegacyItem, ...] = (
    _repair_phase(1, "Resume/reset/discovery semantics", ("opening",), "orientation", "guarded", "test_harness_snapshot_and_reset_support_terminal_resume_gate"),
    _repair_phase(2, "Single group order and next-action authority", ALL_CURRENT_GROUPS, "coordinator", "superseded", "test_twenty_groups_have_exactly_one_of_eight_domain_owners", "test_group_specs_own_dependencies_and_invalidation_metadata"),
    _repair_phase(3, "Single typed question engine", ALL_CURRENT_GROUPS, "coordinator", "superseded", "test_visible_choice_requires_action_and_postcondition"),
    _repair_phase(4, "Case 1/2 structural ownership", ("chain_identity", "endpoint_process", "workload_rpc"), "chain_rpc", "superseded", "test_twenty_groups_have_exactly_one_of_eight_domain_owners"),
    _repair_phase(5, "Harness/ADK dependency boundary", ALL_CURRENT_GROUPS, "llm_boundary", "superseded", "test_deepseek_runtime_does_not_require_google_adk", "test_free_form_intent_has_one_llm_entry_point"),
    _repair_phase(6, "Dead code and prompt-contract cleanup", ALL_CURRENT_GROUPS, "architecture", "superseded", "test_legacy_group_owner_is_absent", "test_harness_has_no_script_style_import_fallbacks"),
    _repair_phase(7, "Dual-AI live chaos verification", ALL_CURRENT_GROUPS, "test_harness", "open", note="Historical transcript claims are not accepted for the current revision."),
    _repair_phase(8, "Deferred-debt re-audit", ALL_CURRENT_GROUPS, "architecture", "open", note="Current open items are enumerated in KNOWN_ISSUES_OPEN."),
    *tuple(_repair_phase(phase, f"Historical chaos/repair phase {phase}", ALL_CURRENT_GROUPS, "test_harness", "open", note="Defect-level evidence is mapped under A.*; dynamic acceptance must be rerun on the current revision.") for phase in range(9, 24)),
    _repair_phase(24, "ADK retirement and search boundary", ("chain_identity", "endpoint_process", "sync_observe"), "llm_boundary", "superseded", "test_deepseek_runtime_does_not_require_google_adk", "test_web_research_status_dict_key_matches_harness_contract"),
    *tuple(_repair_phase(phase, f"Historical family/Case coverage phase {phase}", ("chain_identity", "endpoint_process", "target_samples_fixtures"), "chain_rpc", "open", note="Current-revision live endpoint/fixture/job evidence is required.") for phase in range(25, 30)),
    _repair_phase(30, "Interrupted sync-observe demo and duplicate-response work", ("sync_observe", "preflight_smoke_execution", "opening"), "coordinator", "open", note="The retired plan explicitly marked this phase unfinished."),
)


ALL_ITEMS: tuple[LegacyItem, ...] = KNOWN_ISSUES_A + KNOWN_ISSUES_OPEN + REPAIR_PLAN_PHASES


def _test_names() -> set[str]:
    names: set[str] = set()
    for path in (REPO_ROOT / "tests").rglob("*.py"):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.lstrip()
            if stripped.startswith("def test_"):
                names.add(stripped.split("(", 1)[0].removeprefix("def "))
    return names


def _test_locations() -> dict[str, tuple[str, ...]]:
    locations: dict[str, list[str]] = {}
    for path in sorted((REPO_ROOT / "tests").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.lstrip()
            if not stripped.startswith("def test_"):
                continue
            name = stripped.split("(", 1)[0].removeprefix("def ")
            locations.setdefault(name, []).append(f"{relative}::{name}")
    return {name: tuple(paths) for name, paths in locations.items()}


def validate() -> list[str]:
    errors: list[str] = []
    identifiers = [item.legacy_id for item in ALL_ITEMS]
    if len(identifiers) != len(set(identifiers)):
        errors.append("duplicate legacy ids")
    expected_a = {f"A.{number}" for number in range(1, 72)}
    actual_a = {item.legacy_id for item in KNOWN_ISSUES_A}
    if actual_a != expected_a:
        errors.append(f"A-series mismatch: missing={sorted(expected_a - actual_a)} extra={sorted(actual_a - expected_a)}")
    expected_phases = {f"RP.{number}" for number in range(1, 31)}
    actual_phases = {item.legacy_id for item in REPAIR_PLAN_PHASES}
    if actual_phases != expected_phases:
        errors.append(f"repair-phase mismatch: missing={sorted(expected_phases - actual_phases)} extra={sorted(actual_phases - expected_phases)}")

    known_tests = _test_names()
    allowed_component_owners = set(GROUP_OWNER.values()) | {
        "architecture", "coordinator", "llm_boundary", "terminal", "test_harness"
    }
    for item in ALL_ITEMS:
        if not item.groups:
            errors.append(f"{item.legacy_id}: no current group mapping")
        unknown_groups = sorted(set(item.groups) - set(GROUP_OWNER))
        if unknown_groups:
            errors.append(f"{item.legacy_id}: unknown groups {unknown_groups}")
        if item.disposition not in {"guarded", "superseded", "open"}:
            errors.append(f"{item.legacy_id}: invalid disposition {item.disposition}")
        if item.owner not in allowed_component_owners:
            errors.append(f"{item.legacy_id}: unknown responsible owner {item.owner}")
        if item.disposition in {"guarded", "superseded"}:
            if not item.evidence:
                errors.append(f"{item.legacy_id}: {item.disposition} without evidence")
            missing_tests = sorted(set(item.evidence) - known_tests)
            if missing_tests:
                errors.append(f"{item.legacy_id}: missing current tests {missing_tests}")
    return errors


def report() -> dict[str, object]:
    locations = _test_locations()
    items: list[dict[str, object]] = []
    for item in ALL_ITEMS:
        payload = asdict(item)
        payload["group_owners"] = {group: GROUP_OWNER[group] for group in item.groups}
        payload["evidence_locations"] = [
            location
            for evidence_name in item.evidence
            for location in locations.get(evidence_name, ())
        ]
        items.append(payload)
    return {
        "schema_version": 1,
        "source_documents": [
            ".agent/task-docs/2026-07-10-langgraph-harness-repair-plan.md (retired)",
            ".agent/task-docs/2026-07-11-agent-known-issues.md (retired)",
        ],
        "items": items,
        "validation_errors": validate(),
    }


if __name__ == "__main__":
    payload = report()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    raise SystemExit(1 if payload["validation_errors"] else 0)
