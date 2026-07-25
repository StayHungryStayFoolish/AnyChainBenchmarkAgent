"""Focused contracts for the finite Phase 8C G4 obligation catalog."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from types import SimpleNamespace

from agent.harness.domains.rpc_receipts import evidence_hash
from tests.agent_live.coverage_evidence import RuntimeTurnEvent, content_hash
from tests.agent_live.dynamic_dual_ai_chaos import (
    JourneyDecisionProvenance,
    JourneyVerifierContext,
)
from tests.agent_live.formal_journey_catalog import (
    FACTOR_OBSERVATION_VALUES,
    FORMAL_JOURNEY_VERIFIER_REGISTRY,
    factor_observation_postcondition_id,
    reviewed_seed_factor_proof,
)
from tests.agent_live.product_chaos_obligations import (
    PRODUCT_CHAOS_SEED,
    build_product_chaos_obligations,
    product_chaos_obligation_report,
    validate_product_chaos_obligations,
)
from tests.agent_live.runtime_checkpoint import reviewed_scenario


REVISION = {"commit": "a" * 40, "worktree_hash": "b" * 64}


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _receipt(body, *, rpc=False):
    digest = evidence_hash(body) if rpc else content_hash(body)
    return {**body, "receipt_id": digest}


def _event(
    turn_index,
    *,
    fingerprint="event",
    actions=(),
    receipts=(),
    material=(),
    values=None,
    turn_shape="prose",
    unresolved=(),
    active_group="opening",
):
    diffs = {
        path: {"before": "0" * 64, "after": _hash([turn_index, path])}
        for path in material
    }
    return RuntimeTurnEvent(
        schema_version=3,
        event_type="turn_committed",
        thread_id="factor-test",
        session_purpose="product-chaos",
        before_fingerprint=f"{fingerprint}-before",
        after_fingerprint=fingerprint,
        turn_index=turn_index,
        active_group=active_group,
        pending_question_id="",
        action_queue_types=(),
        admitted_action_types=tuple(actions),
        turn_receipt_summary={
            "input_shape": turn_shape,
            "unresolved_unit_ids": list(unresolved),
        },
        control_receipts=tuple(receipts),
        material_state_diff_hashes=diffs,
        state_diff_hashes=diffs,
        after_value_hashes={
            path: _hash(value) for path, value in (values or {}).items()
        },
    )


def _factor_context(
    factor_name,
    *,
    scenario_id="opening",
    subject_group="qps_profile",
    initial_fingerprint="initial",
    messages=(),
    events=(),
):
    risks = tuple(
        f"{factor_name}:{value}"
        for value in FACTOR_OBSERVATION_VALUES[factor_name]
    )
    decisions = tuple(
        JourneyDecisionProvenance(
            turn_index=index,
            previous_response_hash="a" * 64,
            selected_at_ns=index,
            submitted_at_ns=index,
            user_message_hash="b" * 64,
            persona="operator",
            mission="factor verification",
            rationale="response driven",
            risk_factor_ids=risks,
        )
        for index, _ in enumerate(events, start=1)
    )
    initial = _event(
        0,
        fingerprint=initial_fingerprint,
    )
    return JourneyVerifierContext(
        schedule=SimpleNamespace(
            start_scenario=scenario_id,
            subject_group=subject_group,
        ),
        initial_event=initial,
        current_event=events[-1] if events else initial,
        completed_turns=(),
        transcript=tuple((message, "Agent response") for message in messages),
        observed_edge_keys=(),
        latest_turn=None,
        completed_events=tuple(events),
        completed_decisions=decisions,
    )


def _observe(context, factor_name, factor_value):
    postcondition_id = factor_observation_postcondition_id(
        factor_name, factor_value
    )
    definition = FORMAL_JOURNEY_VERIFIER_REGISTRY.definitions[postcondition_id]
    return definition.verifier(
        JourneyVerifierContext(
            **{
                **context.__dict__,
                "evaluating_postcondition_id": postcondition_id,
            }
        )
    )


def _pending_receipt(turn_index):
    return _receipt({
        "receipt_type": "pending_resolution",
        "turn_index": turn_index,
        "pending_id": "question",
        "pending_group": "opening",
        "pending_contract_hash": "1" * 64,
        "resolution_path": "exact_contract",
        "selected_option_id": "1",
        "selected_value_hash": "2" * 64,
        "resolved_action_id": "action-1",
        "input_hash": "3" * 64,
        "normalizer": "exact",
        "verdict": "accepted",
    })


def _planner_receipt(turn_index):
    return _receipt({
        "receipt_type": "semantic_planner",
        "turn_index": turn_index,
        "input_hash": "1" * 64,
        "pending_contract_hash": "2" * 64,
        "resolver_invoked": True,
        "result_reason_hash": "3" * 64,
        "planned_action_types": ["consult"],
        "semantic_units": [{"unit_id": "unit-1", "disposition": "mapped"}],
        "planner_metrics": {"unit_count": 1},
    })


def _domain_commit(
    turn_index,
    path,
    *,
    owner="orientation",
    navigation="",
    origin="",
    target="",
    group_state_transitions=(),
):
    paths = (path,) if isinstance(path, str) else tuple(path)
    return _receipt({
        "receipt_type": "domain_commit",
        "turn_index": turn_index,
        "owner": owner,
        "completion": "completed",
        "group_registry_contract_hash": "1" * 64,
        "pending_before_hash": "2" * 64,
        "pending_after_hash": "3" * 64,
        "consumed_action_ids": ["action-1"],
        "invalidated_groups": [],
        "invalidated_fields": [],
        "response_fragments": [],
        "reconfigured_groups": [],
        "group_state_transitions": list(group_state_transitions),
        "material_delta": [
            {
                "operation": "write",
                "path": item,
                "value_hash": "4" * 64,
            }
            for item in paths
        ],
        "navigation_operation": navigation,
        "navigation_origin_group": origin,
        "navigation_target_group": target,
        "pending_after_id": "",
    })


def _rpc_schema_receipt(turn_index, fields):
    body = {
        "receipt_type": "rpc_schema_provenance",
        "receipt_version": 1,
        "turn_index": turn_index,
        "owner": "rpc_catalog",
        "method": "eth_test",
        "method_hash": evidence_hash("eth_test"),
        "catalog_revision": turn_index,
        "fields": fields,
        "fields_hash": evidence_hash(fields),
    }
    return _receipt(body, rpc=True)


def _rpc_workload_receipt(turn_index, mode, *, custom):
    body = {
        "receipt_type": "rpc_workload_commit",
        "receipt_version": 1,
        "turn_index": turn_index,
        "owner": "rpc_workload",
        "case": "custom_rpc" if custom else "known",
        "rpc_mode": mode,
        "choice": "custom_rpc" if custom else "template_default",
        "methods": ["eth_test"],
        "method_hashes": [evidence_hash("eth_test")],
        "mixed_weight_entries": (
            [{"method_hash": evidence_hash("eth_test"), "weight": 100}]
            if mode == "mixed"
            else []
        ),
        "replace_defaults": custom,
        "job_local_override": custom,
        "catalog_revision": 1,
        "fixture_required": False,
        "fixture_status": "",
    }
    return _receipt(body, rpc=True)


def _response_receipt(turn_index, language):
    return _receipt({
        "receipt_type": "response_composition",
        "turn_index": turn_index,
        "language": language,
        "active_group": "opening",
        "source_action_ids": [],
        "pending_contract_hash": "1" * 64,
        "fragments": [],
    })


class ProductChaosObligationCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = build_product_chaos_obligations(revision=REVISION)

    def test_catalog_has_one_finite_denominator_from_both_models(self) -> None:
        report = product_chaos_obligation_report(self.rows, revision=REVISION)
        self.assertEqual(report["catalog_seed"], PRODUCT_CHAOS_SEED)
        self.assertEqual(report["required_denominator"], len(self.rows))
        self.assertEqual(report["generated_count"], len(self.rows))
        self.assertEqual(report["not_run_count"], len(self.rows))
        self.assertEqual(report["observed_pass_count"], 0)
        self.assertEqual(report["observed_fail_count"], 0)
        self.assertFalse(report["generation_is_execution"])
        self.assertEqual(sum(report["by_model"].values()), len(self.rows))

    def test_obligation_ids_and_source_rows_are_stable_and_unique(self) -> None:
        rebuilt = build_product_chaos_obligations(revision=REVISION)
        self.assertEqual(self.rows, rebuilt)
        self.assertEqual(
            len({row["obligation_id"] for row in self.rows}),
            len(self.rows),
        )
        self.assertEqual(
            len({
                (row["model"]["model_id"], row["source_row_id"])
                for row in self.rows
            }),
            len(self.rows),
        )
        self.assertEqual({row["status"] for row in self.rows}, {"not_run"})

    def test_every_row_binds_an_executable_reviewed_start_and_real_verifiers(self) -> None:
        verifier_ids = set(FORMAL_JOURNEY_VERIFIER_REGISTRY.definitions)
        for row in self.rows:
            scenario = reviewed_scenario(row["start_contract"]["scenario_id"])
            self.assertTrue(scenario.seed_state)
            self.assertEqual(
                row["start_contract"]["scenario_state_fingerprint"],
                scenario.state_fingerprint,
            )
            verifier = row["verifier_contract"]
            self.assertTrue(set(verifier["required_postcondition_ids"]) <= verifier_ids)
            self.assertTrue(set(verifier["forbidden_postcondition_ids"]) <= verifier_ids)
            self.assertTrue(verifier["required_bindings"])
            factor_contracts = verifier["factor_observation_contracts"]
            self.assertEqual(
                {
                    (item["factor_name"], item["factor_value"])
                    for item in factor_contracts
                },
                set(row["factors"].items()),
            )
            self.assertTrue(all(
                item["postcondition_id"] in verifier_ids
                and item["risk_factor_id"]
                == f"{item['factor_name']}:{item['factor_value']}"
                for item in factor_contracts
            ))
            attestation = row["simulator_contract"]["attestation"]
            self.assertEqual(
                attestation["identity_strength"],
                "auditable_declaration_only",
            )
            self.assertFalse(attestation["cryptographic_identity_claimed"])

    def test_pending_and_session_factors_cannot_be_hidden_by_opening(self) -> None:
        for row in self.rows:
            factors = row["factors"]
            scenario_id = row["start_contract"]["scenario_id"]
            scenario = reviewed_scenario(scenario_id)
            question = dict(scenario.seed_state.get("pending_question") or {})
            pending = factors["pending_state"]
            if pending == "none" and scenario_id != "resume_quarantine":
                self.assertFalse(question, row["obligation_id"])
            elif pending == "manual":
                self.assertTrue(question.get("manual_input_allowed"), row["obligation_id"])
            elif pending == "choice":
                self.assertTrue(question.get("options"), row["obligation_id"])
            if factors.get("session_state") not in {None, "fresh"}:
                self.assertNotEqual(scenario_id, "opening", row["obligation_id"])
            if factors.get("session_state") == "quarantine":
                self.assertEqual(scenario_id, "resume_quarantine")

    def test_validation_fails_closed_on_opening_fallback_for_nonfresh_state(self) -> None:
        rows = copy.deepcopy(list(self.rows))
        index = next(
            i for i, row in enumerate(rows)
            if row["factors"].get("session_state") == "partial"
        )
        opening = reviewed_scenario("opening")
        rows[index]["start_contract"]["scenario_id"] = "opening"
        rows[index]["start_contract"]["scenario_state_fingerprint"] = (
            opening.state_fingerprint
        )
        with self.assertRaisesRegex(ValueError, "start mapping drifted"):
            validate_product_chaos_obligations(rows, revision=REVISION)

    def test_validation_fails_closed_on_execution_claim_or_unknown_verifier(self) -> None:
        executed = copy.deepcopy(list(self.rows))
        executed[0]["status"] = "passed"
        with self.assertRaisesRegex(ValueError, "claimed execution"):
            validate_product_chaos_obligations(executed, revision=REVISION)

        unknown = copy.deepcopy(list(self.rows))
        unknown[0]["verifier_contract"]["required_postcondition_ids"] = ["invented"]
        with self.assertRaisesRegex(ValueError, "unknown product Chaos verifiers"):
            validate_product_chaos_obligations(unknown, revision=REVISION)

    def test_validation_fails_closed_on_stale_revision(self) -> None:
        with self.assertRaisesRegex(ValueError, "stale product Chaos revision"):
            validate_product_chaos_obligations(
                copy.deepcopy(list(self.rows)),
                revision={"commit": "c" * 40, "worktree_hash": "d" * 64},
            )

    def test_model_strength_seed_and_factors_are_preserved_per_row(self) -> None:
        seeds = set()
        for row in self.rows:
            self.assertIsInstance(row["seed"], int)
            self.assertGreater(row["seed"], 0)
            seeds.add(row["seed"])
            self.assertEqual(row["revision_binding"], REVISION)
            self.assertEqual(row["model"]["strength"], 2 if row["model"]["lane"] == "pairwise" else 3)
            self.assertTrue(row["source_row_id"])
            self.assertTrue(row["factors"])
            simulator = row["simulator_contract"]
            self.assertEqual(simulator["selection_mode"], "response_driven")
            self.assertTrue(simulator["prewritten_future_turns_forbidden"])
            self.assertTrue(simulator["read_complete_agent_response_before_each_turn"])
        self.assertEqual(len(seeds), len(self.rows))

    def test_cross_row_seed_transplant_is_rejected_even_when_rehashed(self) -> None:
        transplanted = copy.deepcopy(list(self.rows))
        transplanted[0]["seed"] = transplanted[1]["seed"]
        unsigned = dict(transplanted[0])
        unsigned.pop("contract_hash")
        transplanted[0]["contract_hash"] = content_hash(unsigned)
        with self.assertRaisesRegex(ValueError, "seed drifted"):
            validate_product_chaos_obligations(
                transplanted,
                revision=REVISION,
            )

    def test_seed_factor_values_are_exact_or_explicitly_blocked(self) -> None:
        scenarios = {
            "pending_state": {
                "none": "action_change_group",
                "manual": "custom_needs_method",
                "choice": "opening",
            },
            "session_state": {
                "fresh": "opening",
                "partial": "custom_needs_schema_evidence",
                "complete": "execution",
                "quarantine": "resume_quarantine",
            },
        }
        for factor_name, values in scenarios.items():
            for expected, scenario_id in values.items():
                proof = reviewed_seed_factor_proof(scenario_id, factor_name)
                context = _factor_context(
                    factor_name,
                    scenario_id=scenario_id,
                    initial_fingerprint=proof["scenario_state_fingerprint"],
                    messages=("continue",),
                    events=(_event(1),),
                )
                with self.subTest(factor=factor_name, value=expected):
                    self.assertTrue(_observe(context, factor_name, expected).satisfied)
                    for neighbor in FACTOR_OBSERVATION_VALUES[factor_name]:
                        if neighbor != expected:
                            self.assertFalse(
                                _observe(context, factor_name, neighbor).satisfied
                            )

        group_statuses = {
            "partial": "in_progress",
            "completed": "completed",
            "invalidated": "invalidated",
        }
        for value, status in group_statuses.items():
            context = _factor_context(
                "group_state",
                messages=("continue",),
                events=(_event(
                    1,
                    receipts=(_domain_commit(
                        1,
                        (),
                        group_state_transitions=({
                            "group": "qps_profile",
                            "before": "",
                            "after": status,
                        },),
                    ),),
                ),),
            )
            for candidate in FACTOR_OBSERVATION_VALUES["group_state"]:
                self.assertEqual(
                    _observe(context, "group_state", candidate).satisfied,
                    candidate == value,
                    (value, candidate),
                )

        wrong_group = _factor_context(
            "group_state",
            subject_group="qps_profile",
            messages=("continue",),
            events=(_event(
                1,
                receipts=(_domain_commit(
                    1,
                    (),
                    group_state_transitions=({
                        "group": "observability",
                        "before": "",
                        "after": "completed",
                    },),
                ),),
            ),),
        )
        self.assertFalse(
            _observe(wrong_group, "group_state", "completed").satisfied
        )

        active_group_only = _factor_context(
            "subject_group",
            subject_group="qps_profile",
            messages=("continue",),
            events=(_event(1, active_group="qps_profile"),),
        )
        self.assertFalse(
            _observe(
                active_group_only,
                "subject_group",
                "qps_profile",
            ).satisfied
        )
        committed_subject_group = _factor_context(
            "subject_group",
            subject_group="qps_profile",
            messages=("continue",),
            events=(_event(
                1,
                active_group="qps_profile",
                receipts=(_domain_commit(
                    1,
                    (),
                    group_state_transitions=({
                        "group": "qps_profile",
                        "before": "in_progress",
                        "after": "completed",
                    },),
                ),),
            ),),
        )
        self.assertTrue(
            _observe(
                committed_subject_group,
                "subject_group",
                "qps_profile",
            ).satisfied
        )
        for subject_group in FACTOR_OBSERVATION_VALUES["subject_group"]:
            context = _factor_context(
                "subject_group",
                subject_group=subject_group,
                messages=("continue",),
                events=(_event(
                    1,
                    active_group=subject_group,
                    receipts=(_domain_commit(
                        1,
                        (),
                        group_state_transitions=({
                            "group": subject_group,
                            "before": "in_progress",
                            "after": "completed",
                        },),
                    ),),
                ),),
            )
            with self.subTest(subject_group=subject_group):
                self.assertTrue(
                    _observe(
                        context,
                        "subject_group",
                        subject_group,
                    ).satisfied
                )

    def test_input_shape_values_are_mutually_exclusive(self) -> None:
        contexts = {
            "exact": _factor_context(
                "input_shape",
                messages=("1",),
                events=(_event(1, receipts=(_pending_receipt(1),)),),
            ),
            "natural": _factor_context(
                "input_shape",
                messages=("please explain the next step",),
                events=(_event(1, receipts=(_planner_receipt(1),)),),
            ),
            "multiline": _factor_context(
                "input_shape",
                messages=("first line\nsecond line",),
                events=(_event(1),),
            ),
            "structured": _factor_context(
                "input_shape",
                messages=('{"CLOUD_REGION":"test"}',),
                events=(_event(1, turn_shape="structured"),),
            ),
            "contradictory": _factor_context(
                "input_shape",
                messages=("use A but do not use A",),
                events=(_event(
                    1,
                    receipts=(_planner_receipt(1),),
                    unresolved=("unit-1",),
                ),),
            ),
        }
        for expected, context in contexts.items():
            for candidate in FACTOR_OBSERVATION_VALUES["input_shape"]:
                with self.subTest(expected=expected, candidate=candidate):
                    self.assertEqual(
                        _observe(context, "input_shape", candidate).satisfied,
                        candidate == expected,
                    )

    def test_evidence_shape_values_use_typed_provenance_and_order(self) -> None:
        request_field = {
            "field_path": "params_json",
            "source_kind": "protocol_request_parser",
            "source_revisions": [1],
            "value_hash": "1" * 64,
        }
        response_field = {
            "field_path": "response_sample",
            "source_kind": "model_extraction_from_user_evidence",
            "source_revisions": [1],
            "value_hash": "2" * 64,
        }
        docs_field = {
            "field_path": "method",
            "source_kind": "official_document",
            "source_revisions": [1],
            "value_hash": "3" * 64,
        }
        contexts = {
            "none": _factor_context(
                "evidence_shape", messages=("continue",), events=(_event(1),)
            ),
            "request": _factor_context(
                "evidence_shape",
                messages=('{"method":"eth_test","params":[]}',),
                events=(_event(
                    1,
                    receipts=(_rpc_schema_receipt(1, [request_field]),),
                    material=("custom_rpc.catalog",),
                ),),
            ),
            "response": _factor_context(
                "evidence_shape",
                messages=('{"result":"0x1"}',),
                events=(_event(
                    1,
                    receipts=(_rpc_schema_receipt(1, [response_field]),),
                    material=("custom_rpc.catalog",),
                ),),
            ),
            "docs": _factor_context(
                "evidence_shape",
                messages=("official documentation evidence",),
                events=(_event(
                    1,
                    receipts=(_rpc_schema_receipt(1, [docs_field]),),
                    material=("custom_rpc.catalog",),
                ),),
            ),
        }
        split_response = {**response_field, "source_revisions": [1, 2]}
        contexts["split"] = _factor_context(
            "evidence_shape",
            messages=('{"method":"eth_test","params":[]}', '{"result":"0x1"}'),
            events=(
                _event(
                    1,
                    receipts=(_rpc_schema_receipt(1, [request_field]),),
                    material=("custom_rpc.catalog",),
                ),
                _event(
                    2,
                    receipts=(_rpc_schema_receipt(
                        2, [{**request_field, "source_revisions": [1, 2]}, split_response]
                    ),),
                    material=("custom_rpc.catalog",),
                ),
            ),
        )
        reverse_split = _factor_context(
            "evidence_shape",
            messages=('{"result":"0x1"}', '{"method":"eth_test","params":[]}'),
            events=(
                _event(
                    1,
                    receipts=(_rpc_schema_receipt(1, [split_response]),),
                    material=("custom_rpc.catalog",),
                ),
                _event(
                    2,
                    receipts=(_rpc_schema_receipt(
                        2,
                        [{**request_field, "source_revisions": [1, 2]}],
                    ),),
                    material=("custom_rpc.catalog",),
                ),
            ),
        )
        self.assertFalse(
            _observe(reverse_split, "evidence_shape", "split").satisfied
        )
        for expected, context in contexts.items():
            for candidate in FACTOR_OBSERVATION_VALUES["evidence_shape"]:
                with self.subTest(expected=expected, candidate=candidate):
                    self.assertEqual(
                        _observe(context, "evidence_shape", candidate).satisfied,
                        candidate == expected,
                    )

    def test_recovery_requires_action_provenance_and_matching_commit(self) -> None:
        specifications = {
            "back": ("go_back", "interruption_stack", "go_back"),
            "jump": ("change_group", "active_group", "change_group"),
            "correct": ("correct_failure", "failure_recovery", ""),
            "retry": ("retry_failure", "failure_recovery", ""),
            "reset": ("reset_session", "confirmed_config", ""),
        }
        contexts = {
            "none": _factor_context(
                "recovery", messages=("continue",), events=(_event(1),)
            )
        }
        for value, (action, path, navigation) in specifications.items():
            contexts[value] = _factor_context(
                "recovery",
                messages=(value,),
                events=(_event(
                    1,
                    actions=(action,),
                    receipts=(_domain_commit(
                        1,
                        path,
                        navigation=navigation,
                        origin="opening" if navigation else "",
                        target="qps_profile" if navigation else "",
                    ),),
                    material=(path,),
                ),),
            )
        for expected, context in contexts.items():
            for candidate in FACTOR_OBSERVATION_VALUES["recovery"]:
                self.assertEqual(
                    _observe(context, "recovery", candidate).satisfied,
                    candidate == expected,
                    (expected, candidate),
                )
        answer_only = _factor_context(
            "recovery",
            messages=("Y",),
            events=(_event(
                1,
                actions=("answer_pending",),
                receipts=(_pending_receipt(1),),
            ),),
        )
        self.assertFalse(_observe(answer_only, "recovery", "correct").satisfied)
        cross_turn = _factor_context(
            "recovery",
            messages=("correct", "commit"),
            events=(
                _event(1, actions=("correct_failure",)),
                _event(
                    2,
                    receipts=(_domain_commit(2, "failure_recovery"),),
                    material=("failure_recovery",),
                ),
            ),
        )
        self.assertFalse(_observe(cross_turn, "recovery", "correct").satisfied)

    def test_runtime_factor_values_require_exact_owner_receipts(self) -> None:
        for case in FACTOR_OBSERVATION_VALUES["chain_case"]:
            context = _factor_context(
                "chain_case",
                messages=(case,),
                events=(_event(
                    1,
                    receipts=(_domain_commit(
                        1, "chain_identity", owner="chain_identity"
                    ),),
                    material=("chain_identity",),
                    values={
                        "chain_identity.case": case,
                        "chain_identity.status": "confirmed",
                    },
                ),),
            )
            for candidate in FACTOR_OBSERVATION_VALUES["chain_case"]:
                self.assertEqual(
                    _observe(context, "chain_case", candidate).satisfied,
                    candidate == case,
                )
            unconfirmed = copy.deepcopy(context)
            unconfirmed.completed_events[0].after_value_hashes[
                "chain_identity.status"
            ] = _hash("candidate")
            self.assertFalse(
                _observe(unconfirmed, "chain_case", case).satisfied
            )

        for workload in FACTOR_OBSERVATION_VALUES["workload"]:
            if workload == "not_applicable":
                event = _event(
                    1,
                    receipts=(_domain_commit(1, "workflow_mode"),),
                    material=("workflow_mode",),
                    values={"workflow_mode": "sync_observe"},
                )
            else:
                mode = "single" if workload.endswith("single") else "mixed"
                event = _event(
                    1,
                    receipts=(_rpc_workload_receipt(
                        1, mode, custom=workload.startswith("custom")
                    ),),
                )
            context = _factor_context(
                "workload", messages=(workload,), events=(event,)
            )
            for candidate in FACTOR_OBSERVATION_VALUES["workload"]:
                self.assertEqual(
                    _observe(context, "workload", candidate).satisfied,
                    candidate == workload,
                    (workload, candidate),
                )

    def test_mode_language_and_interruption_values_are_mutually_exclusive(self) -> None:
        modes = {
            "fake": ("rpc_benchmark", "fake-node"),
            "real": ("rpc_benchmark", "real-node"),
            "sync": ("sync_observe", "sync-observe"),
        }
        for expected, (workflow_mode, target_mode) in modes.items():
            context = _factor_context(
                "workflow_mode",
                messages=(expected,),
                events=(_event(
                    1,
                    receipts=(_domain_commit(
                        1, ("workflow_mode", "target_mode")
                    ),),
                    material=("workflow_mode", "target_mode"),
                    values={
                        "workflow_mode": workflow_mode,
                        "target_mode": target_mode,
                    },
                ),),
            )
            for candidate in FACTOR_OBSERVATION_VALUES["workflow_mode"]:
                self.assertEqual(
                    _observe(context, "workflow_mode", candidate).satisfied,
                    candidate == expected,
                )

        for expected in FACTOR_OBSERVATION_VALUES["language"]:
            context = _factor_context(
                "language",
                messages=("language",),
                events=(_event(
                    1,
                    receipts=(_response_receipt(1, expected),),
                    values={"language": expected},
                ),),
            )
            for candidate in FACTOR_OBSERVATION_VALUES["language"]:
                self.assertEqual(
                    _observe(context, "language", candidate).satisfied,
                    candidate == expected,
                )

        depth_events = {
            "0": (_event(1),),
            "1": (_event(
                1,
                actions=("change_group",),
                receipts=(_domain_commit(
                    1,
                    "interruption_stack",
                    navigation="change_group",
                    origin="opening",
                    target="qps_profile",
                ),),
                material=("interruption_stack",),
            ),),
            "2+": (
                _event(
                    1,
                    actions=("change_group",),
                    receipts=(_domain_commit(
                        1,
                        "interruption_stack",
                        navigation="change_group",
                        origin="opening",
                        target="qps_profile",
                    ),),
                    material=("interruption_stack",),
                ),
                _event(
                    2,
                    actions=("change_group",),
                    receipts=(_domain_commit(
                        2,
                        "interruption_stack",
                        navigation="change_group",
                        origin="qps_profile",
                        target="chain_identity",
                    ),),
                    material=("interruption_stack",),
                ),
            ),
        }
        for expected, events in depth_events.items():
            context = _factor_context(
                "interruption_depth",
                messages=tuple("jump" for _ in events),
                events=events,
            )
            for candidate in FACTOR_OBSERVATION_VALUES["interruption_depth"]:
                self.assertEqual(
                    _observe(context, "interruption_depth", candidate).satisfied,
                    candidate == expected,
                )


if __name__ == "__main__":
    unittest.main()
