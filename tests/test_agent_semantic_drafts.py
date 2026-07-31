from __future__ import annotations

import unittest
from copy import deepcopy
import json
from unittest.mock import patch

from agent.harness.plan_coverage import PlanCoverageResult
from agent.harness.semantic_drafts import (
    build_semantic_draft_finalization_receipt,
    build_semantic_plan_draft,
    cancel_semantic_plan_draft,
    mark_semantic_plan_draft_stale,
    reopen_previous_semantic_draft_atom,
    resolve_semantic_draft_atom,
    missing_semantic_draft_secret_bindings,
    semantic_draft_staleness_reasons,
    semantic_hash,
    validate_semantic_plan_draft,
    workflow_precondition_projection,
)
from agent.harness.state import (
    STATE_SCHEMA_VERSION,
    has_incomplete_semantic_finalization,
    migrate_state,
    new_state,
    quarantine_inflight_semantic_finalization,
)
from agent.harness.contracts import ActionProposal, AdmissionResult
from agent.harness.invariants import validate_state


def _bind_product_head(state: dict) -> dict:
    thread_id = str(state.get("thread_id") or "test")
    state.setdefault("turn_context", {})["product_head"] = {
        "product_authority_id": f"authority:{thread_id}",
        "revision": int(state.get("turn_index") or 0),
        "checkpoint_thread_id": f"checkpoint-thread:{thread_id}",
        "checkpoint_id": f"checkpoint:{thread_id}",
        "state_fingerprint": semantic_hash({"thread_id": thread_id}),
    }
    return state


class SemanticPlanDraftTests(unittest.TestCase):
    def _draft(self, *, active_group: str = "opening") -> dict:
        state = new_state("draft-test", language="en")
        state["turn_index"] = 7
        state["active_group"] = active_group
        _bind_product_head(state)
        return build_semantic_plan_draft(
            state,
            original_input=(
                "Explain fake-node and change the unknown setting "
                "https://example.invalid/abcdefghijklmnopqrstuvwxyz123456"
            ),
            source_clauses=[{
                "clause_id": "clause-1",
                "text": "Explain fake-node and change the unknown setting",
                "input_shape": "prose",
            }],
            source_partition=[
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "Explain fake-node",
                    "operation": "consultation",
                    "owner_routes": [{"owner": "orientation", "group": "opening"}],
                    "reason": "read-only explanation",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-1",
                    "source_text": "change the unknown setting",
                    "operation": "unresolved",
                    "owner_routes": [],
                    "reason": "target is ambiguous",
                },
            ],
            semantic_units=[
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "Explain fake-node",
                    "disposition": "action",
                    "action_indexes": [0],
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-1",
                    "source_text": "change the unknown setting",
                    "disposition": "unresolved",
                    "action_indexes": [],
                    "reason": "target is ambiguous",
                },
            ],
            candidate_actions=[{
                "type": "answer_opening_question",
                "topic": "mode_comparison",
                "source_evidence": "Explain fake-node",
            }],
            validation=PlanCoverageResult(
                valid=False,
                errors=(),
                unresolved_clauses=("change the unknown setting",),
                unresolved_units=({
                    "unit_id": "unit-2",
                    "clause_id": "clause-1",
                    "start": 22,
                    "end": 48,
                    "source_text": "change the unknown setting",
                    "source_path": "",
                    "reason": "target is ambiguous",
                },),
            ),
        )

    def test_draft_preserves_candidates_without_admission_metadata(self) -> None:
        draft = self._draft()
        self.assertEqual(draft["status"], "awaiting_clarification")
        self.assertEqual(len(draft["candidates"]), 1)
        action = draft["candidates"][0]["action"]
        self.assertNotIn("_plan_transaction_hash", action)
        self.assertNotIn("_admission_action_id", action)
        self.assertIn("***REDACTED***", draft["original_input"])
        self.assertNotIn(
            "abcdefghijklmnopqrstuvwxyz123456",
            json.dumps(draft, ensure_ascii=False, default=str),
        )

    def test_atom_resolution_is_revision_bound_and_becomes_ready(self) -> None:
        draft = self._draft()
        atom_id = draft["active_atom_id"]
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            resolve_semantic_draft_atom(
                draft,
                draft_id=draft["draft_id"],
                revision=99,
                atom_id=atom_id,
                resolution="change the QPS profile",
            )
        resolved = resolve_semantic_draft_atom(
            draft,
            draft_id=draft["draft_id"],
            revision=draft["revision"],
            atom_id=atom_id,
            resolution="change the QPS profile",
        )
        self.assertEqual(resolved["status"], "ready_for_review")
        self.assertEqual(resolved["revision"], 2)
        self.assertEqual(resolved["active_atom_id"], "")
        validate_semantic_plan_draft(resolved)

    def test_pending_answer_is_canonicalized_to_atomic_draft_resolution(
        self,
    ) -> None:
        from agent.harness.coordinator import _semantic_draft_question
        from agent.harness.plan_coverage import segment_user_turn
        from agent.harness.semantic_admission import (
            prepare_hierarchical_candidate,
        )

        draft = self._draft(active_group="chain_identity")
        state = new_state(draft["session_id"], language="en")
        state["turn_index"] = draft["creation_turn"]
        state["active_group"] = "chain_identity"
        _bind_product_head(state)
        state["semantic_plan_draft"] = draft
        state["pending_question"] = _semantic_draft_question(draft)
        clauses = tuple(segment_user_turn("real-node"))
        candidate = {
            "actions": [{
                "type": "answer_pending",
                "answer": "real-node",
                "source_evidence": "real-node",
            }],
            "semantic_units": [{
                "unit_id": "unit-1",
                "clause_id": clauses[0].clause_id,
                "source_text": "real-node",
                "disposition": "action",
                "action_indexes": [0],
                "reason": "manual draft resolution",
            }],
        }

        prepared, validation = prepare_hierarchical_candidate(
            json.dumps(candidate),
            state,
            clauses,
            pending_choice_unit_ids=frozenset({"unit-1"}),
        )

        self.assertTrue(validation.valid, validation.errors)
        action = json.loads(prepared)["actions"][0]
        self.assertEqual(action["type"], "resolve_semantic_draft_atom")
        self.assertEqual(action["draft_id"], draft["draft_id"])
        self.assertEqual(action["revision"], draft["revision"])
        self.assertEqual(action["atom_id"], draft["active_atom_id"])
        self.assertEqual(action["resolution"], "real-node")

    def test_secret_resolution_persists_only_scoped_reference_and_hash(
        self,
    ) -> None:
        from agent.harness.contracts import handler_result_to_dict
        from agent.harness.coordinator import (
            _apply_handler_result,
            _prepare_ready_semantic_draft_finalization,
            _semantic_draft_question,
            apply_coordinator_action,
        )
        from agent.harness.secret_refs import clear_secret_references

        draft = self._draft()
        state = new_state(draft["session_id"], language="en")
        state["turn_index"] = draft["creation_turn"]
        _bind_product_head(state)
        state["semantic_plan_draft"] = deepcopy(draft)
        state["pending_question"] = _semantic_draft_question(draft)
        token = "abcdefghijklmnopqrstuvwxyz123456"
        endpoint = f"https://rpc.example.invalid/{token}"
        result = apply_coordinator_action(
            state,
            ActionProposal(
                action_id="resolve-secret",
                action_type="resolve_semantic_draft_atom",
                arguments={
                    "draft_id": draft["draft_id"],
                    "revision": draft["revision"],
                    "atom_id": draft["active_atom_id"],
                    "resolution": endpoint,
                    "source_evidence": endpoint,
                },
                confidence="high",
            ),
        )
        serialized = json.dumps(
            handler_result_to_dict(result),
            ensure_ascii=False,
        )
        self.assertNotIn(token, serialized)
        self.assertIn("semantic-secret:", serialized)
        committed = _apply_handler_result(
            state,
            result,
            owner="coordinator",
        )
        durable = json.dumps(
            committed["semantic_plan_draft"],
            ensure_ascii=False,
        )
        self.assertNotIn(token, durable)
        self.assertEqual(
            semantic_draft_staleness_reasons(
                committed["semantic_plan_draft"],
                committed,
            ),
            (),
            {
                "expected": workflow_precondition_projection(state),
                "actual": workflow_precondition_projection(committed),
            },
        )
        clear_secret_references()
        self.assertEqual(
            semantic_draft_staleness_reasons(
                committed["semantic_plan_draft"],
                committed,
            ),
            (),
        )
        missing = missing_semantic_draft_secret_bindings(
            committed["semantic_plan_draft"]
        )
        self.assertEqual(len(missing), 1)
        transition = _prepare_ready_semantic_draft_finalization(committed)
        self.assertEqual(
            transition,
            {
                "phase": "compose",
                "reason": "semantic_draft_secret_reentry_required",
            },
        )
        question = committed["pending_question"]
        self.assertEqual(
            question["manual_action"]["type"],
            "reenter_secret_reference",
        )
        from agent.harness.questions import validate_pending_question_contract

        tampered_question = deepcopy(question)
        tampered_question["secret_reentry_binding"]["value_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "integrity hash"):
            validate_pending_question_contract(tampered_question)
        turn_tamper = deepcopy(question)
        turn_tamper["created_turn_index"] = (
            int(turn_tamper.get("created_turn_index") or 0) + 1
        )
        with self.assertRaisesRegex(ValueError, "integrity hash"):
            validate_pending_question_contract(turn_tamper)
        validate_state(committed)
        binding = dict(question["secret_reentry_binding"])
        from types import SimpleNamespace
        from agent.harness.coordinator import _prepare_pending_answer_result

        routed = _prepare_pending_answer_result(
            committed,
            {
                "answer": endpoint,
                "selected_value": endpoint,
                "source_evidence": endpoint,
            },
            SimpleNamespace(action_id="route-secret-answer"),
        )
        self.assertEqual(
            [item["type"] for item in routed.followup_actions],
            ["reenter_secret_reference"],
            routed,
        )
        self.assertEqual(
            routed.followup_actions[0]["value_hash"],
            binding["value_hash"],
        )
        rejected = apply_coordinator_action(
            committed,
            ActionProposal(
                action_id="restore-wrong-secret",
                action_type="reenter_secret_reference",
                arguments={
                    **binding,
                    "secret_value": "https://rpc.example.invalid/wrong-secret",
                    "source_evidence": "replacement attempt",
                },
                confidence="high",
            ),
        )
        self.assertIsNotNone(rejected.blocker)
        self.assertTrue(
            missing_semantic_draft_secret_bindings(
                committed["semantic_plan_draft"]
            )
        )
        restored = apply_coordinator_action(
            committed,
            ActionProposal(
                action_id="restore-secret",
                action_type="reenter_secret_reference",
                arguments={
                    **binding,
                    "secret_value": endpoint,
                    "source_evidence": endpoint,
                },
                confidence="high",
            ),
        )
        self.assertIsNone(restored.blocker)
        committed = _apply_handler_result(
            committed,
            restored,
            owner="coordinator",
        )
        self.assertFalse(
            missing_semantic_draft_secret_bindings(
                committed["semantic_plan_draft"]
            )
        )
        self.assertEqual(
            workflow_precondition_projection(committed),
            workflow_precondition_projection(state),
        )
        self.assertEqual(
            semantic_draft_staleness_reasons(
                committed["semantic_plan_draft"],
                committed,
            ),
            (),
        )
        transition = _prepare_ready_semantic_draft_finalization(committed)
        self.assertEqual(transition["phase"], "plan", transition)

    def test_initial_input_secret_is_referenced_until_domain_invocation(
        self,
    ) -> None:
        from types import SimpleNamespace

        from agent.harness.coordinator import (
            _materialize_semantic_secret_bindings,
            _semantic_secret_bindings,
        )
        from agent.harness.hierarchical_planner import (
            _replace_secret_values,
            _source_secret_projection,
        )
        from agent.harness.secret_refs import store_secret_reference

        token = "abcdefghijklmnopqrstuvwxyz123456"
        endpoint = f"https://rpc.example.invalid/{token}"
        replacements, bindings, raw_by_reference = (
            _source_secret_projection(
                f"Use endpoint {endpoint} and ask me for the QPS profile."
            )
        )
        projected = _replace_secret_values(endpoint, replacements)
        reference = bindings[0]["reference"]
        self.assertNotIn(token, projected)
        self.assertIn(reference, projected)
        draft = {
            "draft_id": "draft-source-secret",
            "source_secret_bindings": bindings,
            "unresolved_atoms": [],
        }
        store_secret_reference(
            raw_by_reference[reference],
            draft_id=draft["draft_id"],
            atom_id=bindings[0]["atom_id"],
            reference=reference,
        )
        action = {
            "type": "set_rpc_endpoint",
            "endpoint": projected,
            "source_evidence": projected,
        }
        action_bindings = _semantic_secret_bindings(draft, action)
        durable = json.dumps(
            {"action": action, "bindings": action_bindings},
            ensure_ascii=False,
        )
        self.assertNotIn(token, durable)
        materialized = _materialize_semantic_secret_bindings(
            deepcopy(action),
            SimpleNamespace(
                admission_metadata={
                    "semantic_secret_bindings": action_bindings,
                }
            ),
        )
        self.assertEqual(materialized["endpoint"], endpoint)
        self.assertNotIn(token, materialized["source_evidence"])

    def test_secret_reentry_projection_is_contract_driven_and_turn_scoped(
        self,
    ) -> None:
        from agent.harness.secret_refs import (
            project_secret_input,
            release_unowned_input_secret_bindings,
            resolve_secret_reference,
        )

        raw = "short-secret-that-lexical-detection-would-miss"
        projected, bindings = project_secret_input(
            raw,
            scope_id="turn:test:1",
            force_secret=True,
        )
        self.assertTrue(projected.startswith("semantic-secret:"))
        binding = bindings[0]
        self.assertEqual(
            resolve_secret_reference(
                projected,
                draft_id=binding["draft_id"],
                atom_id=binding["atom_id"],
                expected_hash=binding["value_hash"],
            ),
            raw,
        )
        release_unowned_input_secret_bindings(bindings, {})
        self.assertIsNone(
            resolve_secret_reference(
                projected,
                draft_id=binding["draft_id"],
                atom_id=binding["atom_id"],
                expected_hash=binding["value_hash"],
            )
        )

    def test_turn_secret_cleanup_preserves_material_transferred_to_state(
        self,
    ) -> None:
        from agent.harness.secret_refs import (
            project_secret_input,
            reconcile_state_secret_bindings,
            register_state_secret_bindings,
            release_unowned_input_secret_bindings,
            resolve_secret_reference,
        )

        endpoint = "https://rpc.example.invalid/transferred-secret-token"
        projected, bindings = project_secret_input(
            endpoint,
            scope_id="turn:transfer:1",
            force_secret=True,
        )
        state = new_state("secret-transfer", language="en")
        state["confirmed_config"]["LOCAL_RPC_URL"] = projected
        register_state_secret_bindings(state, bindings)
        reconcile_state_secret_bindings(state)

        release_unowned_input_secret_bindings(bindings, state)

        binding = state["secret_bindings"][0]
        self.assertEqual(
            resolve_secret_reference(
                projected,
                draft_id=binding["scope_id"],
                atom_id=binding["atom_id"],
                expected_hash=binding["value_hash"],
            ),
            endpoint,
        )

    def test_state_secret_registry_covers_durable_evidence_and_action_roots(
        self,
    ) -> None:
        from agent.harness.secret_refs import (
            project_secret_input,
            reconcile_state_secret_bindings,
            register_state_secret_bindings,
            resolve_secret_reference,
        )

        raw = "Authorization: Bearer durable-evidence-token-123456789"
        projected, bindings = project_secret_input(
            raw,
            scope_id="turn:evidence:1",
        )
        state = new_state("durable-evidence", language="en")
        state["evidence_buffer"] = [{"content": projected}]
        register_state_secret_bindings(state, bindings)
        reconcile_state_secret_bindings(state)

        self.assertEqual(len(state["secret_bindings"]), 1)
        binding = state["secret_bindings"][0]
        self.assertEqual(
            resolve_secret_reference(
                binding["reference"],
                draft_id=binding["scope_id"],
                atom_id=binding["atom_id"],
                expected_hash=binding["value_hash"],
            ),
            "durable-evidence-token-123456789",
        )

    def test_durable_execution_plan_retains_only_secret_reference(
        self,
    ) -> None:
        from agent.harness.invariants import StateInvariantError, validate_state
        from agent.harness.secret_refs import (
            materialize_state_secret_references,
            project_state_secret_values,
            register_state_secret_bindings,
            store_secret_reference,
        )

        endpoint = "https://user:plan-secret@rpc.example.invalid"
        reference, value_hash = store_secret_reference(
            endpoint,
            draft_id="turn:durable-plan",
            atom_id="input-secret-1",
        )
        state = new_state("durable-plan", language="en")
        state["confirmed_config"]["LOCAL_RPC_URL"] = reference
        register_state_secret_bindings(state, [{
            "reference": reference,
            "scope_id": "turn:durable-plan",
            "atom_id": "input-secret-1",
            "value_hash": value_hash,
        }])
        raw_plan = {
            "execution": {
                "environment": {
                    "LOCAL_RPC_URL": endpoint,
                },
            },
        }
        state["plan"] = raw_plan
        with self.assertRaisesRegex(
            StateInvariantError,
            "unprojected secret material",
        ):
            validate_state(state)

        state["plan"] = project_state_secret_values(raw_plan, state)
        validate_state(state)
        self.assertEqual(
            state["plan"]["execution"]["environment"]["LOCAL_RPC_URL"],
            reference,
        )
        self.assertEqual(
            materialize_state_secret_references(state["plan"], state),
            raw_plan,
        )

    def test_control_metadata_may_name_a_sensitive_field(self) -> None:
        from agent.harness.secret_refs import raw_secret_paths_in_state

        state = new_state("control-metadata", language="en")
        scope = "chain_auxiliary_endpoints::RPC_API_KEY::option:skip"
        state["action_queue"] = [{
            "action_id": "action-1",
            "type": "answer_pending",
            "_plan_scope": scope,
        }]
        state["selected_action"] = {
            "plan_scope": scope,
            "idempotency_key": f"{scope}:action-1",
            "admission_metadata": {
                "semantic_secret_bindings": [{
                    "draft_id": scope,
                    "scope_id": scope,
                    "atom_id": "RPC_API_KEY",
                }],
            },
        }
        state["current_action"] = {
            "_plan_scope": scope,
        }

        self.assertEqual(raw_secret_paths_in_state(state), ())
        state["current_action"]["source_evidence"] = (
            "Authorization: Bearer durable-evidence-token-123456789"
        )
        self.assertEqual(
            raw_secret_paths_in_state(state),
            ("current_action.source_evidence",),
        )

    def test_durable_endpoint_secret_survives_commit_and_requires_reentry_after_restart(
        self,
    ) -> None:
        from types import SimpleNamespace

        from agent.harness.coordinator import (
            _apply_handler_result,
            _prepare_pending_answer_result,
            apply_coordinator_action,
            prepare_turn_step,
        )
        from agent.harness.domains.execution_runtime import _prepare_kwargs
        from agent.harness.secret_refs import (
            clear_secret_references,
            missing_state_secret_bindings,
            reconcile_state_secret_bindings,
            register_state_secret_bindings,
            resolve_secret_reference,
            store_secret_reference,
        )

        endpoint = "https://rpc.example.invalid/durable-secret-token"
        reference, value_hash = store_secret_reference(
            endpoint,
            draft_id="turn:durable",
            atom_id="input-secret-1",
        )
        state = new_state("durable-secret", language="en")
        state["target_mode"] = "real-node"
        state["confirmed_config"]["LOCAL_RPC_URL"] = reference
        binding = {
            "reference": reference,
            "draft_id": "turn:durable",
            "atom_id": "input-secret-1",
            "value_hash": value_hash,
        }
        register_state_secret_bindings(state, [binding])
        reconcile_state_secret_bindings(state)
        self.assertEqual(_prepare_kwargs(state)["target_rpc_url"], endpoint)

        clear_secret_references()
        self.assertEqual(len(missing_state_secret_bindings(state)), 1)
        state["last_user_input"] = "continue"
        prepared = prepare_turn_step(state)
        question = prepared["pending_question"]
        self.assertEqual(
            question["manual_action"]["type"],
            "reenter_secret_reference",
        )
        self.assertEqual(
            question["secret_reentry_binding"]["owner_kind"],
            "durable_state",
        )
        routed = _prepare_pending_answer_result(
            prepared,
            {
                "answer": endpoint,
                "selected_value": endpoint,
                "source_evidence": endpoint,
            },
            SimpleNamespace(action_id="durable-reentry-answer"),
        )
        followup = routed.followup_actions[0]
        restored = apply_coordinator_action(
            prepared,
            ActionProposal(
                action_id="durable-reentry",
                action_type=followup["type"],
                arguments={
                    key: value
                    for key, value in followup.items()
                    if key != "type"
                },
                confidence="high",
            ),
        )
        self.assertIsNone(restored.blocker)
        committed = _apply_handler_result(
            prepared,
            restored,
            owner="coordinator",
        )
        durable_binding = committed["secret_bindings"][0]
        self.assertEqual(
            resolve_secret_reference(
                reference,
                draft_id=durable_binding["scope_id"],
                atom_id=durable_binding["atom_id"],
                expected_hash=durable_binding["value_hash"],
            ),
            endpoint,
        )
        self.assertEqual(_prepare_kwargs(committed)["target_rpc_url"], endpoint)

        committed["confirmed_config"].pop("LOCAL_RPC_URL")
        reconcile_state_secret_bindings(committed)
        self.assertEqual(committed["secret_bindings"], [])
        self.assertIsNone(
            resolve_secret_reference(
                reference,
                draft_id=durable_binding["scope_id"],
                atom_id=durable_binding["atom_id"],
                expected_hash=durable_binding["value_hash"],
            )
        )

    def test_semantic_question_hash_rejects_prompt_and_option_drift(self) -> None:
        from agent.harness.coordinator import _semantic_draft_question
        from agent.harness.questions import validate_pending_question_contract

        question = _semantic_draft_question(self._draft())
        validate_pending_question_contract(question)
        changed_prompt = deepcopy(question)
        changed_prompt["prompt_ref"]["arguments"]["source"] = "different atom"
        with self.assertRaisesRegex(ValueError, "integrity hash"):
            validate_pending_question_contract(changed_prompt)
        missing_cancel = deepcopy(question)
        missing_cancel["options"] = []
        with self.assertRaisesRegex(ValueError, "integrity hash"):
            validate_pending_question_contract(missing_cancel)
        changed_queue_policy = deepcopy(question)
        changed_queue_policy["resume_action_queue"] = not bool(
            question.get("resume_action_queue")
        )
        with self.assertRaisesRegex(ValueError, "integrity hash"):
            validate_pending_question_contract(changed_queue_policy)

    def test_candidate_identity_and_authority_are_revalidated(self) -> None:
        draft = self._draft()
        tampered = deepcopy(draft)
        tampered["candidates"][0]["action"]["topic"] = "supported_chains"
        with self.assertRaisesRegex(ValueError, "candidate identity mismatch"):
            validate_semantic_plan_draft(tampered)

        tampered = deepcopy(draft)
        tampered["candidates"][0]["owner"] = "execution"
        tampered["candidates"][0]["candidate_id"] = semantic_hash({
            "action": tampered["candidates"][0]["action"],
            "source_atom_ids": tuple(
                tampered["candidates"][0]["source_atom_ids"]
            ),
            "owner": "execution",
            "group": tampered["candidates"][0]["group"],
        })
        with self.assertRaisesRegex(ValueError, "candidate owner mismatch"):
            validate_semantic_plan_draft(tampered)

        tampered = deepcopy(draft)
        tampered["candidates"][0]["contract_hashes"]["action_registry"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "contract hashes mismatch"):
            validate_semantic_plan_draft(tampered)

        tampered = deepcopy(draft)
        tampered["candidates"][0]["validation_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "validation hash mismatch"):
            validate_semantic_plan_draft(tampered)

    def test_lifecycle_receipts_prove_resolution_status_and_revision(self) -> None:
        draft = self._draft()
        ready = resolve_semantic_draft_atom(
            draft,
            draft_id=draft["draft_id"],
            revision=draft["revision"],
            atom_id=draft["active_atom_id"],
            resolution="QPS profile",
        )
        tampered = deepcopy(ready)
        tampered["unresolved_atoms"][0]["resolution"] = "another group"
        with self.assertRaisesRegex(
            ValueError,
            "resolution (hash mismatch|does not match lifecycle)",
        ):
            validate_semantic_plan_draft(tampered)

        tampered = deepcopy(ready)
        tampered["revision"] += 1
        with self.assertRaisesRegex(ValueError, "revision does not match lifecycle"):
            validate_semantic_plan_draft(tampered)

        tampered = deepcopy(ready)
        tampered["status"] = "stale"
        with self.assertRaisesRegex(ValueError, "status does not match lifecycle"):
            validate_semantic_plan_draft(tampered)

    def test_owner_compilation_gap_is_missing_user_evidence(self) -> None:
        state = new_state("draft-missing-evidence", language="en")
        _bind_product_head(state)
        draft = build_semantic_plan_draft(
            state,
            original_input="configure the endpoint",
            source_clauses=[{
                "clause_id": "clause-1",
                "text": "configure the endpoint",
                "input_shape": "prose",
            }],
            source_partition=[{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "configure the endpoint",
                "operation": "domain_request",
                "owner_routes": [{
                    "owner": "rpc_endpoint",
                    "group": "endpoint_process",
                }],
                "reason": "endpoint configuration request",
            }],
            semantic_units=[{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "configure the endpoint",
                "disposition": "unresolved",
                "action_indexes": [],
                "reason": "the endpoint value is missing",
            }],
            candidate_actions=[],
            validation=PlanCoverageResult(
                valid=False,
                errors=(),
                unresolved_clauses=("configure the endpoint",),
                unresolved_units=({
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "configure the endpoint",
                    "source_path": "",
                    "reason": "the endpoint value is missing",
                },),
            ),
        )
        self.assertEqual(
            draft["unresolved_atoms"][0]["reason"],
            "missing_user_evidence",
        )

    def test_staleness_covers_session_schema_registry_and_workflow(self) -> None:
        draft = self._draft()
        state = new_state("draft-test", language="en")
        state["turn_index"] = draft["creation_turn"]
        _bind_product_head(state)
        self.assertEqual(semantic_draft_staleness_reasons(draft, state), ())

        changed = deepcopy(state)
        changed["session"]["id"] = "different"
        changed["schema_version"] += 1
        changed["confirmed_config"]["CLOUD_REGION"] = "us-test1"
        self.assertEqual(
            semantic_draft_staleness_reasons(draft, changed),
            (
                "session_changed",
                "checkpoint_schema_changed",
                "workflow_precondition_changed",
            ),
        )
        stale = mark_semantic_plan_draft_stale(
            draft,
            reasons=("session_changed",),
        )
        self.assertEqual(stale["status"], "stale")
        self.assertEqual(stale["active_atom_id"], "")

    def test_staleness_is_bound_to_product_head_lineage(self) -> None:
        draft = self._draft()
        state = new_state("draft-test", language="en")
        state["turn_index"] = draft["creation_turn"]
        _bind_product_head(state)

        later = deepcopy(state)
        later["turn_context"]["product_head"]["revision"] += 1
        later["turn_context"]["product_head"]["checkpoint_id"] = "later"
        later["turn_context"]["product_head"]["state_fingerprint"] = semantic_hash(
            {"checkpoint": "later"}
        )
        self.assertEqual(semantic_draft_staleness_reasons(draft, later), ())

        replaced = deepcopy(state)
        replaced["turn_context"]["product_head"]["checkpoint_id"] = "replacement"
        self.assertEqual(
            semantic_draft_staleness_reasons(draft, replaced),
            ("product_head_checkpoint_replaced",),
        )

        other_authority = deepcopy(state)
        other_authority["turn_context"]["product_head"][
            "product_authority_id"
        ] = "other"
        self.assertEqual(
            semantic_draft_staleness_reasons(draft, other_authority),
            ("product_head_authority_changed",),
        )

        regressed = deepcopy(state)
        regressed["turn_context"]["product_head"]["revision"] = (
            draft["source_checkpoint_revision"] - 1
        )
        self.assertEqual(
            semantic_draft_staleness_reasons(draft, regressed),
            ("product_head_revision_regressed",),
        )

    def test_multiple_atoms_support_previous_and_cancel_without_business_state(self) -> None:
        from agent.harness.coordinator import (
            _semantic_draft_question,
            apply_coordinator_action,
        )

        state = new_state("draft-multiple", language="en")
        _bind_product_head(state)
        draft = build_semantic_plan_draft(
            state,
            original_input="change two unknown settings",
            source_clauses=[{
                "clause_id": "clause-1",
                "text": "change two unknown settings",
                "input_shape": "prose",
            }],
            source_partition=[
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "first unknown",
                    "operation": "unresolved",
                    "owner_routes": [],
                    "reason": "first target is ambiguous",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-1",
                    "source_text": "second unknown",
                    "operation": "unresolved",
                    "owner_routes": [],
                    "reason": "second target is ambiguous",
                },
            ],
            semantic_units=[
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "first unknown",
                    "disposition": "unresolved",
                    "action_indexes": [],
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-1",
                    "source_text": "second unknown",
                    "disposition": "unresolved",
                    "action_indexes": [],
                },
            ],
            candidate_actions=[],
            validation=PlanCoverageResult(
                valid=False,
                errors=(),
                unresolved_clauses=("first unknown", "second unknown"),
                unresolved_units=(
                    {
                        "unit_id": "unit-1",
                        "clause_id": "clause-1",
                        "source_text": "first unknown",
                        "source_path": "",
                        "reason": "first target is ambiguous",
                    },
                    {
                        "unit_id": "unit-2",
                        "clause_id": "clause-1",
                        "source_text": "second unknown",
                        "source_path": "",
                        "reason": "second target is ambiguous",
                    },
                ),
            ),
        )
        first_atom = draft["active_atom_id"]
        initial_question = _semantic_draft_question(draft)
        second = resolve_semantic_draft_atom(
            draft,
            draft_id=draft["draft_id"],
            revision=draft["revision"],
            atom_id=first_atom,
            resolution="QPS profile",
        )
        self.assertNotEqual(second["active_atom_id"], first_atom)
        question = _semantic_draft_question(second)
        previous_action = question["options"][0]["action"]
        self.assertEqual(previous_action["type"], "previous_semantic_draft_atom")
        self.assertEqual(previous_action["draft_id"], second["draft_id"])
        self.assertEqual(previous_action["revision"], second["revision"])
        self.assertEqual(previous_action["atom_id"], first_atom)
        for option in question["options"]:
            action = dict(option["action"])
            tampered_state = deepcopy(state)
            tampered_state["semantic_plan_draft"] = deepcopy(second)
            tampered_state["pending_question"] = deepcopy(question)
            tampered_state["pending_question"]["prompt_ref"]["arguments"][
                "source"
            ] = "tampered source"
            result = apply_coordinator_action(
                tampered_state,
                ActionProposal(
                    action_id=f"stale-{action['type']}",
                    action_type=str(action.pop("type")),
                    arguments=action,
                    confidence="high",
                ),
            )
            self.assertIsNone(result.blocker)
            self.assertEqual(
                result.semantic_draft_command.operation,
                "invalidate",
            )
        restart_state = deepcopy(state)
        restart_state["semantic_plan_draft"] = deepcopy(second)
        restart_state["pending_question"] = deepcopy(question)
        restarted = migrate_state(
            restart_state,
            thread_id="draft-multiple",
            language="en",
            session_purpose="user",
        )
        old_cancel = dict(initial_question["options"][-1]["action"])
        old_result = apply_coordinator_action(
            restarted,
            ActionProposal(
                action_id="old-cancel-after-restart",
                action_type=str(old_cancel.pop("type")),
                arguments=old_cancel,
                confidence="high",
            ),
        )
        self.assertIsNotNone(old_result.blocker)
        with self.assertRaisesRegex(ValueError, "previous atom binding mismatch"):
            reopen_previous_semantic_draft_atom(
                second,
                draft_id=second["draft_id"],
                revision=second["revision"],
                atom_id=second["active_atom_id"],
            )
        reopened = reopen_previous_semantic_draft_atom(
            second,
            draft_id=second["draft_id"],
            revision=second["revision"],
            atom_id=first_atom,
        )
        self.assertEqual(reopened["active_atom_id"], first_atom)
        self.assertFalse(reopened["unresolved_atoms"][0]["resolution"])
        cancelled = cancel_semantic_plan_draft(reopened, reason="user_cancelled")
        self.assertEqual(cancelled["status"], "cancelled")

    def test_finalization_receipt_binds_only_new_transaction_actions(self) -> None:
        draft = self._draft()
        ready = resolve_semantic_draft_atom(
            draft,
            draft_id=draft["draft_id"],
            revision=draft["revision"],
            atom_id=draft["active_atom_id"],
            resolution="QPS profile",
        )
        actions = [
            {
                "type": "answer_opening_question",
                "topic": "mode_comparison",
                "source_evidence": "Explain fake-node",
                "action_id": "new-action-1",
                "_plan_transaction_hash": "a" * 64,
                "_source_unit_ids": ["unit-1"],
            },
            {
                "type": "change_group",
                "group": "qps_profile",
                "source_evidence": "change the QPS profile",
                "action_id": "new-action-2",
                "_plan_transaction_hash": "a" * 64,
            },
        ]
        receipt = build_semantic_draft_finalization_receipt(ready, actions)
        self.assertEqual(
            receipt["final_action_ids"],
            ("new-action-1", "new-action-2"),
        )
        self.assertEqual(receipt["admission_transaction_hash"], "a" * 64)
        self.assertEqual(len(receipt["receipt_hash"]), 64)

    def test_finalization_rejects_omitted_frozen_candidate(self) -> None:
        draft = self._draft()
        ready = resolve_semantic_draft_atom(
            draft,
            draft_id=draft["draft_id"],
            revision=draft["revision"],
            atom_id=draft["active_atom_id"],
            resolution="QPS profile",
        )
        actions = [{
            "type": "change_group",
            "group": "qps_profile",
            "source_evidence": "change the QPS profile",
            "action_id": "replacement-only",
            "_plan_transaction_hash": "c" * 64,
        }]
        with self.assertRaisesRegex(
            ValueError,
            "omitted or changed candidate",
        ):
            build_semantic_draft_finalization_receipt(ready, actions)
        source_detached = [{
            "type": "answer_opening_question",
            "topic": "mode_comparison",
            "source_evidence": "Explain fake-node",
            "action_id": "source-detached",
            "_plan_transaction_hash": "c" * 64,
            "_source_unit_ids": [],
        }]
        with self.assertRaisesRegex(
            ValueError,
            "omitted or changed candidate",
        ):
            build_semantic_draft_finalization_receipt(
                ready,
                source_detached,
            )

    def test_admission_recomputes_final_plan_hash(self) -> None:
        from agent.harness.admission import _validate_admission_transaction
        from agent.harness.invariants import StateInvariantError

        draft = self._draft()
        ready = resolve_semantic_draft_atom(
            draft,
            draft_id=draft["draft_id"],
            revision=draft["revision"],
            atom_id=draft["active_atom_id"],
            resolution="QPS profile",
        )
        actions = [{
            "type": "answer_opening_question",
            "topic": "mode_comparison",
            "source_evidence": "Explain fake-node",
            "action_id": "new-action-1",
            "_plan_transaction_hash": "d" * 64,
            "_source_unit_ids": ["unit-1"],
        }]
        receipt = build_semantic_draft_finalization_receipt(ready, actions)
        actions[0]["_semantic_draft_finalization_receipt"] = receipt
        state = new_state(ready["session_id"], language="en")
        _validate_admission_transaction(
            state,
            actions,
            current_submission=True,
        )
        actions[0]["topic"] = "supported_chains"
        with self.assertRaisesRegex(
            StateInvariantError,
            "plan hash mismatch",
        ):
            _validate_admission_transaction(
                state,
                actions,
                current_submission=True,
            )

    def test_v18_migration_clears_inflight_semantic_state(self) -> None:
        raw = deepcopy(new_state("legacy"))
        raw["schema_version"] = 18
        raw["semantic_planning"] = {
            "contract_version": 1,
            "status": "review_plan",
        }
        raw["semantic_plan_draft"] = self._draft()
        migrated = migrate_state(
            raw,
            thread_id="legacy",
            language="en",
            session_purpose="user",
        )
        self.assertEqual(migrated["semantic_planning"], {})
        self.assertEqual(migrated["semantic_plan_draft"], {})
        self.assertTrue(any(
            item.get("event") == "semantic_draft_migration_invalidated"
            for item in migrated["audit_events"]
        ))

    def test_v19_migration_clears_bound_drafts_and_preserves_business_state(
        self,
    ) -> None:
        from agent.harness.coordinator import _semantic_draft_question

        awaiting = self._draft()
        ready = resolve_semantic_draft_atom(
            awaiting,
            draft_id=awaiting["draft_id"],
            revision=awaiting["revision"],
            atom_id=awaiting["active_atom_id"],
            resolution="QPS profile",
        )
        for draft in (awaiting, ready):
            with self.subTest(status=draft["status"]):
                raw = new_state(draft["session_id"], language="en")
                raw["schema_version"] = 19
                raw["confirmed_config"]["CLOUD_REGION"] = "us-test1"
                raw["semantic_plan_draft"] = deepcopy(draft)
                if draft["status"] == "awaiting_clarification":
                    raw["pending_question"] = _semantic_draft_question(draft)
                migrated = migrate_state(
                    raw,
                    thread_id=draft["session_id"],
                    language="en",
                    session_purpose="user",
                )
                self.assertEqual(
                    migrated["schema_version"],
                    STATE_SCHEMA_VERSION,
                )
                self.assertEqual(migrated["semantic_plan_draft"], {})
                self.assertEqual(migrated["pending_question"], {})
                self.assertEqual(
                    migrated["confirmed_config"]["CLOUD_REGION"],
                    "us-test1",
                )
                migration_events = [
                    item
                    for item in migrated["audit_events"]
                    if item.get("event")
                    == "semantic_draft_migration_invalidated"
                    and item.get("from_schema_version") == 19
                    and item.get("to_schema_version") == STATE_SCHEMA_VERSION
                ]
                self.assertEqual(len(migration_events), 1)
                restarted = migrate_state(
                    deepcopy(migrated),
                    thread_id=draft["session_id"],
                    language="en",
                    session_purpose="user",
                )
                self.assertEqual(
                    restarted["audit_events"],
                    migrated["audit_events"],
                )

    def test_v20_migration_quarantines_all_inflight_draft_authority(
        self,
    ) -> None:
        from agent.harness.coordinator import _semantic_draft_question

        draft = self._draft()
        raw = new_state(draft["session_id"], language="en")
        raw["schema_version"] = 20
        raw["confirmed_config"]["CLOUD_REGION"] = "us-test1"
        raw["semantic_plan_draft"] = deepcopy(draft)
        raw["pending_question"] = _semantic_draft_question(draft)
        raw["action_queue"] = [{
            "action_id": "old-finalized",
            "action_type": "choose_chain",
            "admission_metadata": {
                "semantic_draft_finalization_receipt": {"legacy": True},
            },
        }]
        raw["selected_action"] = deepcopy(raw["action_queue"][0])
        raw["audit_events"].extend([
            {
                "event": "semantic_draft_finalized",
                "draft_id": draft["draft_id"],
                "final_action_ids": ["old-finalized"],
            },
            {
                "event": "semantic_draft_finalization_action_applied",
                "draft_id": draft["draft_id"],
                "action_id": "old-finalized",
            },
        ])
        raw["turn_context"] = {
            "semantic_draft_finalization_receipt": {"legacy": True},
        }
        migrated = migrate_state(
            raw,
            thread_id=draft["session_id"],
            language="en",
            session_purpose="user",
        )
        self.assertEqual(migrated["schema_version"], STATE_SCHEMA_VERSION)
        self.assertEqual(migrated["semantic_plan_draft"], {})
        self.assertEqual(migrated["pending_question"], {})
        self.assertEqual(migrated["action_queue"], [])
        self.assertEqual(migrated["selected_action"], {})
        self.assertFalse(any(
            item.get("event") in {
                "semantic_draft_finalized",
                "semantic_draft_finalization_action_applied",
            }
            for item in migrated["audit_events"]
        ))
        self.assertNotIn(
            "semantic_draft_finalization_receipt",
            migrated["turn_context"],
        )
        self.assertEqual(migrated["confirmed_config"], {})
        self.assertEqual(
            migrated["checkpoint_recovery"]["error_type"],
            "PartialSemanticFinalizationCheckpoint",
        )
        self.assertTrue(any(
            item.get("event")
            == "checkpoint_partial_semantic_finalization_quarantined"
            and item.get("from_schema_version") == 20
            and item.get("to_schema_version") == STATE_SCHEMA_VERSION
            and item.get("source_audit_hash")
            and item.get("transactions")
            for item in migrated["audit_events"]
        ))
        validate_state(migrated)

    def test_current_checkpoint_preserves_exact_bound_draft(self) -> None:
        from agent.harness.coordinator import _semantic_draft_question
        from agent.harness.invariants import validate_state

        raw = new_state("current")
        _bind_product_head(raw)
        draft = build_semantic_plan_draft(
            raw,
            original_input="change an unknown setting",
            source_clauses=[{
                "clause_id": "clause-1",
                "text": "change an unknown setting",
                "input_shape": "prose",
            }],
            source_partition=[{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "change an unknown setting",
                "operation": "unresolved",
                "owner_routes": [],
                "reason": "target is ambiguous",
            }],
            semantic_units=[{
                "unit_id": "unit-1",
                "clause_id": "clause-1",
                "source_text": "change an unknown setting",
                "disposition": "unresolved",
                "action_indexes": [],
            }],
            candidate_actions=[],
            validation=PlanCoverageResult(
                valid=False,
                errors=(),
                unresolved_clauses=("change an unknown setting",),
                unresolved_units=({
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "change an unknown setting",
                    "source_path": "",
                    "reason": "target is ambiguous",
                },),
            ),
        )
        raw["semantic_plan_draft"] = draft
        raw["pending_question"] = _semantic_draft_question(draft)
        restored = migrate_state(
            deepcopy(raw),
            thread_id="current",
            language="en",
            session_purpose="user",
        )
        self.assertEqual(restored["semantic_plan_draft"], draft)
        validate_state(restored)

    def test_current_checkpoint_invalidates_draft_after_authority_drift(
        self,
    ) -> None:
        from agent.harness.coordinator import _semantic_draft_question
        from agent.harness.invariants import validate_state

        raw = new_state("authority-drift")
        raw["turn_index"] = 7
        _bind_product_head(raw)
        draft = self._draft()
        raw["thread_id"] = draft["session_id"]
        raw["session"]["id"] = draft["session_id"]
        raw["semantic_plan_draft"] = deepcopy(draft)
        raw["pending_question"] = _semantic_draft_question(draft)
        with patch(
            "agent.harness.semantic_drafts.action_registry_contract_hash",
            return_value="0" * 64,
        ):
            restored = migrate_state(
                raw,
                thread_id=draft["session_id"],
                language="en",
                session_purpose="user",
            )
        self.assertEqual(restored["semantic_plan_draft"]["status"], "stale")
        self.assertEqual(restored["pending_question"], {})
        self.assertTrue(any(
            item.get("event") == "semantic_draft_authority_invalidated"
            for item in restored["audit_events"]
        ))
        validate_state(restored)

    def test_current_checkpoint_preserves_terminal_and_ready_lifecycle_states(
        self,
    ) -> None:
        from agent.harness.invariants import validate_state

        base = self._draft()
        ready = resolve_semantic_draft_atom(
            base,
            draft_id=base["draft_id"],
            revision=base["revision"],
            atom_id=base["active_atom_id"],
            resolution="QPS profile",
        )
        lifecycle_states = (
            ready,
            mark_semantic_plan_draft_stale(
                ready,
                reasons=("workflow_precondition_changed",),
            ),
            cancel_semantic_plan_draft(base, reason="user_cancelled"),
        )
        for draft in lifecycle_states:
            with self.subTest(status=draft["status"]):
                raw = new_state(draft["session_id"], language="en")
                raw["turn_index"] = draft["creation_turn"]
                raw["semantic_plan_draft"] = draft
                restored = migrate_state(
                    deepcopy(raw),
                    thread_id=draft["session_id"],
                    language="en",
                    session_purpose="user",
                )
                self.assertEqual(restored["semantic_plan_draft"], draft)
                validate_state(restored)

    def test_reset_records_and_removes_semantic_draft(self) -> None:
        from agent.harness.contracts import CheckpointCommand
        from agent.harness.coordinator import (
            _apply_checkpoint_command,
            _semantic_draft_question,
        )

        draft = self._draft()
        state = new_state(draft["session_id"], language="en")
        state["turn_index"] = draft["creation_turn"]
        state["semantic_plan_draft"] = draft
        state["pending_question"] = _semantic_draft_question(draft)
        reset = _apply_checkpoint_command(state, CheckpointCommand("reset"))
        self.assertEqual(reset["semantic_plan_draft"], {})
        self.assertEqual(reset["pending_question"], {})
        self.assertTrue(any(
            item.get("event") == "semantic_draft_reset"
            and item.get("draft_id") == draft["draft_id"]
            for item in reset["audit_events"]
        ))

    def test_coordinator_resolves_bound_atom_then_prepares_full_recompile(
        self,
    ) -> None:
        from agent.harness.coordinator import (
            _apply_handler_result,
            _prepare_ready_semantic_draft_finalization,
            _semantic_draft_question,
            apply_coordinator_action,
        )
        from agent.harness.invariants import validate_state

        state = new_state("draft-control", language="en")
        draft = self._draft()
        state["thread_id"] = draft["session_id"]
        state["session"]["id"] = draft["session_id"]
        state["turn_index"] = draft["creation_turn"]
        _bind_product_head(state)
        state["semantic_plan_draft"] = draft
        state["pending_question"] = _semantic_draft_question(draft)
        business_group_states = deepcopy(state["group_states"])
        business_active_group = state["active_group"]
        validate_state(state)

        result = apply_coordinator_action(
            state,
            ActionProposal(
                action_id="resolve-1",
                action_type="resolve_semantic_draft_atom",
                arguments={
                    "draft_id": draft["draft_id"],
                    "revision": draft["revision"],
                    "atom_id": draft["active_atom_id"],
                    "resolution": "change the QPS profile",
                    "source_evidence": "change the QPS profile",
                },
                confidence="high",
            ),
        )
        committed = _apply_handler_result(state, result, owner="coordinator")
        self.assertEqual(
            committed["semantic_plan_draft"]["status"],
            "ready_for_review",
        )
        self.assertEqual(committed["pending_question"], {})
        self.assertEqual(committed["group_states"], business_group_states)
        self.assertEqual(committed["active_group"], business_active_group)

        transition = _prepare_ready_semantic_draft_finalization(committed)
        self.assertEqual(transition["phase"], "plan")
        self.assertEqual(
            committed["turn_context"]["text"],
            draft["original_input"],
        )
        self.assertEqual(committed["action_queue"], [])
        self.assertEqual(
            committed["turn_receipt"]["submitted_input_hash"],
            draft["original_input_hash"],
        )
        validate_state(committed)

    def test_draft_command_guard_invalidates_replaced_product_head(self) -> None:
        from agent.harness.coordinator import (
            _apply_handler_result,
            _semantic_draft_question,
            apply_coordinator_action,
        )

        draft = self._draft()
        state = new_state(draft["session_id"], language="en")
        state["turn_index"] = draft["creation_turn"]
        _bind_product_head(state)
        state["turn_context"]["product_head"]["checkpoint_id"] = "replacement"
        state["semantic_plan_draft"] = deepcopy(draft)
        state["pending_question"] = _semantic_draft_question(draft)
        result = apply_coordinator_action(
            state,
            ActionProposal(
                action_id="stale-resolve",
                action_type="resolve_semantic_draft_atom",
                arguments={
                    "draft_id": draft["draft_id"],
                    "revision": draft["revision"],
                    "atom_id": draft["active_atom_id"],
                    "resolution": "QPS profile",
                    "source_evidence": "QPS profile",
                },
                confidence="high",
            ),
        )
        self.assertIsNone(result.blocker)
        self.assertTrue(result.response_fragments)
        self.assertEqual(
            result.semantic_draft_command.operation,
            "invalidate",
        )
        committed = _apply_handler_result(state, result, owner="coordinator")
        self.assertEqual(
            committed["semantic_plan_draft"]["status"],
            "stale",
        )
        self.assertEqual(committed["pending_question"], {})

    def test_draft_installation_does_not_mutate_business_workflow_state(self) -> None:
        from agent.harness.coordinator import _consume_planner_queue
        from agent.harness.invariants import validate_state

        state = new_state("draft-test", language="en")
        state["turn_index"] = 7
        state["active_group"] = "qps_profile"
        state["group_states"]["qps_profile"] = {"status": "completed"}
        draft = self._draft(active_group="qps_profile")
        before = {
            "active_group": state["active_group"],
            "group_states": deepcopy(state["group_states"]),
            "confirmed_config": deepcopy(state["confirmed_config"]),
            "action_queue": deepcopy(state["action_queue"]),
        }
        updated = _consume_planner_queue(
            state,
            {
                "actions": [],
                "semantic_units": [],
                "semantic_draft": draft,
                "reason": "semantic plan requires atom clarification",
            },
        )
        self.assertEqual(updated["active_group"], before["active_group"])
        self.assertEqual(updated["group_states"], before["group_states"])
        self.assertEqual(updated["confirmed_config"], before["confirmed_config"])
        self.assertEqual(updated["action_queue"], before["action_queue"])
        self.assertTrue(
            updated["pending_question"]["semantic_draft_binding"]
        )
        validate_state(updated)

    def test_awaiting_draft_without_bound_question_fails_closed(self) -> None:
        from agent.harness.invariants import StateInvariantError, validate_state

        state = new_state("draft-test", language="en")
        state["turn_index"] = 7
        state["semantic_plan_draft"] = self._draft()
        with self.assertRaisesRegex(
            StateInvariantError,
            "no matching pending question",
        ):
            validate_state(state)

    def test_finalization_receipt_is_carried_only_by_new_envelopes(self) -> None:
        from agent.harness.coordinator import admit_turn_step

        draft = self._draft()
        ready = resolve_semantic_draft_atom(
            draft,
            draft_id=draft["draft_id"],
            revision=draft["revision"],
            atom_id=draft["active_atom_id"],
            resolution="change the QPS profile",
        )
        state = new_state("draft-admit", language="en")
        state["thread_id"] = ready["session_id"]
        state["session"]["id"] = ready["session_id"]
        state["turn_index"] = ready["creation_turn"]
        state["semantic_plan_draft"] = ready
        state["turn_context"] = {"text": ready["original_input"]}
        action = {
            "type": "answer_opening_question",
            "topic": "mode_comparison",
            "source_evidence": "Explain fake-node",
            "_plan_transaction_hash": "b" * 64,
            "_source_unit_ids": ["unit-1"],
        }
        state["proposed_actions"] = [action]
        with (
            patch(
                "agent.harness.coordinator.validate_action_plan",
                return_value=AdmissionResult(
                    status="accepted",
                    actions=(action,),
                ),
            ),
            patch("agent.harness.coordinator._validate_admission_transaction"),
        ):
            admitted = admit_turn_step(state)
        self.assertEqual(admitted["semantic_plan_draft"], {})
        self.assertEqual(len(admitted["action_queue"]), 1)
        metadata = admitted["action_queue"][0]["admission_metadata"]
        receipt = metadata["semantic_draft_finalization_receipt"]
        self.assertEqual(
            receipt["final_action_ids"],
            (admitted["action_queue"][0]["action_id"],),
        )
        self.assertEqual(
            admitted["turn_context"]["semantic_draft_finalization_receipt"],
            receipt,
        )

    def test_external_execution_is_rejected_from_draft_finalization(self) -> None:
        from agent.harness.coordinator import admit_turn_step

        draft = self._draft()
        ready = resolve_semantic_draft_atom(
            draft,
            draft_id=draft["draft_id"],
            revision=draft["revision"],
            atom_id=draft["active_atom_id"],
            resolution="run preflight",
        )
        state = new_state("draft-external", language="en")
        state["thread_id"] = ready["session_id"]
        state["session"]["id"] = ready["session_id"]
        state["turn_index"] = ready["creation_turn"]
        state["semantic_plan_draft"] = ready
        state["turn_context"] = {"text": ready["original_input"]}
        action = {
            "type": "approve_preflight_smoke",
            "_plan_transaction_hash": "c" * 64,
        }
        state["proposed_actions"] = [action]
        with (
            patch(
                "agent.harness.coordinator.validate_action_plan",
                return_value=AdmissionResult(
                    status="accepted",
                    actions=(action,),
                ),
            ),
            patch("agent.harness.coordinator._validate_admission_transaction"),
        ):
            rejected = admit_turn_step(state)
        self.assertEqual(
            rejected["semantic_plan_draft"]["status"],
            "stale",
        )
        self.assertEqual(rejected["action_queue"], [])
        self.assertEqual(
            rejected["control"]["phase"],
            "fallback",
        )
        self.assertEqual(
            rejected.get("turn_context", {}).get("admitted_actions", []),
            [],
        )

    def test_empty_final_plan_marks_ready_draft_stale(self) -> None:
        from agent.harness.coordinator import admit_turn_step

        draft = self._draft()
        ready = resolve_semantic_draft_atom(
            draft,
            draft_id=draft["draft_id"],
            revision=draft["revision"],
            atom_id=draft["active_atom_id"],
            resolution="QPS profile",
        )
        state = new_state(ready["session_id"], language="en")
        state["turn_index"] = ready["creation_turn"]
        state["semantic_plan_draft"] = ready
        state["turn_context"] = {"text": ready["original_input"]}
        with patch(
            "agent.harness.coordinator.validate_action_plan",
            return_value=AdmissionResult(status="accepted", actions=()),
        ):
            updated = admit_turn_step(state)
        self.assertEqual(
            updated["semantic_plan_draft"]["status"],
            "stale",
        )
        self.assertEqual(updated["action_queue"], [])

    def test_finalization_clears_draft_before_execution(self) -> None:
        from agent.harness.coordinator import admit_turn_step

        draft = self._draft()
        ready = resolve_semantic_draft_atom(
            draft,
            draft_id=draft["draft_id"],
            revision=draft["revision"],
            atom_id=draft["active_atom_id"],
            resolution="QPS profile",
        )
        state = new_state(ready["session_id"], language="en")
        state["turn_index"] = ready["creation_turn"]
        state["semantic_plan_draft"] = ready
        state["turn_context"] = {"text": ready["original_input"]}
        state["pending_question"] = {
            "id": "source-question",
            "group": "target_mode",
            "kind": "yes_no",
            "accepted_action_types": ["answer_pending"],
            "queue_barrier": True,
        }
        candidate_action = {
            "type": "answer_opening_question",
            "topic": "mode_comparison",
            "source_evidence": "Explain fake-node",
            "_plan_transaction_hash": "e" * 64,
            "_source_unit_ids": ["unit-1"],
        }
        blocked_action = {
            "type": "use_default_workload",
            "_plan_transaction_hash": "e" * 64,
        }
        state["proposed_actions"] = [candidate_action, blocked_action]
        with (
            patch(
                "agent.harness.coordinator.validate_action_plan",
                return_value=AdmissionResult(
                    status="accepted",
                    actions=(candidate_action, blocked_action),
                ),
            ),
            patch("agent.harness.coordinator._validate_admission_transaction"),
        ):
            updated = admit_turn_step(state)
        self.assertEqual(updated["control"]["phase"], "execute")
        self.assertEqual(updated["semantic_plan_draft"], {})
        self.assertEqual(len(updated["action_queue"]), 2)
        self.assertTrue(any(
            item.get("event") == "semantic_draft_finalized"
            for item in updated["audit_events"]
        ))

    def test_incompatible_sibling_mutation_rejects_draft_resolution(self) -> None:
        from agent.harness.admission import validate_action_plan
        from agent.harness.coordinator import _semantic_draft_question

        draft = self._draft(active_group="qps_profile")
        state = new_state("draft-test", language="en")
        state["turn_index"] = draft["creation_turn"]
        state["active_group"] = "qps_profile"
        state["semantic_plan_draft"] = draft
        state["pending_question"] = _semantic_draft_question(draft)
        result = validate_action_plan(
            state,
            [
                {
                    "type": "resolve_semantic_draft_atom",
                    "draft_id": draft["draft_id"],
                    "revision": draft["revision"],
                    "atom_id": draft["active_atom_id"],
                    "resolution": "change QPS",
                    "source_evidence": "change QPS",
                },
                {
                    "type": "set_qps_mode",
                    "qps_mode": "quick",
                    "mutation_explicit": True,
                    "source_evidence": "use quick",
                },
            ],
        )
        self.assertEqual(result.status, "rejected")
        self.assertEqual(
            result.rejections[0].code,
            "pending_answer_invalidated",
        )

    def test_reset_and_navigation_siblings_invalidate_draft_resolution(self) -> None:
        from agent.harness.admission import validate_action_plan
        from agent.harness.coordinator import _semantic_draft_question

        draft = self._draft(active_group="qps_profile")
        base = new_state("draft-test", language="en")
        base["turn_index"] = draft["creation_turn"]
        base["active_group"] = "qps_profile"
        base["semantic_plan_draft"] = draft
        base["pending_question"] = _semantic_draft_question(draft)
        resolution = {
            "type": "resolve_semantic_draft_atom",
            "draft_id": draft["draft_id"],
            "revision": draft["revision"],
            "atom_id": draft["active_atom_id"],
            "resolution": "change QPS",
            "source_evidence": "change QPS",
        }
        siblings = (
            {"type": "reset_session"},
            {
                "type": "change_group",
                "group": "observability",
                "navigation_explicit": True,
                "group_navigation_semantic_verified": True,
                "source_evidence": "then configure observability",
            },
        )
        for sibling in siblings:
            with self.subTest(action_type=sibling["type"]):
                result = validate_action_plan(
                    deepcopy(base),
                    [resolution, sibling],
                )
                self.assertEqual(result.status, "rejected")
                self.assertEqual(
                    result.rejections[0].code,
                    "pending_answer_invalidated",
                )

    def test_incompatible_complete_turn_repartitions_without_business_mutation(
        self,
    ) -> None:
        from agent.harness.coordinator import (
            _semantic_draft_question,
            admit_turn_step,
        )

        draft = self._draft(active_group="qps_profile")
        state = new_state("draft-test", language="en")
        state["turn_index"] = draft["creation_turn"]
        state["active_group"] = "qps_profile"
        state["semantic_plan_draft"] = draft
        state["pending_question"] = _semantic_draft_question(draft)
        state["turn_context"] = {
            "text": "change QPS, then reset the session",
        }
        state["proposed_actions"] = [
            {
                "type": "resolve_semantic_draft_atom",
                "draft_id": draft["draft_id"],
                "revision": draft["revision"],
                "atom_id": draft["active_atom_id"],
                "resolution": "change QPS",
                "source_evidence": "change QPS",
            },
            {"type": "reset_session"},
        ]
        before = {
            "confirmed_config": deepcopy(state["confirmed_config"]),
            "group_states": deepcopy(state["group_states"]),
            "active_group": state["active_group"],
        }
        updated = admit_turn_step(state)
        self.assertEqual(updated["control"]["phase"], "plan")
        self.assertEqual(updated["semantic_plan_draft"]["status"], "stale")
        self.assertEqual(updated["pending_question"], {})
        self.assertEqual(updated["proposed_actions"], [])
        for key, value in before.items():
            self.assertEqual(updated[key], value)

    def test_read_only_consultation_pauses_and_resumes_exact_draft(self) -> None:
        from agent.harness.coordinator import (
            _semantic_draft_question,
            admit_turn_step,
            commit_selected_action_step,
            execute_selected_owner_step,
            prepare_turn_step,
            select_action_step,
        )

        draft = self._draft(active_group="qps_profile")
        state = new_state("draft-test", language="en")
        state["turn_index"] = draft["creation_turn"]
        state["active_group"] = "qps_profile"
        state["semantic_plan_draft"] = deepcopy(draft)
        state["pending_question"] = _semantic_draft_question(draft)
        state["last_user_input"] = "what can this Agent do?"
        state = prepare_turn_step(state)
        original_pending = deepcopy(state["pending_question"])
        state["proposed_actions"] = [{"type": "ask_capabilities"}]

        admitted = admit_turn_step(state)
        selected = select_action_step(admitted)
        prepared = execute_selected_owner_step(
            selected,
            expected_owner="orientation",
        )
        committed = commit_selected_action_step(prepared)
        self.assertEqual(committed["semantic_plan_draft"], draft)
        self.assertEqual(committed["pending_question"], original_pending)
        self.assertEqual(committed["action_queue"], [])
        self.assertEqual(committed["control"]["phase"], "compose")

    def test_pending_cancel_option_commits_with_semantic_draft_atomically(
        self,
    ) -> None:
        from types import SimpleNamespace

        from agent.harness.coordinator import (
            _apply_handler_result,
            _prepare_pending_answer_result,
            _semantic_draft_question,
        )

        draft = self._draft(active_group="opening")
        state = new_state("draft-test", language="en")
        state["turn_index"] = draft["creation_turn"]
        state["active_group"] = "opening"
        _bind_product_head(state)
        state["semantic_plan_draft"] = deepcopy(draft)
        state["pending_question"] = _semantic_draft_question(draft)

        result = _prepare_pending_answer_result(
            state,
            {
                "answer": "Cancel the complete pending plan.",
                "selected_value": "cancel",
                "source_evidence": "Cancel the complete pending plan.",
            },
            SimpleNamespace(action_id="cancel-answer", confidence="high"),
        )

        self.assertEqual(result.followup_actions, ())
        self.assertIsNotNone(result.semantic_draft_command)
        self.assertEqual(result.semantic_draft_command.operation, "cancel")
        committed = _apply_handler_result(
            state,
            result,
            owner="coordinator",
        )
        self.assertEqual(committed["semantic_plan_draft"]["status"], "cancelled")
        self.assertEqual(committed["pending_question"], {})
        validate_state(committed)

    def test_structured_multi_atom_draft_finalizes_one_complete_transaction(
        self,
    ) -> None:
        from agent.harness.coordinator import admit_turn_step
        from agent.harness.invariants import (
            StateInvariantError,
            _validate_semantic_finalization_action_sets,
        )

        state = new_state("structured-draft", language="en")
        state["turn_index"] = 11
        _bind_product_head(state)
        draft = build_semantic_plan_draft(
            state,
            original_input='{"chain":"solana","qps":{"mode":null,"max":null}}',
            source_clauses=[{
                "clause_id": "clause-1",
                "text": '{"chain":"solana","qps":{"mode":null,"max":null}}',
                "input_shape": "json",
            }],
            source_partition=[
                {
                    "unit_id": "unit-chain",
                    "clause_id": "clause-1",
                    "source_path": "$.chain",
                    "source_text": "solana",
                    "operation": "mutation",
                    "owner_routes": [{"owner": "chain_rpc", "group": "chain_identity"}],
                },
                {
                    "unit_id": "unit-mode",
                    "clause_id": "clause-1",
                    "source_path": "$.qps.mode",
                    "source_text": "",
                    "operation": "domain_request",
                    "owner_routes": [{"owner": "performance", "group": "qps_profile"}],
                },
                {
                    "unit_id": "unit-max",
                    "clause_id": "clause-1",
                    "source_path": "$.qps.max",
                    "source_text": "",
                    "operation": "domain_request",
                    "owner_routes": [{"owner": "performance", "group": "qps_profile"}],
                },
            ],
            semantic_units=[
                {
                    "unit_id": "unit-chain",
                    "clause_id": "clause-1",
                    "source_path": "$.chain",
                    "source_text": "solana",
                    "disposition": "action",
                    "action_indexes": [0],
                },
                {
                    "unit_id": "unit-mode",
                    "clause_id": "clause-1",
                    "source_path": "$.qps.mode",
                    "source_text": "",
                    "disposition": "unresolved",
                    "action_indexes": [],
                },
                {
                    "unit_id": "unit-max",
                    "clause_id": "clause-1",
                    "source_path": "$.qps.max",
                    "source_text": "",
                    "disposition": "unresolved",
                    "action_indexes": [],
                },
            ],
            candidate_actions=[{
                "type": "choose_chain",
                "chain_text": "solana",
                "source_evidence": "solana",
            }],
            validation=PlanCoverageResult(
                valid=False,
                errors=(),
                unresolved_clauses=("$.qps.mode", "$.qps.max"),
                unresolved_units=(
                    {
                        "unit_id": "unit-mode",
                        "clause_id": "clause-1",
                        "source_path": "$.qps.mode",
                        "source_text": "",
                        "reason": "QPS mode is missing",
                    },
                    {
                        "unit_id": "unit-max",
                        "clause_id": "clause-1",
                        "source_path": "$.qps.max",
                        "source_text": "",
                        "reason": "maximum QPS is missing",
                    },
                ),
            ),
        )
        first = draft["active_atom_id"]
        draft = resolve_semantic_draft_atom(
            draft,
            draft_id=draft["draft_id"],
            revision=draft["revision"],
            atom_id=first,
            resolution="quick",
        )
        self.assertEqual(
            draft["unresolved_atoms"][1]["source_path"],
            "$.qps.max",
        )
        draft = resolve_semantic_draft_atom(
            draft,
            draft_id=draft["draft_id"],
            revision=draft["revision"],
            atom_id=draft["active_atom_id"],
            resolution="100",
        )
        self.assertEqual(draft["status"], "ready_for_review")

        actions = [
            {
                "type": "choose_chain",
                "chain_text": "solana",
                "source_evidence": "solana",
                "_plan_transaction_hash": "f" * 64,
                "_source_unit_ids": ["unit-chain"],
            },
            {
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "mutation_explicit": True,
                "source_evidence": "quick",
                "_plan_transaction_hash": "f" * 64,
            },
        ]
        state["semantic_plan_draft"] = draft
        state["turn_context"] = {"text": draft["original_input"]}
        state["proposed_actions"] = actions
        with (
            patch(
                "agent.harness.coordinator.validate_action_plan",
                return_value=AdmissionResult(
                    status="accepted",
                    actions=tuple(actions),
                ),
            ),
            patch("agent.harness.coordinator._validate_admission_transaction"),
        ):
            admitted = admit_turn_step(state)
        self.assertEqual(admitted["semantic_plan_draft"], {})
        self.assertEqual(len(admitted["action_queue"]), 2)
        receipts = {
            json.dumps(
                item["admission_metadata"]["semantic_draft_finalization_receipt"],
                sort_keys=True,
            )
            for item in admitted["action_queue"]
        }
        self.assertEqual(len(receipts), 1)
        _validate_semantic_finalization_action_sets(admitted)
        self.assertTrue(has_incomplete_semantic_finalization(admitted))
        recovery_source = deepcopy(admitted)
        recovery_source["confirmed_config"]["CLOUD_REGION"] = "partial-write"
        quarantined = quarantine_inflight_semantic_finalization(
            recovery_source,
            thread_id=state["thread_id"],
            language="en",
            session_purpose="user",
            error_type="InvariantFailureDuringSemanticFinalization",
        )
        self.assertEqual(quarantined["confirmed_config"], {})
        self.assertEqual(
            quarantined["checkpoint_recovery"]["error_type"],
            "InvariantFailureDuringSemanticFinalization",
        )
        self.assertTrue(
            quarantined["audit_events"][0]["transactions"][0][
                "missing_action_ids"
            ]
        )
        validate_state(quarantined)
        incomplete = deepcopy(admitted)
        incomplete["action_queue"] = incomplete["action_queue"][:1]
        with self.assertRaisesRegex(
            StateInvariantError,
            "finalization action set is incomplete",
        ):
            _validate_semantic_finalization_action_sets(incomplete)
        total_loss = deepcopy(admitted)
        total_loss["action_queue"] = []
        with self.assertRaisesRegex(
            StateInvariantError,
            "finalization action set is incomplete",
        ):
            _validate_semantic_finalization_action_sets(total_loss)
        binding_tamper = deepcopy(admitted)
        binding_tamper["action_queue"][0]["admission_metadata"][
            "semantic_secret_bindings"
        ] = [{
            "path": ["chain_text"],
            "reference": "semantic-secret:tampered",
            "draft_id": draft["draft_id"],
            "atom_id": "tampered",
            "value_hash": "0" * 64,
        }]
        with self.assertRaisesRegex(
            StateInvariantError,
            "secret binding changed",
        ):
            _validate_semantic_finalization_action_sets(binding_tamper)
        completed = deepcopy(total_loss)
        canonical_receipt = next(
            item
            for item in completed["audit_events"]
            if item.get("event") == "semantic_draft_finalized"
        )
        for action_id in canonical_receipt["final_action_ids"]:
            completed["audit_events"].append({
                "event": "semantic_draft_finalization_action_applied",
                "draft_id": canonical_receipt["draft_id"],
                "receipt_hash": canonical_receipt["receipt_hash"],
                "admission_transaction_hash": (
                    canonical_receipt["admission_transaction_hash"]
                ),
                "action_id": action_id,
            })
        _validate_semantic_finalization_action_sets(completed)
        legacy_completed = deepcopy(completed)
        legacy_completed["schema_version"] = 20
        for item in legacy_completed["audit_events"]:
            if item.get("event") != "semantic_draft_finalized":
                continue
            for field in (
                "candidate_convergence_hash",
                "secret_binding_manifest_hash",
                "secret_binding_hashes",
                "source_product_authority_id",
            ):
                item.pop(field, None)
        migrated_completed = migrate_state(
            legacy_completed,
            thread_id=state["thread_id"],
            language="en",
            session_purpose="user",
        )
        preserved = [
            item
            for item in migrated_completed["audit_events"]
            if item.get("event") == "legacy_semantic_finalization_preserved"
            and item.get("receipt_hash") == canonical_receipt["receipt_hash"]
        ]
        self.assertEqual(len(preserved), 1)
        self.assertTrue(preserved[0]["complete"])
        self.assertEqual(
            preserved[0]["applied_action_ids"],
            sorted(canonical_receipt["final_action_ids"]),
        )
        self.assertFalse(any(
            item.get("event") in {
                "semantic_draft_finalized",
                "semantic_draft_finalization_action_applied",
            }
            for item in migrated_completed["audit_events"]
        ))
        _validate_semantic_finalization_action_sets(migrated_completed)
        from agent.harness.contracts import (
            FailureDescriptor,
            HandlerResult,
            PendingDomainResult,
            action_envelope_from_dict,
            pending_domain_result_to_dict,
        )
        from agent.harness.coordinator import (
            _prepared_state_hash,
            commit_selected_action_step,
            select_action_step,
        )

        blocked_source = deepcopy(admitted)
        blocked_source["turn_receipt"]["turn_id"] = "blocked-finalization"
        blocked_source["turn_receipt"]["input_hash"] = semantic_hash(
            draft["original_input"]
        )
        blocked = select_action_step(blocked_source)
        envelope = action_envelope_from_dict(blocked["selected_action"])
        blocked["pending_domain_result"] = pending_domain_result_to_dict(
            PendingDomainResult(
                action=envelope,
                result=HandlerResult(
                    blocker=FailureDescriptor(
                        code="harness.failure.internal_contract_violation",
                        arguments={"reason": "domain validation blocked"},
                        source=__name__,
                    ),
                    completion="blocked",
                ),
                prepared_state_hash=_prepared_state_hash(blocked),
            )
        )
        with self.assertRaisesRegex(
            StateInvariantError,
            "transaction blocked before publish",
        ):
            commit_selected_action_step(blocked)
        self.assertEqual(
            blocked["action_queue"][0]["action_id"],
            envelope.action_id,
        )
        self.assertFalse(any(
            item.get("event")
            == "semantic_draft_finalization_action_applied"
            and item.get("action_id") == envelope.action_id
            for item in blocked["audit_events"]
        ))
        _validate_semantic_finalization_action_sets(blocked)
        tampered = deepcopy(admitted)
        audit_receipt = next(
            item
            for item in tampered["audit_events"]
            if item.get("event") == "semantic_draft_finalized"
        )
        audit_receipt["final_action_ids"] = audit_receipt["final_action_ids"][:1]
        with self.assertRaisesRegex(
            StateInvariantError,
            "receipt hash mismatch",
        ):
            _validate_semantic_finalization_action_sets(tampered)


if __name__ == "__main__":
    unittest.main()
