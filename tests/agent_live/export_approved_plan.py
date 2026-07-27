"""Export one immutable Agent-approved plan from an exact LangGraph checkpoint."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from langgraph.checkpoint.sqlite import SqliteSaver

from agent.harness.control_receipts import validate_coordinator_control_receipt
from agent.harness.invariants import validate_state
from agent.harness.runtime_identity import repository_revision
from agent.harness.state import migrate_state, project_checkpoint_state
from agent.harness.turn_transactions import product_authority_id, read_product_head
from agent.runners.execution_scenarios import workflow_type_from_plan
from agent.runners.job_manager import verify_job_receipt


SCHEMA_VERSION = 3
ARTIFACT_TYPE = "agent_approved_plan"
APPROVAL_CONTRACTS = {
    "preflight_smoke_confirm": "approve_preflight_smoke",
    "real_node_smoke_confirm": "approve_preflight_smoke",
    "real_node_final_benchmark_confirm": "approve_final_benchmark",
}
APPROVAL_QUESTIONS = frozenset(APPROVAL_CONTRACTS)
EMPTY_WORKTREE_HASH = hashlib.sha256(b"").hexdigest()
SECURITY_CONTRACT = {
    "classification": "confidential_local_acceptance_evidence",
    "directory_mode": "0700",
    "file_mode": "0400",
    "publish_allowed": False,
    "passing_retention": "delete_after_phase8_passing_decision",
    "failure_retention": "retain_owner_only_for_failure_investigation",
    "failure_cleanup": "explicit_cleanup_required_before_publication_or_copy",
}
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication requires renameat2")
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(f"immutable destination exists: {destination}")
    raise OSError(error, os.strerror(error), str(destination))


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _rename_noreplace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
        raise


def _seal(path: Path) -> None:
    path.chmod(0o400)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _assert_regular_nonsymlink(
    path: Path,
    *,
    description: str,
    immutable: bool = False,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is missing, not regular, or a symlink")
    if immutable and path.stat().st_mode & 0o222:
        raise ValueError(f"{description} is writable")
    cursor = path.parent
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValueError(f"{description} has a symlinked parent")
        cursor = cursor.parent


def _isolated_checkpoint_bytes(
    source_path: Path,
    *,
    thread_id: str,
    session_purpose: str,
) -> tuple[bytes, dict[str, Any]]:
    """Copy one exact checkpoint row into a single-thread SQLite database."""

    _assert_regular_nonsymlink(source_path, description="Agent checkpoint")
    product_head = read_product_head(
        source_path,
        product_authority_id(thread_id, session_purpose),
    )
    checkpoint_thread_id = (
        product_head.checkpoint_thread_id if product_head else thread_id
    )
    checkpoint_id = product_head.checkpoint_id if product_head else ""
    source_uri = f"file:{source_path}?mode=ro"
    with tempfile.TemporaryDirectory() as tmpdir:
        isolated = Path(tmpdir) / "approval-checkpoint.sqlite"
        try:
            with closing(sqlite3.connect(source_uri, uri=True)) as source:
                source.execute("BEGIN")
                row = source.execute(
                    """
                    SELECT thread_id, checkpoint_ns, checkpoint_id,
                           parent_checkpoint_id, type, checkpoint, metadata
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ''
                      AND (? = '' OR checkpoint_id = ?)
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    (checkpoint_thread_id, checkpoint_id, checkpoint_id),
                ).fetchone()
                if row is None:
                    raise ValueError("Agent checkpoint has no requested thread")
                writes = source.execute(
                    """
                    SELECT thread_id, checkpoint_ns, checkpoint_id, task_id,
                           idx, channel, type, value
                    FROM writes
                    WHERE thread_id = ? AND checkpoint_ns = ?
                          AND checkpoint_id = ?
                    ORDER BY task_id, idx
                    """,
                    (row[0], row[1], row[2]),
                ).fetchall()
                with closing(sqlite3.connect(isolated)) as destination:
                    destination.executescript(
                        """
                        CREATE TABLE checkpoints (
                            thread_id TEXT NOT NULL,
                            checkpoint_ns TEXT NOT NULL DEFAULT '',
                            checkpoint_id TEXT NOT NULL,
                            parent_checkpoint_id TEXT,
                            type TEXT,
                            checkpoint BLOB,
                            metadata BLOB,
                            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                        );
                        CREATE TABLE writes (
                            thread_id TEXT NOT NULL,
                            checkpoint_ns TEXT NOT NULL DEFAULT '',
                            checkpoint_id TEXT NOT NULL,
                            task_id TEXT NOT NULL,
                            idx INTEGER NOT NULL,
                            channel TEXT NOT NULL,
                            type TEXT,
                            value BLOB,
                            PRIMARY KEY (
                                thread_id, checkpoint_ns, checkpoint_id,
                                task_id, idx
                            )
                        );
                        """
                    )
                    destination.execute(
                        "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)",
                        row,
                    )
                    destination.executemany(
                        "INSERT INTO writes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        writes,
                    )
                    destination.commit()
                source.rollback()
        except sqlite3.Error as exc:
            raise ValueError(f"Agent checkpoint cannot be isolated: {exc}") from exc
        payload = isolated.read_bytes()
    identity = {
        "thread_id": str(row[0]),
        "logical_thread_id": thread_id,
        "checkpoint_ns": str(row[1]),
        "checkpoint_id": str(row[2]),
        "parent_checkpoint_id": str(row[3] or ""),
        "checkpoint_blob_hash": hashlib.sha256(bytes(row[5])).hexdigest(),
        "metadata_blob_hash": hashlib.sha256(bytes(row[6])).hexdigest(),
        "write_count": len(writes),
    }
    return payload, identity


def _read_checkpoint_state(
    path: Path,
    *,
    checkpoint_thread_id: str,
    logical_thread_id: str,
    checkpoint_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one exact checkpoint without invoking migrations or graph writes."""

    _assert_regular_nonsymlink(path, description="isolated checkpoint evidence")
    uri = f"file:{path}?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True, check_same_thread=False)) as connection:
        saver = SqliteSaver(connection)
        config = {
            "configurable": {
                "thread_id": checkpoint_thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
            }
        }
        item = saver.get_tuple(config)
        if item is None:
            raise ValueError("isolated checkpoint does not contain the exact identity")
        state = deepcopy(dict(item.checkpoint.get("channel_values") or {}))
        if str((item.config.get("configurable") or {}).get("checkpoint_id") or "") != checkpoint_id:
            raise ValueError("isolated checkpoint resolved a different checkpoint")
        validate_state(state)
        projected = project_checkpoint_state(state)
        if projected != state:
            raise ValueError("checkpoint contains non-durable runtime state")
        migrated = migrate_state(
            deepcopy(state),
            thread_id=logical_thread_id,
            language=str(state.get("language") or "en"),
            session_purpose=str((state.get("session") or {}).get("purpose") or ""),
        )
        if migrated != state:
            raise ValueError("checkpoint requires migration and is not exportable")
        metadata = {
            "config": deepcopy(dict(item.config)),
            "parent_config": deepcopy(dict(item.parent_config or {})),
            "metadata": deepcopy(dict(item.metadata or {})),
            "channel_versions_hash": _content_hash(
                item.checkpoint.get("channel_versions") or {}
            ),
            "versions_seen_hash": _content_hash(
                item.checkpoint.get("versions_seen") or {}
            ),
        }
    return state, metadata


def _checkpoint_identity_from_isolated(
    path: Path,
    *,
    thread_id: str,
    checkpoint_id: str,
    logical_thread_id: str = "",
) -> dict[str, Any]:
    uri = f"file:{path}?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True, check_same_thread=False)) as connection:
        row = connection.execute(
            """
            SELECT thread_id, checkpoint_ns, checkpoint_id,
                   parent_checkpoint_id, type, checkpoint, metadata
            FROM checkpoints
            WHERE thread_id = ? AND checkpoint_ns = '' AND checkpoint_id = ?
            """,
            (thread_id, checkpoint_id),
        ).fetchone()
        if row is None:
            raise ValueError("isolated checkpoint identity row is missing")
        write_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM writes
                WHERE thread_id = ? AND checkpoint_ns = '' AND checkpoint_id = ?
                """,
                (thread_id, checkpoint_id),
            ).fetchone()[0]
        )
        checkpoint_count = int(
            connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        )
        unrelated_writes = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM writes
                WHERE thread_id != ? OR checkpoint_ns != '' OR checkpoint_id != ?
                """,
                (thread_id, checkpoint_id),
            ).fetchone()[0]
        )
    if checkpoint_count != 1 or unrelated_writes:
        raise ValueError("isolated checkpoint contains unrelated state")
    identity = {
        "thread_id": str(row[0]),
        "checkpoint_ns": str(row[1]),
        "checkpoint_id": str(row[2]),
        "parent_checkpoint_id": str(row[3] or ""),
        "checkpoint_blob_hash": hashlib.sha256(bytes(row[5])).hexdigest(),
        "metadata_blob_hash": hashlib.sha256(bytes(row[6])).hexdigest(),
        "write_count": write_count,
    }
    if logical_thread_id:
        identity["logical_thread_id"] = logical_thread_id
    return identity


def _approval_chain(
    state: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    turn_index = int(state.get("turn_index") or 0)
    context = dict(state.get("turn_context") or {})
    receipts = [
        dict(item)
        for item in context.get("control_receipts") or ()
        if isinstance(item, Mapping)
    ]
    pending = [
        item
        for item in receipts
        if item.get("receipt_type") == "pending_resolution"
        and item.get("pending_id") in APPROVAL_QUESTIONS
        and item.get("verdict") == "accepted"
    ]
    approvals = [
        item
        for item in receipts
        if item.get("receipt_type") == "execution_approval"
    ]
    if len(pending) != 1 or len(approvals) != 1:
        raise ValueError("checkpoint has no unique approval receipt chain")
    pending_receipt = pending[0]
    approval_receipt = approvals[0]
    for receipt in (pending_receipt, approval_receipt):
        valid, reason = validate_coordinator_control_receipt(
            receipt,
            turn_index=turn_index,
        )
        if not valid:
            raise ValueError(f"checkpoint has invalid coordinator receipt: {reason}")
    if (
        approval_receipt["pending_resolution_receipt_id"]
        != pending_receipt["receipt_id"]
        or approval_receipt["answer_action_id"]
        != pending_receipt["resolved_action_id"]
        or approval_receipt["approval_question_id"]
        != pending_receipt["pending_id"]
    ):
        raise ValueError("execution approval does not bind the pending resolution")

    admitted = [
        dict(item)
        for item in context.get("admitted_actions") or ()
        if isinstance(item, Mapping)
        and str(item.get("action_id") or "")
        == str(pending_receipt["resolved_action_id"])
        and str(item.get("type") or item.get("action_type") or "")
        == "answer_pending"
    ]
    committed_answers = [
        dict(item)
        for item in state.get("completed_actions") or ()
        if isinstance(item, Mapping)
        and str(item.get("action_id") or "")
        == str(pending_receipt["resolved_action_id"])
        and str(item.get("type") or item.get("action_type") or "")
        == "answer_pending"
    ]
    approval_action_type = str(approval_receipt["approval_action_type"])
    completed = [
        dict(item)
        for item in state.get("completed_actions") or ()
        if isinstance(item, Mapping)
        and str(item.get("action_id") or "")
        == str(approval_receipt["approval_action_id"])
        and str(item.get("type") or item.get("action_type") or "")
        == approval_action_type
    ]
    if (
        len(admitted) != 1
        or len(committed_answers) != 1
        or len(completed) != 1
    ):
        raise ValueError("approval receipt chain has no unique committed actions")
    answer_value = (
        committed_answers[0].get("selected_value")
        if "selected_value" in committed_answers[0]
        else committed_answers[0].get("answer")
    )
    if (
        APPROVAL_CONTRACTS.get(str(pending_receipt["pending_id"]))
        != approval_action_type
        or answer_value is not True
        or pending_receipt.get("selected_value_hash") != _content_hash(True)
    ):
        raise ValueError("approval receipt chain is not an affirmative selection")
    execution_order = [
        str(item)
        for item in (state.get("turn_receipt") or {}).get("execution_order") or ()
    ]
    answer_id = str(pending_receipt["resolved_action_id"])
    approval_id = str(approval_receipt["approval_action_id"])
    if (
        answer_id not in execution_order
        or approval_id not in execution_order
        or execution_order.index(answer_id) >= execution_order.index(approval_id)
    ):
        raise ValueError("approval actions are absent or out of order")
    intent = dict(state.get("side_effect_intent") or {})
    side_effect_receipt = dict(state.get("side_effect_receipt") or {})
    execution_receipts = dict(
        (state.get("job") or {}).get("execution_receipts") or {}
    )
    submission = dict(
        execution_receipts.get("submission_attempt")
        or execution_receipts.get("submission")
        or {}
    )
    job_id = str((state.get("job") or {}).get("job_id") or "")
    if (
        intent.get("status") != "succeeded"
        or intent.get("intent_id")
        != approval_receipt.get("side_effect_intent_id")
        or _content_hash(intent)
        != approval_receipt.get("side_effect_intent_hash")
        or intent.get("action_id") != approval_id
        or intent.get("operation") != approval_action_type
        or intent.get("request_fingerprint")
        != approval_receipt.get("request_fingerprint")
        or _content_hash(intent.get("request") or {})
        != intent.get("request_fingerprint")
        or _content_hash(intent.get("idempotency_key"))
        != approval_receipt.get("idempotency_key_hash")
        or side_effect_receipt.get("status") != "succeeded"
        or side_effect_receipt.get("receipt_id")
        != approval_receipt.get("side_effect_receipt_id")
        or _content_hash(side_effect_receipt)
        != approval_receipt.get("side_effect_receipt_hash")
        or side_effect_receipt.get("intent_id") != intent.get("intent_id")
        or side_effect_receipt.get("action_id") != approval_id
        or side_effect_receipt.get("idempotency_key")
        != intent.get("idempotency_key")
        or side_effect_receipt.get("job_id") != job_id
        or not verify_job_receipt(submission)
        or submission.get("job_id") != job_id
        or submission.get("receipt_id")
        != approval_receipt.get("job_submission_receipt_id")
        or _content_hash(submission)
        != approval_receipt.get("job_submission_receipt_hash")
        or submission.get("approved_plan_hash")
        != approval_receipt.get("approved_plan_hash")
        or approval_receipt.get("approved_plan_hash")
        != _content_hash(state.get("plan") or {})
    ):
        raise ValueError("execution approval does not bind its submitted side effect")
    return (
        pending_receipt,
        approval_receipt,
        committed_answers[0],
        completed[0],
    )


def _target_mode_from_plan(plan: Mapping[str, Any]) -> str:
    workflow = workflow_type_from_plan(plan)
    if workflow == "sync_observe":
        return "sync-observe"
    return "fake-node" if plan.get("use_fake_node") is True else "real-node"


def _validate_state_and_plan(
    state: Mapping[str, Any],
    plan_bytes: bytes,
    *,
    thread_id: str,
    session_purpose: str,
) -> tuple[
    dict[str, Any],
    str,
    str,
    tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
]:
    if dict(state.get("session") or {}).get("id") != thread_id:
        raise ValueError("checkpoint thread identity does not match")
    if dict(state.get("session") or {}).get("purpose") != session_purpose:
        raise ValueError("checkpoint session purpose does not match")
    plan = json.loads(plan_bytes)
    if not isinstance(plan, dict) or dict(state.get("plan") or {}) != plan:
        raise ValueError("checkpoint plan differs from immutable plan bytes")
    workflow = workflow_type_from_plan(plan)
    target_mode = _target_mode_from_plan(plan)
    if (
        str(state.get("workflow_mode") or "") != workflow
        or str(state.get("target_mode") or "") != target_mode
    ):
        raise ValueError("checkpoint workflow or target differs from the plan")
    chain = _approval_chain(state)
    approval = chain[1]
    approval_action_type = str(approval.get("approval_action_type") or "")
    if (
        approval_action_type == "approve_preflight_smoke"
        and (state.get("preflight") or {}).get("approved") is not True
    ):
        raise ValueError("checkpoint has no approved preflight decision")
    if (
        approval_action_type == "approve_final_benchmark"
        and (state.get("final_benchmark") or {}).get("approved") is not True
    ):
        raise ValueError("checkpoint has no approved final benchmark decision")
    if (
        approval["plan_hash"] != _content_hash(plan)
        or approval["workflow_type"] != workflow
        or approval["target_mode"] != target_mode
        or approval["job_id"] != str((state.get("job") or {}).get("job_id") or "")
    ):
        raise ValueError("execution approval does not bind the approved plan and job")
    return plan, workflow, target_mode, chain


def export_approved_plan(
    *,
    repo_root: Path,
    checkpoint_path: Path,
    thread_id: str,
    session_purpose: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    revision = repository_revision(repo_root)
    if revision["worktree_hash"] != EMPTY_WORKTREE_HASH:
        raise ValueError("approved plan export requires a clean tracked worktree")
    output_dir = output_dir.resolve()
    if output_dir.is_symlink():
        raise ValueError("approved-plan output directory is a symlink")
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    output_dir.chmod(0o700)
    if output_dir.stat().st_mode & 0o077:
        raise ValueError("approved-plan output directory is not owner-only")
    checkpoint_payload, checkpoint_identity = _isolated_checkpoint_bytes(
        checkpoint_path,
        thread_id=thread_id,
        session_purpose=session_purpose,
    )
    checkpoint_hash = hashlib.sha256(checkpoint_payload).hexdigest()
    exported_checkpoint = output_dir / f"checkpoint-{checkpoint_hash}.sqlite"
    if exported_checkpoint.exists():
        _assert_regular_nonsymlink(
            exported_checkpoint,
            description="immutable checkpoint export",
        )
        if exported_checkpoint.read_bytes() != checkpoint_payload:
            raise ValueError("immutable checkpoint export path has different bytes")
    else:
        _write_exclusive(exported_checkpoint, checkpoint_payload)
        _seal(exported_checkpoint)

    state, checkpoint_metadata = _read_checkpoint_state(
        exported_checkpoint,
        checkpoint_thread_id=str(checkpoint_identity["thread_id"]),
        logical_thread_id=thread_id,
        checkpoint_id=str(checkpoint_identity["checkpoint_id"]),
    )
    raw_plan_path = Path(str(state.get("plan_file") or ""))
    if not raw_plan_path.is_absolute():
        raw_plan_path = repo_root / raw_plan_path
    prepared_root = (repo_root / ".agent" / "prepared").resolve()
    _assert_regular_nonsymlink(raw_plan_path, description="prepared plan")
    source_plan = raw_plan_path.resolve()
    if source_plan.parent != prepared_root:
        raise ValueError("checkpoint plan is outside prepared-plan authority")
    plan_bytes = source_plan.read_bytes()
    plan, workflow, target_mode, chain = _validate_state_and_plan(
        state,
        plan_bytes,
        thread_id=thread_id,
        session_purpose=session_purpose,
    )
    approval_receipt = chain[1]
    if dict(approval_receipt["repository_revision"]) != revision:
        raise ValueError("approval-time repository revision differs from export revision")

    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    exported_plan = output_dir / f"plan-{plan_hash}.json"
    if exported_plan.exists():
        _assert_regular_nonsymlink(exported_plan, description="immutable plan export")
        if exported_plan.read_bytes() != plan_bytes:
            raise ValueError("immutable plan export path has different bytes")
    else:
        _write_exclusive(exported_plan, plan_bytes)
        _seal(exported_plan)
    snapshot_projection = {
        "thread_id": thread_id,
        "session_purpose": session_purpose,
        "turn_index": int(state.get("turn_index") or 0),
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_metadata_hash": _content_hash(checkpoint_metadata),
        "target_mode": target_mode,
        "workflow_type": workflow,
        "approval_action_type": str(
            approval_receipt["approval_action_type"]
        ),
        "chain": str(
            (state.get("chain_identity") or {}).get("canonical")
            or plan.get("chain")
            or ""
        ),
        "plan_content_hash": _content_hash(plan),
        "plan_sha256": plan_hash,
        "job_id": str((state.get("job") or {}).get("job_id") or ""),
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "security_contract": SECURITY_CONTRACT,
        "revision": revision,
        "thread_id": thread_id,
        "session_purpose": session_purpose,
        "checkpoint_file": {
            "path": str(exported_checkpoint),
            "sha256": checkpoint_hash,
            "size_bytes": len(checkpoint_payload),
        },
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_snapshot": snapshot_projection,
        "checkpoint_snapshot_hash": _content_hash(snapshot_projection),
        "pending_resolution_receipt": chain[0],
        "execution_approval_receipt": approval_receipt,
        "answer_action": chain[2],
        "approval_action": chain[3],
        "source_plan_file": str(source_plan),
        "source_plan_sha256": plan_hash,
        "exported_plan_file": str(exported_plan),
        "exported_plan_sha256": plan_hash,
        "workflow_type": workflow,
        "target_mode": target_mode,
        "approval_action_type": str(
            approval_receipt["approval_action_type"]
        ),
    }
    artifact["artifact_hash"] = _content_hash(artifact)
    artifact_path = output_dir / f"approval-{artifact['artifact_hash']}.json"
    _write_exclusive(artifact_path, _canonical_bytes(artifact))
    _seal(artifact_path)
    return exported_plan, artifact_path


def validate_approved_plan_artifact(
    artifact_path: Path,
    *,
    expected_revision: Mapping[str, str],
    expected_plan_file: Path,
    expected_workflow: str,
    expected_target_mode: str,
    expected_approval_action: str | None = None,
) -> tuple[bool, str]:
    try:
        _assert_regular_nonsymlink(
            artifact_path,
            description="approved-plan artifact",
            immutable=True,
        )
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(artifact, Mapping):
            raise ValueError("approved-plan artifact must be an object")
        unsigned = dict(artifact)
        artifact_hash = str(unsigned.pop("artifact_hash", ""))
        if (
            artifact.get("schema_version") != SCHEMA_VERSION
            or artifact.get("artifact_type") != ARTIFACT_TYPE
            or artifact.get("security_contract") != SECURITY_CONTRACT
            or artifact_hash != _content_hash(unsigned)
            or artifact_path.name != f"approval-{artifact_hash}.json"
        ):
            raise ValueError("approved-plan artifact identity is invalid")
        if (
            artifact_path.parent.is_symlink()
            or artifact_path.parent.stat().st_mode & 0o077
        ):
            raise ValueError("approved-plan evidence directory is not owner-only")
        if dict(artifact.get("revision") or {}) != dict(expected_revision):
            raise ValueError("approved-plan artifact revision differs")
        if dict(expected_revision).get("worktree_hash") != EMPTY_WORKTREE_HASH:
            raise ValueError("approved-plan expected revision is dirty")

        plan_file = Path(str(artifact.get("exported_plan_file") or ""))
        expected_plan = expected_plan_file.resolve()
        _assert_regular_nonsymlink(
            plan_file,
            description="approved plan",
            immutable=True,
        )
        if (
            plan_file.resolve() != expected_plan
            or plan_file.name != f"plan-{_sha256(plan_file)}.json"
            or artifact.get("exported_plan_sha256") != _sha256(plan_file)
        ):
            raise ValueError("approved-plan bytes or identity differ")
        plan_bytes = plan_file.read_bytes()

        checkpoint_record = dict(artifact.get("checkpoint_file") or {})
        checkpoint_file = Path(str(checkpoint_record.get("path") or ""))
        _assert_regular_nonsymlink(
            checkpoint_file,
            description="checkpoint evidence",
            immutable=True,
        )
        if (
            checkpoint_record.get("sha256") != _sha256(checkpoint_file)
            or checkpoint_record.get("size_bytes")
            != checkpoint_file.stat().st_size
            or checkpoint_file.name
            != f"checkpoint-{checkpoint_record.get('sha256')}.sqlite"
        ):
            raise ValueError("checkpoint evidence hash or identity differs")
        identity = dict(artifact.get("checkpoint_identity") or {})
        observed_identity = _checkpoint_identity_from_isolated(
            checkpoint_file,
            thread_id=str(identity.get("thread_id") or ""),
            checkpoint_id=str(identity.get("checkpoint_id") or ""),
            logical_thread_id=str(artifact.get("thread_id") or ""),
        )
        if identity != observed_identity:
            raise ValueError("checkpoint identity differs from isolated bytes")
        state, metadata = _read_checkpoint_state(
            checkpoint_file,
            checkpoint_thread_id=str(identity.get("thread_id") or ""),
            logical_thread_id=str(artifact.get("thread_id") or ""),
            checkpoint_id=str(identity.get("checkpoint_id") or ""),
        )
        plan, workflow, target_mode, chain = _validate_state_and_plan(
            state,
            plan_bytes,
            thread_id=str(artifact.get("thread_id") or ""),
            session_purpose=str(artifact.get("session_purpose") or ""),
        )
        if (
            workflow != expected_workflow
            or target_mode != expected_target_mode
            or workflow_type_from_plan(plan) != expected_workflow
            or _target_mode_from_plan(plan) != expected_target_mode
        ):
            raise ValueError("approved plan workflow or target differs")
        approval_action_type = str(
            chain[1].get("approval_action_type") or ""
        )
        if (
            expected_approval_action is not None
            and approval_action_type != expected_approval_action
        ):
            raise ValueError("approved plan has the wrong approval stage")
        if artifact.get("approval_action_type") != approval_action_type:
            raise ValueError("approval action type differs from checkpoint")
        if dict(chain[1].get("repository_revision") or {}) != dict(expected_revision):
            raise ValueError("approval-time repository revision differs")
        if chain[0] != artifact.get("pending_resolution_receipt"):
            raise ValueError("pending-resolution receipt differs from checkpoint")
        if chain[1] != artifact.get("execution_approval_receipt"):
            raise ValueError("execution-approval receipt differs from checkpoint")
        if chain[2] != artifact.get("answer_action"):
            raise ValueError("answer action differs from checkpoint")
        if chain[3] != artifact.get("approval_action"):
            raise ValueError("approval action differs from checkpoint")
        if metadata != artifact.get("checkpoint_metadata"):
            raise ValueError("checkpoint metadata differs")
        expected_snapshot = {
            "thread_id": str(artifact.get("thread_id") or ""),
            "session_purpose": str(artifact.get("session_purpose") or ""),
            "turn_index": int(state.get("turn_index") or 0),
            "checkpoint_identity": observed_identity,
            "checkpoint_metadata_hash": _content_hash(metadata),
            "target_mode": target_mode,
            "workflow_type": workflow,
            "approval_action_type": approval_action_type,
            "chain": str(
                (state.get("chain_identity") or {}).get("canonical")
                or plan.get("chain")
                or ""
            ),
            "plan_content_hash": _content_hash(plan),
            "plan_sha256": _sha256(plan_file),
            "job_id": str((state.get("job") or {}).get("job_id") or ""),
        }
        snapshot = dict(artifact.get("checkpoint_snapshot") or {})
        if (
            artifact.get("checkpoint_snapshot_hash") != _content_hash(snapshot)
            or snapshot != expected_snapshot
        ):
            raise ValueError("checkpoint snapshot projection differs")
    except (OSError, sqlite3.Error, json.JSONDecodeError, TypeError, ValueError) as exc:
        return False, f"approved-plan artifact is invalid: {exc}"
    return True, ""


def _load_cleanup_record(
    path: Path,
    *,
    expected_type: str,
    expected_status: str,
    expected_revision: Mapping[str, str],
) -> dict[str, Any]:
    _assert_regular_nonsymlink(
        path,
        description=f"{expected_status} cleanup record",
        immutable=True,
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("cleanup record must be an object")
    hash_field = (
        "receipt_hash"
        if expected_type == "approved_plan_cleanup_receipt"
        else "journal_hash"
    )
    unsigned = dict(record)
    observed_hash = str(unsigned.pop(hash_field, ""))
    if (
        record.get("schema_version") != 1
        or record.get("artifact_type") != expected_type
        or record.get("status") != expected_status
        or record.get("security_contract") != SECURITY_CONTRACT
        or dict(record.get("revision") or {}) != dict(expected_revision)
        or observed_hash != _content_hash(unsigned)
    ):
        raise ValueError("cleanup record identity is invalid")
    expected_prefix = (
        "approved-plan-cleanup-"
        if expected_type == "approved_plan_cleanup_receipt"
        else f"approved-plan-cleanup-{expected_status}-"
    )
    if path.name != f"{expected_prefix}{observed_hash}.json":
        raise ValueError("cleanup record filename is invalid")
    return record


def _validate_quarantine(
    quarantine: Path,
    deleted_files: Sequence[Mapping[str, Any]],
) -> None:
    if quarantine.is_symlink() or not quarantine.is_dir():
        raise ValueError("confidential cleanup quarantine is invalid")
    expected_names = {
        Path(str(item.get("path") or "")).name for item in deleted_files
    }
    observed = set(quarantine.iterdir())
    observed_names = {path.name for path in observed}
    if not observed_names.issubset(expected_names):
        raise ValueError("confidential cleanup quarantine contents differ")
    expected_by_name = {
        Path(str(item.get("path") or "")).name: item
        for item in deleted_files
    }
    for name in observed_names:
        item = expected_by_name[name]
        path = quarantine / name
        _assert_regular_nonsymlink(
            path,
            description="quarantined confidential evidence",
            immutable=True,
        )
        if (
            _sha256(path) != item.get("sha256")
            or path.stat().st_size != item.get("size_bytes")
        ):
            raise ValueError("quarantined confidential evidence differs")


def _validate_prepared_cleanup(
    path: Path,
    *,
    expected_revision: Mapping[str, str],
    expected_evidence_root: Path,
) -> dict[str, Any]:
    prepared = _load_cleanup_record(
        path,
        expected_type="approved_plan_cleanup_journal",
        expected_status="prepared",
        expected_revision=expected_revision,
    )
    evidence_root = Path(str(prepared.get("evidence_root") or ""))
    quarantine = Path(str(prepared.get("quarantine_path") or ""))
    deleted = list(prepared.get("deleted_files") or ())
    validated = list(prepared.get("validated_contracts") or ())
    if (
        not evidence_root.is_absolute()
        or evidence_root != expected_evidence_root
        or not quarantine.is_absolute()
        or quarantine.parent != evidence_root.parent
        or not quarantine.name.startswith(f".{evidence_root.name}.cleanup-")
        or not deleted
        or int(prepared.get("deleted_file_count") or 0) != len(deleted)
        or not validated
    ):
        raise ValueError("prepared cleanup journal contract is invalid")
    names: set[str] = set()
    for item in deleted:
        if not isinstance(item, Mapping):
            raise ValueError("prepared cleanup file identity is invalid")
        source = Path(str(item.get("path") or ""))
        if (
            set(item) != {"kind", "path", "sha256", "size_bytes"}
            or not source.is_absolute()
            or source.parent != evidence_root
            or source.name in names
            or str(item.get("kind") or "")
            not in {"approval", "plan", "checkpoint"}
            or len(str(item.get("sha256") or "")) != 64
            or isinstance(item.get("size_bytes"), bool)
            or not isinstance(item.get("size_bytes"), int)
            or int(item["size_bytes"]) < 0
        ):
            raise ValueError("prepared cleanup file identity is invalid")
        names.add(source.name)
    for item in validated:
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "approval_artifact_sha256",
                "plan_sha256",
                "approval_action",
                "workflow",
                "target_mode",
            }
            or len(str(item.get("approval_artifact_sha256") or "")) != 64
            or len(str(item.get("plan_sha256") or "")) != 64
            or not all(
                str(item.get(field) or "")
                for field in ("approval_action", "workflow", "target_mode")
            )
        ):
            raise ValueError("prepared cleanup contract identity is invalid")
    return prepared


def _validate_committed_cleanup(
    path: Path,
    *,
    expected_revision: Mapping[str, str],
    expected_evidence_root: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    committed = _load_cleanup_record(
        path,
        expected_type="approved_plan_cleanup_journal",
        expected_status="committed",
        expected_revision=expected_revision,
    )
    prepared_path = Path(str(committed.get("prepared_journal") or ""))
    if (
        not prepared_path.is_absolute()
        or prepared_path.parent != output_dir
        or committed.get("prepared_journal_sha256") != _sha256(prepared_path)
    ):
        raise ValueError("committed cleanup prepared-journal binding differs")
    prepared = _validate_prepared_cleanup(
        prepared_path,
        expected_revision=expected_revision,
        expected_evidence_root=expected_evidence_root,
    )
    for field in (
        "evidence_root",
        "quarantine_path",
        "validated_contracts",
        "deleted_files",
        "deleted_file_count",
    ):
        if committed.get(field) != prepared.get(field):
            raise ValueError("committed cleanup journal chain differs")
    return committed, prepared_path, prepared


def _contract_projection(
    contracts: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    return sorted(
        (
            {
                "approval_action": str(item.get("approval_action") or ""),
                "workflow": str(item.get("workflow") or ""),
                "target_mode": str(item.get("target_mode") or ""),
                "approval_artifact_sha256": str(
                    item.get("approval_artifact_sha256") or ""
                ),
                "plan_sha256": str(item.get("plan_sha256") or ""),
            }
            for item in contracts
        ),
        key=lambda item: (
            item["workflow"],
            item["target_mode"],
            item["approval_action"],
            item["approval_artifact_sha256"],
            item["plan_sha256"],
        ),
    )


def validate_completed_cleanup_receipt(
    path: Path,
    *,
    expected_revision: Mapping[str, str],
    expected_evidence_root: Path,
    expected_contracts: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    receipt = _load_cleanup_record(
        path,
        expected_type="approved_plan_cleanup_receipt",
        expected_status="completed",
        expected_revision=expected_revision,
    )
    committed_path = Path(str(receipt.get("committed_journal") or ""))
    if (
        not committed_path.is_absolute()
        or committed_path.parent != output_dir
        or receipt.get("committed_journal_sha256") != _sha256(committed_path)
    ):
        raise ValueError("completed cleanup committed-journal binding differs")
    committed, prepared_path, prepared = _validate_committed_cleanup(
        committed_path,
        expected_revision=expected_revision,
        expected_evidence_root=expected_evidence_root,
        output_dir=output_dir,
    )
    for field in (
        "evidence_root",
        "quarantine_path",
        "validated_contracts",
        "deleted_files",
        "deleted_file_count",
    ):
        if receipt.get(field) != committed.get(field):
            raise ValueError("completed cleanup journal chain differs")
    if (
        receipt.get("prepared_journal") != str(prepared_path)
        or receipt.get("prepared_journal_sha256") != _sha256(prepared_path)
        or list(receipt.get("validated_contracts") or ())
        != list(prepared.get("validated_contracts") or ())
        or _contract_projection(
            list(receipt.get("validated_contracts") or ())
        )
        != _contract_projection(expected_contracts)
    ):
        raise ValueError("completed cleanup contract binding differs")
    quarantine = Path(str(receipt.get("quarantine_path") or ""))
    if expected_evidence_root.exists() or quarantine.exists():
        raise RuntimeError("completed cleanup evidence contradicts storage")
    return receipt


def _write_cleanup_committed(
    *,
    prepared_path: Path,
    prepared: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    committed: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "approved_plan_cleanup_journal",
        "status": "committed",
        "security_contract": SECURITY_CONTRACT,
        "revision": dict(prepared.get("revision") or {}),
        "prepared_journal": str(prepared_path),
        "prepared_journal_sha256": _sha256(prepared_path),
        "evidence_root": str(prepared.get("evidence_root") or ""),
        "quarantine_path": str(prepared.get("quarantine_path") or ""),
        "validated_contracts": list(prepared.get("validated_contracts") or ()),
        "deleted_files": list(prepared.get("deleted_files") or ()),
        "deleted_file_count": int(prepared.get("deleted_file_count") or 0),
        "committed_at": (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        ),
    }
    committed["journal_hash"] = _content_hash(committed)
    path = output_dir / (
        f"approved-plan-cleanup-committed-{committed['journal_hash']}.json"
    )
    _write_exclusive(path, _canonical_bytes(committed))
    _seal(path)
    return path, committed


def _finish_cleanup(
    *,
    committed_path: Path,
    committed: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    evidence_root = Path(str(committed.get("evidence_root") or ""))
    quarantine = Path(str(committed.get("quarantine_path") or ""))
    deleted_files = list(committed.get("deleted_files") or ())
    if evidence_root.exists() or evidence_root.is_symlink():
        raise RuntimeError("confidential evidence root reappeared after commit")
    if quarantine.exists() or quarantine.is_symlink():
        _validate_quarantine(quarantine, deleted_files)
        try:
            shutil.rmtree(quarantine)
        except Exception as exc:
            raise RuntimeError(
                f"confidential evidence remains quarantined: {quarantine}"
            ) from exc
        _fsync_directory(quarantine.parent)
    if evidence_root.exists() or quarantine.exists():
        raise RuntimeError("confidential evidence cleanup did not complete")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "approved_plan_cleanup_receipt",
        "status": "completed",
        "security_contract": SECURITY_CONTRACT,
        "revision": dict(committed.get("revision") or {}),
        "prepared_journal": str(committed.get("prepared_journal") or ""),
        "prepared_journal_sha256": str(
            committed.get("prepared_journal_sha256") or ""
        ),
        "committed_journal": str(committed_path),
        "committed_journal_sha256": _sha256(committed_path),
        "evidence_root": str(evidence_root),
        "quarantine_path": str(quarantine),
        "validated_contracts": list(
            committed.get("validated_contracts") or ()
        ),
        "deleted_files": deleted_files,
        "deleted_file_count": int(
            committed.get("deleted_file_count") or 0
        ),
        "completed_at": (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        ),
    }
    receipt["receipt_hash"] = _content_hash(receipt)
    path = output_dir / f"approved-plan-cleanup-{receipt['receipt_hash']}.json"
    _write_exclusive(path, _canonical_bytes(receipt))
    _seal(path)
    return path


def _recover_cleanup(
    *,
    evidence_root: Path,
    expected_revision: Mapping[str, str],
    expected_contracts: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> tuple[Path | None, tuple[Path, dict[str, Any]] | None]:
    completed: list[tuple[Path, dict[str, Any]]] = []
    committed: list[tuple[Path, dict[str, Any]]] = []
    prepared: list[tuple[Path, dict[str, Any]]] = []
    for path in output_dir.glob("approved-plan-cleanup-*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        status = str(raw.get("status") or "")
        artifact_type = str(raw.get("artifact_type") or "")
        if artifact_type == "approved_plan_cleanup_receipt":
            record = validate_completed_cleanup_receipt(
                path,
                expected_revision=expected_revision,
                expected_evidence_root=evidence_root,
                expected_contracts=expected_contracts,
                output_dir=output_dir,
            )
            completed.append((path, record))
        elif status == "prepared":
            record = _validate_prepared_cleanup(
                path,
                expected_revision=expected_revision,
                expected_evidence_root=evidence_root,
            )
            prepared.append((path, record))
        elif status == "committed":
            record, _prepared_path, _prepared = (
                _validate_committed_cleanup(
                    path,
                    expected_revision=expected_revision,
                    expected_evidence_root=evidence_root,
                    output_dir=output_dir,
                )
            )
            committed.append((path, record))
        else:
            raise ValueError("unknown cleanup record is present")
    if len(completed) > 1 or len(committed) > 1 or len(prepared) > 1:
        raise RuntimeError("confidential cleanup history is ambiguous")
    if completed:
        path, record = completed[0]
        return path, None
    if committed:
        path, record = committed[0]
        receipt = _finish_cleanup(
            committed_path=path,
            committed=record,
            output_dir=output_dir,
        )
        return receipt, None
    if prepared:
        path, record = prepared[0]
        quarantine = Path(str(record.get("quarantine_path") or ""))
        if quarantine.exists() and not evidence_root.exists():
            _validate_quarantine(
                quarantine,
                list(record.get("deleted_files") or ()),
            )
            committed_path, committed_record = _write_cleanup_committed(
                prepared_path=path,
                prepared=record,
                output_dir=output_dir,
            )
            receipt = _finish_cleanup(
                committed_path=committed_path,
                committed=committed_record,
                output_dir=output_dir,
            )
            return receipt, None
        if evidence_root.exists() and not quarantine.exists():
            return None, (path, record)
        raise RuntimeError("prepared cleanup storage state is inconsistent")
    return None, None


def cleanup_approved_plan_evidence(
    contracts: Sequence[Mapping[str, Any]],
    *,
    expected_revision: Mapping[str, str],
    output_dir: Path,
) -> Path:
    """Delete confidential approval sources after terminal Phase 8 validation."""

    if not contracts:
        raise ValueError("approved-plan cleanup requires validated contracts")
    evidence_root = Path(
        str(contracts[0].get("artifact_file") or "")
    ).resolve().parent
    if any(
        Path(str(contract.get("artifact_file") or "")).resolve().parent
        != evidence_root
        for contract in contracts
    ):
        raise ValueError(
            "confidential approval evidence must share one atomic directory"
        )
    output_dir = output_dir.resolve()
    if output_dir == evidence_root or evidence_root in output_dir.parents:
        raise ValueError("cleanup receipts must be outside confidential evidence")
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    output_dir.chmod(0o700)
    recovered, existing_prepared = _recover_cleanup(
        evidence_root=evidence_root,
        expected_revision=expected_revision,
        expected_contracts=contracts,
        output_dir=output_dir,
    )
    if recovered is not None:
        return recovered

    validated: list[dict[str, Any]] = []
    protected_paths: dict[Path, dict[str, Any]] = {}
    for contract in contracts:
        artifact_path = Path(str(contract.get("artifact_file") or "")).resolve()
        plan_path = Path(str(contract.get("plan_file") or "")).resolve()
        valid, reason = validate_approved_plan_artifact(
            artifact_path,
            expected_revision=expected_revision,
            expected_plan_file=plan_path,
            expected_workflow=str(contract.get("workflow") or ""),
            expected_target_mode=str(contract.get("target_mode") or ""),
            expected_approval_action=str(contract.get("approval_action") or ""),
        )
        if not valid:
            raise ValueError(
                f"approved-plan cleanup rejected unvalidated evidence: {reason}"
            )
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact_sha256 = _sha256(artifact_path)
        plan_sha256 = _sha256(plan_path)
        if (
            str(contract.get("approval_artifact_sha256") or "")
            != artifact_sha256
            or str(contract.get("plan_sha256") or "") != plan_sha256
        ):
            raise ValueError(
                "approved-plan cleanup contract digest differs from source"
            )
        checkpoint_path = Path(
            str((artifact.get("checkpoint_file") or {}).get("path") or "")
        ).resolve()
        for kind, path in (
            ("approval", artifact_path),
            ("plan", plan_path),
            ("checkpoint", checkpoint_path),
        ):
            _assert_regular_nonsymlink(
                path,
                description=f"confidential {kind} evidence",
                immutable=True,
            )
            observed = {
                "kind": kind,
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            previous = protected_paths.setdefault(path, observed)
            if previous != observed:
                raise ValueError("confidential evidence identity is inconsistent")
        validated.append({
            "approval_artifact_sha256": artifact_sha256,
            "plan_sha256": plan_sha256,
            "approval_action": str(contract.get("approval_action") or ""),
            "workflow": str(contract.get("workflow") or ""),
            "target_mode": str(contract.get("target_mode") or ""),
        })

    deleted = sorted(
        protected_paths.values(),
        key=lambda item: (item["kind"], item["path"]),
    )
    evidence_roots = {path.parent for path in protected_paths}
    if evidence_roots != {evidence_root}:
        raise ValueError(
            "confidential approval evidence must share one atomic directory"
        )
    observed_entries = set(evidence_root.iterdir())
    if observed_entries != set(protected_paths):
        raise ValueError(
            "confidential approval directory contains unrelated entries"
        )
    prepared: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "approved_plan_cleanup_journal",
        "status": "prepared",
        "security_contract": SECURITY_CONTRACT,
        "revision": dict(expected_revision),
        "validated_contracts": validated,
        "deleted_files": deleted,
        "deleted_file_count": len(deleted),
        "evidence_root": str(evidence_root),
        "prepared_at": (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        ),
    }
    prepared["journal_hash"] = _content_hash(prepared)
    quarantine = evidence_root.with_name(
        f".{evidence_root.name}.cleanup-{prepared['journal_hash'][:16]}"
    )
    prepared["quarantine_path"] = str(quarantine)
    prepared["journal_hash"] = _content_hash({
        key: value for key, value in prepared.items() if key != "journal_hash"
    })
    if existing_prepared is None:
        journal = output_dir / (
            f"approved-plan-cleanup-prepared-{prepared['journal_hash']}.json"
        )
        _write_exclusive(journal, _canonical_bytes(prepared))
        _seal(journal)
    else:
        journal, prepared = existing_prepared
        if (
            list(prepared.get("validated_contracts") or ()) != validated
            or list(prepared.get("deleted_files") or ()) != deleted
        ):
            raise RuntimeError("prepared cleanup journal differs from sources")
        quarantine = Path(str(prepared.get("quarantine_path") or ""))

    if quarantine.exists() or quarantine.is_symlink():
        raise RuntimeError("confidential cleanup quarantine already exists")
    os.rename(evidence_root, quarantine)
    _fsync_directory(evidence_root.parent)
    try:
        committed_path, committed = _write_cleanup_committed(
            prepared_path=journal,
            prepared=prepared,
            output_dir=output_dir,
        )
    except Exception:
        os.rename(quarantine, evidence_root)
        _fsync_directory(evidence_root.parent)
        raise
    return _finish_cleanup(
        committed_path=committed_path,
        committed=committed,
        output_dir=output_dir,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--session-purpose", default="user")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan, artifact = export_approved_plan(
        repo_root=args.repo_root.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        thread_id=args.thread_id,
        session_purpose=args.session_purpose,
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(
        {"plan_file": str(plan), "approval_artifact": str(artifact)},
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
