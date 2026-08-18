"""Process-local secret references for semantic clarification evidence."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import uuid
from contextlib import AbstractContextManager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from agent.utils.redaction import secret_values
from .state import (
    DURABLE_SECRET_CONTROL_METADATA_LEAF_KEYS,
    DURABLE_SECRET_CAPABLE_LEAF_KEYS,
    DURABLE_SECRET_CAPABLE_STATE_ROOTS,
    SecretBinding,
)
from .contracts import SECRET_REFERENCE_RE, is_secret_reference

_REFERENCE_RE = SECRET_REFERENCE_RE
_REFERENCE_HASH_RE = re.compile(r"secret-reference-hash:[0-9a-f]{64}")


@dataclass(frozen=True)
class _SecretRecord:
    draft_id: str
    atom_id: str
    value: str
    value_hash: str


_LOCK = threading.RLock()
_VALUES: dict[str, _SecretRecord] = {}
_SCOPES: dict[str, set[tuple[str, str]]] = {}
_VERSIONS: dict[str, int] = {}
_RESERVATIONS: dict[str, str] = {}
_UNCERTAIN_TRANSACTIONS: dict[str, "SecretRegistryTransaction"] = {}
_CLEAR_RESERVATION = ""
_CLEAR_EPOCH = 0
_REGISTRY_REVISION = 0
_ACTIVE_TRANSACTION: ContextVar["SecretRegistryTransaction | None"] = (
    ContextVar("anychain_secret_registry_transaction", default=None)
)


class SecretRegistryConflict(RuntimeError):
    """Fail-closed optimistic concurrency conflict in the secret registry."""


@dataclass(frozen=True)
class _RegistrySnapshot:
    record: _SecretRecord | None
    scopes: frozenset[tuple[str, str]]
    version: int


def _snapshot_locked(reference: str) -> _RegistrySnapshot:
    return _RegistrySnapshot(
        record=_VALUES.get(reference),
        scopes=frozenset(_SCOPES.get(reference) or ()),
        version=int(_VERSIONS.get(reference) or 0),
    )


def _assert_global_mutation_available_locked(reference: str) -> None:
    if _CLEAR_RESERVATION:
        raise SecretRegistryConflict(
            "secret registry clear is prepared by another transaction"
        )
    if reference in _RESERVATIONS:
        raise SecretRegistryConflict(
            "secret reference is prepared by another transaction"
        )


def _publish_snapshot_locked(
    reference: str,
    desired: _RegistrySnapshot,
) -> None:
    global _REGISTRY_REVISION

    if desired.record is None:
        _VALUES.pop(reference, None)
        _SCOPES.pop(reference, None)
    else:
        _VALUES[reference] = desired.record
        _SCOPES[reference] = set(desired.scopes)
    _VERSIONS[reference] = int(_VERSIONS.get(reference) or 0) + 1
    _REGISTRY_REVISION += 1


class SecretRegistryTransaction(AbstractContextManager["SecretRegistryTransaction"]):
    """Stage registry mutations until Product Head is ready to publish."""

    def __init__(self) -> None:
        self.transaction_id = f"secret-tx:{uuid.uuid4().hex}"
        self._observed: dict[str, _RegistrySnapshot] = {}
        self._writes: dict[str, _RegistrySnapshot] = {}
        self._clear_requested = False
        self._start_clear_epoch = 0
        self._clear_base_revision = 0
        self._reserved_references: set[str] = set()
        self._prepared = False
        self._published = False
        self._product_head_committed = False
        self._committed = False
        self._clear_before: dict[str, _RegistrySnapshot] = {}
        self._external_commit_probe: Callable[[], bool] | None = None
        self._reconciliation_pending = False
        self._token: Any = None
        self._local_lock = threading.RLock()

    def __enter__(self) -> "SecretRegistryTransaction":
        if _ACTIVE_TRANSACTION.get() is not None:
            raise RuntimeError("secret registry transactions cannot be nested")
        _reconcile_uncertain_transactions()
        with _LOCK:
            self._start_clear_epoch = _CLEAR_EPOCH
            self._clear_base_revision = _REGISTRY_REVISION
        try:
            self._token = _ACTIVE_TRANSACTION.set(self)
            return self
        except BaseException:
            raise

    @property
    def prepared(self) -> bool:
        return self._prepared

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def product_head_committed(self) -> bool:
        return self._product_head_committed

    def read(self, reference: str) -> _RegistrySnapshot:
        reference = str(reference)
        with self._local_lock:
            if reference in self._writes:
                return self._writes[reference]
            if self._clear_requested:
                return _RegistrySnapshot(None, frozenset(), 0)
            if reference in self._observed:
                return self._observed[reference]
            if self._prepared or self._committed:
                raise SecretRegistryConflict(
                    "secret registry cannot add reads after prepare"
                )
            with _LOCK:
                if _CLEAR_EPOCH != self._start_clear_epoch:
                    raise SecretRegistryConflict(
                        "secret registry clear epoch changed"
                    )
                if _CLEAR_RESERVATION and _CLEAR_RESERVATION != self.transaction_id:
                    raise SecretRegistryConflict(
                        "secret registry clear is prepared by another transaction"
                    )
                owner = _RESERVATIONS.get(reference)
                if owner and owner != self.transaction_id:
                    raise SecretRegistryConflict(
                        "secret reference is prepared by another transaction"
                    )
                snapshot = _snapshot_locked(reference)
            self._observed[reference] = snapshot
            return snapshot

    def stage(self, reference: str, desired: _RegistrySnapshot) -> None:
        reference = str(reference)
        with self._local_lock:
            if self._prepared or self._committed:
                raise SecretRegistryConflict(
                    "secret registry cannot mutate after prepare"
                )
            if not self._clear_requested and reference not in self._observed:
                self.read(reference)
            self._writes[reference] = desired

    def clear(self) -> None:
        with self._local_lock:
            if self._prepared or self._committed:
                raise SecretRegistryConflict(
                    "secret registry cannot mutate after prepare"
                )
            self._clear_requested = True
            self._observed.clear()
            self._writes.clear()

    def prepare(self) -> None:
        global _CLEAR_RESERVATION

        with self._local_lock:
            if self._committed:
                return
            if self._prepared:
                return
            references = set(self._observed) | set(self._writes)
            with _LOCK:
                if _CLEAR_EPOCH != self._start_clear_epoch:
                    raise SecretRegistryConflict(
                        "secret registry clear epoch changed"
                    )
                if _CLEAR_RESERVATION and _CLEAR_RESERVATION != self.transaction_id:
                    raise SecretRegistryConflict(
                        "secret registry clear is prepared by another transaction"
                    )
                if self._clear_requested:
                    if _REGISTRY_REVISION != self._clear_base_revision:
                        raise SecretRegistryConflict(
                            "secret registry changed before clear prepare"
                        )
                    if any(
                        owner != self.transaction_id
                        for owner in _RESERVATIONS.values()
                    ):
                        raise SecretRegistryConflict(
                            "secret references are prepared by another transaction"
                        )
                    self._clear_before = {
                        reference: _snapshot_locked(reference)
                        for reference in {*_VALUES, *_SCOPES, *_VERSIONS}
                    }
                else:
                    for reference in references:
                        owner = _RESERVATIONS.get(reference)
                        if owner and owner != self.transaction_id:
                            raise SecretRegistryConflict(
                                "secret reference is prepared by another transaction"
                            )
                        observed = self._observed.get(reference)
                        if (
                            observed is not None
                            and _snapshot_locked(reference).version
                            != observed.version
                        ):
                            raise SecretRegistryConflict(
                                "secret reference version changed"
                            )
                if self._clear_requested:
                    _CLEAR_RESERVATION = self.transaction_id
                else:
                    for reference in references:
                        _RESERVATIONS[reference] = self.transaction_id
                    self._reserved_references = references
                self._prepared = True

    def publish_prepared(self) -> None:
        global _CLEAR_EPOCH, _CLEAR_RESERVATION, _REGISTRY_REVISION

        with self._local_lock:
            if self._committed:
                return
            if self._published:
                return
            if not self._prepared:
                raise SecretRegistryConflict(
                    "secret registry transaction must prepare before publish"
                )
            with _LOCK:
                if self._clear_requested:
                    if _CLEAR_RESERVATION != self.transaction_id:
                        raise SecretRegistryConflict(
                            "secret registry clear reservation was lost"
                        )
                    for reference in {*_VALUES, *_SCOPES}:
                        _VERSIONS[reference] = (
                            int(_VERSIONS.get(reference) or 0) + 1
                        )
                    _VALUES.clear()
                    _SCOPES.clear()
                    _CLEAR_EPOCH += 1
                    _REGISTRY_REVISION += 1
                else:
                    for reference in self._reserved_references:
                        if _RESERVATIONS.get(reference) != self.transaction_id:
                            raise SecretRegistryConflict(
                                "secret reference reservation was lost"
                            )
                for reference, desired in self._writes.items():
                    _publish_snapshot_locked(reference, desired)
            self._published = True

    def finalize(self) -> None:
        """Release reservations after the Product Head commit succeeds."""

        with self._local_lock:
            if self._committed:
                return
            if not self._published:
                raise SecretRegistryConflict(
                    "secret registry transaction must publish before finalize"
                )
            with _LOCK:
                self._release_reservations_locked()
            self._committed = True
            self._prepared = False
            self._published = False
            self._clear_local_state()

    def mark_product_head_committed(self) -> None:
        """Prevent rollback after the external Product Head becomes durable."""

        with self._local_lock:
            if not self._published:
                raise SecretRegistryConflict(
                    "Product Head cannot commit before registry publish"
                )
            self._product_head_committed = True

    def bind_external_commit_probe(self, probe: Callable[[], bool]) -> None:
        """Bind the durable commit decision before publishing the overlay."""

        with self._local_lock:
            if self._prepared or self._published or self._committed:
                raise SecretRegistryConflict(
                    "external commit probe must be bound before prepare"
                )
            self._external_commit_probe = probe

    def _external_commit_status(self) -> bool | None:
        probe = self._external_commit_probe
        if probe is None:
            return False
        try:
            return bool(probe())
        except BaseException:
            return None

    def commit(self) -> None:
        """Commit a registry-only transaction without an external participant."""

        if not self._prepared:
            self.prepare()
        self.publish_prepared()
        self.finalize()

    def rollback(self) -> None:
        """Discard or reverse a published overlay before Product Head commit."""

        reconciliation_pending = False
        with self._local_lock:
            commit_status: bool | None = False
            if self._published:
                commit_status = (
                    True
                    if self._product_head_committed
                    else self._external_commit_status()
                )
            with _LOCK:
                if self._published:
                    if commit_status is True:
                        self._release_reservations_locked()
                        self._committed = True
                    elif commit_status is None:
                        _UNCERTAIN_TRANSACTIONS[self.transaction_id] = self
                        self._reconciliation_pending = True
                        reconciliation_pending = True
                    else:
                        self._restore_published_overlay_locked()
                if not reconciliation_pending:
                    self._release_reservations_locked()
            if not reconciliation_pending:
                self._prepared = False
                self._published = False
                self._clear_local_state()
        if reconciliation_pending:
            raise SecretRegistryConflict(
                "secret registry commit status is uncertain; "
                "reconciliation is required"
            )

    def reconcile_uncertain(self) -> None:
        """Resolve one preserved overlay only from an authoritative decision."""

        with self._local_lock:
            if not self._reconciliation_pending:
                return
            commit_status = self._external_commit_status()
            if commit_status is None:
                raise SecretRegistryConflict(
                    "secret registry commit status remains uncertain"
                )
            with _LOCK:
                if _UNCERTAIN_TRANSACTIONS.get(self.transaction_id) is not self:
                    raise SecretRegistryConflict(
                        "secret registry reconciliation ownership was lost"
                    )
                if commit_status:
                    self._release_reservations_locked()
                    self._committed = True
                else:
                    self._restore_published_overlay_locked()
                    self._release_reservations_locked()
                _UNCERTAIN_TRANSACTIONS.pop(self.transaction_id, None)
            self._prepared = False
            self._published = False
            self._reconciliation_pending = False
            self._clear_local_state()

    def _restore_published_overlay_locked(self) -> None:
        global _CLEAR_EPOCH, _REGISTRY_REVISION

        if self._clear_requested:
            _VALUES.clear()
            _SCOPES.clear()
            for reference, snapshot in self._clear_before.items():
                if snapshot.record is not None:
                    _VALUES[reference] = snapshot.record
                    _SCOPES[reference] = set(snapshot.scopes)
                _VERSIONS[reference] = (
                    int(_VERSIONS.get(reference) or 0) + 1
                )
            _CLEAR_EPOCH += 1
            _REGISTRY_REVISION += 1
            return
        for reference in self._reserved_references:
            original = self._observed.get(
                reference,
                _RegistrySnapshot(None, frozenset(), 0),
            )
            _publish_snapshot_locked(reference, original)

    def _release_reservations_locked(self) -> None:
        global _CLEAR_RESERVATION

        for reference in self._reserved_references:
            if _RESERVATIONS.get(reference) == self.transaction_id:
                _RESERVATIONS.pop(reference, None)
        self._reserved_references.clear()
        if _CLEAR_RESERVATION == self.transaction_id:
            _CLEAR_RESERVATION = ""

    def _clear_local_state(self) -> None:
        self._observed.clear()
        self._writes.clear()
        self._clear_requested = False
        self._clear_before.clear()
        self._product_head_committed = False
        self._external_commit_probe = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if not self._committed:
                self.rollback()
        finally:
            if self._token is not None:
                _ACTIVE_TRANSACTION.reset(self._token)


def secret_registry_transaction() -> SecretRegistryTransaction:
    """Create one registry transaction around a Product Head attempt."""

    return SecretRegistryTransaction()


def _reconcile_uncertain_transactions() -> None:
    """Resolve preserved commit gaps before admitting another transaction."""

    with _LOCK:
        pending = tuple(_UNCERTAIN_TRANSACTIONS.values())
    for transaction in pending:
        transaction.reconcile_uncertain()


def _active_registry_transaction() -> SecretRegistryTransaction | None:
    return _ACTIVE_TRANSACTION.get()


def new_secret_reference() -> str:
    return f"semantic-secret:{secrets.token_urlsafe(24)}"


def secret_value_hash(value: str) -> str:
    """Return the deterministic hash used only for non-secret semantics."""

    encoded = json.dumps(
        str(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def secret_value_verifier(value: str, reference: str) -> str:
    """Return a memory-hard verifier salted by an opaque reference."""

    if not is_secret_reference(reference):
        raise ValueError("secret verifier requires an opaque reference")
    encoded = json.dumps(
        str(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.scrypt(
        encoded,
        salt=str(reference).encode("utf-8"),
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    ).hex()


def secret_value_matches(value: str, reference: str, expected_hash: str) -> bool:
    """Verify secret material without persisting or exposing the raw value."""

    try:
        actual = secret_value_verifier(value, reference)
    except ValueError:
        return False
    return secrets.compare_digest(actual, str(expected_hash))


def store_secret_reference(
    value: str,
    *,
    draft_id: str,
    atom_id: str,
    reference: str = "",
) -> tuple[str, str]:
    """Store one secret in memory and return its opaque reference and hash."""

    raw = str(value)
    if not raw or not draft_id or not atom_id:
        raise ValueError("secret reference requires value and exact scope")
    reference = str(reference or new_secret_reference())
    if not is_secret_reference(reference):
        raise ValueError("secret reference has an invalid identity")
    value_hash = secret_value_verifier(raw, reference)
    transaction = _active_registry_transaction()
    if transaction is not None:
        current = transaction.read(reference)
        transaction.stage(
            reference,
            _RegistrySnapshot(
                record=_SecretRecord(
                    draft_id=draft_id,
                    atom_id=atom_id,
                    value=raw,
                    value_hash=value_hash,
                ),
                scopes=frozenset({
                    *current.scopes,
                    (draft_id, atom_id),
                }),
                version=current.version,
            ),
        )
        return reference, value_hash
    with _LOCK:
        _assert_global_mutation_available_locked(reference)
        current = _snapshot_locked(reference)
        _publish_snapshot_locked(
            reference,
            _RegistrySnapshot(
                record=_SecretRecord(
                    draft_id=draft_id,
                    atom_id=atom_id,
                    value=raw,
                    value_hash=value_hash,
                ),
                scopes=frozenset({
                    *current.scopes,
                    (draft_id, atom_id),
                }),
                version=current.version,
            ),
        )
    return reference, value_hash


def authorize_secret_reference(
    reference: str,
    *,
    draft_id: str,
    atom_id: str,
    expected_hash: str,
) -> None:
    """Authorize an existing reference for one additional exact scope."""

    reference = str(reference)
    transaction = _active_registry_transaction()
    if transaction is not None:
        current = transaction.read(reference)
        if current.record is None or current.record.value_hash != expected_hash:
            raise ValueError("secret reference is unavailable or has changed")
        transaction.stage(
            reference,
            _RegistrySnapshot(
                record=current.record,
                scopes=frozenset({
                    *current.scopes,
                    (draft_id, atom_id),
                }),
                version=current.version,
            ),
        )
        return
    with _LOCK:
        _assert_global_mutation_available_locked(reference)
        current = _snapshot_locked(reference)
        if current.record is None or current.record.value_hash != expected_hash:
            raise ValueError("secret reference is unavailable or has changed")
        _publish_snapshot_locked(
            reference,
            _RegistrySnapshot(
                record=current.record,
                scopes=frozenset({
                    *current.scopes,
                    (draft_id, atom_id),
                }),
                version=current.version,
            ),
        )


def project_secret_input(
    value: str,
    *,
    scope_id: str,
    force_secret: bool = False,
) -> tuple[str, tuple[dict[str, str], ...]]:
    """Replace recognized credentials before state reaches LangGraph."""

    raw = str(value)
    if force_secret and raw.strip():
        secret = raw.strip()
        reference = new_secret_reference()
        atom_id = "input-secret-1"
        _, value_hash = store_secret_reference(
            secret,
            draft_id=scope_id,
            atom_id=atom_id,
            reference=reference,
        )
        return reference, ({
            "reference": reference,
            "draft_id": scope_id,
            "atom_id": atom_id,
            "value_hash": value_hash,
        },)
    projected = raw
    bindings: list[dict[str, str]] = []
    for index, secret in enumerate(secret_values(raw)):
        reference = new_secret_reference()
        atom_id = f"input-secret-{index + 1}"
        _, value_hash = store_secret_reference(
            secret,
            draft_id=scope_id,
            atom_id=atom_id,
            reference=reference,
        )
        projected = projected.replace(secret, reference)
        bindings.append({
            "reference": reference,
            "draft_id": scope_id,
            "atom_id": atom_id,
            "value_hash": value_hash,
        })
    return projected, tuple(bindings)


def _owned_secret_references(state: Mapping[str, Any]) -> set[str]:
    references: set[str] = set()
    draft = state.get("semantic_plan_draft") or {}
    if isinstance(draft, Mapping):
        for binding in draft.get("source_secret_bindings") or ():
            if isinstance(binding, Mapping):
                reference = str(binding.get("reference") or "")
                if reference:
                    references.add(reference)
        for atom in draft.get("unresolved_atoms") or ():
            if isinstance(atom, Mapping):
                reference = str(atom.get("resolution_ref") or "")
                if reference:
                    references.add(reference)

    def collect_envelope(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        metadata = value.get("admission_metadata") or {}
        if isinstance(metadata, Mapping):
            for binding in metadata.get("semantic_secret_bindings") or ():
                if isinstance(binding, Mapping):
                    reference = str(binding.get("reference") or "")
                    if reference:
                        references.add(reference)

    for item in state.get("action_queue") or ():
        collect_envelope(item)
    collect_envelope(state.get("selected_action"))
    collect_envelope(state.get("current_action"))
    pending_result = state.get("pending_domain_result") or {}
    if isinstance(pending_result, Mapping):
        collect_envelope(pending_result.get("action"))
    return references


def secret_references_in_value(value: Any) -> set[str]:
    """Return opaque references present in an arbitrary durable value."""

    if isinstance(value, str):
        return {match.group(0) for match in _REFERENCE_RE.finditer(value)}
    if isinstance(value, Mapping):
        return {
            reference
            for key, child in value.items()
            for candidate in (key, child)
            for reference in secret_references_in_value(candidate)
        }
    if isinstance(value, (list, tuple)):
        return {
            reference
            for child in value
            for reference in secret_references_in_value(child)
        }
    return set()


def redact_secret_references(value: Any) -> Any:
    """Replace opaque references in non-owning audit projections."""

    if isinstance(value, str):
        return _REFERENCE_RE.sub(
            lambda match: (
                "secret-reference-hash:"
                + hashlib.sha256(match.group(0).encode("utf-8")).hexdigest()
            ),
            value,
        )
    if isinstance(value, Mapping):
        projected: dict[Any, Any] = {}
        for key, child in value.items():
            projected_key = redact_secret_references(key)
            if projected_key in projected:
                raise ValueError(
                    "secret reference redaction creates a mapping key collision"
                )
            projected[projected_key] = redact_secret_references(child)
        return projected
    if isinstance(value, list):
        return [redact_secret_references(child) for child in value]
    if isinstance(value, tuple):
        return tuple(redact_secret_references(child) for child in value)
    return value


def state_owned_secret_references(state: Mapping[str, Any]) -> set[str]:
    return {
        reference
        for root in DURABLE_SECRET_CAPABLE_STATE_ROOTS
        for reference in secret_references_in_value(state.get(root))
    }


def raw_secret_paths_in_state(state: Mapping[str, Any]) -> tuple[str, ...]:
    """Return durable secret-capable paths containing unprojected material."""

    sensitive_keys = {
        str(key).casefold() for key in DURABLE_SECRET_CAPABLE_LEAF_KEYS
    }
    control_metadata_keys = {
        str(key).casefold()
        for key in DURABLE_SECRET_CONTROL_METADATA_LEAF_KEYS
    }
    paths: list[str] = []

    def mapping_key_contains_raw_material(key: str) -> bool:
        stripped = key.strip()
        if (
            not stripped
            or is_secret_reference(stripped)
            or _REFERENCE_HASH_RE.fullmatch(stripped)
        ):
            return False
        return bool(
            secret_values(stripped)
            or "://" in stripped
            or "\n" in stripped
            or "\r" in stripped
        )

    def collect(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                key_path = (*path, key_text)
                if mapping_key_contains_raw_material(key_text):
                    paths.append(".".join(key_path))
                collect(child, key_path)
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                collect(child, (*path, str(index)))
            return
        if not isinstance(value, str) or not value.strip():
            return
        raw = value.strip()
        if (
            raw in {"none", "<none>", "<not selected>", "***REDACTED***"}
            or is_secret_reference(raw)
            or _REFERENCE_HASH_RE.fullmatch(raw)
        ):
            return
        leaf = path[-1].casefold() if path else ""
        if leaf in control_metadata_keys:
            return
        if leaf in sensitive_keys or secret_values(raw):
            paths.append(".".join(path))

    for root in sorted(DURABLE_SECRET_CAPABLE_STATE_ROOTS):
        collect(state.get(root), (root,))
    return tuple(dict.fromkeys(paths))


def normalize_state_secret_binding(value: Mapping[str, Any]) -> SecretBinding:
    """Normalize and validate the sole durable secret-binding contract."""

    binding: SecretBinding = {
        "reference": str(value.get("reference") or ""),
        "scope_id": str(
            value.get("scope_id")
            or value.get("draft_id")
            or ""
        ),
        "atom_id": str(value.get("atom_id") or ""),
        "value_hash": str(value.get("value_hash") or ""),
    }
    if (
        not is_secret_reference(binding["reference"])
        or not binding["scope_id"]
        or not binding["atom_id"]
        or not re.fullmatch(r"[0-9a-f]{64}", binding["value_hash"])
    ):
        raise ValueError("state secret binding is invalid")
    return binding


def normalize_secret_reentry_binding(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize the coordinator binding around one durable secret owner."""

    expected_keys = {
        "scope_id",
        "owner_kind",
        "owner_id",
        "owner_revision",
        "reference",
        "atom_id",
        "value_hash",
    }
    if set(value) != expected_keys:
        raise ValueError("secret_reentry_binding has an invalid contract")
    owner_kind = str(value.get("owner_kind") or "")
    normalized: dict[str, Any] = {
        "scope_id": str(value.get("scope_id") or ""),
        "owner_kind": owner_kind,
        "owner_id": str(value.get("owner_id") or ""),
        "owner_revision": int(value.get("owner_revision") or 0),
        "reference": str(value.get("reference") or ""),
        "atom_id": str(value.get("atom_id") or ""),
        "value_hash": str(value.get("value_hash") or ""),
    }
    normalize_state_secret_binding(normalized)
    if (
        owner_kind not in {"semantic_draft", "durable_state"}
        or not normalized["owner_id"]
        or normalized["owner_revision"] < 0
    ):
        raise ValueError("secret_reentry_binding requires exact identities")
    return normalized


def register_state_secret_bindings(
    state: dict[str, Any],
    bindings: Any,
) -> None:
    """Transfer admitted binding metadata into the durable state authority."""

    existing = {
        str(item.get("reference") or ""): normalize_state_secret_binding(item)
        for item in state.get("secret_bindings") or ()
        if isinstance(item, Mapping)
    }
    for raw in bindings or ():
        if not isinstance(raw, Mapping):
            raise ValueError("state secret binding must be an object")
        binding = normalize_state_secret_binding(raw)
        prior = existing.get(binding["reference"])
        if prior is not None and prior != binding:
            raise ValueError("state secret reference binding changed")
        existing[binding["reference"]] = binding
    state["secret_bindings"] = [
        existing[key] for key in sorted(existing)
    ]


def reconcile_state_secret_bindings(state: dict[str, Any]) -> None:
    """Retain exactly the bindings referenced by durable workflow state."""

    live = state_owned_secret_references(state)
    existing = {
        str(item.get("reference") or ""): normalize_state_secret_binding(item)
        for item in state.get("secret_bindings") or ()
        if isinstance(item, Mapping)
    }
    missing = sorted(live - set(existing))
    if missing:
        raise ValueError(
            "durable state contains unbound secret references: "
            + ", ".join(missing)
        )
    for reference in set(existing) - live:
        discard_secret_reference(reference)
    state["secret_bindings"] = [
        existing[reference] for reference in sorted(live)
    ]


def missing_state_secret_bindings(
    state: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Return exact durable bindings whose process-local material is absent."""

    missing: list[dict[str, str]] = []
    live = state_owned_secret_references(state)
    for raw in state.get("secret_bindings") or ():
        if not isinstance(raw, Mapping):
            continue
        binding = normalize_state_secret_binding(raw)
        if binding["reference"] not in live:
            continue
        if not secret_reference_available(
            binding["reference"],
            draft_id=binding["scope_id"],
            atom_id=binding["atom_id"],
            expected_hash=binding["value_hash"],
        ):
            missing.append(binding)
    return missing


def materialize_state_secret_references(
    value: Any,
    state: Mapping[str, Any],
    *,
    skip_keys: frozenset[str] = frozenset(),
) -> Any:
    """Resolve state-owned references only at an authorized invocation edge."""

    bindings = {
        str(item.get("reference") or ""): normalize_state_secret_binding(item)
        for item in state.get("secret_bindings") or ()
        if isinstance(item, Mapping)
    }

    def materialize(item: Any) -> Any:
        if isinstance(item, str):
            result = item
            for reference in secret_references_in_value(item):
                binding = bindings.get(reference)
                if binding is None:
                    raise ValueError("secret reference has no durable binding")
                secret = resolve_secret_reference(
                    reference,
                    draft_id=binding["scope_id"],
                    atom_id=binding["atom_id"],
                    expected_hash=binding["value_hash"],
                )
                if secret is None:
                    raise ValueError("state secret reference is unavailable")
                result = result.replace(reference, secret)
            return result
        if isinstance(item, Mapping):
            result: dict[Any, Any] = {}
            for key, child in item.items():
                materialized_key = materialize(key)
                if materialized_key in result:
                    raise ValueError(
                        "secret materialization creates a mapping key collision"
                    )
                result[materialized_key] = (
                    child if str(key) in skip_keys else materialize(child)
                )
            return result
        if isinstance(item, list):
            return [materialize(child) for child in item]
        if isinstance(item, tuple):
            return tuple(materialize(child) for child in item)
        return item

    return materialize(value)


def project_state_secret_values(
    value: Any,
    state: Mapping[str, Any],
) -> Any:
    """Replace raw state-owned material before a value becomes durable."""

    replacements: dict[str, str] = {}
    for raw in state.get("secret_bindings") or ():
        if not isinstance(raw, Mapping):
            continue
        binding = normalize_state_secret_binding(raw)
        secret = resolve_secret_reference(
            binding["reference"],
            draft_id=binding["scope_id"],
            atom_id=binding["atom_id"],
            expected_hash=binding["value_hash"],
        )
        if secret is None:
            raise ValueError("state secret reference is unavailable")
        replacements[secret] = binding["reference"]

    def project(item: Any) -> Any:
        if isinstance(item, str):
            result = item
            for secret, reference in replacements.items():
                result = result.replace(secret, reference)
            return result
        if isinstance(item, Mapping):
            result: dict[Any, Any] = {}
            for key, child in item.items():
                projected_key = project(key)
                if projected_key in result:
                    raise ValueError(
                        "secret projection creates a mapping key collision"
                    )
                result[projected_key] = project(child)
            return result
        if isinstance(item, list):
            return [project(child) for child in item]
        if isinstance(item, tuple):
            return tuple(project(child) for child in item)
        return item

    return project(value)


def validate_state_secret_bindings(state: Mapping[str, Any]) -> None:
    bindings = [
        normalize_state_secret_binding(item)
        for item in state.get("secret_bindings") or ()
        if isinstance(item, Mapping)
    ]
    if len(bindings) != len(state.get("secret_bindings") or ()):
        raise ValueError("state secret bindings must contain only objects")
    references = [item["reference"] for item in bindings]
    if len(references) != len(set(references)):
        raise ValueError("state secret bindings contain duplicate references")
    live = state_owned_secret_references(state)
    if live != set(references):
        raise ValueError("state secret binding ownership does not match durable state")


def release_unowned_input_secret_bindings(
    bindings: tuple[Mapping[str, str], ...],
    state: Mapping[str, Any],
) -> None:
    """Release turn-scoped material not transferred to a durable owner."""

    retained = {
        *_owned_secret_references(state),
        *state_owned_secret_references(state),
    }
    for binding in bindings:
        reference = str(binding.get("reference") or "")
        if reference and reference not in retained:
            discard_secret_reference(reference)


def all_owned_secret_references(state: Mapping[str, Any]) -> set[str]:
    """Return every reference owned by durable state or in-flight envelopes."""

    return {
        *_owned_secret_references(state),
        *state_owned_secret_references(state),
        *(
            str(item.get("reference") or "")
            for item in state.get("secret_bindings") or ()
            if isinstance(item, Mapping)
            and str(item.get("reference") or "")
        ),
    }


def discard_state_secret_references(state: Mapping[str, Any]) -> None:
    """Release every process-local secret currently owned by one state."""

    for reference in all_owned_secret_references(state):
        discard_secret_reference(reference)


def resolve_secret_reference(
    reference: str,
    *,
    draft_id: str,
    atom_id: str,
    expected_hash: str,
) -> str | None:
    """Resolve only a reference bound to the exact draft atom and hash."""

    reference = str(reference)
    transaction = _active_registry_transaction()
    if transaction is not None:
        snapshot = transaction.read(reference)
    else:
        with _LOCK:
            _assert_global_mutation_available_locked(reference)
            snapshot = _snapshot_locked(reference)
    if (
        snapshot.record is None
        or snapshot.record.value_hash != expected_hash
        or (draft_id, atom_id) not in snapshot.scopes
    ):
        return None
    return snapshot.record.value


def secret_reference_available(
    reference: str,
    *,
    draft_id: str,
    atom_id: str,
    expected_hash: str,
) -> bool:
    """Return whether exact scoped material is available in this process."""

    return resolve_secret_reference(
        reference,
        draft_id=draft_id,
        atom_id=atom_id,
        expected_hash=expected_hash,
    ) is not None


def restore_secret_reference(
    value: str,
    *,
    reference: str,
    draft_id: str,
    atom_id: str,
    expected_hash: str,
) -> None:
    """Restore one lost process-local reference without changing its identity."""

    raw = str(value).strip()
    candidates = tuple(dict.fromkeys((raw, *secret_values(raw))))
    matching = [
        candidate
        for candidate in candidates
        if candidate
        and secret_value_matches(candidate, reference, expected_hash)
    ]
    if len(matching) != 1:
        raise ValueError("re-entered secret does not match the bound value hash")
    store_secret_reference(
        matching[0],
        draft_id=draft_id,
        atom_id=atom_id,
        reference=reference,
    )


def discard_secret_reference(reference: str) -> None:
    if not reference:
        return
    reference = str(reference)
    transaction = _active_registry_transaction()
    if transaction is not None:
        current = transaction.read(reference)
        transaction.stage(
            reference,
            _RegistrySnapshot(
                record=None,
                scopes=frozenset(),
                version=current.version,
            ),
        )
        return
    with _LOCK:
        _assert_global_mutation_available_locked(reference)
        current = _snapshot_locked(reference)
        _publish_snapshot_locked(
            reference,
            _RegistrySnapshot(
                record=None,
                scopes=frozenset(),
                version=current.version,
            ),
        )


def discard_draft_secret_references(draft: Mapping[str, object]) -> None:
    for binding in draft.get("source_secret_bindings") or ():
        if isinstance(binding, Mapping):
            discard_secret_reference(str(binding.get("reference") or ""))
    for atom in draft.get("unresolved_atoms") or ():
        if isinstance(atom, Mapping):
            discard_secret_reference(str(atom.get("resolution_ref") or ""))


def clear_secret_references() -> None:
    """Clear ephemeral references during tests or process shutdown."""

    global _CLEAR_EPOCH, _REGISTRY_REVISION

    transaction = _active_registry_transaction()
    if transaction is not None:
        transaction.clear()
        return
    with _LOCK:
        if _CLEAR_RESERVATION or _RESERVATIONS:
            raise SecretRegistryConflict(
                "secret registry cannot clear prepared references"
            )
        for reference in {*_VALUES, *_SCOPES}:
            _VERSIONS[reference] = int(_VERSIONS.get(reference) or 0) + 1
        _VALUES.clear()
        _SCOPES.clear()
        _CLEAR_EPOCH += 1
        _REGISTRY_REVISION += 1
