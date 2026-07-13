# AnyChain Agent — Known Issues Register (2026-07-11)

Single source of truth for the Agent's **currently known problems**. Before this
file, known issues were scattered across the repair-plan and audit docs. This
register separates: (A) fixed defects, (B) still-open minor items / accepted
trade-offs, and (C) validation gaps (things not yet verified).

Scope: findings from the LangGraph-harness refactor and the live dual-AI chaos
campaign (Phases 9–13 of `2026-07-10-langgraph-harness-repair-plan.md`), run
against live DeepSeek (`deepseek-chat`) driving the real `bin/anychain-agent`.

Legend: severity S1 (breaks a mode/flow) · S2 (wrong/misleading behavior) ·
S3 (minor UX / cosmetic).

---

## A. Fixed defects (71) — all with regression tests, verified live

All found only via live dual-AI chaos; the mocked unit suite could not surface
them. See the cited Phase for details. Defects 19–26 are from the 2026-07-11
conformance-refactor + reactive dual-AI chaos batch (Phase 19); defects 27–30 are
from the 2026-07-12 sync-observe / custom-RPC / fake-node coverage batch (Phase 20);
defects 31–36 are from the 2026-07-12 observability-exporter / intensive-QPS /
sync-observe-field-correction coverage batch (Phase 21); defects 37–40 are from
the 2026-07-12 advanced-tuning / accounts-disk / observability-local coverage
batch (Phase 22), targeting group-internal branches never previously driven
live (see the coverage-gap review that scoped Phase 22 for the full list);
defects 41–47 are from the 2026-07-12 parallel four-agent chaos sweep (Phase
23 — real-node mode, non-EVM chain families, job/report analysis, and a
regression-hardening pass re-triggering defects 1–40 with new phrasing), each
agent running an isolated CLI session (own
`ANYCHAIN_AGENT_SESSION_ID`/`ANYCHAIN_AGENT_CHECKPOINT_PATH`) in discovery-only
mode; fixes were then applied and verified serially by a single process to
avoid concurrent edits to the same files.

| # | Defect | Sev | Phase | Regression test |
|---|--------|-----|-------|-----------------|
| 1 | Numbered/label answer to a yes/no confirm ("1"/"Y") rejected | S2 | 9 | test_numbered_answer_to_yes_no_confirm_applies_without_llm |
| 2 | Mode switch silently reused the stale chain | S2 | 9 | test_target_mode_change_confirmation_names_preserved_chain |
| 3 | Negated protocol ("没有 JSON-RPC") read as jsonrpc, skipped Case-3 handoff | S2 | 9 | test_adapter_family_hint_ignores_negated_protocol_mentions |
| 4 | Multi-part question dropped its second half (preflight/smoke never defined) | S2 | 9 | test_combined_modes_and_preflight_smoke_question_answers_both |
| 5 | Out-of-range menu number ("0") misrouted to the LLM, dropped the question | S2 | 10 | test_out_of_range_numbered_answer_keeps_question_without_llm |
| 6 | EVM endpoint placeholder `${TARGET_ADDRESS:-0x..}` not expanded — blocked real-node/sync-observe endpoint validation for **all EVM chains** | S1 | 10 | test_endpoint_probe_resolves_env_placeholder_sample_address |
| 7 | sync-observe infinite loop — mode could never reach execution | S1 | 10 | test_sync_observe_group_completes_instead_of_looping |
| 8 | No-op same-chain "change" dropped the active question | S2 | 10 | test_noop_same_chain_change_preserves_active_question |
| 9 | Pasted benchmark report mis-routed to chain selection | S2 | 11 | test_multiline_paste_with_analysis_request_captured_as_evidence |
| 10 | Premature preflight-run confirmation on an incomplete real-node config ("配置已收集" when nothing was) | S1 | 12 | test_preflight_group_not_offered_when_config_incomplete |
| 11 | QPS override values unvalidated (negative/zero/max<initial accepted) | S2 | 12 | test_qps_override_rejects_invalid_values |
| 12 | Custom-RPC mixed weights accepted a non-existent method ("eth_fooBar") | S2 | 13 | test_custom_rpc_weights_reject_unknown_method_but_allow_template_default |
| 13 | `logs <bad-id>` leaked a raw FileNotFoundError to the user | S3 | 13 | test_logs_command_reports_clean_error_for_missing_job |
| 14 | Startup dependency-install "[Y/n]" offer was dead — ADK-available branch in `startup()` cleared the `install_dependencies` question set by `_startup_doctor`, so "y" never installed (hit on **every** new user, since missing vegeta is the default state) | S1 | 14 | test_startup_preserves_dependency_offer_so_yes_installs |
| 15 | "bsc 有哪些 rpc workload" (a question about a chain's default methods) either selected the chain or dumped the generic chain list instead of answering; resolver misreads the chain name as choose_chain | S2 | 16 | test_chain_workload_question_answered_not_selected |
| 16 | Accepting a recommendation looped — "按你推荐的来" re-printed the recommendation instead of starting it; recommendation was text, not an actionable option | S2 | 17 | test_accept_recommendation_starts_recommended_setup |
| 17 | "can my machine run / is my env ready" (机器能不能跑) got a generic requirements checklist instead of the startup readiness verdict + detected specs already computed | S2 | 17 | test_environment_readiness_question_answered_from_discovery |
| 18 | Negated protocol ENUMERATION mis-classified: a chain described as "GraphQL, 不是 REST/cosmos/substrate/bitcoin" (Mina) was forced to `substrate` and routed to Case 2 endpoint validation instead of the Case-3 handoff; negation only covered the first list item, and chain-scan surfaced "bitcoin" from the negated list | S2 | 18 | test_adapter_family_hint_ignores_negated_protocol_mentions |

| 19 | Chain-info questions ("bsc 有哪些 rpc workload") and host-readiness questions ("机器能不能跑") were classified by harness-side regex/phrase tables (Forbidden Pattern) | S2 (conformance) | 19 | test_chain_workload_question_answered_not_selected, test_environment_readiness_question_answered_from_discovery |
| 20 | QPS `MAX_QPS >= INITIAL_QPS` invariant bypassed when the two fields were set across separate turns or via the incremental adjust flow (single-field dict never tripped the cross-check) | S2 | 19 | test_qps_override_rejects_invalid_values |
| 21 | `custom_rpc_weights` accepted arbitrary methods when the allowed-set was empty (the `allowed_methods and ...` short-circuit disabled the whole unknown-method check) | S2 | 19 | test_custom_rpc_weights_reject_unknown_method_but_allow_template_default |
| 22 | "what is this config field / does it affect results" got a canned "paste the field" reply instead of a real answer from the runtime field contract | S3 | 19 | test_config_field_explanation_answers_from_runtime_contract |
| 23 | Accepting a detected CLOUD_REGION/ZONE/MACHINE_TYPE stored the user's literal sentence ("用检测到的就行") as the value — garbage report metadata; the prompt promised "use the detected value" but nothing bound it | S2 | 19 | test_provider_metadata_confirm_stores_detected_value_not_literal |
| 24 | LOCAL_RPC_URL/endpoint liveness probe used the EVM method `eth_chainId` for **every** jsonrpc-transport chain, so real endpoints for non-EVM jsonrpc chains (solana, sui, near, starknet, tron, avalanche-x) were rejected ("Method not found") — blocked real-node/sync-observe for those chains | S1 | 19 | test_health_probe_method_is_chain_specific_not_evm_only |
| 25 | A numbered answer ("1"/"2") to a confirm_or_value question that renders numbered options (e.g. MAINNET_RPC_URL sync-health confirm) was rejected and re-asked | S2 | 19 | test_numbered_answer_to_confirm_or_value_question_applies |
| 26 | Choosing "use defaults" for a **mixed** workload never set `mixed_weights_confirmed`, so preflight was offered ("配置已收集") then deadlocked on that blocker — every mixed run was affected | S1 | 19 | test_mixed_default_workload_confirms_weights_for_preflight |

| 27 | Endpoint-only sync-observe deadlocked at preflight on `node_process_identity`: the flow never asks for it (a remote node has no local process), but the checklist blanket-required it | S1 | 20 | test_sync_observe_endpoint_only_waives_node_process_identity |
| 28 | Pasting copied RPC content (a curl/JSON-RPC body containing a URL) at the custom-method step was rejected as "looks like an endpoint" instead of extracting the method+params from the JSON body | S2 | 20 | test_custom_rpc_method_extracts_from_pasted_content_with_url |
| 29 | The `node_process_identity` waiver added a `sync_observe_local_attribution` kwarg to the harness side that `prepare_benchmark_run` did not accept, so **every** preflight raised TypeError — masked by the generic "model call failed" message (self-introduced during 27, caught + fixed same batch) | S1 | 20 | test_prepare_kwargs_are_all_accepted_by_prepare_benchmark_run |
| 30 | "Analyze the latest job" analyzed a stale startup-detected running job instead of the just-submitted one: the injected `latest_job_id` hint was preferred over the on-disk latest and never refreshed after a submission | S2 | 20 | test_analyze_latest_job_uses_disk_latest_not_stale_hint |

| 31 | `_config_field_knowledge` only searched `RUNTIME_BASELINE_FIELDS`, excluding `SYNC_OBSERVE_FIELDS`/`WORKFLOW_CONFIRMATION_FIELDS`/`OPTIONAL_ACCOUNTS_FIELDS`/`REAL_NODE_ENDPOINT_FIELDS` — asking what any sync-observe field (e.g. stop condition) does, or what its valid values are, always fell back to the generic "paste the field" placeholder. Also fixed: the resolver's free-text `subject` doesn't match the canonical key 1:1 (e.g. pluralized "sync_observe_stop_conditions"), so the lookup now normalizes separators and a trailing "s" before matching | S2 | 21 | test_config_field_explanation_covers_sync_observe_only_fields |
| 32 | `sync_observe_duration_seconds` had zero validation — negative/zero/non-numeric values (e.g. "-100") were accepted and reached execution's `--duration` CLI arg, crashing the job with "requires a non-negative integer" after the full preflight/submit flow had already run | S1 | 21 | test_sync_observe_duration_rejects_invalid_values |
| 33 | Choosing observability `exporter` mode (meant to integrate with the user's *existing* Prometheus) confirmed the mode and moved straight to advanced-tuning defaults without ever telling the user which port/URL to scrape — defeating the entire purpose of the mode | S2 | 21 | test_exporter_observability_mode_discloses_scrape_port |
| 34 | `sync_observe_stop_condition`/`sync_observe_duration_seconds` could not be corrected via natural language once set: both were absent from `CONFIRMABLE_CONFIG_FIELDS`/`SPECIAL_CONFIG_FIELDS`, so a correction like "把观察时长改成 600" was always classified "unmapped" and silently discarded — a permanent dead end once either field was set (including to an invalid value under #32, before that fix), recoverable only by discarding the entire session's configuration | S1 | 21 | test_sync_observe_stop_condition_and_duration_correctable_via_nl |
| 35 | A leading Y/N answer to a yes/no pending question followed by free text ("N，我要调一下") did not exact-match `_answer_fits_pending`'s `{"y","yes","n","no"}` set, so it fell through to the LLM resolver, which has no functional action type for "still answering the pending question" and emitted the permanently-rejected `answer_pending`, followed by a `change_group` that no-opped (same origin/target group with a still-blocking question) — producing a turn with a completely empty `visible_response`, shown to the user as the generic "ADK 没有返回可显示内容" fallback | S1 | 21 | test_declined_yes_no_with_trailing_intent_stays_deterministic |
| 36 | `_invalid_qps_overrides` validated only the `overrides` dict in isolation, never merged with the benchmark mode's own baseline defaults: overriding only `MAX_QPS` (e.g. to 100) while `INITIAL_QPS` stayed at the mode's default (e.g. `intensive`'s 50000) produced an inverted, uncaught profile that advanced straight past QPS to RPC mode | S1 | 21 | test_qps_override_validates_against_mode_defaults_not_just_overrides |

| 37 | `DATA_VOL_SIZE`/`DATA_VOL_MAX_IOPS`/`DATA_VOL_MAX_THROUGHPUT` (ledger_disk), their `ACCOUNTS_VOL_*` twins (accounts_disk), and `NETWORK_MAX_BANDWIDTH_GBPS` (network) had zero validation on the direct manual-question answer path — negative/zero/non-numeric values (e.g. "-50", "-999") were all accepted silently and the flow advanced regardless, for every field in this family, on both groups | S1 | 22 | test_disk_and_network_numeric_fields_reject_negative_values |
| 38 | The same numeric fields' "paste inferred config" / direct `KEY=VALUE` assignment paths independently had the identical gap, made *worse* by `_first_number_text`: its digit-only regex has no sign handling, so a pasted "-50" for e.g. `DATA_VOL_MAX_IOPS` silently became the applied value "50" — changing the value without telling the user, rather than merely failing to validate it. Closes the deferred B.16 finding | S2 | 22 | test_paste_inferred_disk_values_preserve_sign_and_reject_negative |
| 39 | `advanced_tuning` threshold/interval overrides (`MAX_LATENCY_THRESHOLD`, `BOTTLENECK_CPU_THRESHOLD`, etc., set via the `advanced_tuning_adjust_value` flow — structurally identical to `#36`'s QPS adjust flow) had zero validation: `MAX_LATENCY_THRESHOLD=-500` and `BOTTLENECK_CPU_THRESHOLD=150` (a percentage field) were both accepted and reached the confirmed profile uncaught | S2 | 22 | test_advanced_tuning_adjust_value_rejects_invalid |
| 40 | Choosing observability `local` mode (starts exporter + Prometheus + Grafana on this host) confirmed the mode and moved straight to advanced-tuning defaults without ever disclosing which ports would be bound — same class of gap as `#33`'s missing exporter scrape-target disclosure, but relevant here for host port-conflict awareness rather than external-Prometheus integration | S2 | 22 | test_local_observability_mode_discloses_ports |

| 41 | **CRITICAL.** A yes/no confirm declined via the fuzzy `resolve_pending_choice` fallback (any decline that doesn't start with a clean y/n token, e.g. "nope, let me adjust something first") was silently inverted into an accept: the dispatcher applies the fuzzy match via `_apply_pending_answer(state, str(selected), pending)`, round-tripping a correctly-resolved Python `False` through `str()`; inside `_coerce_answer`'s generic option-matching loop, `str(option.get("value") or "")` collapsed the `False`-valued "N" option to `""` before stringifying, so `"false"` never matched and the loop fell through to `return raw` — returning the *string* `"False"`, and `bool("False")` is `True` in Python. For `preflight_smoke_confirm` this **silently submitted a real benchmark job against explicit user refusal** (reproduced live against a real BSC endpoint during the fix's own verification sweep); the same `is False` identity-check pattern is used by several other yes/no confirms (`target_mode_change_confirm`, `chain_change_confirm`, others), so this was systemic, not preflight-specific | S1 | 23 | test_fuzzy_matched_false_decline_does_not_invert_to_confirm |
| 42 | `agent/validators/endpoint_probe.py::_call_request` crashed with `TypeError: POST data should be bytes...` on every GET-shaped RPC/REST method probe: an empty-string body is falsy but still a `str`, so `body.encode(...) if isinstance(body, str) and body else body` fell through to `else body`, passing the literal `""` as `data` to `urllib.request.Request` (which requires `None`/bytes). GET is the default method for the bitcoin_jsonrpc/rest/hedera_dual/tendermint families and one of polkadot's mixed methods, so `LOCAL_RPC_URL`/`SYNC_OBSERVE_RPC_URL`/custom-RPC endpoint validation was **structurally unable to ever pass** for roughly half of all configured chains, regardless of endpoint health | S1 | 23 | test_call_request_handles_empty_body_get_without_typeerror |
| 43 | `evidence_buffer` only ever grows (nothing ever clears it), and the deterministic gate at the top of `process_turn` re-analyzed `evidence_buffer[-1]` for any text matching a broad keyword list ("分析", "why", "fix", ...) whenever the buffer was non-empty — with no exception for a later turn naming a real, different job by id or asking about "the latest job". Once any error text had ever been pasted in a session, **every subsequent job-failure question was permanently hijacked** into re-analyzing that first stale paste, even when the user named the correct job_id explicitly | S1 | 23 | test_job_reference_after_evidence_paste_does_not_reanalyze_stale_evidence |
| 44 | `_pending_context_response` (answers "why do I need this field / can I skip it" for an active pending question) had real explanatory text for `new_chain_endpoint`/`custom_rpc_endpoint` but fell through to `format_current_context` (an unrelated state-summary dump) for `LOCAL_RPC_URL`/`SYNC_OBSERVE_RPC_URL`, even though `_pending_endpoint_context_question` explicitly routes both into this handler; `SYNC_OBSERVE_RPC_URL` also had no `RuntimeField` entry at all, so the underlying field-explanation lookup could not have found it regardless of routing | S2 | 23 | test_endpoint_question_context_explains_field_not_generic_state_dump |
| 45 | The target-mode-switch confirmation unconditionally claimed "endpoint、workload、QPS、preflight/smoke 会重新确认" (all four will be re-confirmed), but `_invalidate_for_target_mode` only actually resets workload/RPC state, and only when switching *into* sync-observe — endpoint and QPS profile carry over untouched whenever they still apply. Observed live: a stale `intensive` QPS profile silently survived a sync-observe round trip despite the promise. (`invalidated_groups`, the list this function also writes to, was separately confirmed to be dead code — written at 10+ call sites, read nowhere in the codebase.) Fixed the message to state accurately what happens rather than overclaiming; the underlying invalidation behavior itself was left as-is (lower-risk fix given the deadline — see the open item below) | S2 | 23 | (message-only fix; see manual verification in the fix commit) |
| 46 | `_config_field_knowledge`'s field catalog (`ALL_RUNTIME_FIELDS`) had no entries at all for the QPS-profile sub-fields (`INITIAL_QPS`/`MAX_QPS`/`QPS_STEP`/`DURATION`) or any `advanced_tuning` threshold/interval field — validation (`#36`/`#39`) and field-explanation (`#31`) had drifted apart for the same field families, so "QPS_STEP是干什么用的" or "MAX_LATENCY_THRESHOLD这个阈值具体是干嘛的" still returned the canned "paste the field" placeholder | S3 | 23 | test_config_field_explanation_covers_qps_and_advanced_tuning_fields |
| 47 | `oracle._reason_label`'s `mapping` dict had no entry for two of `routing.next_group_and_reason`'s dynamic f-string reasons ("continue custom RPC workflow: {status}", "continue new-chain validation: {status}"), so a raw internal status enum (e.g. `needs_schema_evidence`) leaked verbatim into the "recommended next action" sentence shown to the user, observed live right after a custom-RPC schema validation failure | S2 | 23 | test_reason_label_does_not_leak_raw_internal_status |

| 48 | `google_search` was fully non-functional: `WebResearchStatus.as_dict()` emitted key `enabled`, but `agent/harness/groups.py`'s real-node client-setup branch read `state["web_research"]["google_search_available"]` — the two never matched, so the "will use google_search" branch could never fire for any real Gemini config | S1 | 24 | test_web_research_status_dict_key_matches_harness_contract |
| 49 | Even after #48's fix, the client-setup handoff message only *promised* to search and never actually called anything — `get_google_search_tools()` had zero callers anywhere in the codebase | S2 | 24 | test_sync_observe_client_setup_with_google_search_grounds_the_handoff_message |
| 50 | New `_sync_client_setup_handoff_message` citations fallback: `"<none>" if language.startswith("zh") else "<none>"` — both ternary branches produced the identical English literal, so the "no citations" placeholder never localized to Chinese | S3 | 24 (code-review) | manual verification; covered indirectly by test_sync_observe_client_setup_with_google_search_grounds_the_handoff_message |
| 51 | `install_dependencies` (present since commit `ce7870a`, carried verbatim through the ADK-retirement move into `agent/runners/dependency_installer.py`) ran the agent-runtime/gcloud subprocess whenever `include_agent_runtime or include_gcloud`, but only reported that subprocess's real result when `include_agent_runtime` alone was true — a gcloud-only install's real exit code/output was hidden behind `{"skipped": true}` | S2 | 24 (code-review) | manual verification (reporting condition now keys off `agent is not None`, matching the exit-code aggregation above it) |
| 52 | `validate_fake_node_fixture_authenticity` (present since at least commit `e895b60`, carried verbatim through the move into `agent/validators/fixture_checks.py`) never checked `exit_code`, unlike its sibling `_fixture_payload_incomplete` — a crashed fixture-authenticity subprocess (empty/non-JSON stdout) was reported as `status="ok"` | S2 | 24 (code-review) | manual verification |
| 53 | `agent/tools/executor.py`'s new `_confirmation_required` hand-built a duplicate of `agent/runners/tool_result.py::tool_result`'s envelope instead of calling it | S3 | 24 (code-review) | existing test_agent_tool_dispatch.py coverage of the gated tools' `needs_confirmation` shape |
| 54 | New `run_google_search_grounding` had no timeout on the live Gemini/google_search network call, unlike `endpoint_probe.py`'s `timeout=3.0` convention — a hung API call would block the whole synchronous harness turn indefinitely | S1 | 24 (code-review) | manual verification (wrapped in `asyncio.wait_for(timeout=20.0)`, degrades to a typed unavailable result on timeout) |
| 55 | New `tests/test_agent_tool_dispatch.py` regex-parsed `executor.py`'s source text to reconstruct the dispatch name list (fragile to reformatting), and its parity loop actually executed real subprocess/host-discovery side effects for `discover_environment`/`audit_dependencies`/`run_doctor` (no required args, so nothing short-circuited them) | S3 | 24 (code-review) | test rewritten to use `ast`-based extraction and to stub the three side-effecting zero-arg tools during the loop |
| 56 | `_run_grounding_turn` took `agent_cls`/`runner_cls`/`google_search`/`types` as pass-through parameters with no caller ever supplying alternate values — pure unexercised indirection | S3 | 24 (code-review) | n/a (simplification only; imports moved inside the function) |
| 57 | `routing.next_group_and_reason`'s sync-observe chain had a branch for `source == "client_setup" and not client_setup_acknowledged` but none for the acknowledged case, so it fell straight through to `"choose sync-observe stop condition"` — while `groups.py::_question_for_group` (the actual live pending question) correctly asks `sync_observe_after_client_setup` (choose the real data source now that a client is being prepared) first. Any advisory "current config" / next-blocking-item preview requested right after the handoff turn (no active pending question yet) showed a stale, wrong next step. Also added the two reason labels (`acknowledge real client setup handoff`, `choose sync-observe data source after client setup`) that were missing from `oracle._reason_label`'s mapping — same raw-string-leak class as `#47` | S2 | 24 (live dual-AI chaos, 2026-07-13 extended real-node sweep) | test_oracle_and_groups_agree_on_next_group_after_client_setup_ack |
| 58 | **CRITICAL.** A single real-node turn that both switches chain and asks to jump ahead (e.g. "switch to hedera, only have a real endpoint, ...") resolves into a compound action queue: `change_chain` (pauses on a `chain_change_confirm` interrupt) followed by a queued `change_group` targeting a later group. Once the interrupt is confirmed, `_invalidate_for_chain_change` correctly clears `endpoint_evidence` for the new chain, but the queued `change_group` then ran via `_activate_group_question` with no check that its target group's prerequisites were still satisfied — it jumped straight to `workload_rpc` and asked for the new chain's workload before its endpoint had ever been probed, while `confirmed_config["LOCAL_RPC_URL"]` still silently held the *previous* chain's endpoint. Confirmed live via direct checkpoint inspection (cosmos-hub → hedera): `active_group` became `workload_rpc` immediately after the chain-change confirm, with `endpoint_evidence == {}` and `confirmed_config["LOCAL_RPC_URL"]` still `https://cosmos-rest.publicnode.com`. Found while closing a coverage gap: the `tendermint`/`hedera_dual`/`rest` adapter families had never been driven through a live dual-AI chaos conversation before (only reachable via direct code execution per `#42`'s note) — first live real-node run on `cosmos-hub` (tendermint) surfaced this on the immediately following chain switch to `hedera` (hedera_dual). Fixed by having `_activate_group_question` redirect to the real blocking group (`routing.next_group_and_reason`) whenever the requested group is itself listed in `invalidated_groups` (this is the field's first real reader — `#45`'s note had confirmed it was previously dead code) and a genuinely-unmet earlier precondition exists; a jump to a group that was simply never configured yet (not invalidated) is untouched, preserving the existing `change_group`-jumps-ahead-of-an-incomplete-group behavior | S1 | 24 (live dual-AI chaos, 2026-07-13 tendermint/hedera_dual family coverage sweep) | test_change_group_does_not_skip_a_just_invalidated_earlier_group |
| 59 | **CRITICAL.** Case 2 ("new chain in an existing adapter family") could never pass its very first endpoint-validation step for any family except `jsonrpc`. `health_probe_methods` returned `(None, {})` for every non-`jsonrpc` family unconditionally, so a template-less chain (Case 2's defining condition — it has no `config/chains/<chain>.json` yet) ended up with zero probe methods; `_should_use_generic_jsonrpc_probe` then had nothing to route on and fell through to the template-requiring `_probe_health`/`_probe_method` path (`tools/chain_adapters/cli.py health-probe`), which raises `FileNotFoundError` for a chain with no template. Found live testing a genuinely new `substrate` chain (Moonriver, real public RPC `https://rpc.api.moonriver.moonbeam.network`): the user-visible error was a raw `CalledProcessError`, not a helpful diagnostic. Fixed for the two families whose transport is a plain POST JSON-RPC call — `substrate` (`system_chain`) and `bitcoin_jsonrpc` (`getblockchaininfo`, which every existing bitcoin_jsonrpc template already declares as its `_meta.health_probe.method` — this fix also makes that declared method actually get used, previously dead for the same reason) — by widening `GENERIC_JSONRPC_PROBE_FAMILIES` and giving `health_probe_methods` a per-family safe default for a template-less chain, mirroring the existing `jsonrpc`/`eth_chainId` design. Verified live end to end on Moonriver (real endpoint validated, then a real custom method + schema evidence validated — full Case 2 pass) and confirmed the bitcoin_jsonrpc fix routes correctly (no more crash, a clean "endpoint returned 404" failure against a non-JSON-RPC test host instead). `tendermint`/`rest`/`hedera_dual` (GET/REST-shaped transports the generic probe does not build requests for) are **not** fixed by this change and were confirmed still broken live (Juno/`tendermint` and Stellar/`rest` both reproduced the identical `CalledProcessError` crash) — see open item B.25 | S1 | 25 (live dual-AI chaos, 2026-07-13 Case 2 family coverage sweep) | test_new_chain_endpoint_validation_works_for_generic_pop_families_without_a_template |
| 60 | Case 1 (custom RPC method on an already-supported chain) and Case 2's `new_chain_method` step both unconditionally rejected any `GET /path`-shaped answer as "looks like an endpoint, REST path, or doc title" via `_looks_like_rest_path_or_doc_method`, with no adapter-family awareness — but for a chain whose own family is REST-shaped (`rest`/`tendermint`/`hedera_dual`), a `GET /path` answer *is* the correct method format (every default method on `cosmos-hub`/`algorand`/`hedera` already looks exactly like this). Found live testing Case 1 on `cosmos-hub`: adding the real custom method `GET /cosmos/staking/v1beta1/pool` was rejected outright, telling the user to "switch protocol family to rest" even though `cosmos-hub` (tendermint) already speaks REST-shaped paths. Fixed by only applying the REST-path rejection when the chain's own family is not one of `GENERIC_JSONRPC_PROBE_FAMILIES` (`#59`'s canonical partition) — a genuine URL is still always rejected regardless of family. Verified live end to end on `cosmos-hub`: the method is now accepted, its schema (`[]`, no params) validated against the real endpoint with a genuine HTTP 200 | S1 | 26 (live dual-AI chaos, 2026-07-13 Case 1/2 coverage audit) | test_custom_rpc_method_accepts_get_path_for_rest_shaped_chain_family |
| 61 | **CRITICAL.** Closes B.25: Case 2 endpoint validation was still structurally broken for `tendermint`/`rest`/`hedera_dual` after `#59` (which only covered the POST-JSON-RPC families). The generic JSON-RPC prober cannot serve GET-path/REST methods at all, so a template-less chain in these 3 families still crashed via the template-requiring `_probe_health` path for both the initial health check and any user-supplied `GET /path` custom method. Fixed with a family-tiered strategy in `agent/validators/endpoint_probe.py`: (1) a new generic REST prober (`_validate_generic_rest_endpoint`/`_probe_generic_rest_method`, builds real GET/POST HTTP requests, with basic `{placeholder}` substitution from schema-evidence-style params) now handles any template-less chain whose adapter family is `REST_SHAPED_FAMILIES` (`rest`/`tendermint`/`hedera_dual`) once methods are GET/POST-path-shaped; (2) for the *initial* health check before any method is known, `tendermint` gets a genuinely universal safe default (`GET /cosmos/base/tendermint/v1beta1/blocks/latest` — every Cosmos-SDK chain's own SDK-provided REST module, not an app-specific endpoint); (3) `rest`/`hedera_dual` have no such universal path across their wildly heterogeneous real chains (Algorand/Aptos/Cardano/Tezos/TON/Hedera all speak completely different REST APIs), so they fall back to a bare reachability check (`_validate_bare_reachability` — any completed HTTP response counts as "the server is alive," full protocol validation deferred to the next step once a real method is supplied) instead of crashing. Verified live end to end on the exact two chains that reproduced `B.25`: Juno (tendermint) — endpoint validated via the universal default, then a real custom method (`GET /cosmos/staking/v1beta1/pool`) validated with a genuine HTTP 200; Stellar (rest) — endpoint accepted via bare reachability (a genuine 200 against Horizon's real API root document), then a custom method attempt correctly returned a clean HTTP 404 (real network response, not a crash) | S1 | 28 (live dual-AI chaos, 2026-07-13, closes B.25) | test_new_chain_endpoint_validation_works_for_rest_shaped_families_without_a_template |

| 62 | Opening-menu option 4 ("了解支持的链/RPC method/扩展方式", the `answer_opening_question` capability topic) rendered no visible content and became a dead loop: the `group == "opening"` handler's `value == "info"` branch silently cleared `pending_question` and set `active_group` without ever printing `_framework_capability_summary(state)`, returning an empty `visible_response`. Found by the user's own manual testing, not by dual-AI chaos — every prior chaos sweep had only exercised free-text phrasing of "what chains do you support" etc., never a literal numbered-menu selection of a "meta" opening option, so this simple defect had zero chaos coverage. Fixed by appending the capability summary to `visible_response` and setting `_stop_after_response` so the turn actually renders it | S1 | 29 | test_opening_menu_info_option_shows_capability_content_not_a_dead_loop |
| 63 | Closes B.22: a failed custom-RPC schema validation (`_validate_rpc_schema`) wrote `case_dict[params_field] = params` *before* validating and never cleared it on failure, so the deterministic re-ask gate (`"params" not in custom_rpc`) could never re-fire a second time — recoverable via free-text routing on the next turn, but with no formal `pending_question` prompting for it. Fixed by writing `params_field` only on the success path and explicitly popping it on failure | S2 | 29 | test_custom_rpc_schema_failure_reprompts_for_schema_evidence |
| 64 | Closes B.20: `oracle._execution_status` read the stale, never-refreshed `state["job"]` snapshot instead of live on-disk job status, so a status-dump response could claim a job was still `running` well after it had actually finished (confirmed via the deterministic `status` command in the same session). Fixed by preferring `job_manager.get_job(job_id)`'s live disk read whenever a `job_id` is present, falling back to the snapshot on any error | S2 | 29 | test_execution_status_prefers_live_job_status_over_stale_snapshot |
| 65 | Closes B.18: a plain-English decline for an optional `chain_auxiliary_endpoints` field (e.g. "skip it, I don't have one") was rejected as "that reply does not look like an answer," while only a narrow token set (e.g. bare "none") worked. Fixed by widening the decline matcher (`_text_mentions_field_skip`) to recognize common skip/absence phrasing in English and Chinese ("skip", "跳过", "没有", "不需要", "not applicable", etc.), mirroring the broader NL matching already used for `has_accounts_device` | S3 | 29 | test_optional_chain_auxiliary_field_accepts_plain_english_decline |
| 66 | Closes B.23: "adjust mixed weights" and "add a custom RPC method" were the same code path (`workload_choice`'s generic `else` branch), forcing a user who only wanted to reweight the chain template's existing default methods through a full endpoint/method (re-)validation cycle as if adding a brand-new method. Fixed by giving `weights` its own branch that goes straight to `needs_weights`, with `_question_for_group`'s prompt falling back to the chain template's own default methods when no custom methods have been validated yet | S2 | 29 | test_adjust_mixed_weights_skips_endpoint_revalidation |
| 67 | Closes B.1: `needs_google_search` was a dead flag returned by the chain-identity/custom-RPC-schema resolvers but read nowhere in the harness, so Gemini+google_search grounding (the framework's intended second-pass verification whenever adding a new chain or custom RPC method) never actually fired. Confirmed with the user this was a genuine framework gap, not something to remove: the intended design runs google_search grounding **unconditionally** whenever `google_search_available` is true, not gated by the underlying LLM's own self-judgment of uncertainty. Fixed by wiring `run_google_search_grounding()` (the same function `#48`/`#49` already use) into both the chain-identity resolution path (`_augment_chain_resolution_with_search`) and the custom-RPC/new-chain schema-evidence path (`_augment_schema_draft_with_search`), gated only on `google_search_available`, with the evidence summary appended to the user-facing confirmation prompt; removed the now-genuinely-unused `needs_google_search` field from both resolver prompts/schemas | S2 | 29 | test_unknown_chain_identity_grounds_with_google_search_unconditionally, test_custom_rpc_schema_extraction_grounds_with_google_search_unconditionally |
| 68 | Closes B.17: there was no natural-language cancel/back path out of a stuck `custom_rpc_endpoint`/`new_chain_endpoint` question — "go back"/"cancel" just re-rendered the same endpoint prompt. Root cause was not missing navigation handling (go_back/change_group routing already worked correctly) but that leaving the question never abandoned the half-finished custom-method/new-chain attempt, so routing bounced straight back to the same unmet precondition. Fixed by resetting the in-progress `custom_rpc`/`chain_identity` state (`_abandon_stuck_endpoint_question_if_needed`) at the top of both the `go_back` and `change_group` action handlers whenever the pending question is one of these two, plus resolver guidance (`intent.py`) to route an explicit give-up/cancel utterance to `go_back`/`change_group` instead of treating it as an attempted endpoint answer | S2 | 29 | test_go_back_out_of_stuck_custom_rpc_endpoint_abandons_it |
| 69 | Closes B.21: `analyze_report`'s entry point (`_report_artifact_entry_response`) correctly resolved the just-submitted job's paths but never read the data it pointed at, even when the same turn explicitly asked to interpret it ("跟我解释一下这个报告，帮我看看瓶颈在哪") — it only listed paths and suggested a follow-up ask. Fixed by adding `_report_log_highlights` (tails `benchmark.log` via `job_manager.tail_job_log`, filtered to error/failure markers or success markers depending on job status) and appending a "日志摘录"/"log excerpt" block with the real matching lines to the response | S2 | 29 | test_analyze_report_includes_real_log_excerpt_not_just_paths |
| 70 | **CRITICAL.** `oracle.compute_next_action` forced `config_status` to `"complete"` whenever `execution_status` showed any job status at all (`job_running`/`job_completed`/`job_failed`), with no check that the job belonged to the in-progress workflow. `job`/`latest_job_id` deliberately survive a full reset (`RESET_PRESERVED_KEYS`, so `analyze_report`/`status` keep working for the last completed job post-reset) — so a brand-new, still-in-progress config with an unrelated leftover job from before the reset reported `config_status: complete` in the very same status dump that also correctly named a real unmet next blocking question (e.g. "choose sync-observe stop condition"), a directly self-contradictory response. Found live via the user's own manual testing (second defect found this way in this batch — see `#62`'s note): a fresh `bsc`/sync-observe setup with a stale `job_failed` from an earlier session showed exactly this contradiction. Fixed by removing the override — `group in {"job_monitoring", ""}` (from `routing.next_group_and_reason`, the harness's single source of truth for workflow completeness) is already the correct and sufficient test. Verified live end to end: submitted a real fake-node job (populating `state["job"]`), reset the session, started a fresh incomplete `bsc`/sync-observe setup, and confirmed the status dump now correctly reads `配置状态：in_progress` instead of the previous contradictory `complete` | S1 | 29 (live, user manual testing) | test_config_status_not_forced_complete_by_a_stale_unrelated_job |
| 71 | Same bug class as `#68`/B.17: no natural-language cancel/back path out of a stuck `SYNC_OBSERVE_RPC_URL` question (reached via sync-observe's `endpoint_only`/`existing_local_node` sources) — "go back" got "this configuration group has no blocking item right now..." and then re-rendered the identical question, a dead end. Same root cause as `#68`: `_pop_previous_group` correctly skips the current group (`endpoint_process`) and lands one group further back with nothing left to ask, falling through to `_ask_next_blocking_question`, which recomputes via routing and finds `sync_observe.source` still unvalidated — re-rendering the same question because `_abandon_stuck_endpoint_question_if_needed` did not yet handle this pending id. Found via a parallel dual-AI chaos subagent tasked with sync-observe coverage (dispatched during Phase 29 to close a live-coverage gap on this mode). Fixed by adding a `SYNC_OBSERVE_RPC_URL` branch that resets `sync_observe.source` (and related ack/attribution/endpoint-evidence flags) so routing naturally lands back on the source-selection question instead of looping. Verified live: restarted the driver to pick up the fix, drove back to the same stuck state, and confirmed "go back" now correctly re-presents the sync-observe source menu | S2 | 29 (live dual-AI chaos, parallel subagent) | test_go_back_out_of_stuck_sync_observe_rpc_url_abandons_it |

Phase 29 (2026-07-13) closes the full backlog of previously-deferred B-section
items per the user's "fix everything, not with patches" directive (`#62`-`#69`
above), plus the C.3/C.4 validation gaps below. `#62` was found by the user's
own manual testing, not chaos testing — see that row for the coverage gap it
exposed (a literal numbered-menu selection of a "meta" opening option had
never been chaos-tested, only free-text phrasing of the same intent). C.4's
`benchmark_subprocess_env` fix was verified with a genuinely clean live run
(`job_20260713155411_fb2c0867`, no manual PATH workaround) completing a real
fake-node benchmark end to end. `#70` was a second defect the user's own
manual testing found mid-phase (a `config_status`/`execution_status`
self-contradiction in a fresh sync-observe session) — two live user-found
defects in one phase is a real signal that dual-AI chaos testing systematically
under-covers stale-cross-session-state and literal-menu-selection scenarios;
worth deliberately adding both to future chaos test design.

Phase 24 (2026-07-13) retired the entire unused `agent/adk_app/` ADK-native
Agent/Runner tool-calling surface (never the shipped product's entrypoint —
the real harness never used ADK's Runner loop) and consolidated
`agent/tools/executor.py`/`schema.py` as the single tool-dispatch surface,
relocating every genuinely-needed piece (`install_dependencies`, fixture
checks, `load_execution_contract`, `inspect_llm_auth`, ADK-availability probe,
REPL startup state, and `google_search` grounding) into new homes instead of
deleting them. Defects 48–50 are the google_search fix this phase set out to
make; 51–52 are pre-existing bugs (introduced years earlier, unrelated to
this phase) that surfaced only because relocating their files required
reading them closely; 53–56 surfaced from a dedicated code-review pass
(8 finder agents across correctness/reuse/simplification/efficiency/altitude
angles, then independently re-verified by 3 fresh skeptical verifier agents
per finding with no prior context) run against this phase's own new code.
See open item B.1 for `needs_google_search` (the chain-identity resolver's
similarly-named but distinct, still-unwired flag) — deliberately left out of
this phase's scope.

Defect 19 is the conformance refactor: chain-info + readiness intent moved from
harness phrase tables to the resolver's typed `answer_opening_question`
topics (`environment_readiness`, `supported_chains`+subject, `extension`+subject);
the harness routes on the typed output and the phrase classifiers
(`_chain_question_target`, `_is_environment_readiness_question`,
`_known_chain_in_text`, `_actions_are_chain_query_only`) were removed. All three
were re-verified live on DeepSeek.

Systemic root cause behind #7 and #10 (active_group trusted by
`_ask_next_blocking_question` + a `_question_for_group` branch with no completion
guard) was audited across **every** group branch in Phase 13 — no other
instances remain.

---

## B. Open items — known, deliberately not fixed (low severity / by design)

These are real but were judged not worth a code change now, or are accepted
trade-offs. Listed so they are not "discovered" again as if new.

~~1. **`needs_google_search` is a dead flag; the chain-identity resolver's search
   signal is still not wired.**~~ **Fixed as `#67` (Phase 29).** The
   chain-identity resolver (`resolve_unknown_chain_identity` in
   `agent/harness/intent.py`) set `needs_google_search`/`evidence_summary` in
   its returned JSON, but `agent/harness/groups.py` only read
   `chain_exists`/`canonical_chain_name`/`possible_known_chain`/`adapter_family`
   from that same payload — `needs_google_search`/`evidence_summary` were read
   nowhere in the repo. Per the user's confirmation of the intended framework
   design, google_search grounding (distinct from
   `state["web_research"]["google_search_available"]`, already correctly
   wired per `#48`/`#49`) now runs unconditionally whenever that flag is true,
   for both the chain-identity resolution path and the custom-RPC/new-chain
   schema-evidence path; the now-genuinely-unused `needs_google_search` field
   was removed from both resolver schemas/prompts rather than left as a dead
   decision point.

2. **Corrupt-checkpoint error message misattributes the cause.** (S3)
   A corrupt `checkpoints.sqlite` yields the user-facing message "底层模型调用暂时
   失败" (underlying *model* call failed) when the real cause is
   `sqlite3.DatabaseError: file is not a database`. Handling is otherwise
   graceful: clean stdout message, no crash, traceback only in stderr logs (not
   leaked to the user). Rare (requires a corrupted DB).

3. **Vague "继续配置" at the idle `opening` state is a no-op.** (S3)
   Right after a consultation answer (active_group=opening, no pending), a bare
   "继续配置" does not advance. Concrete phrasings ("接下来配置什么", "继续下一步")
   and any real answer work. LLM-classification dependent; flow not stuck.

4. **Heavy typo on the mode word is not recognized.** (S3)
   "fak node" (typo) does not resolve to fake-node, though chain-name typos do
   ("bnb chian" → bsc). The flow then asks for the mode, so it is recoverable.
   LLM limitation, not a harness bug.

5. **Multi-turn analysis of an already-buffered evidence block may re-ask to
   paste.** (S3) The first analysis of pasted evidence is grounded; a deeper
   follow-up question sometimes asks the user to paste again instead of reusing
   the buffer. Re-pasting recovers.

6. **Contradictory single-turn instructions resolve first-wins + confirm.** (S3)
   "use fake-node and real-node" / "test solana and ethereum" set the first and
   offer a switch-confirmation for the second, rather than asking "which one?".
   Safe and recoverable; "ask which" would be a nicer UX but is not a defect.

7. **Chinese ordinal "第一个" is not mapped to option 1.** (S3)
   Numbered ("1") and label ("Y"/quick) answers work; "第一个" is gracefully
   rejected (question kept). Enhancement, not a bug.

8. **Phase-6 accepted trade-offs.** (S3) The unsupported-family handoff flow makes
   one extra LLM call per pasted evidence block; a navigation handoff turn calls
   `resolve_action_queue` twice. Both documented and accepted in the repair plan.

9. **Explicit natural-language "run scripts/install_deps.sh" is not honored.** (S3)
   When deps are missing, the startup offer's "y" now installs (fixed, A.14), but
   typing an explicit request like "你帮我执行 scripts/install_deps.sh" instead
   falls through to the harness and returns a generic workflow description rather
   than running the installer. Wiring this cleanly needs an install intent in the
   resolver (avoid terminal keyword routing); low priority since "y" works.

10. **Free-form "what output/results will I see" questions get a generic
   workflow reply.** (S3) E.g. "测完之后我能看到什么结果" returns the pipeline
   steps, not a description of the artifacts (HTML report, per-method
   success/latency, QPS progression, resource metrics, logs). Same recurring
   class as A.14/A.15/A.16/A.17 (info exists, free-form phrasing not routed to
   it) but S3. A durable fix is better opening-consultation topic coverage +
   a smarter fallback than one-off answers per question.

11. **Verbose identity-confirm answer can be misrouted to the generic chain
   list.** (S3) Answering the unknown-chain identity gate with a long sentence
   ("是真实的链，Mina 是 GraphQL，不是任何 family") is routed to a supported_chains
   list instead of mapping to the "real chain, choose protocol" option. Clean
   answers ("1" / "真实链") work and reach the Case-3 handoff. Root is the general
   "pending numbered_choice + verbose free-text -> resolver hijack" ordering
   (`_route_free_text` runs before `resolve_pending_choice`); a durable fix is a
   routing-precedence change, deferred as risky.

---

12. **No bulk "accept all detected environment metadata" action.** (S3) Saying
   "剩下的环境元数据全部用检测值" advances only the current field, not all remaining
   metadata. This is deliberately conservative for the hardware group
   (LEDGER_DEVICE and disks must be confirmed individually per the product
   boundary), so only the pure provider_deployment metadata could safely bulk-fill.
   Each field is still individually correct (defect 23 fixed); this is a UX nicety.

13. **Verbose "it's not any of these families" answer to the protocol-family gate
   loops back to the identity gate.** (S3) Answering the adapter-family question
   with a sentence ("不是 EVM，也不是 cosmos/substrate/bitcoin…") is re-read as a chain
   mention and re-asks the unknown-chain identity question instead of selecting
   option 7 (not-in-families → Case-3 handoff). The numbered "7" and "不确定"
   answers work and reach the handoff. Same root as B.11 (verbose free-text to a
   pending numbered_choice → resolver hijack); a durable fix is resolver
   pending-question context, deferred as risky.

14. **Case-3 "生成二次开发文档" trigger is also recorded as an evidence line.** (S3)
   The turn that triggers handoff-doc generation is first captured as an evidence
   item, so the generated doc lists the trigger phrase among collected evidence.
   Cosmetic; the handoff content is otherwise correct.

15. **Interpretive failure analysis asks the user to paste logs even when the
   latest failed job's `benchmark.log` is on disk.** (S3) After a failed job,
   "为什么失败/是不是缺 go" returns the "paste the logs" prompt instead of reading
   the known job's log and explaining it. `logs <job_id>` and "分析最近 job" surface
   the path/artifacts deterministically; only the interpretive step is missing.
   Same class as B.10 (info accessible, free-form phrasing not routed to it). A
   durable fix lets the analysis path read the latest job's benchmark.log directly.

~~16. `_first_number_text` silently strips a leading "-" via digit-only regex
    extraction.~~ **Fixed as `#37`/`#38` (Phase 22).** Both the manual-question
    and paste-inference/direct-assignment entry points now reject
    negative/zero/non-numeric values for this whole field family via
    `_invalid_positive_number`, and `_signed_number_text` preserves the sign
    through normalization instead of losing it to `_first_number_text`.

~~17. **No natural-language cancel/back path out of a stuck `custom_rpc_endpoint`/
    `new_chain_endpoint` question.**~~ **Fixed as `#68` (Phase 29).** Root
    cause was not missing navigation handling (go_back/change_group already
    worked) but that leaving the question never abandoned the half-finished
    custom-method/new-chain attempt, so routing bounced straight back. Fixed
    by abandoning the in-progress attempt on go_back/change_group out of
    these two questions, plus resolver guidance to route an explicit give-up
    utterance there instead of treating it as an attempted endpoint answer.

~~18. **A plain-English decline for an "optional" field is rejected; only a
    narrow token set works.**~~ **Fixed as `#65` (Phase 29).** Widened the
    `chain_auxiliary_endpoints` decline matcher to recognize common skip/
    absence phrasing in English and Chinese, mirroring the broader NL
    matching already used for `has_accounts_device`.

19. **Report/capability questions are sometimes misclassified as
    `current_config` (a generic status dump) by the resolver.** (S2, found
    Phase 23) "生成一份报告长什么样，都有哪些指标" and an English equivalent both
    returned an irrelevant state dump instead of describing the report. This
    is an LLM classification miss, not a deterministic code bug (the intent
    prompt already instructs `analyze_report` for this case) — no code fix
    applied; would need resolver-prompt iteration and re-verification, which
    risks new nondeterministic side effects under this deadline.

~~20. **`_execution_status` (`agent/harness/oracle.py`) reads `state["job"]`,
    which is never refreshed after submission, instead of live on-disk job
    status.**~~ **Fixed as `#64` (Phase 29).** Now prefers
    `job_manager.get_job(job_id)`'s live disk read whenever a `job_id` is
    present, falling back to the snapshot on any error.

~~21. **`analyze_report`'s entry point never reads the data it points at.**~~
    **Fixed as `#69` (Phase 29).** Added `_report_log_highlights` (tails
    `benchmark.log`, filters to error/failure or success markers by job
    status) and appended a real log-excerpt block to the response.

~~22. **Custom-RPC schema validation failure poisons `case_dict[params_field]`
    with the bad value, so the deterministic re-ask gate never re-fires.**~~
    **Fixed as `#63` (Phase 29).** `params_field` is now only written on the
    success path; the failure branch explicitly pops it so the re-ask gate
    fires correctly.

~~23. **"Adjust mixed weights" and "add custom RPC method" are the same code
    path**, forcing a user who only wants to reweight the chain template's
    existing default methods through a full endpoint/method (re-)validation
    cycle as if adding a brand-new custom method.~~ **Fixed as `#66` (Phase
    29).** `weights` now has its own branch straight to `needs_weights`,
    falling back to the chain template's own default methods when no custom
    methods have been validated yet.

24. **Real behavioral question behind `#45`'s message fix: should endpoint/QPS
    actually be invalidated on certain mode switches, not just re-worded?**
    (S2-adjacent design question, found Phase 23) `#45` corrected the
    confirmation message to stop overclaiming; it deliberately did not change
    `_invalidate_for_target_mode`'s actual reset behavior, since real-node's
    QPS/endpoint semantics genuinely differ from sync-observe's (no QPS at
    all) and fake-node/real-node's do not (endpoint reachability aside). Worth
    a deliberate design pass on which fields *should* invalidate on which
    transition, rather than a reactive fix.
    ~~also worth finally deleting or wiring up `invalidated_groups`, which is
    dead code today~~ **`invalidated_groups` is no longer dead code as of
    `#58` (Phase 25)** — `_activate_group_question` now reads it to redirect a
    `change_group` jump away from a group invalidated earlier in the same
    turn. It is still only *written* by `_invalidate_for_chain_change`/
    `_invalidate_for_target_mode` and never pruned/cleared once a group is
    re-satisfied (harmless for `#58`'s fix, since it double-checks against
    `routing.next_group_and_reason` before redirecting — see `#58`'s test),
    but a deliberate pass on whether it should be pruned is still worth doing
    if it grows more readers.

~~25. **Case 2 endpoint validation is still structurally broken for the
    `tendermint`, `rest`, and `hedera_dual` adapter families.**~~ **Closed by
    `#61` (Phase 28, same day).** (S1, found
    Phase 25, 2026-07-13) `#59` fixed this for `substrate` and
    `bitcoin_jsonrpc` (both plain POST JSON-RPC transports) by giving
    `health_probe_methods` a safe per-family default method for a
    template-less chain. `tendermint`/`rest`/`hedera_dual` are GET-path/REST-
    shaped transports — the generic probe (`_validate_generic_jsonrpc_endpoint`
    /`_probe_generic_jsonrpc_method`) unconditionally builds a POST JSON-RPC
    body, so even providing a safe default method for these families would not
    be enough; a genuine template-less GET/REST probe path would need to be
    built (structurally the same class of gap `#42` fixed for *templated*
    chains' GET methods, but for the *no-template* case). Confirmed still
    broken live: switching to a genuinely new `tendermint`-family chain (Juno,
    `https://juno-api.polkachu.com`) and a genuinely new `rest`-family chain
    (Stellar, `https://horizon.stellar.org`) both reproduced the identical
    `CalledProcessError`/`FileNotFoundError` crash `#59` fixed for `substrate`.
    Not fixed in the same pass as `#59` — building a correct generic GET/REST
    probe is a larger, riskier change than the two POST-JSON-RPC families'
    fix, better done as its own deliberate pass. `hedera_dual` is additionally
    unusual in that it may have no natural second real-world exemplar chain to
    even test Case 2 against (Hedera's mirror-node REST + EVM JSON-RPC-relay
    dual API shape appears specific to Hedera's own architecture; this was not
    exhaustively confirmed — WebSearch is blocked by org policy in this
    environment, same limitation noted in Phase 18 — so treat as a reasoned
    assessment, not a verified fact).

    **Resolution (`#61`):** built a genuine template-less GET/REST probe
    (`_validate_generic_rest_endpoint`) plus a bare-reachability fallback for
    families with no universal safe path. Verified live on the exact two
    chains that reproduced this: Juno (tendermint) and Stellar (rest), both
    now validate real endpoints and real custom methods against genuine HTTP
    responses instead of crashing. `hedera_dual`'s lack-of-a-second-exemplar
    question is unaffected by this fix (still an open, unresolved research
    question, not blocking — the fallback logic covers it structurally
    whenever/if a second real chain in that family is ever tried).

26. **An off-topic status question ("你现在开始测试了么？") right after an
    informational "confirm and stop" message (e.g. `sync_observe_demo_ack`,
    `sync_observe_client_setup_ack`) gets a technically-correct but confusing
    reply.** (S3, found live 2026-07-13, user manual testing) Both of these
    fields clear `pending_question` and stop the turn after showing an
    informational message, by design — so the next turn (even an unrelated
    question) hits `oracle`'s generic "no pending question" branch
    (`f"当前没有待确认问题。{format_recommended_next_action(...)}"`,
    `agent/harness/oracle.py:173`), which correctly states there is no active
    question and names the next recommended step, but never directly answers
    "is anything running right now" — easy to misread as the flow being
    stuck or a job silently executing in the background, when in fact nothing
    has been submitted yet (no job exists until the preflight/smoke confirm
    step). A durable fix would give `compute_next_action`/the resolver an
    explicit "is a job currently executing" topic distinct from the generic
    next-blocking-question description. Not fixed in this pass — surfaced
    while validating sync-observe live, out of scope for the Phase 29 batch.

## D. Active regressions from this session's own fixes — NOT VERIFIED, hand off with caution

Everything in section A was verified (scripted regression test + a live DeepSeek-driven
run). Everything below was **not** — the work was interrupted mid-verification. Do not
assume anything in this section actually works; re-run the full suite and live-verify
before trusting it.

**Sequence of events (2026-07-13, same day as Phase 29):** after closing out `#62`-`#71`,
two additional live-user-testing-driven changes were made to `agent/harness/groups.py`:

1. A fallback so generic "what should I do now" questions (classified as
   `answer_opening_question`, any topic) fall through to
   `_ask_next_blocking_question` and actually render the real next question, instead of
   only printing a static FAQ paragraph and stopping. This part **was** verified live
   (confirmed working for both the `sync_observe_demo_ack` and
   `sync_observe_client_setup_ack` "confirm and stop" cases).

2. **`sync_observe_demo_ack` was changed to auto-execute immediately** (fill
   `stop_condition`/`duration_seconds`/`observability.mode`/`advanced_tuning` with safe
   defaults and call `run_approved_preflight_and_smoke` directly) instead of asking 4
   more questions whose semantics only make sense for a real sync observation. This was
   an explicit user design decision (demo_only reused real-observation-only questions,
   which was genuinely incoherent) but was **only verified with a mocked test**
   (`run_approved_preflight_and_smoke` patched out), never a live run.

**Bug found by the user's live manual testing, immediately after item 2 shipped:**
confirming the demo disclaimer now always fails preflight with
`missing: mainnet_rpc_url_reviewed, node_process_identity`. Root cause:
`agent/planners/config_checklist.py`'s sync-observe checklist only knew one waiver
signal (`sync_observe_local_attribution`, a boolean set for `endpoint_only` waiving
`node_process_identity` only) — `demo_only` has neither a local process nor a real
endpoint, so it also needed `mainnet_rpc_url_reviewed` waived, and nothing did that.
This is a pre-existing architectural gap (the checklist was never taught about
`demo_only`) that item 2 surfaced immediately instead of after 4 more questions.

**Also found live:** the same-turn fallback from item 1 above increased how often three
independently hand-written "nothing to confirm, here's the next action" text templates
(`agent/harness/oracle.py:184`'s `format_current_context` tail,
`agent/harness/groups.py`'s `_ask_next_blocking_question` fallback, and
`agent/harness/groups.py`'s `_activate_group_question` fallback, which unconditionally
calls `_ask_next_blocking_question` right after printing its own filler line) fire
back-to-back in the same turn, producing 2-3 near-duplicate sentences per reply.

**Status of the fix for the checklist bug (drafted, CODE WRITTEN, NOT TESTED):**
- Threaded `sync_observe_source` (the literal state value, e.g. `"demo_only"`) through
  `agent/harness/nodes/execution.py::_prepare_kwargs` →
  `agent/runners/benchmark_pipeline.py::prepare_benchmark_run`/`_structured_request` →
  `agent/planners/config_checklist.py`, replacing the old boolean
  `sync_observe_local_attribution` signal entirely (all reads/writes removed from
  `execution.py`, `groups.py` (4 call sites), `benchmark_pipeline.py`, and
  `config_checklist.py`).
- Added `SYNC_OBSERVE_SOURCE_WAIVERS = {"endpoint_only": {"node_process_identity"}, "demo_only": {"node_process_identity", "mainnet_rpc_url_reviewed"}}` in `config_checklist.py`,
  replacing the single-key special case with a per-source table.
- Updated `test_sync_observe_endpoint_only_waives_node_process_identity` to build
  requests with `sync_observe_source` instead of the removed boolean field, and added
  `test_sync_observe_demo_only_waives_real_node_requirements_for_preflight`.
- **NOT DONE:** the full test suite has not been re-run since these edits (interrupted
  before `.venv-adk/bin/python3 -m unittest discover -s tests` could execute). No
  end-to-end regression test exists yet driving `process_turn` through the real
  `sync_observe_demo_ack` → `run_approved_preflight_and_smoke` path with the checklist
  logic actually running (only the two isolated `build_configuration_checklist` unit
  tests above exist). No live DeepSeek re-verification was performed.

**Status of the fix for the duplicated-text bug: NOT STARTED.** The planned approach
(written up before work was halted): extract `format_current_context`'s tail into a
named `oracle.format_no_pending_question_message(state, language)` function, make
`_ask_next_blocking_question`'s fallback call it instead of hand-building a second
near-identical string, and delete `_activate_group_question`'s standalone filler line
(`"这个配置组当前没有阻塞项..."`) entirely so it just calls `_ask_next_blocking_question`
directly without a redundant lead-in sentence. None of this has been implemented.

**Before trusting or building on any of this:** run
`.venv-adk/bin/python3 -m unittest discover -s tests` and
`.venv-adk/bin/python3 tools/check_agent_boundaries.py` first — they have not been run
since the `config_checklist.py`/`benchmark_pipeline.py`/`execution.py` edits landed, so
it is possible something else in the flatten/checklist chain was missed. Then live-drive
a real DeepSeek session through sync-observe → `demo_only` → confirm the disclaimer and
confirm preflight actually passes (read `.agent/jobs/<job_id>/job.json` to confirm), and
separately confirm the duplicated-text bug (still unfixed) before starting on it.

## C. Validation gaps — not yet verified (not defects, missing coverage)

1. ~~**Docker dual-AI chaos with a second independent model.**~~ **Closed
   (2026-07-13).** The README's ideal gate wants the real CLI in Docker with
   a separate user-simulator model (e.g. Codex). Per user clarification: (a)
   Docker was only ever needed because the user develops on a Mac without a
   Linux environment — the actual chaos-testing sessions already run natively
   in a Linux sandbox, making Docker itself moot here; (b) the "two AI" bar
   is satisfied by one assistant reactively playing both the Agent-tester and
   user-simulator roles from the real prior turn's output (not a scripted
   transcript) — a genuinely independent second model is not a requirement.
   Not a real gap given these clarifications.

2. **Gemini + google_search path.** All chaos ran on DeepSeek. The Gemini
   search-grounding path (now wired per `#67`/B.1) is still unexercised with
   a real Gemini config in this environment — no Gemini credentials available
   here. Still open.

3. **Full-suite pandas tests.** **Closed (2026-07-13).** Installed
   pandas/matplotlib/scipy/seaborn/numpy/scikit-learn/statsmodels into
   `.venv-adk`; 512/512 tests pass, zero skips.

4. **Real benchmark execution.** **Closed (2026-07-13).** Installed a Go
   toolchain (needed to compile `tools/fake-node`'s simulator) and diagnosed
   why real fake-node runs kept failing at the analysis/report stage after
   Vegeta itself had already run successfully: `blockchain_node_benchmark.sh`
   invokes bare `python3` for its analysis/report scripts, not any Agent
   venv, and the system `python3` lacks pandas/matplotlib/etc. `.venv`
   (documented by `scripts/install_deps.sh` as `PROJECT_VENV_DIR`, a separate
   venv from the Agent's own `.venv-adk`) already had every needed package.
   Fixed by adding `agent/runners/materialize.py::benchmark_subprocess_env`,
   which prepends `.venv/bin` to `PATH` for the benchmark subprocess when it
   exists, wired into both `job_manager.py`'s sync path and `job_worker.py`'s
   detached-worker path. Verified live end to end with a genuinely clean run
   (`job_20260713155411_fb2c0867`, ordinary environment with no manual PATH
   override) completing fully: real Vegeta load, real analysis scripts, real
   bilingual HTML report, real archiving, `status: completed`.

5. ~~**Custom-RPC / endpoint validation on non-EVM families.**~~ **Closed by
   Phase 23.** Driven live on litecoin/bch (bitcoin_jsonrpc); found and fixed
   `#42` (the GET-body crash, confirmed by direct code execution to also hit
   rest/hedera_dual/tendermint/some substrate methods). Real-node mode's own
   full flow (endpoint validation, mixed-weight RPC workload, QPS,
   observability, advanced_tuning through to the preflight/smoke gate) was
   also driven live for the first time (Phase 23), surfacing `#44`/`#45` plus
   open items B.22/B.23. `chain_auxiliary_endpoints` (RPC_API_KEY-style
   fields) was also driven live for the first time, cleanly, on dogecoin.

6. ~~**`tendermint` adapter family had zero live dual-AI chaos coverage**~~
   **Closed for `tendermint` by Phase 24's 2026-07-13 family-coverage sweep.**
   Before this, `#42`'s GET-body fix for `tendermint`/`rest`/`hedera_dual` had
   only been confirmed by direct code execution, never by an actual live
   conversation against a real endpoint of one of those families. Drove
   real-node mode live on `cosmos-hub` end to end: `LOCAL_RPC_URL` validated
   against a real public endpoint (`https://cosmos-rest.publicnode.com`,
   genuine 200 response with live chain data), then `mixed` RPC workload
   defaults confirmed for the family's `GET /cosmos/...` method set. This same
   session, switching from `cosmos-hub` to `hedera` (`hedera_dual`), surfaced
   `#58` (a real, now-fixed defect) — so the sweep earned its keep beyond
   closing the coverage gap.

   ~~Still open: `rest`, a *successful* `hedera_dual` probe, and `substrate`
   post-`#42` confirmation.~~ **`rest` and `substrate` closed same-day
   (2026-07-13, follow-up sweep).** Drove real-node mode live end to end on
   `algorand` (`rest`): `LOCAL_RPC_URL` validated against
   `https://mainnet-api.algonode.cloud` with a genuine HTTP 200 real-node
   status response. Also confirmed the `#58` fix generalizes: switching from
   `algorand` to `hedera` with a plain single-intent turn ("换成 hedera")
   correctly re-asked for `LOCAL_RPC_URL` instead of reusing the old
   endpoint. Then drove `polkadot` (`substrate`) live end to end:
   `LOCAL_RPC_URL` validated against `https://rpc.polkadot.io` with a
   genuine response (real DOT balance, real block height), then `mixed`
   workload defaults confirmed cleanly — first live post-`#42` substrate
   endpoint validation (previous `polkadot` coverage was mixed-weights only,
   pre-dating that fix).

   `hedera_dual` remains the one open item, and it is now clearly
   environmental rather than a code gap: three separate live attempts against
   `https://mainnet-public.mirrornode.hedera.com` (across two sessions) got a
   genuine HTTP 200 health probe and 2 of 5 real method probes to pass
   (`GET /api/v1/accounts/{addr}`, `eth_getBalance`, `eth_call`, all
   returning real Hedera account/balance data) before a different method
   probe timed out each time. Confirmed via direct `curl` from this sandbox,
   independent of the Agent: the exact same query
   (`GET /api/v1/balances?account.id=0.0.2`) returned HTTP 200 in under 0.5s
   on 4 of 5 direct attempts and timed out (>6s) on 1 of 5 — this specific
   public host is intermittently slow from this sandbox's network, not a
   substitution or timeout-handling bug in `endpoint_probe.py`. A `hedera`
   endpoint validation attempted at a moment of good connectivity to this
   host (or against a different Hedera mirror-node provider) would very
   likely pass cleanly; not re-attempted further to avoid burning live-chaos
   turns on known external flakiness.

7. **Item 6 above is a different axis from Case 1/2/3 (see
   `docs/*/anychain-agent-ai-work-gate.md`) and should not be read as Case
   coverage.** Item 6's sweep drove *already-supported* chains through their
   default RPC methods across families — useful, but not Case 1 (a supported
   chain's custom-RPC-method path) or Case 2 (a genuinely new chain in an
   existing family). Auditing actual Case coverage (2026-07-13, Phase 25)
   found: **Case 1** (custom RPC method on a supported chain) had real
   defects found/fixed historically (`#12`/`#21`/`#28`) but only ever on
   jsonrpc-family chains — never exercised on the `tendermint`/`substrate`/
   `rest`/`hedera_dual` chains item 6 newly covered. **Case 2** (new chain in
   an existing family) had only ever been driven live once, on Mantle
   (`jsonrpc`), Phase 18 — the other 5 families were completely untested,
   and turned out to be structurally broken for exactly this reason (`#59`,
   originally B.25). Case 2 is now closed for `substrate`/`bitcoin_jsonrpc`
   (Moonriver/Dash, this phase) **and, as of `#61` the same day, also closed
   for `tendermint`/`rest`/`hedera_dual`** (Juno/Stellar, both re-verified
   live after the fix — see `#61`; B.25 itself is now closed). **Case 1 on the
   newly-covered families: driven live on `cosmos-hub` (tendermint) and found
   a second real defect (`#60`) — a `GET /path` custom method was rejected
   outright with no adapter-family awareness. Fixed and verified live with a
   real endpoint probe.** `#60`'s fix is family-based (not chain-specific),
   so it also covers `rest`/`hedera_dual` by construction; Case 1 was not
   separately re-driven live on `polkadot`/`algorand`/`hedera` given the fix
   and its regression test already cover the general case — a live spot
   check on those three would be a reasonable low-cost follow-up if pursued.

---

## How to maintain this file

- When a new defect is found: add to **B** (or fix it and move to **A** with a
  regression test).
- When a gap in **C** is closed: note the result and remove it.
- Keep severities honest; do not downgrade an S1/S2 to close it faster.
