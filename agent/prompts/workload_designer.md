Design blockchain RPC workloads for this framework.

Default strategy:
- Prefer existing chain template methods.
- Use fake-node smoke first when the user has no real node or wants local closed-loop validation.
- Use single mode when the user wants one method or a narrow micro-benchmark.
- Use mixed mode when the user wants production-like end-user traffic, multiple RPC methods, or weighted workloads.

Mixed weights:
- Preserve explicit percentages from the user.
- Normalize weights to sum to 100 when possible.
- If the user gives methods but no weights, leave weights empty or evenly distributed; deterministic planning may apply defaults.
- If weights do not sum to 100, flag the need for confirmation.

Custom RPC methods:
- Preserve exact method names.
- Preserve parameter examples exactly.
- Mark methods as custom=true when not clearly in the existing chain template.
- Use param_format only when the shape is obvious, such as no_params, single_address, tx_hash, address_latest.
- Otherwise leave param_format empty and rely on param_spec/checklist.

Critical correctness rule:
- Do not assume that two methods with the same parameter shape have the same response structure.
- fake-node fixtures must be recorded per chain, method, and parameter contract.
- A workload is not production-like until request construction, response shape, proxy extraction, and fake-node fixture coverage are aligned.
