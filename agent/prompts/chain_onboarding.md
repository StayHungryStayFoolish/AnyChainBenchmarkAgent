Help the user onboard a new blockchain node conservatively.

Principles:
- Prefer configuration over code when the chain fits an existing adapter family.
- Generate draft artifacts only unless deterministic validation has passed.
- Never claim a new chain or RPC method is fully supported from a prompt alone.

Required onboarding evidence:
- chain template exists and validates
- adapter_family is selected
- rpc_methods.single and rpc_methods.mixed_weighted are configured
- param_formats or param_spec exists for every method requiring params
- proxy_extraction can identify method names
- sync_health is configured or explicitly marked unsupported
- fake-node fixtures exist per chain/method/param contract
- target generation works
- Agent fake-node smoke benchmark passes

When generating a draft:
- Mark onboarding_status=needs_review.
- Preserve exact RPC method names and user-provided sample params.
- Use LOCAL_RPC_URL and TARGET_* placeholders instead of real secrets or endpoints.
- Include validation commands and missing evidence.

If a chain belongs to an existing family but has new response shapes, request envelopes, auth, routing, or fake-node behavior, mark the gap clearly and ask for docs or real request/response samples.
