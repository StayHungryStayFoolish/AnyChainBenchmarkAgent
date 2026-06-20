Convert the user prompt into a benchmark request JSON object.

Return JSON only. Use this shape when fields are known:
{
  "chain": "",
  "goal": "smoke | baseline | max_stable_qps | stress | bottleneck_confirmation | regression",
  "rpc_mode": "single | mixed",
  "use_fake_node": false,
  "local_rpc_url": "",
  "mainnet_rpc_url": "",
  "deployment": {"type": "vm | kubernetes | unknown", "provider": "gcp | aws | azure | other |"},
  "observability": {"enabled": false, "mode": "local | exporter"},
  "dependency_mode": "audit | isolated | managed",
  "qps": {"initial": 0, "max": 0, "step": 0, "duration_seconds": 0},
  "workload": {
    "methods": [
      {"name": "", "weight": 0, "params": [], "custom": false, "param_format": ""}
    ]
  },
  "bottleneck_focus": ["cpu", "memory", "disk", "network", "sync_health", "rpc_errors"],
  "confirmations": []
}

Extraction rules:
- If the user mentions fake-node, mock, closed-loop, or no real node, set use_fake_node=true.
- If the user gives a real RPC URL, preserve it exactly in local_rpc_url.
- If the user names a chain, normalize it to lowercase unless the name contains a meaningful dash such as avalanche-c.
- If the user says "max QPS", "maximum stable", or "capacity", use goal=max_stable_qps.
- If the user says "smoke", "quick", or "1 QPS", use goal=smoke.
- If the user asks for resource bottlenecks, include bottleneck_focus from the prompt.
- If the user gives mixed RPC percentages, set rpc_mode=mixed and preserve method weights.
- If the user gives custom RPC methods, preserve exact method names and parameter examples.
- If no RPC mode is clear, leave it to deterministic defaults.

Do not invent:
- endpoint URLs
- API keys or service accounts
- disk devices
- node process names/systemd units
- cloud project IDs
- exact production machine facts
- RPC parameter samples that the user did not provide

When information is missing, omit the field or leave it empty so deterministic checklists can ask.
