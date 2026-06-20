Classify the user prompt for the Agent.

Return JSON only:
{
  "intent": "benchmark_request | framework_question | artifact_question | plan_edit | onboarding_request | out_of_scope",
  "confidence": 0.0,
  "reason": "short explanation",
  "requires_existing_plan": false,
  "requires_job_artifacts": false,
  "suggested_next_tool": "draft_request | capability_answer | artifact_qa | modify_plan | out_of_scope"
}

Routing rules:
- benchmark_request: the user wants to create, plan, or run a benchmark. Examples: "test my Solana node", "run a fake-node smoke benchmark", "find max stable QPS".
- framework_question: the user asks what the framework supports or how to configure/use it. Examples: supported chains, RPC methods, fake-node, Kubernetes, Prometheus/Grafana, config files, runtime.env.
- artifact_question: the user asks about generated reports, charts, CSVs, logs, bottlenecks, evidence, empty graphs, or previous jobs.
- plan_edit: the user modifies an existing plan. Examples: "set max qps to 5000", "change mixed weights to getSlot 70%".
- onboarding_request: the user wants to extend or embed the framework. Examples: integrate an enterprise Agent platform, integrate an enterprise KB, add a chain in an existing protocol family, add a new protocol family, add a custom RPC method, draft a chain template, or ask where to develop and how to validate secondary development.
- out_of_scope: the request is unrelated to blockchain node benchmarking.

Important distinctions:
- "How many chains/RPC methods do you support?" is framework_question, not benchmark_request.
- "Why is the chart empty?" is artifact_question.
- "Add a new chain" is framework_question unless the user asks to generate a benchmark plan.
- "How do I add a new chain/RPC method/KB integration?" is onboarding_request, not a generic framework_question.
- "How do I integrate this Agent into our internal Agent platform?" is onboarding_request.
- "Generate a plan for adding chain X with methods A and B" is onboarding_request.
- "Add a protocol family that is not one of the six families" is onboarding_request.
- "Use method X at 70% and method Y at 30%" is plan_edit if a plan exists, otherwise benchmark_request.

Prefer the safest route when ambiguous:
- If the prompt could cause execution, route to benchmark_request so preflight and approval gates apply.
- If the prompt asks for status, evidence, or report interpretation, route to artifact_question.
