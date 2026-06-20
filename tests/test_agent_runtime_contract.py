import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
AGENT = REPO / "agent" / "cli.py"
AGENT_BIN = REPO / "bin" / "anychain-agent"
sys.path.insert(0, str(REPO / "agent"))

from discovery.environment import discover_environment  # noqa: E402
from knowledge.framework_capabilities import load_framework_capabilities  # noqa: E402
from llm import config as llm_config_module  # noqa: E402
from llm.config import load_llm_config  # noqa: E402
from llm.google_auth import credential_plan  # noqa: E402
from llm.orchestrator import PROMPT_BUNDLES, prompt_coverage  # noqa: E402
from llm.prompt_loader import compose_prompt, load_prompt  # noqa: E402
from llm.providers import provider_from_config  # noqa: E402
from llm.types import LLMResponse  # noqa: E402
from memory.compactor import compact_session_state, should_auto_compact  # noqa: E402
from planners.request_modifier import apply_request_modification  # noqa: E402
from runners.guardrails import validate_execution_plan  # noqa: E402
from runners.job_manager import get_job, submit_job  # noqa: E402
from chat import ChatSession  # noqa: E402
from diagnostics.doctor import format_doctor_report, run_doctor  # noqa: E402
from onboarding.chain_onboarding import generate_onboarding_package  # noqa: E402
from wizard import run_wizard  # noqa: E402


def run_agent(*args, env=None):
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    completed = subprocess.run(
        [sys.executable, str(AGENT), *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        env=command_env,
    )
    return json.loads(completed.stdout)


class AgentRuntimeContractTest(unittest.TestCase):
    def test_read_only_discovery_with_injected_commands(self):
        def fake_runner(command, timeout):
            joined = " ".join(command)
            if "config/cloud_provider.sh" in joined:
                return 0, "gcp,gcp_gvnic,eth0", ""
            if "ip route" in joined:
                return 0, "eth0\n", ""
            if command[:3] == ["lsblk", "-J", "-o"]:
                return 0, json.dumps({
                    "blockdevices": [
                        {"name": "sda", "type": "disk", "size": "100G", "mountpoint": "/", "fstype": "ext4", "label": ""},
                        {"name": "sdb", "type": "disk", "size": "2T", "mountpoint": "/var/lib/solana/ledger", "fstype": "xfs", "label": "ledger"},
                    ]
                }), ""
            if "command -v" in joined:
                return 0, "/usr/bin/mock\n", ""
            return 1, "", "not mocked"

        discovery = discover_environment(command_runner=fake_runner)
        self.assertEqual(discovery["cloud"]["provider"], "gcp")
        self.assertEqual(discovery["cloud"]["platform"], "gce")
        self.assertEqual(discovery["network"]["default_interface"], "eth0")
        self.assertEqual(discovery["disks"]["proposed_ledger_device"], "sdb")
        self.assertEqual(discovery["dependencies"]["mode"], "audit")

    def test_prompt_to_plan_preflight_and_mock_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            request_file = tmp_path / "request.json"
            plan_file = tmp_path / "plan.json"
            jobs_dir = tmp_path / "jobs"

            request = run_agent(
                "draft-request",
                "--prompt",
                "Test Solana maximum stable QPS on GKE with fake-node smoke first and focus on disk P99.",
                "--output",
                str(request_file),
            )
            self.assertEqual(request["chain"], "solana")
            self.assertEqual(request["goal"], "max_stable_qps")
            self.assertTrue(request["use_fake_node"])
            self.assertEqual(request["deployment"]["type"], "kubernetes")
            self.assertIn("disk", request["bottleneck_focus"])
            self.assertEqual(request["recommended_initial_validation"], "smoke")

            llm_request = run_agent(
                "draft-request",
                "--prompt",
                "Test Solana maximum stable QPS on GKE with fake-node smoke first and focus on disk P99.",
                "--mock-llm",
            )
            self.assertEqual(llm_request["chain"], "solana")
            self.assertEqual(llm_request["llm_status"], "accepted")

            route = run_agent(
                "route-intent",
                "--prompt",
                "How do I configure mixed RPC weights?",
                "--mock-llm",
            )
            self.assertIn(route["intent"], {"benchmark_request", "framework_question"})

            framework_answer = run_agent(
                "ask",
                "--prompt",
                "How do I use fake-node for local closed loop testing?",
            )
            self.assertEqual(framework_answer["intent"], "framework_question")
            self.assertTrue(framework_answer["sources"])

            out_of_scope = run_agent("ask", "--prompt", "Write me a stock trading robot")
            self.assertEqual(out_of_scope["intent"], "out_of_scope")

            smoke = run_agent("llm-smoke", "--mock")
            self.assertEqual(smoke["provider"], "fake")

            capabilities = run_agent("capabilities")
            self.assertEqual(capabilities["chain_count"], 36)
            self.assertEqual(capabilities["family_count"], 6)
            self.assertGreater(capabilities["unique_rpc_method_count"], 100)

            capability_answer = run_agent(
                "ask",
                "--prompt",
                "How many chains and RPC methods does the framework support?",
            )
            self.assertEqual(capability_answer["intent"], "framework_question")
            self.assertEqual(capability_answer["capabilities"]["chain_count"], 36)
            self.assertIn("families", capability_answer["capabilities"])

            gap = run_agent("gap-analysis", "--chain", "solana", "--method", "getBalance", "--method", "missingMethod")
            self.assertEqual(gap["chain"], "solana")
            self.assertIn("getBalance", gap["supported_methods"])
            self.assertIn("missingMethod", gap["missing_methods"])
            self.assertTrue(gap["onboarding_plan"])

            gap_answer = run_agent("ask", "--prompt", "Does solana support missingMethod RPC method?")
            self.assertEqual(gap_answer["intent"], "framework_question")
            self.assertIn("gap_analysis", gap_answer)

            plan = run_agent("plan", "--request", str(request_file), "--output", str(plan_file), "--dry-run")
            self.assertEqual(plan["chain"], "solana")
            self.assertEqual(plan["strategy"], "ramp")
            self.assertTrue(plan["use_fake_node"])
            self.assertEqual(plan["execution"]["runner_mode"], "detached")
            self.assertEqual(plan["required_inputs"], [])
            self.assertIn("--fake-node", plan["execution"]["command"])
            self.assertIn("STANDARD_INITIAL_QPS", plan["execution"]["environment"])
            self.assertNotIn("QUICK_INITIAL_QPS", plan["execution"]["environment"])
            self.assertIn("confidence", plan)
            self.assertIn("plan_execution", plan["approval_checkpoints"])
            self.assertTrue(plan["redaction_policy"]["enabled"])
            self.assertGreaterEqual(len(plan["config_snapshot"]["files"]), 2)
            self.assertIn("required_questions", plan)
            self.assertIn("risk", plan)
            self.assertIn(plan["risk"]["risk_level"], {"low", "medium", "high"})
            self.assertIn("configuration_checklist", plan)
            self.assertFalse(plan["configuration_checklist"]["missing_blockers"])

            risk = run_agent("risk-score", "--plan", str(plan_file))
            self.assertEqual(risk["risk_level"], plan["risk"]["risk_level"])

            discovered_plan_file = tmp_path / "discovered_plan.json"
            discovered_plan = run_agent(
                "plan",
                "--request",
                str(request_file),
                "--output",
                str(discovered_plan_file),
                "--discover",
            )
            self.assertNotEqual(discovered_plan["discovery"]["source"], "not_collected")

            doctor = run_agent("doctor")
            self.assertIn(doctor["status"], {"ready", "ready_without_llm", "needs_dependencies"})
            self.assertEqual(doctor["capabilities"]["chain_count"], 36)
            self.assertIn("next_actions", doctor)

            validation = run_agent("validate-plan", str(plan_file))
            self.assertTrue(validation["valid"])

            self.assertTrue(validate_execution_plan(plan, approved=False))
            self.assertEqual(validate_execution_plan(plan, approved=True), [])

            preflight = run_agent("preflight", "--plan", str(plan_file))
            self.assertTrue(preflight["passed"])

            job = run_agent("submit", "--plan", str(plan_file), "--jobs-dir", str(jobs_dir), "--mock")
            self.assertEqual(job["status"], "completed")
            self.assertTrue(Path(job["artifact_index"]).is_file())
            self.assertTrue(Path(job["runtime_env_file"]).is_file())
            runtime_env = Path(job["runtime_env_file"]).read_text(encoding="utf-8")
            self.assertIn("export BLOCKCHAIN_NODE='solana'", runtime_env)
            self.assertIn("export RPC_MODE='single'", runtime_env)
            artifact_index = json.loads(Path(job["artifact_index"]).read_text(encoding="utf-8"))
            self.assertEqual(artifact_index["evidence"]["runtime_env_file"], job["runtime_env_file"])

            status = run_agent("status", "--job-id", job["job_id"], "--jobs-dir", str(jobs_dir))
            self.assertEqual(status["job_id"], job["job_id"])
            self.assertEqual(status["status"], "completed")

            analysis = run_agent("analyze", "--job-id", job["job_id"], "--jobs-dir", str(jobs_dir))
            self.assertEqual(analysis["job_id"], job["job_id"])
            self.assertEqual(analysis["grade"], "WARNING")
            self.assertEqual(analysis["evidence"]["runtime_env_file"], job["runtime_env_file"])
            self.assertTrue(analysis["recommendations"])

            artifact_answer = run_agent(
                "artifact-qa",
                "--question",
                "Why is the report evidence incomplete?",
                "--job-id",
                job["job_id"],
                "--jobs-dir",
                str(jobs_dir),
            )
            self.assertEqual(artifact_answer["intent"], "framework_question")
            self.assertIn("runtime_env_file", artifact_answer["answer"])

            updated_request, runner_changes = apply_request_modification(request, "run in foreground")
            self.assertEqual(updated_request["runner_mode"], "foreground")
            self.assertIn("runner_mode -> foreground", runner_changes)

            jobs = run_agent("jobs", "--jobs-dir", str(jobs_dir))
            self.assertEqual(jobs["jobs"][0]["job_id"], job["job_id"])
            resume = run_agent("resume", "--job-id", job["job_id"], "--jobs-dir", str(jobs_dir))
            self.assertEqual(resume["job_id"], job["job_id"])
            self.assertIn("analyze", resume["next_actions"])
            logs = run_agent("logs", "--job-id", job["job_id"], "--jobs-dir", str(jobs_dir))
            self.assertFalse(logs["exists"])

            synthetic_csv = tmp_path / "performance.csv"
            synthetic_csv.write_text("timestamp,cpu\n1,50\n2,60\n", encoding="utf-8")
            empty_csv = tmp_path / "proxy_method.csv"
            empty_csv.write_text("timestamp,method,status\n", encoding="utf-8")
            summary_json = tmp_path / "summary.json"
            summary_json.write_text(json.dumps({"bottleneck_detected": True, "bottleneck_types": ["disk"]}), encoding="utf-8")
            report_html = tmp_path / "report.html"
            report_html.write_text("<html><body>report</body></html>", encoding="utf-8")
            synthetic_index = tmp_path / "artifact_index.json"
            synthetic_index.write_text(json.dumps({
                "evidence": {
                    "performance_csv": str(synthetic_csv),
                    "proxy_method_csv": str(empty_csv),
                    "archive_summary": str(summary_json),
                    "html_report": str(report_html),
                    "sync_health_csv": str(tmp_path / "missing.csv"),
                }
            }), encoding="utf-8")
            synthetic_artifact_answer = run_agent(
                "artifact-qa",
                "--question",
                "Why are charts empty and where is the bottleneck evidence?",
                "--artifact-index",
                str(synthetic_index),
            )
            self.assertIn("CSV has 2 data rows", synthetic_artifact_answer["answer"])
            self.assertIn("CSV exists but has no data rows", synthetic_artifact_answer["answer"])
            self.assertIn("missing file", synthetic_artifact_answer["answer"])
            self.assertIn("Report chart explanation", synthetic_artifact_answer["answer"])
            self.assertIn("Bottleneck diagnostics", synthetic_artifact_answer["answer"])
            self.assertIn("per_method_attribution", {chart["chart_id"] for chart in synthetic_artifact_answer["chart_explanation"]["charts"]})

            bottleneck_csv = tmp_path / "bottleneck_performance.csv"
            bottleneck_csv.write_text(
                "timestamp,cpu_usage,cpu_iowait,data_avg_await,data_util,total_iops,total_throughput_mibs\n"
                "1,92,18,35,94,10000,600\n"
                "2,94,20,40,96,12000,700\n",
                encoding="utf-8",
            )
            proxy_csv = tmp_path / "proxy_method_errors.csv"
            proxy_csv.write_text(
                "timestamp,method,status,latency_ms\n"
                "1,eth_getBalance,500,1200\n"
                "2,eth_getBalance,200,1100\n",
                encoding="utf-8",
            )
            bottleneck_index = tmp_path / "bottleneck_index.json"
            bottleneck_index.write_text(json.dumps({
                "evidence": {
                    "performance_csv": str(bottleneck_csv),
                    "proxy_method_csv": str(proxy_csv),
                    "sync_health_csv": "",
                }
            }), encoding="utf-8")
            diagnostics = run_agent("diagnose-artifacts", "--artifact-index", str(bottleneck_index))
            categories = {finding["category"] for finding in diagnostics["findings"]}
            self.assertIn("disk_latency", categories)
            self.assertIn("rpc_errors", categories)

            clean_csv = tmp_path / "clean_performance.csv"
            clean_csv.write_text(
                "timestamp,cpu_usage,cpu_iowait,data_vda_avg_await,data_vda_util,cgroup_cpu_usage_usec\n"
                "1,2,0.1,0.5,1.0,123456789\n"
                "2,3,0.2,0.6,1.2,223456789\n",
                encoding="utf-8",
            )
            clean_index = tmp_path / "clean_index.json"
            clean_index.write_text(json.dumps({
                "evidence": {
                    "performance_csv": str(clean_csv),
                    "proxy_method_csv": "",
                    "sync_health_csv": "",
                }
            }), encoding="utf-8")
            clean_diagnostics = run_agent("diagnose-artifacts", "--artifact-index", str(clean_index))
            self.assertNotIn(
                "cpu",
                {finding["category"] for finding in clean_diagnostics["findings"]},
            )

            runbook_file = tmp_path / "runbook.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(AGENT),
                    "runbook",
                    "--plan",
                    str(plan_file),
                    "--output",
                    str(runbook_file),
                ],
                cwd=REPO,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            self.assertIn("Benchmark Agent Runbook", completed.stdout)
            self.assertTrue(runbook_file.is_file())

            wizard_dir = tmp_path / "wizard"
            wizard_result = run_agent(
                "wizard",
                "--prompt",
                "Test Solana maximum stable QPS on GKE with fake-node smoke first and focus on disk P99.",
                "--output-dir",
                str(wizard_dir),
                "--yes",
                "--mock",
                "--mock-llm",
            )
            self.assertEqual(wizard_result["status"], "submitted")
            self.assertEqual(wizard_result["job"]["status"], "completed")
            self.assertTrue((wizard_dir / "runbook.md").is_file())
            wizard_plan = json.loads((wizard_dir / "plan.json").read_text())
            self.assertNotEqual(wizard_plan["discovery"]["source"], "not_collected")
            self.assertIn("required_questions", wizard_plan)

            changed_request_file = tmp_path / "changed_request.json"
            changed_plan_file = tmp_path / "changed_plan.json"
            changed_request = dict(request)
            changed_request["qps"] = {"initial": 100, "max": 20000, "step": 1000, "duration_seconds": 120}
            changed_request_file.write_text(json.dumps(changed_request), encoding="utf-8")
            run_agent("plan", "--request", str(changed_request_file), "--output", str(changed_plan_file))
            diff = run_agent("diff-plan", "--old", str(plan_file), "--new", str(changed_plan_file))
            self.assertTrue(diff["changed"])

            archives_dir = tmp_path / "archives"
            first = archives_dir / "run_1" / "test_summary.json"
            second = archives_dir / "run_2" / "test_summary.json"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text(json.dumps({
                "run_id": "run_1",
                "benchmark_mode": "standard",
                "max_successful_qps": 1000,
                "bottleneck_detected": False,
                "bottleneck_types": [],
                "archived_at": "2026-06-16 00:00:00",
            }), encoding="utf-8")
            second.write_text(json.dumps({
                "run_id": "run_2",
                "benchmark_mode": "standard",
                "max_successful_qps": 1500,
                "bottleneck_detected": True,
                "bottleneck_types": ["disk"],
                "archived_at": "2026-06-16 01:00:00",
            }), encoding="utf-8")
            history = run_agent("history", "--archives-dir", str(archives_dir), "--limit", "2")
            self.assertEqual(len(history["runs"]), 2)
            comparison = run_agent("history", "--archives-dir", str(archives_dir), "--compare-latest")
            self.assertEqual(comparison["status"], "compared")

            answers_file = tmp_path / "answers.json"
            answers_file.write_text(json.dumps({
                "ledger_device_confirmation": "sdb",
                "dependency_mode_confirmation": "audit",
            }), encoding="utf-8")
            answers_wizard_dir = tmp_path / "answers_wizard"
            answers_result = run_agent(
                "wizard",
                "--prompt",
                "Test Solana maximum stable QPS on GKE with fake-node smoke first.",
                "--output-dir",
                str(answers_wizard_dir),
                "--answers-file",
                str(answers_file),
                "--quiet",
            )
            self.assertIn(answers_result["status"], {"planned", "needs_input"})

            new_chain_gap = run_agent("ask", "--prompt", "How do I add new chain foochain?")
            self.assertEqual(new_chain_gap["intent"], "onboarding_request")
            self.assertIn("Onboarding package for foochain", new_chain_gap["answer"])
            self.assertIn("docs/en/secondary-development-guide.md", new_chain_gap["answer"])

            kb_onboarding = run_agent("ask", "--prompt", "How do I integrate our enterprise Knowledge Base?")
            self.assertEqual(kb_onboarding["intent"], "onboarding_request")
            self.assertIn("Enterprise Knowledge Base onboarding plan", kb_onboarding["answer"])

            platform_onboarding = run_agent("ask", "--prompt", "How do we embed this into our internal Agent platform?")
            self.assertEqual(platform_onboarding["intent"], "onboarding_request")
            self.assertIn("Enterprise Agent platform integration plan", platform_onboarding["answer"])
            self.assertIn("tool-schema", platform_onboarding["answer"])
            self.assertIn("tool-call", platform_onboarding["answer"])

            protocol_onboarding = run_agent("ask", "--prompt", "Generate a plan to add a new protocol family for FooVM")
            self.assertEqual(protocol_onboarding["intent"], "onboarding_request")
            self.assertIn("New protocol family onboarding plan", protocol_onboarding["answer"])

            rpc_onboarding = run_agent("ask", "--prompt", "Add custom RPC method foo_getBalance to chain foochain")
            self.assertEqual(rpc_onboarding["intent"], "onboarding_request")
            self.assertIn("foo_getBalance", rpc_onboarding["answer"])
            onboarding = run_agent(
                "onboarding-plan",
                "--chain",
                "foochain",
                "--method",
                "foo_getBalance",
                "--adapter-family",
                "jsonrpc",
            )
            self.assertEqual(onboarding["status"], "needs_onboarding")
            self.assertEqual(onboarding["adapter_family"], "jsonrpc")
            self.assertTrue(onboarding["validation_commands"])

            draft_template_file = tmp_path / "foochain.json"
            draft_template = run_agent(
                "draft-chain-template",
                "--chain",
                "FooChain",
                "--adapter-family",
                "jsonrpc",
                "--method",
                "foo_getBalance",
                "--method",
                "foo_getTransaction",
                "--output",
                str(draft_template_file),
            )
            self.assertEqual(draft_template["status"], "draft")
            self.assertEqual(draft_template["template"]["_meta"]["onboarding_status"], "needs_review")
            self.assertEqual(sum(item["weight"] for item in draft_template["template"]["rpc_methods"]["mixed_weighted"]), 100)
            self.assertTrue(draft_template_file.is_file())

            kb_disabled = run_agent("knowledge-smoke")
            self.assertFalse(kb_disabled["status"]["enabled"])

            schema = run_agent("tool-schema")
            tool_names = {tool["function"]["name"] for tool in schema["tools"]}
            self.assertIn("submit_job", tool_names)
            self.assertIn("draft_chain_template", tool_names)
            self.assertIn("knowledge_search", tool_names)
            self.assertTrue(all(tool["type"] == "function" for tool in schema["tools"]))
            for tool in schema["tools"]:
                params = tool["function"]["parameters"]
                self.assertEqual(params["type"], "object")
                self.assertIn("required", params)
            capability_tool = run_agent("tool-call", "--name", "load_capabilities")
            self.assertEqual(capability_tool["chain_count"], 36)
            draft_tool = run_agent(
                "tool-call",
                "--name",
                "draft_request",
                "--arguments",
                '{"prompt":"Create a Solana fake-node smoke benchmark at 1 QPS"}',
            )
            self.assertEqual(draft_tool["chain"], "solana")
            generated_tool_plan = run_agent(
                "tool-call",
                "--name",
                "generate_plan",
                "--arguments",
                json.dumps({
                    "request": {
                        **draft_tool,
                        "ledger_device": "sdb",
                        "confirmations": ["ledger_device_confirmation"],
                    }
                }),
            )
            self.assertEqual(generated_tool_plan["chain"], "solana")
            self.assertEqual(generated_tool_plan["execution"]["runner_mode"], "detached")

    def test_terminal_chat_agent_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat_dir = Path(tmp) / "chat"
            session = ChatSession(output_dir=chat_dir)
            answer = session.handle("How many chains and RPC methods are supported?")
            self.assertIn("Current chain templates define", answer)
            doctor_answer = session.handle("doctor")
            self.assertIn("Agent doctor report", doctor_answer)
            self.assertIn("capabilities:", doctor_answer)

            plan_answer = session.handle("Create a Solana fake-node smoke benchmark at 1 QPS")
            self.assertIn("Created a benchmark plan", plan_answer)
            self.assertIn("solana", plan_answer)
            self.assertTrue((chat_dir / "plan.json").is_file())
            checklist_view = session.handle("checklist")
            self.assertIn("Checklist passed", checklist_view)
            checklist_plan = json.loads((chat_dir / "plan.json").read_text(encoding="utf-8"))
            self.assertFalse([
                question
                for question in checklist_plan["required_questions"]
                if question.get("severity") == "blocker"
            ])
            qps_update = session.handle("set max qps to 5000 and duration 120 seconds")
            self.assertIn("Updated the current benchmark plan", qps_update)
            updated_plan = json.loads((chat_dir / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(updated_plan["advanced_defaults"]["qps"]["max"], 5000)
            self.assertEqual(updated_plan["advanced_defaults"]["qps"]["duration_seconds"], 120)
            weight_update = session.handle("change mixed weights to getSlot 70%, getBlockHeight 30%")
            self.assertIn("mixed workload methods", weight_update)
            updated_request = json.loads((chat_dir / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(updated_request["rpc_mode"], "mixed")
            self.assertEqual(sum(item["weight"] for item in updated_request["workload"]["methods"]), 100)
            question_answer = session.handle("What does qps mean in this framework?")
            self.assertNotIn("I could not identify a supported plan edit", question_answer)
            self.assertIn("Preflight passed", session.handle("preflight"))
            mock_answer = session.handle("run mock")
            self.assertIn("Submitted mock job", mock_answer)
            self.assertIn("job_", mock_answer)
            self.assertIn("Mock lifecycle completed", session.handle("analyze"))
            self.assertIn("runtime_env_file", session.handle("qa What evidence was generated?"))
            trace_answer = session.handle("trace")
            self.assertIn("workflow_trace.jsonl", trace_answer)
            trace_rows = [
                json.loads(line)
                for line in (chat_dir / "workflow_trace.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            workflows = {row["workflow"] for row in trace_rows}
            self.assertIn("benchmark_request", workflows)
            self.assertIn("checklist", workflows)
            self.assertIn("plan_edit", workflows)
            restored_session = ChatSession(output_dir=chat_dir)
            restored_greeting = restored_session.greeting()
            self.assertIn("Latest job:", restored_greeting)
            self.assertIn("(completed)", restored_greeting)
            self.assertIn("Mock lifecycle completed", restored_session.handle("analyze"))
            compact_answer = session.handle("compact")
            self.assertIn("Context compacted", compact_answer)
            self.assertTrue((chat_dir / "memory.json").is_file())
            memory = json.loads((chat_dir / "memory.json").read_text(encoding="utf-8"))
            self.assertEqual(memory["preserved_state"]["chain"], "solana")
            self.assertTrue(memory["preserved_state"]["job_id"].startswith("job_"))
            self.assertEqual(memory["thresholds"]["context_window_tokens"], 1000000)
            self.assertEqual(memory["thresholds"]["token_threshold"], 700000)
            self.assertLessEqual(len(session.turns), 9)
            self.assertIn("preserved_state", session.handle("memory"))
            self.assertIn("Mock lifecycle completed", session.handle("analyze"))

            one_shot = subprocess.run(
                [
                    str(AGENT_BIN),
                    "--prompt",
                    "How many chains and RPC methods are supported?",
                    "--output-dir",
                    str(Path(tmp) / "one-shot"),
                ],
                cwd=REPO,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            self.assertIn("AnyChain Benchmark Agent", one_shot.stdout)
            self.assertIn("Mode: deterministic/offline", one_shot.stdout)
            self.assertIn("Current chain templates define", one_shot.stdout)

            repl = subprocess.run(
                [str(AGENT_BIN), "--output-dir", str(Path(tmp) / "repl")],
                cwd=REPO,
                input="Create a Solana fake-node smoke benchmark at 1 QPS\nset max qps to 5000\nrun mock\ncompact\nmemory\nstatus\nexit\n",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            self.assertIn("Created a benchmark plan", repl.stdout)
            self.assertIn("Updated the current benchmark plan", repl.stdout)
            self.assertIn("Submitted mock job", repl.stdout)
            self.assertIn("Context compacted", repl.stdout)
            self.assertIn("\"status\": \"completed\"", repl.stdout)
            self.assertTrue((Path(tmp) / "repl" / "memory.json").is_file())

    def test_agent_doctor_formats_readiness_report(self):
        report = run_doctor({
            "cloud": {"provider": "gcp", "platform": "gce", "confidence": 0.8},
            "deployment": {"type": "vm"},
            "network": {"default_interface": "eth0"},
            "disks": {"ambiguous_candidates": ["sdb", "sdc"]},
            "dependencies": {
                "missing_required": ["vegeta"],
                "missing_optional": ["kubectl"],
            },
            "warnings": ["Multiple plausible data disks were found; confirm ledger/accounts devices."],
        })
        self.assertEqual(report["status"], "needs_dependencies")
        self.assertEqual(report["environment"]["dependencies"]["missing_required"], ["vegeta"])
        text = format_doctor_report(report)
        self.assertIn("Agent doctor report", text)
        self.assertIn("required dependencies missing: vegeta", text)

    def test_request_modifier_updates_qps_and_mixed_weights(self):
        request = {"chain": "solana", "rpc_mode": "single", "use_fake_node": True}
        updated, changes = apply_request_modification(
            request,
            "Set max qps to 5000, initial qps 100, step 250, duration 120 seconds.",
        )
        self.assertEqual(updated["qps"]["max"], 5000)
        self.assertEqual(updated["qps"]["initial"], 100)
        self.assertEqual(updated["qps"]["step"], 250)
        self.assertEqual(updated["qps"]["duration_seconds"], 120)
        self.assertTrue(changes)

        updated, changes = apply_request_modification(
            updated,
            "Change mixed weights to getSlot 70%, getBlockHeight 30%",
        )
        self.assertEqual(updated["rpc_mode"], "mixed")
        self.assertEqual(updated["workload"]["methods"][0]["method"], "getSlot")
        self.assertEqual(sum(item["weight"] for item in updated["workload"]["methods"]), 100)
        self.assertIn("mixed_weights_confirmation", updated["confirmations"])

    def test_compactor_preserves_current_state_and_recent_turns(self):
        turns = [{"role": "user", "content": f"turn {idx}"} for idx in range(20)]
        summary = compact_session_state(
            turns=turns,
            request={"chain": "solana", "rpc_mode": "mixed", "use_fake_node": True},
            plan={
                "plan_id": "plan_test",
                "chain": "solana",
                "strategy": "smoke",
                "rpc_mode": "mixed",
                "required_questions": [
                    {"id": "ledger_device_confirmation", "severity": "blocker", "prompt": "Confirm ledger device."}
                ],
            },
            job={"job_id": "job_test", "status": "completed", "artifact_index": "/tmp/artifact_index.json"},
            discovery={"deployment": {"type": "vm"}, "cloud": {"provider": "gcp"}},
            keep_recent=4,
            reason="test",
        )
        self.assertEqual(summary["preserved_state"]["chain"], "solana")
        self.assertEqual(summary["preserved_state"]["job_id"], "job_test")
        self.assertEqual(len(summary["recent_turns"]), 4)
        self.assertEqual(summary["compacted_turn_count"], 16)
        self.assertEqual(summary["thresholds"]["context_window_tokens"], 1000000)
        self.assertEqual(summary["thresholds"]["trigger_ratio"], 0.7)
        self.assertTrue(summary["open_questions"])
        self.assertFalse(should_auto_compact(turns=turns, turn_threshold=1000))
        self.assertTrue(should_auto_compact(turns=turns, turn_threshold=10))
        self.assertTrue(should_auto_compact(
            turns=[{"role": "user", "content": "x" * 100}],
            context_window_tokens=10,
            trigger_ratio=0.5,
            turn_threshold=1000,
        ))

    def test_wizard_applies_required_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_wizard(
                prompt="Run a baseline benchmark on my node.",
                output_dir=tmp,
                discovery_override={
                    "source": "test",
                    "mode": "read_only",
                    "deployment": {"type": "vm"},
                    "cloud": {"provider": "gcp", "platform": "gce", "confidence": 0.8},
                    "disks": {
                        "confidence": 0.0,
                        "proposed_ledger_device": "",
                        "proposed_accounts_device": "",
                        "ambiguous_candidates": ["sdb", "sdc"],
                        "candidates": [
                            {"name": "sdb", "mountpoint": "/data"},
                            {"name": "sdc", "mountpoint": "/accounts"},
                        ],
                    },
                    "dependencies": {"mode": "audit", "missing_required": ["vegeta"], "missing_optional": []},
                    "warnings": [],
                },
                answers={
                    "chain": "solana",
                    "local_rpc_url": "http://127.0.0.1:8899",
                    "ledger_device_confirmation": "sdb",
                    "ledger_device": "sdb",
                    "blockchain_process_names": "agave-validator solana-validator",
                    "data_vol_max_iops": "30000",
                    "data_vol_max_throughput": "700",
                    "network_max_bandwidth_gbps": "25",
                    "dependency_mode_confirmation": "audit",
                },
                yes=False,
                mock=False,
                quiet=True,
            )
            self.assertEqual(result["status"], "planned")
            plan = json.loads((Path(tmp) / "plan.json").read_text())
            self.assertEqual(plan["chain"], "solana")
            self.assertEqual(plan["execution"]["environment"]["LOCAL_RPC_URL"], "http://127.0.0.1:8899")
            self.assertEqual(plan["materialized_config"]["LEDGER_DEVICE"], "sdb")
            self.assertIn("ledger_device_confirmation", plan["confirmed_inputs"])

    def test_onboarding_package_contains_workload_and_validation_contract(self):
        package = generate_onboarding_package("foochain", ["foo_methodA", "foo_methodB"], adapter_family="jsonrpc")
        self.assertEqual(package["workload_plugin"]["rpc_mode"], "mixed")
        self.assertEqual(sum(item["weight"] for item in package["workload_plugin"]["mixed_weighted"]), 100)
        self.assertIn("jsonrpc", package["supported_families"])
        self.assertTrue(package["fake_node_steps"])

    def test_llm_provider_config_supports_api_keys_and_google_auth(self):
        gemini_key_config = load_llm_config({
            "LLM_PROVIDER": "gemini",
            "LLM_MODEL": "gemini-3.1-pro",
            "LLM_AUTH_MODE": "api_key",
            "GEMINI_API_KEY": "test-key",
        })
        self.assertEqual(gemini_key_config.validate(), [])
        self.assertEqual(provider_from_config(gemini_key_config).__class__.__name__, "GeminiAPIKeyProvider")

        claude_key_config = load_llm_config({
            "LLM_PROVIDER": "claude",
            "LLM_MODEL": "claude-opus-4-8",
            "LLM_AUTH_MODE": "api_key",
            "ANTHROPIC_API_KEY": "test-key",
        })
        self.assertEqual(claude_key_config.validate(), [])
        self.assertEqual(provider_from_config(claude_key_config).__class__.__name__, "AnthropicAPIKeyProvider")

        openai_key_config = load_llm_config({
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-5.5",
            "LLM_AUTH_MODE": "api_key",
            "OPENAI_API_KEY": "test-key",
        })
        self.assertEqual(openai_key_config.validate(), [])
        self.assertEqual(provider_from_config(openai_key_config).__class__.__name__, "OpenAIProvider")

        config = load_llm_config({
            "LLM_PROVIDER": "gemini",
            "LLM_MODEL": "gemini-3.1-pro",
            "LLM_AUTH_MODE": "service_account_impersonation",
            "GOOGLE_CLOUD_PROJECT": "example-project",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
            "GOOGLE_SERVICE_ACCOUNT_EMAIL": "benchmark-agent@example-project.iam.gserviceaccount.com",
        })
        self.assertEqual(config.provider, "gemini")
        self.assertEqual(config.validate(), [])
        plan = credential_plan(config).safe_dict()
        self.assertEqual(plan["mode"], "service_account_impersonation")
        self.assertEqual(plan["service_account_email"], "benchmark-agent@example-project.iam.gserviceaccount.com")
        self.assertEqual(provider_from_config(config).__class__.__name__, "VertexGeminiProvider")

        partner_model_config = load_llm_config({
            "LLM_PROVIDER": "claude",
            "LLM_MODEL": "claude-opus-4-8",
            "LLM_AUTH_MODE": "attached_service_account",
            "GOOGLE_CLOUD_PROJECT": "example-project",
            "GOOGLE_CLOUD_LOCATION": "us-east5",
        })
        self.assertEqual(partner_model_config.provider, "claude")
        self.assertEqual(partner_model_config.validate(), [])
        self.assertEqual(provider_from_config(partner_model_config).__class__.__name__, "VertexClaudeProvider")

        json_key_config = load_llm_config({
            "LLM_PROVIDER": "gemini",
            "LLM_MODEL": "gemini-3.1-pro",
            "LLM_AUTH_MODE": "service_account_file",
            "GOOGLE_CLOUD_PROJECT": "example-project",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/example-service-account.json",
        })
        self.assertEqual(json_key_config.validate(), [])

        missing_impersonation = load_llm_config({
            "LLM_PROVIDER": "gemini",
            "LLM_AUTH_MODE": "service_account_impersonation",
            "GOOGLE_CLOUD_PROJECT": "example-project",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
        })
        self.assertIn("GOOGLE_SERVICE_ACCOUNT_EMAIL is required", "; ".join(missing_impersonation.validate()))

    def test_prompt_contracts_are_explicit_and_tool_bounded(self):
        system_prompt = load_prompt("system")
        request_prompt = compose_prompt("system", "safety_guardrails", "config_interviewer", "request_drafter", "workload_designer")
        result_prompt = compose_prompt("system", "result_analyzer", "chart_diagnostics")
        onboarding_prompt = compose_prompt("system", "chain_onboarding")

        self.assertIn("deterministic framework tools", system_prompt)
        self.assertIn("Do not invent", request_prompt)
        self.assertIn("runtime.env", request_prompt)
        self.assertIn("custom RPC methods", request_prompt)
        self.assertIn("weighted", request_prompt)
        self.assertIn("P50/P90/P99", result_prompt)
        self.assertIn("CPU-disk correlation", result_prompt)
        self.assertIn("needs_review", onboarding_prompt)
        self.assertIn("fake-node fixtures", onboarding_prompt)
        prompt_files = {path.stem for path in (REPO / "agent" / "prompts").glob("*.md")}
        self.assertEqual(prompt_files - set(prompt_coverage()), set())
        self.assertIn("artifact_analysis", PROMPT_BUNDLES)
        self.assertIn("onboarding", PROMPT_BUNDLES)

    def test_prompt_orchestrator_is_used_by_artifact_and_onboarding_flows(self):
        from analyzers.artifact_qa import answer_artifact_question

        class SpyProvider:
            def __init__(self):
                self.system_prompts = []

            def complete(self, request):
                self.system_prompts.append(request.messages[0].content)
                return LLMResponse(text="spy synthesis", model="spy", provider="spy", raw={})

        provider = SpyProvider()
        answer = answer_artifact_question("Why are charts empty?", artifact_index=None, llm_provider=provider)
        self.assertEqual(answer["answer"], "spy synthesis")
        self.assertIn("Analyze benchmark artifacts", provider.system_prompts[0])
        self.assertIn("Explain benchmark charts", provider.system_prompts[0])

        provider = SpyProvider()
        package = generate_onboarding_package(
            "foochain",
            ["foo_getBalance"],
            adapter_family="jsonrpc",
            llm_provider=provider,
        )
        self.assertEqual(package["llm_summary"], "spy synthesis")
        self.assertIn("onboard a new blockchain node", provider.system_prompts[0])
        self.assertIn("Design blockchain RPC workloads", provider.system_prompts[0])

    def test_http_knowledge_provider_contract(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                if self.path == "/search":
                    self._json({"results": [{"title": "Solana RPC", "text": "getBalance"}]})
                elif self.path == "/workload/suggest":
                    self._json({"methods": [{"method": "getBalance", "weight": 100}]})
                else:
                    self._json({})

            def do_GET(self):
                if self.path == "/chains/solana/rpc-methods":
                    self._json({"methods": [{"method": "getBalance"}]})
                else:
                    self._json({})

            def log_message(self, format, *args):
                return

            def _json(self, payload):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"
            payload = run_agent(
                "knowledge-smoke",
                "--query",
                "solana",
                "--chain",
                "solana",
                env={
                    "AGENT_KNOWLEDGE_PROVIDER": "http",
                    "AGENT_KNOWLEDGE_BASE_URL": base_url,
                },
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        self.assertTrue(payload["status"]["enabled"])
        self.assertEqual(payload["search"][0]["title"], "Solana RPC")
        self.assertEqual(payload["rpc_methods"][0]["method"], "getBalance")

    def test_job_notification_webhook_is_optional_and_file_backed(self):
        events = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                events.append(json.loads(self.rfile.read(length).decode("utf-8")))
                self.send_response(204)
                self.end_headers()

            def log_message(self, format, *args):
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        old_url = os.environ.get("AGENT_NOTIFY_WEBHOOK_URL")
        old_on = os.environ.get("AGENT_NOTIFY_ON")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                plan_file = Path(tmp) / "plan.json"
                plan_file.write_text(json.dumps({
                    "plan_id": "plan_notify",
                    "chain": "solana",
                    "strategy": "smoke",
                    "rpc_mode": "single",
                    "use_fake_node": True,
                    "required_inputs": [],
                    "execution": {
                        "working_dir": str(REPO),
                        "command": ["./blockchain_node_benchmark.sh", "--quick", "--fake-node"],
                        "environment": {"BLOCKCHAIN_NODE": "solana"},
                    },
                    "materialized_config": {},
                    "artifacts": {},
                }), encoding="utf-8")
                os.environ["AGENT_NOTIFY_WEBHOOK_URL"] = f"http://127.0.0.1:{server.server_port}/notify"
                os.environ["AGENT_NOTIFY_ON"] = "completed"
                job = submit_job(plan_file, jobs_dir=Path(tmp) / "jobs", mock=True)
        finally:
            if old_url is None:
                os.environ.pop("AGENT_NOTIFY_WEBHOOK_URL", None)
            else:
                os.environ["AGENT_NOTIFY_WEBHOOK_URL"] = old_url
            if old_on is None:
                os.environ.pop("AGENT_NOTIFY_ON", None)
            else:
                os.environ["AGENT_NOTIFY_ON"] = old_on
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(events[0]["job_id"], job["job_id"])
        self.assertEqual(events[0]["status"], "completed")

    def test_llm_config_can_load_persistent_agent_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config_dir = repo / "config"
            config_dir.mkdir()
            agent_config = config_dir / "agent_config.sh"
            user_config = config_dir / "user_config.sh"
            agent_config.write_text(
                "\n".join([
                    'LLM_PROVIDER="${LLM_PROVIDER:-openai}"',
                    'LLM_MODEL="${LLM_MODEL:-gpt-5.5}"',
                    'OPENAI_API_KEY="${OPENAI_API_KEY:-test-key}"',
                    'export LLM_PROVIDER LLM_MODEL OPENAI_API_KEY',
                ]),
                encoding="utf-8",
            )
            user_config.write_text('LLM_PROVIDER="${LLM_PROVIDER:-gemini}"\nexport LLM_PROVIDER\n', encoding="utf-8")
            old_root = llm_config_module.REPO_ROOT
            old_agent_config = llm_config_module.AGENT_CONFIG
            old_config = llm_config_module.USER_CONFIG
            try:
                llm_config_module.REPO_ROOT = repo
                llm_config_module.AGENT_CONFIG = agent_config
                llm_config_module.USER_CONFIG = user_config
                config = load_llm_config()
            finally:
                llm_config_module.REPO_ROOT = old_root
                llm_config_module.AGENT_CONFIG = old_agent_config
                llm_config_module.USER_CONFIG = old_config
            self.assertEqual(config.provider, "openai")
            self.assertEqual(config.model, "gpt-5.5")
            self.assertTrue(config.openai_api_key_present)

    def test_agent_runtime_env_overrides_parent_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_file = tmp_path / "runtime_value.txt"
            runner = tmp_path / "blockchain_node_benchmark.sh"
            runner.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$BLOCKCHAIN_NODE\" > \"$1\"\n",
                encoding="utf-8",
            )
            runner.chmod(0o755)
            plan_file = tmp_path / "plan.json"
            plan_file.write_text(json.dumps({
                "plan_id": "plan_runtime_priority",
                "chain": "solana",
                "strategy": "smoke",
                "rpc_mode": "single",
                "use_fake_node": True,
                "required_inputs": [],
                "execution": {
                    "working_dir": str(tmp_path),
                    "command": ["./blockchain_node_benchmark.sh", str(output_file)],
                    "environment": {"BLOCKCHAIN_NODE": "solana"},
                },
                "materialized_config": {},
                "artifacts": {},
            }), encoding="utf-8")
            old_value = os.environ.get("BLOCKCHAIN_NODE")
            os.environ["BLOCKCHAIN_NODE"] = "ethereum"
            try:
                job = submit_job(plan_file, jobs_dir=tmp_path / "jobs", approved=True)
            finally:
                if old_value is None:
                    os.environ.pop("BLOCKCHAIN_NODE", None)
                else:
                    os.environ["BLOCKCHAIN_NODE"] = old_value
            self.assertEqual(job["status"], "completed")
            self.assertEqual(output_file.read_text(encoding="utf-8").strip(), "solana")
            runtime_env = Path(job["runtime_env_file"]).read_text(encoding="utf-8")
            self.assertIn("export BLOCKCHAIN_NODE='solana'", runtime_env)

    def test_detached_real_job_continues_after_submit_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_file = tmp_path / "detached_value.txt"
            runner = tmp_path / "blockchain_node_benchmark.sh"
            runner.write_text(
                "#!/usr/bin/env bash\n"
                "sleep 0.2\n"
                "printf '%s\\n' \"$BLOCKCHAIN_NODE\" > \"$1\"\n",
                encoding="utf-8",
            )
            runner.chmod(0o755)
            plan_file = tmp_path / "plan.json"
            plan_file.write_text(json.dumps({
                "plan_id": "plan_detached",
                "chain": "solana",
                "strategy": "smoke",
                "rpc_mode": "single",
                "use_fake_node": True,
                "required_inputs": [],
                "execution": {
                    "working_dir": str(tmp_path),
                    "command": ["./blockchain_node_benchmark.sh", str(output_file)],
                    "environment": {"BLOCKCHAIN_NODE": "solana"},
                    "runner_mode": "detached",
                },
                "materialized_config": {},
                "artifacts": {},
            }), encoding="utf-8")
            job = submit_job(plan_file, jobs_dir=tmp_path / "jobs", approved=True)
            self.assertEqual(job["status"], "running")
            self.assertEqual(job["runner_mode"], "detached")
            self.assertGreater(job["worker_pid"], 0)
            completed = None
            for _ in range(40):
                current = get_job(job["job_id"], jobs_dir=tmp_path / "jobs")
                if current["status"] != "running":
                    completed = current
                    break
                time.sleep(0.1)
            self.assertIsNotNone(completed)
            assert completed is not None
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(output_file.read_text(encoding="utf-8").strip(), "solana")
            self.assertTrue(Path(completed["artifact_index"]).is_file())

    def test_dynamic_framework_capabilities_are_loaded_from_current_templates(self):
        capabilities = load_framework_capabilities()
        self.assertEqual(capabilities["chain_count"], 36)
        self.assertEqual(capabilities["family_count"], 6)
        self.assertIn("jsonrpc", capabilities["families"])
        self.assertGreater(capabilities["fake_node"]["fixture_file_count"], 0)
        solana = next(chain for chain in capabilities["chains"] if chain["chain"] == "solana")
        self.assertIn("getBalance", solana["methods"])
        self.assertEqual(solana["family"], "jsonrpc")


if __name__ == "__main__":
    unittest.main()
