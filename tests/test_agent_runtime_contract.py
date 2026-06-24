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
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
AGENT = REPO / "agent" / "cli.py"
AGENT_BIN = REPO / "bin" / "anychain-agent"
sys.path.insert(0, str(REPO / "agent"))

from discovery.environment import discover_environment  # noqa: E402
from knowledge.framework_capabilities import load_framework_capabilities  # noqa: E402
from knowledge.framework_context import load_framework_context, render_framework_context_for_prompt  # noqa: E402
from llm import config as llm_config_module  # noqa: E402
from llm.config import load_llm_config  # noqa: E402
from llm.google_auth import credential_plan  # noqa: E402
from llm.providers import provider_from_config  # noqa: E402
from runners.materialize import build_runtime_env  # noqa: E402
from runners.guardrails import validate_execution_plan  # noqa: E402
from runners.job_manager import get_job, submit_job  # noqa: E402
from adk_app.app import status_payload as adk_status_payload  # noqa: E402
from adk_app.callbacks import before_tool_callback  # noqa: E402
from adk_app.compat import adk_feature_report  # noqa: E402
from adk_app.evals.runner import run_offline_evals as run_adk_offline_evals  # noqa: E402
from adk_app.workflow.schemas import validate_intent_route  # noqa: E402
from adk_app.workflow.root_workflow import root_workflow_dry_run  # noqa: E402
from adk_app.agents.router import route_user_intent  # noqa: E402
from adk_app.instructions import ROOT_INSTRUCTION  # noqa: E402
from adk_app.root_agent import resolve_adk_model  # noqa: E402
from adk_app.runtime import run_adk_cli  # noqa: E402
from adk_app.runner_bridge import runner_bridge_status  # noqa: E402
from adk_app.state import load_startup_state as adk_load_startup_state  # noqa: E402
from adk_app.state import preserved_state_for_adk  # noqa: E402
from adk_app.tools.auth import inspect_llm_auth as adk_inspect_llm_auth  # noqa: E402
from adk_app.tools.registry import get_adk_tools  # noqa: E402
from adk_app.tools.actions import install_dependencies as adk_install_dependencies  # noqa: E402
from adk_app.tools.actions import _fake_node_smoke_plan  # noqa: E402
from adk_app.tools.actions import run_fake_node_smoke_benchmark as adk_run_fake_node_smoke_benchmark  # noqa: E402
from adk_app.tools.actions import run_smoke as adk_run_smoke  # noqa: E402
from adk_app.tools.actions import submit_benchmark_job as adk_submit_benchmark_job  # noqa: E402
from adk_app.tools.enterprise import enterprise_integration_manifest  # noqa: E402
from adk_app.tools.planning import draft_benchmark_request as adk_draft_benchmark_request  # noqa: E402
from adk_app.tools.planning import generate_benchmark_plan as adk_generate_benchmark_plan  # noqa: E402
from adk_app.tools.planning import prepare_benchmark_run as adk_prepare_benchmark_run  # noqa: E402
from adk_app.tools.planning import render_runbook as adk_render_runbook  # noqa: E402
from adk_app.tools.planning import run_preflight as adk_run_preflight  # noqa: E402
from adk_app.tools.read_only import audit_dependencies as adk_audit_dependencies  # noqa: E402
from adk_app.tools.read_only import load_execution_contract as adk_load_execution_contract  # noqa: E402
from adk_app.tools.read_only import load_framework_context as adk_load_framework_context  # noqa: E402
from adk_app.tools.read_only import load_framework_capabilities as adk_load_framework_capabilities  # noqa: E402
from adk_app.tools.read_only import list_rpc_methods as adk_list_rpc_methods  # noqa: E402
from adk_app.tools.read_only import list_supported_chains as adk_list_supported_chains  # noqa: E402
from diagnostics.doctor import format_doctor_report, run_doctor  # noqa: E402
from onboarding.chain_onboarding import generate_onboarding_package  # noqa: E402
from onboarding.request_answers import answer_onboarding_request  # noqa: E402


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
            if "computeMetadata/v1/instance/zone" in joined:
                return 0, "projects/123/zones/us-central1-a", ""
            if "computeMetadata/v1/instance/machine-type" in joined:
                return 0, "projects/123/machineTypes/c3-standard-22", ""
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
        self.assertEqual(discovery["cloud"]["region"], "us-central1")
        self.assertEqual(discovery["cloud"]["zone"], "us-central1-a")
        self.assertEqual(discovery["cloud"]["machine_type"], "c3-standard-22")
        self.assertEqual(discovery["network"]["default_interface"], "eth0")
        self.assertEqual(discovery["disks"]["proposed_ledger_device"], "sdb")
        self.assertEqual(discovery["dependencies"]["mode"], "audit")

    def test_prompt_to_plan_preflight_and_mock_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            request_file = tmp_path / "request.json"
            plan_file = tmp_path / "plan.json"
            jobs_dir = tmp_path / "jobs"

            request = {
                "chain": "solana",
                "goal": "max_stable_qps",
                "rpc_mode": "single",
                "use_fake_node": True,
                "deployment": {"type": "kubernetes", "provider": "gcp"},
                "observability": {"enabled": False, "mode": "local"},
                "dependency_mode": "audit",
                "runner_mode": "detached",
                "bottleneck_focus": ["disk"],
                "recommended_initial_validation": "smoke",
                "source_prompt": "Test Solana maximum stable QPS on GKE with fake-node smoke first and focus on disk P99.",
            }
            request_file.write_text(json.dumps(request), encoding="utf-8")
            self.assertEqual(request["chain"], "solana")
            self.assertEqual(request["goal"], "max_stable_qps")
            self.assertTrue(request["use_fake_node"])
            self.assertEqual(request["deployment"]["type"], "kubernetes")
            self.assertIn("disk", request["bottleneck_focus"])
            self.assertEqual(request["recommended_initial_validation"], "smoke")

            capabilities = run_agent("capabilities")
            self.assertEqual(capabilities["chain_count"], 36)
            self.assertEqual(capabilities["family_count"], 6)
            self.assertGreater(capabilities["unique_rpc_method_count"], 100)

            gap = run_agent("gap-analysis", "--chain", "solana", "--method", "getBalance", "--method", "missingMethod")
            self.assertEqual(gap["chain"], "solana")
            self.assertIn("getBalance", gap["supported_methods"])
            self.assertIn("missingMethod", gap["missing_methods"])
            self.assertTrue(gap["onboarding_plan"])

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

            new_chain_gap = answer_onboarding_request("How do I add new chain foochain?")
            self.assertIn("Onboarding package for foochain", new_chain_gap)
            self.assertIn("Evidence policy", new_chain_gap)
            self.assertIn("Coding handoff", new_chain_gap)
            self.assertIn("official RPC docs", new_chain_gap)
            self.assertIn("docs/en/secondary-development-guide.md", new_chain_gap)

            kb_onboarding = answer_onboarding_request("How do I integrate our enterprise Knowledge Base?")
            self.assertIn("Enterprise Knowledge Base onboarding plan", kb_onboarding)

            platform_onboarding = answer_onboarding_request("How do we embed this into our internal Agent platform?")
            self.assertIn("Enterprise Agent platform integration plan", platform_onboarding)
            self.assertIn("tool-schema", platform_onboarding)
            self.assertIn("tool-call", platform_onboarding)

            protocol_onboarding = answer_onboarding_request("Generate a plan to add a new protocol family for FooVM")
            self.assertIn("New protocol family onboarding plan", protocol_onboarding)

            rpc_onboarding = answer_onboarding_request("Add custom RPC method foo_getBalance to chain foochain")
            self.assertIn("foo_getBalance", rpc_onboarding)
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
            self.assertIn("quality_gate", onboarding)
            self.assertIn("coding_brief", onboarding)
            self.assertTrue(onboarding["quality_gate"]["family_known"])
            self.assertIn("official protocol/RPC documentation", onboarding["quality_gate"]["required_chain_evidence"][0])

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
            self.assertIn("load_execution_contract", tool_names)
            self.assertTrue(all(tool["type"] == "function" for tool in schema["tools"]))
            install_tool = next(tool for tool in schema["tools"] if tool["function"]["name"] == "install_dependencies")
            install_props = install_tool["function"]["parameters"]["properties"]
            self.assertFalse(install_props["include_agent_runtime"]["default"])
            self.assertTrue(install_props["include_vegeta"]["default"])
            self.assertTrue(install_props["no_sudo"]["default"])
            self.assertFalse(install_props["include_gcloud"]["default"])
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
                json.dumps({
                    "source_prompt": "Create a Solana fake-node smoke benchmark at 1 QPS",
                    "chain": "solana",
                    "goal": "smoke",
                    "rpc_mode": "single",
                    "use_fake_node": True,
                    "qps_max": 1,
                }),
            )
            self.assertEqual(draft_tool["chain"], "solana")
            structured_draft_tool = run_agent(
                "tool-call",
                "--name",
                "draft_request",
                "--arguments",
                json.dumps({
                    "chain": "ethereum",
                    "goal": "stress",
                    "rpc_mode": "mixed",
                    "use_fake_node": False,
                    "qps_max": 2000,
                    "rpc_methods": ["eth_blockNumber", "eth_getBalance"],
                    "mixed_weights": {"eth_blockNumber": 60, "eth_getBalance": 40},
                }),
            )
            self.assertEqual(structured_draft_tool["chain"], "ethereum")
            self.assertEqual(structured_draft_tool["rpc_mode"], "mixed")
            self.assertEqual(structured_draft_tool["qps"]["max"], 2000)
            self.assertEqual(sum(item["weight"] for item in structured_draft_tool["mixed_weighted"]), 100)
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

    def test_human_entrypoint_uses_product_terminal_not_adk_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_adk = Path(tmp) / "fake-adk"
            fake_adk.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$@\" > \"$FAKE_ADK_ARGS_FILE\"\n"
                "cat > \"$FAKE_ADK_STDIN_FILE\"\n",
                encoding="utf-8",
            )
            fake_adk.chmod(0o755)
            args_file = Path(tmp) / "args.txt"
            stdin_file = Path(tmp) / "stdin.txt"
            env = os.environ.copy()
            env["FAKE_ADK_ARGS_FILE"] = str(args_file)
            env["FAKE_ADK_STDIN_FILE"] = str(stdin_file)

            completed = subprocess.run(
                [
                    str(AGENT_BIN),
                    "--state-file",
                    str(Path(tmp) / "state.json"),
                    "--prompt",
                    "I want to benchmark Solana with fake-node",
                ],
                cwd=REPO,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=True,
            )
            self.assertEqual(completed.returncode, 0)
            self.assertIn("Selected fake-node", completed.stdout)
            self.assertIn("CLOUD", completed.stdout)
            self.assertNotIn("[user]", completed.stdout)
            self.assertFalse(args_file.exists())
            self.assertFalse(stdin_file.exists())

            missing_status = run_adk_cli(["--adk-bin", str(Path(tmp) / "missing-adk")])
            self.assertEqual(missing_status, 2)

    def test_agent_doctor_formats_readiness_report(self):
        old_env = os.environ.copy()
        try:
            os.environ.update({
                "LLM_PROVIDER": "gemini",
                "LLM_MODEL": "gemini-3.1-pro",
                "LLM_AUTH_MODE": "google_adc",
                "GOOGLE_CLOUD_PROJECT": "example-project",
                "GOOGLE_CLOUD_LOCATION": "us-central1",
            })
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
            self.assertEqual(report["google_auth"]["auth_mode"], "google_adc")
            self.assertIn("gcloud_available", report["google_auth"])
            text = format_doctor_report(report)
            self.assertIn("Agent doctor report", text)
            self.assertIn("required dependencies missing: vegeta", text)
            self.assertIn("Google auth mode: google_adc", text)
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_adk_runner_bridge_is_explicit_not_fallback(self):
        status = runner_bridge_status().as_dict()
        self.assertIn("available", status)
        self.assertIn("reason", status)
        self.assertEqual(status["runner_import"], "google.adk.runners.Runner")

    def test_onboarding_package_contains_workload_and_validation_contract(self):
        package = generate_onboarding_package("foochain", ["foo_methodA", "foo_methodB"], adapter_family="jsonrpc")
        self.assertEqual(package["workload_plugin"]["rpc_mode"], "mixed")
        self.assertEqual(sum(item["weight"] for item in package["workload_plugin"]["mixed_weighted"]), 100)
        self.assertIn("jsonrpc", package["supported_families"])
        self.assertTrue(package["fake_node_steps"])
        self.assertIn("quality_gate", package)
        self.assertIn("coding_brief", package)
        self.assertIn("Implementation brief", package["coding_brief"])
        self.assertIn("Existing-family chain path", package["coding_brief"])
        self.assertIn("New-family path", package["coding_brief"])
        self.assertIn("New RPC method path", package["coding_brief"])
        self.assertIn("fake-node smoke", package["coding_brief"])
        self.assertIn("Do not rely on the model's general blockchain knowledge", package["quality_gate"]["llm_policy"][0])

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

        with tempfile.TemporaryDirectory() as tmp:
            adc_file = Path(tmp) / "application_default_credentials.json"
            adc_file.write_text('{"quota_project_id": "adc-quota-project"}\n', encoding="utf-8")
            adc_config = load_llm_config({
                "LLM_PROVIDER": "gemini",
                "LLM_MODEL": "gemini-3.1-pro",
                "LLM_AUTH_MODE": "google_adc",
                "GOOGLE_CLOUD_LOCATION": "us-central1",
                "GOOGLE_APPLICATION_CREDENTIALS": str(adc_file),
            })
        self.assertEqual(adc_config.google_project, "adc-quota-project")
        self.assertEqual(adc_config.validate(), [])

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

    def test_adk_model_resolution_uses_configured_real_model(self):
        old_env = os.environ.copy()
        try:
            os.environ.clear()
            os.environ.update({
                "LLM_PROVIDER": "gemini",
                "LLM_MODEL": "gemini-3.5-flash",
                "LLM_AUTH_MODE": "api_key",
                "GEMINI_API_KEY": "test-key",
            })
            self.assertEqual(resolve_adk_model(), "gemini-3.5-flash")
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_prompt_contracts_are_explicit_and_tool_bounded(self):
        self.assertIn("deterministic benchmark tools", ROOT_INSTRUCTION)
        self.assertIn("Do not invent", ROOT_INSTRUCTION)
        self.assertIn("runtime.env", ROOT_INSTRUCTION)
        self.assertIn("custom RPC methods", ROOT_INSTRUCTION)
        self.assertIn("weighted", ROOT_INSTRUCTION)
        self.assertIn("P50/P90/P99", ROOT_INSTRUCTION)
        self.assertIn("CPU-disk correlation", ROOT_INSTRUCTION)
        self.assertIn("needs_review", ROOT_INSTRUCTION)
        self.assertIn("fake-node fixtures", ROOT_INSTRUCTION)
        self.assertIn("Use a structured router only for intent classification", ROOT_INSTRUCTION)
        self.assertIn("prepare_benchmark_run", ROOT_INSTRUCTION)
        self.assertIn("run_fake_node_smoke_benchmark", ROOT_INSTRUCTION)
        self.assertIn("audit_dependencies", ROOT_INSTRUCTION)
        tool_names = {tool.__name__ for tool in get_adk_tools(include_actions=True)}
        self.assertIn("prepare_benchmark_run", tool_names)
        self.assertIn("draft_benchmark_request", tool_names)
        self.assertIn("generate_benchmark_plan", tool_names)
        self.assertIn("run_smoke", tool_names)
        self.assertIn("run_fake_node_smoke_benchmark", tool_names)
        self.assertIn("install_dependencies", tool_names)
        self.assertIn("submit_benchmark_job", tool_names)
        self.assertFalse((REPO / "agent" / "prompts").exists())

    def test_adk_action_callback_blocks_unapproved_execution(self):
        class Tool:
            name = "submit_benchmark_job"

        for tool_name in ("submit_benchmark_job", "run_fake_node_smoke_benchmark", "install_dependencies"):
            Tool.name = tool_name
            blocked = before_tool_callback(Tool(), {"plan_file": "/tmp/plan.json"}, tool_context=None)
            self.assertIsNotNone(blocked)
            self.assertEqual(blocked["status"], "needs_confirmation")
            self.assertTrue(blocked["requires_user_confirmation"])

        Tool.name = "submit_benchmark_job"
        allowed = before_tool_callback(Tool(), {"plan_file": "/tmp/plan.json", "approved": True}, tool_context=None)
        self.assertIsNone(allowed)

        class ReadOnlyTool:
            name = "run_doctor"

        self.assertIsNone(before_tool_callback(ReadOnlyTool(), {}, tool_context=None))

    def test_dependency_install_defaults_to_benchmark_engine_only(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout="ok", args=command)

        with patch("adk_app.tools.actions.subprocess.run", side_effect=fake_run):
            result = adk_install_dependencies(approved=True)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(calls[0][:2], ["bash", "scripts/install_deps.sh"])
        self.assertIn("--yes", calls[0])
        self.assertEqual(len(calls), 1)
        self.assertTrue(result["data"]["agent_runtime"]["skipped"])

        calls.clear()
        with patch("adk_app.tools.actions.subprocess.run", side_effect=fake_run):
            result = adk_install_dependencies(approved=True, include_gcloud=True)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(calls[0][:2], ["bash", "scripts/install_deps.sh"])
        self.assertEqual(calls[1][:2], ["bash", "scripts/install_agent_deps.sh"])
        self.assertIn("--with-gcloud", calls[1])

    def test_adk_skeleton_is_offline_safe_and_tool_bounded(self):
        status = adk_status_payload()
        self.assertIn(status["status"], {"ready", "not_installed"})
        self.assertIn("adk", status)
        self.assertTrue(status["root_instruction_present"])
        self.assertIn("deterministic benchmark tools", ROOT_INSTRUCTION)
        self.assertIn("Never install dependencies without explicit user confirmation", ROOT_INSTRUCTION)
        self.assertIn("Never launch a real benchmark without explicit user confirmation", ROOT_INSTRUCTION)
        self.assertIn("cite concrete artifact paths", ROOT_INSTRUCTION)
        self.assertIn("Do not rely on the model's general blockchain knowledge", ROOT_INSTRUCTION)
        self.assertIn("produce a coding brief", ROOT_INSTRUCTION)
        self.assertIn("Use a structured router only for intent classification", ROOT_INSTRUCTION)

        cli_status = run_agent("adk-status")
        self.assertEqual(cli_status["root_instruction_present"], status["root_instruction_present"])

    def test_adk_feature_report_is_offline_safe(self):
        report = adk_feature_report()
        self.assertIn("package", report)
        self.assertIn("features", report)
        self.assertIn("workflow", report["features"])
        self.assertIn("implementation_recommendation", report)

        cli_report = run_agent("adk-feature-report")
        self.assertIn("workflow", cli_report["features"])

    def test_adk_native_workflow_smoke_is_credential_free(self):
        payload = run_agent("adk-native-smoke")
        self.assertIn(payload["status"], {"passed", "not_installed", "failed"})
        if payload["status"] == "passed":
            self.assertEqual(payload["workflow"], "anychain_native_workflow_smoke")
            self.assertEqual(payload["nodes"], ["startup_doctor", "intent_route"])
            self.assertGreaterEqual(payload["event_count"], 1)
        elif payload["status"] == "not_installed":
            self.assertIn("recommendation", payload)
        else:
            self.fail(payload.get("error", "native ADK workflow smoke failed"))

    def test_router_outputs_valid_workflow_intent_schema(self):
        route = route_user_intent("请用 fake-node 测试 solana mixed getSlot=70 getBlockHeight=30", default_language="zh")
        self.assertEqual(route["intent"], "START_BENCHMARK")
        self.assertEqual(route["language"], "zh")
        self.assertEqual(route["entities"]["chain"], "solana")
        self.assertEqual(route["entities"]["target"], "fake-node")
        self.assertEqual(validate_intent_route(route), [])

        real_node_route = route_user_intent("我想测试真实 solana 节点", default_language="zh")
        self.assertEqual(real_node_route["intent"], "START_BENCHMARK")
        self.assertEqual(real_node_route["entities"]["target"], "real-node")
        self.assertNotIn("target", real_node_route["missing_clarifications"])

        onboarding = run_agent("route-intent", "--text", "Add custom RPC method foo_getBalance to chain foochain")
        self.assertEqual(onboarding["intent"], "ONBOARD_CHAIN_RPC")
        self.assertEqual(validate_intent_route(onboarding), [])

    def test_root_workflow_dry_run_emits_workflow_events(self):
        payload = root_workflow_dry_run("请用 fake-node 测试 solana", language="zh")
        event_types = [event["event_type"] for event in payload["events"]]
        self.assertEqual(event_types[:4], ["startup_doctor", "framework_context_loaded", "session_resume", "intent_route"])
        self.assertIn("benchmark_workflow_selected", event_types)
        self.assertEqual(payload["status"], "ok")
        context_event = payload["events"][1]
        self.assertEqual(context_event["data"]["capability_summary"]["chain_count"], 36)
        self.assertFalse(context_event["data"]["context_policy"]["load_full_docs_by_default"])

        cli_payload = run_agent("workflow-dry-run", "--text", "Add custom RPC method foo_getBalance to chain foochain")
        cli_events = [event["event_type"] for event in cli_payload["events"]]
        self.assertIn("onboarding_workflow_selected", cli_events)

    def test_adk_read_only_tool_wrappers_are_structured(self):
        tools = get_adk_tools(include_actions=False)
        tool_names = {tool.__name__ for tool in tools}
        self.assertIn("discover_environment", tool_names)
        self.assertIn("run_doctor", tool_names)
        self.assertIn("audit_dependencies", tool_names)
        self.assertIn("load_framework_context", tool_names)
        self.assertIn("load_execution_contract", tool_names)
        self.assertIn("load_framework_capabilities", tool_names)
        self.assertIn("list_supported_chains", tool_names)
        self.assertIn("knowledge_search", tool_names)
        self.assertNotIn("submit_benchmark_job", tool_names)

        root_tool_names = {tool.__name__ for tool in get_adk_tools(include_actions=True)}
        self.assertIn("run_smoke", root_tool_names)
        self.assertIn("run_fake_node_smoke_benchmark", root_tool_names)
        self.assertIn("submit_benchmark_job", root_tool_names)

        capabilities = adk_load_framework_capabilities()
        self.assertEqual(capabilities["status"], "ok")
        self.assertFalse(capabilities["requires_user_confirmation"])
        self.assertEqual(capabilities["data"]["chain_count"], 36)
        self.assertTrue(capabilities["next_actions"])

        context = adk_load_framework_context(language="zh")
        self.assertEqual(context["status"], "ok")
        self.assertEqual(context["data"]["capability_summary"]["chain_count"], 36)
        self.assertIn("runtime.env", [item["name"] for item in context["data"]["configuration_layers"]])
        self.assertTrue(context["data"]["authoritative_docs"])

        contract = adk_load_execution_contract(use_fake_node=False)
        self.assertEqual(contract["status"], "ok")
        self.assertIn("local_rpc_url", contract["data"]["selected_required_keys"])
        self.assertIn("preflight", contract["data"]["mandatory_gates"])

        chains = adk_list_supported_chains()
        self.assertEqual(chains["status"], "ok")
        self.assertIn("solana", chains["data"]["chains"])

        solana_methods = adk_list_rpc_methods("solana")
        self.assertEqual(solana_methods["status"], "ok")
        self.assertEqual(solana_methods["data"]["chain"], "solana")
        self.assertTrue(solana_methods["next_actions"])

    def test_framework_context_is_compact_and_tool_callable(self):
        context = load_framework_context(language="en")
        self.assertEqual(context["capability_summary"]["chain_count"], 36)
        self.assertFalse(context["context_policy"]["load_full_docs_by_default"])
        topics = {item["topic"] for item in context["authoritative_docs"]}
        self.assertIn("add_chain_or_rpc", topics)
        self.assertIn("framework_flow", topics)

        rendered = render_framework_context_for_prompt(language="zh")
        self.assertIn("AnyChain framework context", rendered)
        self.assertIn("36 chains", rendered)
        self.assertIn("runtime flow", rendered)

        payload = run_agent("tool-call", "--name", "load_framework_context", "--arguments", '{"language":"zh"}')
        self.assertEqual(payload["capability_summary"]["chain_count"], 36)
        self.assertTrue(payload["authoritative_docs"])

        contract_payload = run_agent("tool-call", "--name", "load_execution_contract", "--arguments", '{"use_fake_node":true}')
        self.assertIn("ledger_device", contract_payload["data"]["selected_required_keys"])
        self.assertNotIn("local_rpc_url", contract_payload["data"]["selected_required_keys"])

        missing = adk_list_rpc_methods("not-a-chain")
        self.assertEqual(missing["status"], "not_found")
        self.assertTrue(missing["warnings"])

    def test_adk_planning_and_action_tools_are_confirmation_gated(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prepared = adk_prepare_benchmark_run(
                source_prompt="Create a Solana fake-node smoke benchmark at 1 QPS",
                chain="solana",
                goal="smoke",
                rpc_mode="single",
                use_fake_node=True,
                qps_max=1,
                output_dir=str(tmp_path / "prepared"),
            )
            self.assertIn(prepared["status"], {"ok", "blocked"})
            self.assertTrue(Path(prepared["data"]["plan_file"]).is_file())
            self.assertTrue(Path(prepared["data"]["runbook_file"]).is_file())
            self.assertIn("inferred_values", prepared["data"])
            self.assertIn("missing_required", prepared["data"])
            self.assertIn("questions", prepared["data"])

            request = adk_draft_benchmark_request(
                source_prompt="Create a Solana fake-node smoke benchmark at 1 QPS",
                chain="solana",
                goal="smoke",
                rpc_mode="single",
                use_fake_node=True,
                qps_max=1,
            )
            self.assertEqual(request["status"], "ok")
            self.assertEqual(request["data"]["chain"], "solana")

            structured_request = adk_draft_benchmark_request(
                source_prompt="benchmark my node",
                chain="ethereum",
                goal="stress",
                rpc_mode="mixed",
                use_fake_node=False,
                deployment_type="kubernetes",
                cloud_provider="gcp",
                cloud_region="us-central1",
                cloud_zone="us-central1-a",
                machine_type="c3-standard-22",
                blockchain_process_names=["reth", "ethereum"],
                ledger_device="sdb",
                data_vol_type="hyperdisk-extreme",
                data_vol_size="2000",
                data_vol_max_iops="30000",
                data_vol_max_throughput="700",
                accounts_device="sdc",
                accounts_vol_type="hyperdisk-balanced",
                accounts_vol_size="500",
                accounts_vol_max_iops="10000",
                accounts_vol_max_throughput="300",
                network_interface="eth0",
                network_max_bandwidth_gbps="25",
                qps_initial=100,
                qps_max=1000,
                qps_step=100,
                duration_seconds=300,
                rpc_methods=["eth_blockNumber", "eth_getBalance"],
                mixed_weights={"eth_blockNumber": 70, "eth_getBalance": 30},
            )
            self.assertEqual(structured_request["data"]["chain"], "ethereum")
            self.assertEqual(structured_request["data"]["goal"], "stress")
            self.assertEqual(structured_request["data"]["rpc_mode"], "mixed")
            self.assertEqual(structured_request["data"]["deployment"]["type"], "kubernetes")
            self.assertEqual(structured_request["data"]["qps"]["max"], 1000)
            self.assertEqual(sum(item["weight"] for item in structured_request["data"]["mixed_weighted"]), 100)
            self.assertEqual(structured_request["data"]["cloud_region"], "us-central1")
            self.assertEqual(structured_request["data"]["blockchain_process_names"], ["reth", "ethereum"])
            real_node_request = adk_draft_benchmark_request(
                source_prompt="benchmark my real Solana node",
                chain="solana",
                goal="smoke",
                rpc_mode="single",
                use_fake_node=False,
                target_rpc_url="http://127.0.0.1:8899",
            )
            real_node_plan = adk_generate_benchmark_plan(real_node_request["data"])["data"]
            self.assertEqual(real_node_plan["execution"]["environment"]["LOCAL_RPC_URL"], "http://127.0.0.1:8899")
            self.assertNotIn("local_rpc_url", real_node_plan["required_inputs"])
            full_real_plan = adk_generate_benchmark_plan(structured_request["data"])["data"]
            materialized = full_real_plan["materialized_config"]
            self.assertEqual(materialized["CLOUD_REGION"], "us-central1")
            self.assertEqual(materialized["MACHINE_TYPE"], "c3-standard-22")
            self.assertEqual(materialized["BLOCKCHAIN_PROCESS_NAMES_STR"], "reth ethereum")
            self.assertEqual(materialized["DATA_VOL_TYPE"], "hyperdisk-extreme")
            self.assertEqual(materialized["DATA_VOL_SIZE"], "2000")
            self.assertEqual(materialized["ACCOUNTS_DEVICE"], "sdc")
            self.assertEqual(materialized["ACCOUNTS_VOL_MAX_IOPS"], "10000")
            self.assertEqual(materialized["NETWORK_INTERFACE"], "eth0")
            self.assertEqual(materialized["NETWORK_MAX_BANDWIDTH_GBPS"], "25")
            self.assertTrue(full_real_plan["chain_template_requirements"]["mixed_weighted"])
            self.assertIn("custom_rpc_extension_fields", full_real_plan["chain_template_requirements"])
            self.assertIn("param_spec", full_real_plan["chain_template_requirements"]["custom_rpc_extension_fields"])
            question_ids = {item["id"] for item in full_real_plan["required_questions"]}
            self.assertIn("rpc_workload_confirmation", question_ids)
            self.assertIn("custom_rpc_method_review", question_ids)
            self.assertIn("rpc_param_samples_confirmation", question_ids)
            self.assertIn("advanced_config_review", question_ids)
            disk_discovery = {
                "source": "test",
                "deployment": {"type": "vm"},
                "cloud": {"provider": "gcp", "confidence": 0.9},
                "disks": {
                    "candidates": [
                        {"name": "sda", "type": "disk", "size": "100G", "mountpoint": "/", "fstype": "ext4", "label": ""},
                        {"name": "sdb", "type": "disk", "size": "2T", "mountpoint": "/var/lib/solana/ledger", "fstype": "xfs", "label": "ledger"},
                        {"name": "sdc", "type": "disk", "size": "500G", "mountpoint": "/var/lib/solana/accounts", "fstype": "xfs", "label": "accounts"},
                    ],
                    "proposed_ledger_device": "sdb",
                    "proposed_accounts_device": "sdc",
                    "confidence": 0.9,
                },
                "dependencies": {"missing_required": []},
                "warnings": [],
            }
            disk_plan = adk_generate_benchmark_plan(structured_request["data"], discovery=disk_discovery)["data"]
            disk_question = next(
                item for item in disk_plan["required_questions"]
                if item["id"] == "disk_inventory_confirmation"
            )
            self.assertEqual(disk_question["proposed_ledger_device"], "sdb")
            self.assertEqual(disk_question["proposed_accounts_device"], "sdc")
            self.assertEqual({item["name"] for item in disk_question["candidates"]}, {"sda", "sdb", "sdc"})
            runtime_env = build_runtime_env(full_real_plan)
            self.assertEqual(runtime_env["DATA_VOL_TYPE"], "hyperdisk-extreme")
            self.assertEqual(runtime_env["ACCOUNTS_VOL_TYPE"], "hyperdisk-balanced")
            self.assertEqual(runtime_env["NETWORK_INTERFACE"], "eth0")
            self.assertEqual(runtime_env["BLOCKCHAIN_PROCESS_NAMES_STR"], "reth ethereum")

            plan_payload = adk_generate_benchmark_plan(request["data"])
            self.assertEqual(plan_payload["status"], "ok")
            plan = plan_payload["data"]
            self.assertEqual(plan["chain"], "solana")
            self.assertTrue(plan["use_fake_node"])

            preflight = adk_run_preflight(plan)
            self.assertEqual(preflight["status"], "ok")
            self.assertTrue(preflight["data"]["passed"])

            runbook_file = tmp_path / "runbook.md"
            runbook = adk_render_runbook(plan, output=str(runbook_file))
            self.assertEqual(runbook["status"], "ok")
            self.assertTrue(runbook_file.is_file())
            self.assertIn(str(runbook_file), runbook["evidence_paths"])

            plan_file = tmp_path / "plan.json"
            plan_file.write_text(json.dumps(plan), encoding="utf-8")
            smoke_without_approval = adk_run_smoke(str(plan_file), jobs_dir=str(tmp_path / "jobs"), approved=False)
            self.assertEqual(smoke_without_approval["status"], "needs_confirmation")
            self.assertTrue(smoke_without_approval["requires_user_confirmation"])

            smoke = adk_run_smoke(str(plan_file), jobs_dir=str(tmp_path / "jobs"), approved=True)
            self.assertEqual(smoke["status"], "ok")
            self.assertEqual(smoke["data"]["job"]["status"], "completed")
            self.assertTrue(smoke["evidence_paths"])

            fake_node_without_approval = adk_run_fake_node_smoke_benchmark(
                str(plan_file),
                jobs_dir=str(tmp_path / "jobs"),
                approved=False,
            )
            self.assertEqual(fake_node_without_approval["status"], "needs_confirmation")
            self.assertTrue(fake_node_without_approval["requires_user_confirmation"])
            isolated_plan = _fake_node_smoke_plan(plan_file, tmp_path / "jobs" / "fake_node_smoke")
            isolated_env = isolated_plan["execution"]["environment"]
            self.assertTrue(isolated_plan["use_fake_node"])
            self.assertEqual(isolated_plan["execution"]["runner_mode"], "foreground")
            self.assertIn("--fake-node", isolated_plan["execution"]["command"])
            self.assertIn(str(tmp_path / "jobs" / "fake_node_smoke"), isolated_env["BLOCKCHAIN_BENCHMARK_DATA_DIR"])
            self.assertIn(str(tmp_path / "jobs" / "fake_node_smoke"), isolated_env["MEMORY_SHARE_DIR"])
            self.assertNotIn("local_rpc_url", isolated_plan.get("required_inputs", []))
            self.assertNotIn("local_rpc_url", isolated_plan.get("configuration_checklist", {}).get("missing_blockers", []))

            real_without_approval = adk_submit_benchmark_job(str(plan_file), jobs_dir=str(tmp_path / "jobs"), approved=False)
            self.assertEqual(real_without_approval["status"], "needs_confirmation")
            self.assertTrue(real_without_approval["requires_user_confirmation"])

            dependency_install_without_approval = adk_install_dependencies(approved=False)
            self.assertEqual(dependency_install_without_approval["status"], "needs_confirmation")
            self.assertTrue(dependency_install_without_approval["requires_user_confirmation"])

            audit = adk_audit_dependencies()
            self.assertIn(audit["status"], {"ok", "needs_dependencies"})
            self.assertIn("--check", audit["data"]["benchmark"]["command"])
            self.assertIn("--check", audit["data"]["agent_runtime"]["command"])

    def test_adk_offline_eval_includes_router_schema_contract(self):
        payload = run_adk_offline_evals()
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["case_count"], payload["passed_count"])
        self.assertIn("structured router schema", payload["note"])

    def test_adk_auth_diagnostics_are_safe_and_actionable(self):
        old_env = os.environ.copy()
        try:
            for key in (
                "LLM_PROVIDER",
                "LLM_MODEL",
                "LLM_AUTH_MODE",
                "GEMINI_API_KEY",
                "GOOGLE_API_KEY",
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_CLOUD_LOCATION",
                "GOOGLE_SERVICE_ACCOUNT_EMAIL",
                "GOOGLE_APPLICATION_CREDENTIALS",
                "XDG_CONFIG_HOME",
            ):
                os.environ.pop(key, None)
            os.environ.update({
                "LLM_PROVIDER": "gemini",
                "LLM_MODEL": "gemini-3.1-pro",
                "LLM_AUTH_MODE": "google_adc",
                "GOOGLE_CLOUD_PROJECT": "example-project",
                "GOOGLE_CLOUD_LOCATION": "us-central1",
            })
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["XDG_CONFIG_HOME"] = tmp
                auth = adk_inspect_llm_auth()
            self.assertEqual(auth["status"], "ok")
            self.assertEqual(auth["data"]["llm"]["auth_mode"], "google_adc")
            self.assertIn("gcloud", auth["data"])
            self.assertIn("local_adc", auth["data"])
            self.assertFalse(auth["data"]["local_adc"]["well_known_file_exists"])
            self.assertTrue(any("gcloud auth application-default login" in action for action in auth["next_actions"]))

            os.environ["LLM_AUTH_MODE"] = "service_account_file"
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/nonexistent-service-account.json"
            auth = adk_inspect_llm_auth()
            self.assertEqual(auth["status"], "ok")
            self.assertTrue(auth["data"]["service_account_file"]["configured"])
            self.assertFalse(auth["data"]["service_account_file"]["file_exists"])
            serialized = json.dumps(auth).lower()
            self.assertNotIn("private_key", serialized)
            self.assertNotIn("access_token", serialized)
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_adk_startup_state_recovers_latest_file_backed_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            request = adk_draft_benchmark_request(
                source_prompt="Create a Solana fake-node smoke benchmark at 1 QPS",
                chain="solana",
                goal="smoke",
                rpc_mode="single",
                use_fake_node=True,
                qps_max=1,
            )
            plan = adk_generate_benchmark_plan(request["data"])["data"]
            plan_file = tmp_path / "plan.json"
            plan_file.write_text(json.dumps(plan), encoding="utf-8")
            jobs_dir = tmp_path / "jobs"
            smoke = adk_run_smoke(str(plan_file), jobs_dir=str(jobs_dir), approved=True)
            job = smoke["data"]["job"]

            state = adk_load_startup_state(jobs_dir=jobs_dir)
            self.assertTrue(state["resume_available"])
            self.assertEqual(state["latest_job"]["job_id"], job["job_id"])
            self.assertIn("analyze latest job", state["next_actions"])

            preserved = preserved_state_for_adk(state)
            self.assertEqual(preserved["job_id"], job["job_id"])
            self.assertEqual(preserved["job_status"], "completed")
            self.assertTrue(preserved["runtime_env_file"])
            self.assertTrue(preserved["artifact_index"])

    def test_adk_enterprise_manifest_describes_tool_boundaries(self):
        manifest = enterprise_integration_manifest()
        self.assertEqual(manifest["status"], "ok")
        self.assertIn("discover_environment", manifest["read_only_tools"])
        self.assertIn("generate_benchmark_plan", manifest["planning_tools"])
        self.assertIn("submit_benchmark_job", manifest["confirmation_gated_tools"])
        self.assertIn("real benchmark launch must follow preflight and smoke", manifest["requirements"])
        self.assertIn("POST /search", manifest["knowledge_base"]["optional_http_contract"])

    def test_adk_offline_eval_runner_passes(self):
        payload = run_adk_offline_evals()
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["case_count"], payload["passed_count"])
        cli_payload = run_agent("adk-eval")
        self.assertEqual(cli_payload["status"], "passed")

    def test_artifact_flow_is_deterministic(self):
        from analyzers.artifact_qa import answer_artifact_question

        answer = answer_artifact_question("Why are charts empty?", artifact_index=None)
        self.assertIn("No artifact evidence", answer["answer"])

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
