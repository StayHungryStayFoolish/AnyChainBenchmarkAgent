import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
AGENT = REPO / "agent" / "cli.py"
sys.path.insert(0, str(REPO / "agent"))

from discovery.environment import discover_environment  # noqa: E402
from knowledge.framework_capabilities import load_framework_capabilities  # noqa: E402
from llm.config import load_llm_config  # noqa: E402
from llm.google_auth import credential_plan  # noqa: E402
from llm.providers import provider_from_config  # noqa: E402
from runners.guardrails import validate_execution_plan  # noqa: E402
from wizard import run_wizard  # noqa: E402


def run_agent(*args):
    completed = subprocess.run(
        [sys.executable, str(AGENT), *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
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
            self.assertEqual(new_chain_gap["intent"], "framework_question")
            self.assertIn("chain_template", {gap["type"] for gap in new_chain_gap["gap_analysis"]["gaps"]})

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

    def test_llm_provider_config_supports_vertex_service_accounts(self):
        config = load_llm_config({
            "LLM_PROVIDER": "vertex_gemini_openai",
            "LLM_MODEL": "gemini-2.5-pro",
            "GOOGLE_AUTH_MODE": "service_account_impersonation",
            "GOOGLE_CLOUD_PROJECT": "example-project",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
            "GOOGLE_SERVICE_ACCOUNT_EMAIL": "benchmark-agent@example-project.iam.gserviceaccount.com",
        })
        self.assertEqual(config.validate(), [])
        plan = credential_plan(config).safe_dict()
        self.assertEqual(plan["mode"], "service_account_impersonation")
        self.assertEqual(plan["service_account_email"], "benchmark-agent@example-project.iam.gserviceaccount.com")
        self.assertEqual(provider_from_config(config).__class__.__name__, "VertexGeminiOpenAIProvider")

        partner_model_config = load_llm_config({
            "LLM_PROVIDER": "vertex_claude",
            "LLM_MODEL": "claude-3-7-sonnet@20250219",
            "GOOGLE_AUTH_MODE": "attached_service_account",
            "GOOGLE_CLOUD_PROJECT": "example-project",
            "GOOGLE_CLOUD_LOCATION": "us-east5",
        })
        self.assertEqual(partner_model_config.validate(), [])
        self.assertEqual(provider_from_config(partner_model_config).__class__.__name__, "VertexClaudeProvider")

        missing_impersonation = load_llm_config({
            "LLM_PROVIDER": "vertex_gemini_openai",
            "GOOGLE_AUTH_MODE": "service_account_impersonation",
            "GOOGLE_CLOUD_PROJECT": "example-project",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
        })
        self.assertIn("GOOGLE_SERVICE_ACCOUNT_EMAIL is required", "; ".join(missing_impersonation.validate()))

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
