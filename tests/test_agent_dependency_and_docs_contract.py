"""Contracts for the R42 Agent dependency and long-lived documentation split."""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements-adk.txt"
INSTALLER = ROOT / "scripts" / "install_agent_deps.sh"
HANDOFF = ROOT / ".agent" / "task-docs" / "2026-07-10-agent-handoff-for-external-ai.md"

LONG_LIVED_DOCS = (
    ROOT / "README.md",
    ROOT / "README_ZH.md",
    ROOT / "agent" / "README.md",
    ROOT / "docs" / "en" / "adk-agent-architecture.md",
    ROOT / "docs" / "zh" / "adk-agent-architecture.md",
    ROOT / "docs" / "en" / "agent-cli-verification-guide.md",
    ROOT / "docs" / "en" / "anychain-agent-ai-work-gate.md",
    ROOT / "docs" / "zh" / "anychain-agent-ai-work-gate.md",
    ROOT / "docs" / "en" / "secondary-development-guide.md",
    ROOT / "docs" / "zh" / "secondary-development-guide.md",
    ROOT / "tests" / "agent_live" / "README.md",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in _text(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(("-r", "--")):
            continue
        names.add(re.split(r"[<>=!~\[;\s]", line, maxsplit=1)[0].lower())
    return names


class AgentDependencyContractTests(unittest.TestCase):
    def test_core_requirements_do_not_install_google_adk(self) -> None:
        names = _requirement_names(REQUIREMENTS)
        self.assertNotIn("google-adk", names)
        self.assertTrue(
            {"langgraph", "langgraph-checkpoint-sqlite", "openai", "prompt-toolkit"}.issubset(names)
        )

    def test_installer_has_explicit_optional_search_extra_and_compatibility_aliases(self) -> None:
        script = _text(INSTALLER)
        self.assertIn("--with-google-search", script)
        self.assertIn("--agent-venv|--adk-venv", script)
        self.assertIn("--skip-agent-runtime", script)
        self.assertIn("pip install google-adk", script)
        self.assertNotIn("import google.adk\nimport langgraph", script)
        subprocess.run(["bash", "-n", str(INSTALLER)], cwd=ROOT, check=True)

    def test_core_runtime_probe_does_not_import_google_adk(self) -> None:
        script = _text(INSTALLER)
        core_probe = script.split("agent_runtime_ready()", 1)[1].split("google_search_ready()", 1)[0]
        self.assertIn("import langgraph", core_probe)
        self.assertNotIn("google.adk", core_probe)


class AgentDocumentationContractTests(unittest.TestCase):
    def test_entry_docs_describe_default_core_and_explicit_search_extra(self) -> None:
        expectations = {
            "README.md": ("default install does not", "--with-google-search"),
            "README_ZH.md": ("默认安装不包含、也不依赖 Google ADK", "--with-google-search"),
            "agent/README.md": ("default installs the core LangGraph", "--with-google-search"),
        }
        for relative, terms in expectations.items():
            content = _text(ROOT / relative)
            for term in terms:
                self.assertIn(term, content, f"{relative} is missing {term}")

    def test_architecture_mirrors_dependency_custom_rpc_sync_and_coverage_boundaries(self) -> None:
        en = _text(ROOT / "docs/en/adk-agent-architecture.md")
        zh = _text(ROOT / "docs/zh/adk-agent-architecture.md")
        shared_contract_terms = (
            "--with-google-search",
            "single_replace",
            "mixed_replace",
            "mixed_add",
            "sync-observe",
            "observed-pass",
            "externally-blocked",
            "Docker/Linux",
        )
        for term in shared_contract_terms:
            self.assertIn(term, en)
            self.assertIn(term, zh)

    def test_architecture_mirrors_provider_and_semantic_planner_boundaries(self) -> None:
        for relative in (
            "agent/README.md",
            "docs/en/adk-agent-architecture.md",
            "docs/zh/adk-agent-architecture.md",
        ):
            content = _text(ROOT / relative)
            for term in (
                "LLMProvider",
                "generateContent",
                "Anthropic Messages API",
                "rawPredict",
                "hierarchical_planner.py",
                "Stage A",
                "Stage B",
            ):
                self.assertIn(term, content, f"{relative} is missing {term}")

    def test_cli_guide_rejects_fake_node_only_workflow_coverage(self) -> None:
        guide = _text(ROOT / "docs/en/agent-cli-verification-guide.md")
        self.assertNotIn("Fake-node validation is enough for Agent CLI workflow coverage", guide)
        self.assertIn("Fake-node validates only the fake-node closed loop", guide)
        self.assertIn("Acceptance runs are Docker/Linux-only", guide)

    def test_work_gates_mirror_optional_dependency_and_observed_coverage(self) -> None:
        for relative in (
            "docs/en/anychain-agent-ai-work-gate.md",
            "docs/zh/anychain-agent-ai-work-gate.md",
        ):
            content = _text(ROOT / relative)
            for term in (
                "--with-google-search",
                "positional",
                "object params",
                "observed execution",
                "externally-blocked",
                "fake-node",
                "sync-observe",
            ):
                self.assertIn(term, content, f"{relative} is missing {term}")

    def test_secondary_guides_mirror_parameter_and_job_local_semantics(self) -> None:
        for relative in (
            "docs/en/secondary-development-guide.md",
            "docs/zh/secondary-development-guide.md",
        ):
            content = _text(ROOT / relative)
            for term in (
                "--with-google-search",
                "single_replace",
                "mixed_replace",
                "mixed_add",
                "job-local",
                "object key",
            ):
                self.assertIn(term, content, f"{relative} is missing {term}")

    def test_live_readme_separates_generated_and_observed_docker_evidence(self) -> None:
        content = _text(ROOT / "tests/agent_live/README.md")
        for term in (
            "Docker-only",
            "cataloged/generated",
            "observed-pass",
            "observed-fail",
            "not-run",
            "externally-blocked",
            "hashed job artifacts",
        ):
            self.assertIn(term, content)

    def test_live_docs_name_the_single_acceptance_authority_and_phase_boundary(self) -> None:
        for path in (
            ROOT / "agent" / "README.md",
            ROOT / "tests" / "agent_live" / "README.md",
            HANDOFF,
        ):
            content = _text(path)
            for term in (
                "run_product_acceptance.py",
                "Phase 6",
                "G2",
                "Phase 8",
                "G3-G6",
                "subordinate evidence providers",
            ):
                self.assertIn(term, content, f"{path.relative_to(ROOT)} is missing {term}")

    def test_relative_markdown_links_exist(self) -> None:
        link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
        missing: list[str] = []
        for doc in LONG_LIVED_DOCS:
            for target in link_pattern.findall(_text(doc)):
                clean_target = target.strip().split("#", 1)[0]
                if not clean_target or re.match(r"^[a-z][a-z0-9+.-]*:", clean_target, re.I):
                    continue
                clean_target = clean_target.strip("<>").replace("%20", " ")
                if not (doc.parent / clean_target).resolve().exists():
                    missing.append(f"{doc.relative_to(ROOT)} -> {target}")
        self.assertEqual([], missing, "missing local Markdown links:\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
