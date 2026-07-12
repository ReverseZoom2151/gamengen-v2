from pathlib import Path
import re


ROOT = Path(__file__).parents[1]


def test_quality_workflow_uses_immutable_actions_and_pinned_linter():
    workflow = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    assert "permissions:\n  contents: read" in workflow
    assert "ruff==0.15.21" in workflow
    assert re.search(r"actions/checkout@[0-9a-f]{40}", workflow)
    assert re.search(r"actions/setup-python@[0-9a-f]{40}", workflow)


def test_security_workflow_has_secret_and_dependency_gates():
    workflow = (ROOT / ".github/workflows/security.yml").read_text(encoding="utf-8")
    assert re.search(r"gitleaks/gitleaks-action@[0-9a-f]{40}", workflow)
    assert re.search(r"actions/dependency-review-action@[0-9a-f]{40}", workflow)
    assert "fetch-depth: 0" in workflow
