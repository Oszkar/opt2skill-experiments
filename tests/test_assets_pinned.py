"""scripts/fetch_assets.sh and configs/g1_reconcile.yaml pin the same three commits; keep them
in sync by testing it rather than by hoping a future editor updates both files together."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_fetch_assets_commits_match_config():
    cfg = yaml.safe_load((ROOT / "configs" / "g1_reconcile.yaml").read_text(encoding="utf-8"))
    script = (ROOT / "scripts" / "fetch_assets.sh").read_text(encoding="utf-8")

    script_commits: dict[str, str] = {}
    for line in script.splitlines():
        m = re.match(r"fetch\s+(\S+)\s+(\S+)\s+([0-9a-f]{40})\s*$", line.strip())
        if m:
            repo, _subdir, commit = m.groups()
            script_commits[repo] = commit
    assert len(script_commits) == 3, f"expected 3 pinned fetch calls, found {script_commits}"

    config_repos = {k: v for k, v in cfg["assets"].items() if isinstance(v, dict) and "repo" in v}
    assert len(config_repos) == 3
    for key, entry in config_repos.items():
        assert entry["repo"] in script_commits, f"{key}: {entry['repo']} not pinned in scripts/fetch_assets.sh"
        assert script_commits[entry["repo"]] == entry["commit"], (
            f"{key}: fetch_assets.sh pins {script_commits[entry['repo']]} but the config pins {entry['commit']}"
        )
