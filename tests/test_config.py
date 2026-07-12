"""CPU tests validating configs/model.yaml and env/*.env (Track A).

README §7: test_config validates model.yaml and the env file(s) parse and
contain the required keys. The env file uses `: "${VAR:=default}"` form (so
environment overrides win), so the parser understands that shape.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

_PLAIN = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)=(.*)$")
_COLON = re.compile(r'^\s*:\s*"\$\{([A-Z_][A-Z0-9_]*):=(.*)\}"\s*$')


def parse_env_defaults(path: Path) -> dict[str, str]:
    """Extract VAR -> default from both `KEY=val` and `: "${KEY:=val}"` lines."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        m = _COLON.match(line) or _PLAIN.match(line)
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def test_model_yaml_parses_and_has_required_keys():
    cfg = yaml.safe_load((REPO / "configs/model.yaml").read_text())
    for key in ("model", "dtype", "max_model_len", "seed", "sampling", "workload"):
        assert key in cfg, f"model.yaml missing {key}"
    for key in ("temperature", "top_p", "max_tokens"):
        assert key in cfg["sampling"], f"sampling missing {key}"
    assert cfg["dtype"] == "float16"
    assert isinstance(cfg["max_model_len"], int)
    assert cfg["workload"]["request_rates"] == [5, 10, 20, 30, 40, 50, 60]


def test_b0_env_parses_and_has_required_keys():
    env = parse_env_defaults(REPO / "env/b0_vanilla.env")
    for key in ("MODEL", "DTYPE", "MAX_MODEL_LEN", "GPU_MEM_UTIL", "PORT"):
        assert key in env, f"b0_vanilla.env missing {key}"
    assert env["MODEL"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert env["DTYPE"] == "float16"
    assert env["PORT"].isdigit()
    assert 0.0 < float(env["GPU_MEM_UTIL"]) <= 1.0


def test_env_and_yaml_agree_on_model_and_dtype():
    cfg = yaml.safe_load((REPO / "configs/model.yaml").read_text())
    env = parse_env_defaults(REPO / "env/b0_vanilla.env")
    assert env["MODEL"] == cfg["model"]
    assert env["DTYPE"] == cfg["dtype"]


@pytest.mark.parametrize("env_file", ["env/b0_vanilla.env"])
def test_env_files_have_no_trailing_whitespace_keys(env_file):
    # every parsed default should be non-empty
    env = parse_env_defaults(REPO / env_file)
    assert all(v != "" for v in env.values())
