"""setup_guide 后端接入引导引擎测试。

覆盖纯函数（satisfies 版本比较、diagnose 诊断）与引导引擎的交互分支
（注入 input_ / run 模拟「回车即继续」）。
"""

import sys

import pytest

from quonic.backends import setup_guide
from quonic.backends.setup_guide import diagnose, ensure_ready, guided_setup, satisfies

# ---------------------------------------------------------------------------
# satisfies：版本约束比较（纯函数）
# ---------------------------------------------------------------------------

def test_satisfies_lt():
    assert satisfies("2.3.1", "<2.4.0")
    assert not satisfies("2.5.0", "<2.4.0")
    assert not satisfies("2.4.0", "<2.4.0")  # 边界：等于不满足 <


def test_satisfies_gte_eq():
    assert satisfies("1.0", ">=1.0")
    assert not satisfies("0.9", ">=1.0")
    assert satisfies("2.0", "==2.0")


def test_satisfies_numeric_not_lexicographic():
    # 数字比较而非字符串比较："2.10" 应 >= "2.4"（字符串比较会得出相反结论）
    assert not satisfies("2.10", "<2.4")
    assert satisfies("2.4", "<2.10")


def test_satisfies_bad_constraint():
    with pytest.raises(ValueError):
        satisfies("1.0", "noop")


# ---------------------------------------------------------------------------
# diagnose：诊断（纯函数）
# ---------------------------------------------------------------------------

def _fake_setup(**overrides):
    setup = {
        "name": "Test",
        "sdk": {
            "package": "definitely_not_installed_pkg_xyz",
            "pip": "definitely-not-installed-pkg-xyz",
            "install": "definitely-not-installed-pkg-xyz",
        },
        "auth": {"token_file": "/nonexistent/token.json"},
    }
    setup.update(overrides)
    return setup


def test_diagnose_all_ready(tmp_path):
    token = tmp_path / "token.json"
    token.write_text("{}")
    setup = _fake_setup(
        sdk={"package": "os", "pip": "os", "install": "os"},  # os 一定可导入
        auth={"token_file": str(token)},
    )
    d = diagnose(setup)
    assert d.ready
    assert d.sdk_installed
    assert d.auth_ready
    assert d.conflicts == []


def test_diagnose_missing_sdk_and_auth():
    d = diagnose(_fake_setup())
    assert not d.sdk_installed
    assert not d.auth_ready
    assert not d.ready


def test_diagnose_conflict(monkeypatch, tmp_path):
    token = tmp_path / "token.json"
    token.write_text("{}")
    setup = _fake_setup(
        sdk={"package": "os", "pip": "os", "install": "os"},
        auth={"token_file": str(token)},
        conflicts=[{"package": "somepkg", "constraint": "<2.4.0"}],
    )
    monkeypatch.setattr(setup_guide, "_installed_version", lambda pkg: "2.5.0")
    d = diagnose(setup)
    assert d.sdk_installed
    assert d.auth_ready
    assert len(d.conflicts) == 1
    assert d.conflicts[0]["installed"] == "2.5.0"
    assert not d.ready


# ---------------------------------------------------------------------------
# ensure_ready：非交互环境抛异常
# ---------------------------------------------------------------------------

def test_ensure_ready_raises_when_not_interactive(monkeypatch):
    monkeypatch.setattr(setup_guide, "_is_interactive", lambda: False)
    with pytest.raises(ImportError, match="definitely-not-installed-pkg-xyz"):
        ensure_ready(_fake_setup())


def test_ensure_ready_passes_when_ready(monkeypatch):
    monkeypatch.setattr(setup_guide, "_module_available", lambda pkg: True)
    monkeypatch.setattr(setup_guide, "_installed_version", lambda pkg: None)
    monkeypatch.setattr(setup_guide, "_auth_ready", lambda auth: True)
    # 不抛异常即通过
    ensure_ready(_fake_setup())


# ---------------------------------------------------------------------------
# guided_setup：交互分支（注入 input_ / run）
# ---------------------------------------------------------------------------

def test_guided_setup_installs_sdk_on_enter(monkeypatch):
    setup = _fake_setup()
    monkeypatch.setattr(setup_guide, "_module_available", lambda pkg: False)
    monkeypatch.setattr(setup_guide, "_conflicts", lambda s: [])
    monkeypatch.setattr(setup_guide, "_auth_ready", lambda a: True)

    installed = []

    def fake_run(cmd):
        installed.append(cmd)

    def fake_input(prompt=""):
        return ""  # 回车 = 是

    guided_setup(setup, input_=fake_input, run=fake_run)

    assert installed == [[sys.executable, "-m", "pip", "install", setup["sdk"]["install"]]]


def test_guided_setup_skips_sdk_on_n(monkeypatch):
    setup = _fake_setup()
    monkeypatch.setattr(setup_guide, "_module_available", lambda pkg: False)
    monkeypatch.setattr(setup_guide, "_conflicts", lambda s: [])
    monkeypatch.setattr(setup_guide, "_auth_ready", lambda a: True)

    installed = []

    def fake_run(cmd):
        installed.append(cmd)

    def fake_input(prompt=""):
        return "n"  # 跳过

    guided_setup(setup, input_=fake_input, run=fake_run)
    assert installed == []


# ---------------------------------------------------------------------------
# _handle_conflict：默认回退到 venv 方案（不污染主环境）
# ---------------------------------------------------------------------------

def test_conflict_default_is_venv(monkeypatch):
    c = {"package": "qiskit", "constraint": "<2.4.0"}
    calls = []
    monkeypatch.setattr(setup_guide, "_print_venv_guide", lambda: calls.append("venv"))
    monkeypatch.setattr(
        setup_guide, "_run_pip", lambda run, target: calls.append(("pip", target))
    )

    setup_guide._handle_conflict(c, run=lambda *a: None, input_=lambda p="": "")
    # 回车默认 = 打印 venv 引导，不执行 pip 降级
    assert calls == ["venv"]


def test_conflict_option2_downgrades(monkeypatch):
    c = {"package": "qiskit", "constraint": "<2.4.0"}
    calls = []
    monkeypatch.setattr(setup_guide, "_print_venv_guide", lambda: calls.append("venv"))
    monkeypatch.setattr(
        setup_guide, "_run_pip", lambda run, target: calls.append(("pip", target))
    )

    setup_guide._handle_conflict(c, run=lambda *a: None, input_=lambda p="": "2")
    # 显式选 2 = 降级
    assert calls == [("pip", "qiskit<2.4.0")]
