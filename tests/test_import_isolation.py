"""scripts/check_imports.py: checker/ imports nothing from martingale, verl.py nothing from trl.py."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    spec = importlib.util.spec_from_file_location("check_imports", ROOT / "scripts" / "check_imports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_imports = _load_script()


def test_repository_has_no_violations():
    assert check_imports.violations(ROOT) == []


@pytest.mark.parametrize("line", [
    "from martingale.integrations.trl import _prefix_lengths",
    "from martingale.integrations import trl",
    "import martingale.integrations.trl as t",
    "from .trl import pack_row_id",
    "from . import trl",
])
def test_any_spelling_of_a_trl_import_in_the_verl_adapter_is_flagged(tmp_path, line):
    verl = tmp_path / check_imports.VERL_ADAPTER
    verl.parent.mkdir(parents=True)
    verl.write_text(f"{line}\nfrom martingale.integrations._common import pack_row_id\n")
    (tmp_path / "checker").mkdir()
    names = [name for _path, name in check_imports.violations(tmp_path)]
    assert names and all(n.startswith(check_imports.TRL_ADAPTER) for n in names)


def test_shared_and_callback_imports_are_allowed(tmp_path):
    verl = tmp_path / check_imports.VERL_ADAPTER
    verl.parent.mkdir(parents=True)
    verl.write_text("from martingale.integrations._common import pack_row_id\n"
                    "from martingale.integrations._trl_callback import MartingaleMonitor\n")
    (tmp_path / "checker").mkdir()
    assert check_imports.violations(tmp_path) == []


def test_a_checker_import_of_martingale_is_flagged(tmp_path):
    (tmp_path / "checker").mkdir()
    (tmp_path / "checker" / "x.py").write_text("from martingale.record import Recorder\n")
    assert [name for _path, name in check_imports.violations(tmp_path)] == ["martingale.record",
                                                                             "martingale.record.Recorder"]


def test_importing_the_verl_adapter_does_not_load_the_trl_adapter():
    code = ("import sys, martingale.integrations.verl\n"
            "sys.exit(1 if 'martingale.integrations.trl' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0
