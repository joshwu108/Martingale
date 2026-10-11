"""
CI-enforced import isolation:
  - checker/ imports nothing from martingale (src/martingale/).
  - the verl adapter imports nothing from the TRL adapter; what they share lives in
    martingale.integrations._common.
"""
import ast
import pathlib
import sys

VERL_ADAPTER = pathlib.Path("src/martingale/integrations/verl.py")
TRL_ADAPTER = "martingale.integrations.trl"


def get_import_names(node, package=""):
    """Return module names from an import node: the module and, for `from m import x`, `m.x` (x may be
    a submodule). Relative imports resolve against `package`."""
    if isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        if node.level:
            base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
            mod = f"{base}.{mod}" if mod else base
        return [mod] + [f"{mod}.{a.name}" for a in node.names if mod]
    elif isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return []


def imports_of(source, package=""):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield from get_import_names(node, package)


def violations(root="."):
    root = pathlib.Path(root)
    bad = []
    for filepath in (root / "checker").rglob("*.py"):
        for name in imports_of(filepath.read_text()):
            if name and name.startswith("martingale"):
                bad.append((filepath, name))
    verl = root / VERL_ADAPTER
    if verl.exists():
        for name in imports_of(verl.read_text(), package="martingale.integrations"):
            if name == TRL_ADAPTER or name.startswith(TRL_ADAPTER + "."):
                bad.append((verl, name))
    return bad


if __name__ == "__main__":
    bad = violations()
    for filepath, module in bad:
        print(f"FAIL {filepath}: imports {module}")
    sys.exit(1 if bad else 0)
