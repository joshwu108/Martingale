"""
Check that checker/ imports nothing from martingale (src/martingale/).
CI-enforced import isolation.
"""
import ast
import pathlib
import sys


def get_import_names(node):
    """Return module/package names from an import node."""
    if isinstance(node, ast.ImportFrom):
        return [node.module or ""]
    elif isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return []


bad = []
for filepath in pathlib.Path("checker").rglob("*.py"):
    tree = ast.parse(filepath.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in get_import_names(node):
                if name and name.startswith("martingale"):
                    bad.append((filepath, name))

for filepath, module in bad:
    print(f"FAIL {filepath}: imports {module}")

sys.exit(1 if bad else 0)
