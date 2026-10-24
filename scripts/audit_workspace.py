"""Audit provenance and package boundaries without requiring original sources to run the app."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "cashflow_web": ROOT / "apps/web",
    "cashflow": ROOT / "apps/web/cashflow",
    "cashflow_app": ROOT / "packages/integration/cashflow_app",
    "forecasting": ROOT / "packages/analytics/forecasting",
    "transaction_classifier": ROOT / "packages/analytics/transaction_classifier",
    "cashflow_integrated": ROOT / "packages/risk/cashflow_integrated",
}


def imports(node):
    if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "__package__":
        children = node.body  # Standalone-script fallback imports do not run in the installed package.
    else:
        children = ast.iter_child_nodes(node)
    for child in children:
        if isinstance(child, ast.Import):
            yield from (alias.name.split(".")[0] for alias in child.names)
        elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
            yield child.module.split(".")[0]
        yield from imports(child)


def main():
    edges = {name: set() for name in PACKAGES}
    for name, path in PACKAGES.items():
        for source in path.glob("*.py"):
            if source.name.startswith("test_"):
                continue
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            edges[name].update(dependency for dependency in imports(tree)
                               if dependency in PACKAGES and dependency != name)

    def visit(name, ancestors):
        if name in ancestors:
            raise RuntimeError(f"Package import cycle: {ancestors + [name]}")
        for dependency in edges[name]:
            visit(dependency, ancestors + [name])
    for name in edges:
        visit(name, [])
    manifest = json.loads((ROOT / "source-manifest.json").read_text(encoding="utf-8"))
    unchanged = 0
    missing_sources = []
    changed_copies = []
    for name, repo in manifest.items():
        source_root = Path(repo["source"])
        if not source_root.is_dir():
            missing_sources.append(name)
        for relative, expected in repo["sha256"].items():
            if source_root.is_dir():
                actual = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError(f"Original source changed: {name}/{relative}")
                unchanged += 1
            target = ROOT / repo["destination"] / relative
            if not target.is_file():
                raise RuntimeError(f"Copied source missing: {target}")
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                changed_copies.append(target.relative_to(ROOT).as_posix())
    result = {"original_files_unchanged": unchanged, "sources_not_available": missing_sources,
              "modified_copied_files": changed_copies,
              "package_import_graph": {name: sorted(dependencies) for name, dependencies in edges.items()},
              "package_cycles": False}
    output = ROOT / "artifacts/verification/source-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
