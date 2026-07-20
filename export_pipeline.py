from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_NOTEBOOKS = [
    "dataset_description.ipynb",
    "descriptive_statistics.ipynb",
    "hypothesis_testing.ipynb",
]


def run_cmd(args: list[str], *, cwd: Path | None = None, input_text: str | None = None) -> str:
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        input=input_text,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(args)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def find_pandoc() -> str:
    pandoc = shutil.which("pandoc")
    if pandoc:
        return pandoc

    local_appdata = Path.home() / "AppData" / "Local" / "Pandoc" / "pandoc.exe"
    if local_appdata.exists():
        return str(local_appdata)

    raise RuntimeError("Pandoc is not found. Please install pandoc or add it to PATH.")


def notebook_to_markdown(python_exe: str, notebook: Path, md_dir: Path) -> Path:
    run_cmd(
        [
            python_exe,
            "-m",
            "nbconvert",
            "--to",
            "markdown",
            str(notebook),
            "--output-dir",
            str(md_dir),
        ],
        cwd=ROOT,
    )
    return md_dir / f"{notebook.stem}.md"


def _html_table_to_pipe_table(html_table: str, pandoc_bin: str) -> str:
    converted = run_cmd(
        [
            pandoc_bin,
            "--from=html",
            "--to=gfm+pipe_tables",
            "--wrap=none",
        ],
        input_text=html_table,
    ).strip()
    return converted


def normalize_markdown_tables(md_file: Path, pandoc_bin: str) -> None:
    text = md_file.read_text(encoding="utf-8")

    # Remove notebook-generated inline CSS blocks.
    text = re.sub(r"<style scoped>[\s\S]*?</style>\s*", "", text, flags=re.IGNORECASE)

    # Unwrap <div> wrappers around HTML table output from DataFrame display.
    text = re.sub(r"<div>\s*(<table[\s\S]*?</table>)\s*</div>", r"\1", text, flags=re.IGNORECASE)

    table_pattern = re.compile(r"<table[\s\S]*?</table>", flags=re.IGNORECASE)

    def replace_table(match: re.Match[str]) -> str:
        html_table = match.group(0)
        md_table = _html_table_to_pipe_table(html_table, pandoc_bin)
        return f"\n\n{md_table}\n\n"

    text = table_pattern.sub(replace_table, text)

    # Remove leftover empty div tags if present.
    text = re.sub(r"</?div>\s*", "", text, flags=re.IGNORECASE)

    md_file.write_text(text, encoding="utf-8")


def markdown_to_docx(pandoc_bin: str, md_file: Path, md_dir: Path, out_docx: Path) -> None:
    run_cmd(
        [
            pandoc_bin,
            str(md_file),
            "--from=markdown+tex_math_dollars+pipe_tables+raw_html",
            f"--resource-path={md_dir}",
            "--standalone",
            "-o",
            str(out_docx),
        ],
        cwd=ROOT,
    )


def export_pipeline(python_exe: str, notebooks: list[str], md_dir: Path, docx_dir: Path) -> None:
    pandoc_bin = find_pandoc()
    md_dir.mkdir(parents=True, exist_ok=True)
    docx_dir.mkdir(parents=True, exist_ok=True)

    for notebook_name in notebooks:
        notebook = ROOT / notebook_name
        if not notebook.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook}")

        md_file = notebook_to_markdown(python_exe, notebook, md_dir)
        normalize_markdown_tables(md_file, pandoc_bin)
        out_docx = docx_dir / f"{notebook.stem}.docx"
        markdown_to_docx(pandoc_bin, md_file, md_dir, out_docx)
        print(f"Exported: {out_docx}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export notebooks to DOCX via Markdown with DataFrame table normalization."
    )
    parser.add_argument(
        "--python",
        required=True,
        help="Python executable used to run nbconvert (e.g., .venv/Scripts/python.exe).",
    )
    parser.add_argument(
        "--notebooks",
        nargs="*",
        default=DEFAULT_NOTEBOOKS,
        help="Notebook paths relative to repository root.",
    )
    parser.add_argument(
        "--markdown-dir",
        default="export/markdown",
        help="Directory to store generated markdown files.",
    )
    parser.add_argument(
        "--docx-dir",
        default="export/docx",
        help="Directory to store generated docx files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_pipeline(
        python_exe=args.python,
        notebooks=args.notebooks,
        md_dir=ROOT / args.markdown_dir,
        docx_dir=ROOT / args.docx_dir,
    )


if __name__ == "__main__":
    main()
