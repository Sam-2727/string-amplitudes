"""Render the published calculation notebooks into the documentation build."""

from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from nbconvert.writers import FilesWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "_build" / "html" / "notebooks"
PUBLISHED_NOTEBOOKS = (
    (
        PROJECT_ROOT / "tutorials" / "ribbon_graph_generation.ipynb",
        "Ribbon graph generation",
    ),
    (
        PROJECT_ROOT / "tutorials" / "compact_boson_partition_function.ipynb",
        "Compact boson partition function",
    ),
)


def main() -> None:
    """Render the explicitly published notebooks without executing them."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    writer = FilesWriter(build_directory=str(OUTPUT_DIRECTORY))

    for notebook_path, title in PUBLISHED_NOTEBOOKS:
        notebook = nbformat.read(notebook_path, as_version=4)
        body, resources = exporter.from_notebook_node(
            notebook,
            resources={
                "metadata": {
                    "name": title,
                    "path": str(notebook_path.parent),
                }
            },
        )
        writer.write(body, resources, notebook_name=notebook_path.stem)


if __name__ == "__main__":
    main()
