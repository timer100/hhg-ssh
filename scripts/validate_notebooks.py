"""Execute full tutorials in a temporary directory, preserving saved outputs."""
import argparse
import shutil
import tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.validation' / 'notebooks')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ('Tutorial_Static_Analysis.ipynb', 'Tutorial_SSH.ipynb'):
        with tempfile.TemporaryDirectory(prefix='hhg-notebook-') as directory:
            work = Path(directory)
            (work / 'Images').mkdir()
            shutil.copytree(ROOT / 'hhg', work / 'hhg',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            notebook = nbformat.read(ROOT / name, as_version=4)
            client = NotebookClient(notebook, timeout=900, kernel_name='python3',
                                    resources={'metadata': {'path': str(work)}},
                                    allow_errors=False)
            client.execute()
            nbformat.write(notebook, args.output_dir / name)
            print(f'PASS {name}', flush=True)


if __name__ == '__main__':
    main()
