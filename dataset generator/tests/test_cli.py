import os
import subprocess
import sys
from pathlib import Path


def test_cli_module_entrypoint_generates_dataset(tmp_path: Path):
    output_dir = tmp_path / "cli_dataset"
    env = os.environ.copy()
    repo_src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(repo_src)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "procmusic_dataset.cli",
            "--out",
            str(output_dir),
            "--count",
            "1",
            "--seed",
            "5",
            "--min-tracks",
            "1",
            "--max-tracks",
            "1",
            "--min-bars",
            "1",
            "--max-bars",
            "1",
            "--sample-rate",
            "8000",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "project_000000" / "metadata.json").exists()
