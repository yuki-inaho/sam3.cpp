"""Package tracked sources with a relative SHA256 manifest, without models/caches."""

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def package(repository, name, output):
    root = Path(repository).resolve()
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Archive root must be a single directory name")
    files = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    with tempfile.TemporaryDirectory(prefix="evm-package-") as directory:
        stage = Path(directory) / name
        stage.mkdir()
        for relative in sorted(set(files)):
            path = Path(relative)
            if not relative or path.is_absolute() or ".." in path.parts:
                continue
            if path.parts[0] in {"models", "outputs", "build", "data", "dist"}:
                continue
            source = root / path
            if not source.is_file() or source.is_symlink():
                continue
            target = stage / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        manifest = []
        for source in sorted(stage.rglob("*")):
            if source.is_file():
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                manifest.append(f"{digest}  {source.relative_to(stage)}")
        (stage / "FILES.sha256").write_text("\n".join(manifest) + "\n")
        archive = Path(directory) / "source.tar"

        def public_header(info):
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.pax_headers = {}
            return info

        with tarfile.open(archive, "w") as tar:
            tar.add(stage, arcname=name, filter=public_header)
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["zstd", "-T4", "-3", "-f", str(archive), "-o", str(output)], check=True
        )
    print(f"Packaged {len(manifest)} tracked files: {output.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    package(args.repository, args.name, args.output)


if __name__ == "__main__":
    main()
