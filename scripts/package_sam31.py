#!/usr/bin/env python3
"""Package tracked sources, fixtures and current evidence without local weights/cache."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import tarfile

root = Path(__file__).resolve().parents[1]
if (root / '.git').exists():
    # Include staged/tracked sources only; local notes and credentials are not sources.
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
else:
    # A delivered source archive has no Git metadata. Its verified file manifest
    # supplies the same source list without collecting unrelated local files.
    manifest = root / 'FILES.sha256'
    if not manifest.is_file():
        raise SystemExit('Packaging requires a Git checkout or delivery FILES.sha256')
    files = []
    for line in manifest.read_text().splitlines():
        digest, name = line.split('  ', 1)
        if len(digest) != 64 or Path(name).is_absolute() or '..' in Path(name).parts:
            raise SystemExit('Invalid path in delivery manifest')
        files.append(name)
with tempfile.TemporaryDirectory(prefix='sam31-package-') as directory:
    stage = Path(directory) / 'sam3cpp-sam31'
    stage.mkdir()
    for name in sorted(set(files)):
        if not name or name.startswith(('models/', 'data/', 'images/', 'videos/')):
            continue
        src = root / name
        if not src.is_file() or src.is_symlink():
            continue
        dst = stage / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        shutil.copymode(src, dst)
    for src in sorted((root / 'evidence').rglob('*')):
        if not src.is_file() or any(part.startswith('.') for part in src.relative_to(root / 'evidence').parts):
            continue
        dst = stage / src.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in ('.log', '.json', '.txt'):
            value = src.read_text().replace(str(root)+'/', '').replace(str(root), '.')
            value = value.replace(str(Path.home() / 'Downloads') + '/', 'models/')
            value = value.replace('/tmp/sam31-fixture-regenerated', 'regenerated-fixture')
            dst.write_text(value)
        else:
            shutil.copyfile(src, dst)
    manifest = []
    for src in sorted(stage.rglob('*')):
        if src.is_file():
            manifest.append(hashlib.sha256(src.read_bytes()).hexdigest() + '  ' + str(src.relative_to(stage)))
    (stage / 'FILES.sha256').write_text('\n'.join(manifest)+'\n')
    archive = Path(directory) / 'source.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(stage, arcname='sam3cpp-sam31')
    target = root / 'dist/sam3cpp_sam31_source.tar.zst'
    target.parent.mkdir(exist_ok=True)
    subprocess.run(['zstd', '-T4', '-3', '-f', str(archive), '-o', str(target)], check=True)
    print(f'{len(manifest)} files packaged: {target.name}')
