"""Linux lightweight runtime provenance oracle; never loads a model."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

binary = Path(sys.argv[1]).resolve()
checks = 0


def invoke(*arguments, success=True, cwd=None):
    global checks
    run = subprocess.run([str(binary), *map(str, arguments)], capture_output=True, text=True, cwd=cwd)
    if not success:
        assert run.returncode == 1 and run.stderr, run
        checks += 1
        return None
    assert run.returncode == 0, run.stderr
    digest, encoded = run.stdout.splitlines()
    assert hashlib.sha256(encoded.encode()).hexdigest() == digest
    identity = json.loads(encoded)
    assert identity["schema"] == "efficient-sam2-loaded-elf-v1"
    paths = [entry["path"] for entry in identity["objects"]]
    assert paths == sorted(paths)
    assert str(binary) in paths
    for entry in identity["objects"]:
        assert hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    checks += 1
    return digest, identity


first = invoke()
assert invoke() == first  # Different ASLR addresses must not change provenance.
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    source = root / "probe.c"
    source.write_text("int runtime_identity_probe(void) { return VALUE; }\n")
    original = root / "probe.so"
    replacement = root / "replacement.so"
    for value, destination in ((1, original), (2, replacement)):
        subprocess.run(["cc", "-shared", "-fPIC", f"-DVALUE={value}", str(source), "-o", str(destination)], check=True)
    loaded = invoke(original)
    assert loaded[0] != first[0]
    assert str(original) in [item["path"] for item in loaded[1]["objects"]]
    invoke("./probe.so", success=False, cwd=root)
    disposable = root / "delete.so"
    shutil.copyfile(original, disposable)
    invoke(disposable, "delete", success=False)
    disposable = root / "replace.so"
    shutil.copyfile(original, disposable)
    invoke(disposable, "replace", replacement, success=False)
print(f"Runtime identity: {checks} deterministic/hashlib/loaded/deleted/replaced/relative checks PASS")
