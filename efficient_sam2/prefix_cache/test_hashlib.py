"""Independent standard-library cross-check of the unmodified native digest."""
import hashlib
import pathlib
import struct
import subprocess
import sys

output = subprocess.check_output([sys.argv[1], sys.argv[2]], text=True)
records = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
expected = {
    "bytes256": hashlib.sha256(bytes(range(256))).hexdigest(),
    "file_million_a": hashlib.sha256(b"a" * 1_000_000).hexdigest(),
    "rgb2x1": hashlib.sha256(b"efficient-sam2-rgb-v1\n" + struct.pack("<QQ", 2, 1) + bytes(range(6))).hexdigest(),
}
assert records == expected, (records, expected)
assert not pathlib.Path(sys.argv[2]).exists()
print(output, end="")
print("PASS: independent Python hashlib cross-check")
