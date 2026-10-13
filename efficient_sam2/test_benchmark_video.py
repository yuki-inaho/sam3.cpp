"""Fast benchmark orchestration tests; the child is a fixture, never a model."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).with_name('benchmark_video.py')


@unittest.skipUnless(os.name == "posix", "Fixture child uses a POSIX shebang")
class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / 'model.gguf'
        self.model.write_bytes(b'fixture only')
        self.frames = self.root / 'frames'
        self.frames.mkdir()
        for i in range(17):
            (self.frames / f'{i:05d}.jpg').write_bytes(str(i).encode())
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        self.protocol = {
            'threads': 4, 'initialized_sequence_length': 17,
            'pair_orders': ['AB', 'BA', 'AB'], 'consumed_frames': [0, 1, 2, 3],
            'variants': {'A': 'restored_fp32_cpu', 'B': 'experimental_ggml_q8_0_cpu'},
            'model_sha256': sha(self.model), 'box_xyxy': [1, 2, 3, 4],
            'input_files': {f'{i:05d}.jpg': sha(self.frames / f'{i:05d}.jpg') for i in range(4)},
        }
        self.child = self.root / 'fixture_child'
        self.child.write_text('#!' + sys.executable + '\n' + '''
import hashlib,json,os,pathlib,sys
model,frames,output=sys.argv[1:4]
output=pathlib.Path(output); count=int(sys.argv[-1]); q8='EFFICIENT_SAM2_Q8_0' in os.environ
assert not any(k in os.environ for k in ['EFFICIENT_SAM2_PREFIX_CACHE','EFFICIENT_SAM2_DUMP_LOGITS'])
runtime={'model_sha256':hashlib.sha256(pathlib.Path(model).read_bytes()).hexdigest(),
'compute':'experimental_ggml_q8_0_cpu' if q8 else 'restored_fp32_cpu',
'frames':count,'threads':4,'model_load_seconds':.1,'setup_seconds':.2,
'runtime_provenance':{'fixture':'stable'}}
(output/'runtime.json').write_text(json.dumps(runtime))
for i in range(count):
 (output/f'{i:05d}.png').write_bytes(bytes([i]))
 print(json.dumps({'frame_index':i,'seconds':1 if q8 else 2,'score':.9,'object_logit':1}))
print(json.dumps({'event':'efficient_profile','phase':'fixture','part':'total','ms':1}),file=sys.stderr)
''')
        self.child.chmod(0o700)

    def run_fixture(self):
        protocol = self.root / 'protocol.json'
        protocol.write_text(json.dumps(self.protocol))
        return subprocess.run([sys.executable, str(RUNNER), '--executable', str(self.child),
                               '--model', str(self.model), '--frames', str(self.frames),
                               '--protocol', str(protocol), '--output', str(self.root / 'output')],
                              capture_output=True, text=True)

    def test_all_pairs_and_warmups_are_retained(self):
        result = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stderr)
        raw = json.loads((self.root / 'output/raw_samples.json').read_text())
        self.assertEqual([x['label'] for x in raw['samples']],
                         ['warmup_A','warmup_B','pair_0_A','pair_0_B','pair_1_B','pair_1_A','pair_2_A','pair_2_B'])
        self.assertEqual(sum(x['measured'] for x in raw['samples']), 6)
        summary = json.loads((self.root / 'output/summary.json').read_text())
        self.assertEqual(summary['median_paired_inference_ratio_A_over_B'], 2)
        self.assertFalse(summary['quality_claim'])

    def test_reject_protocol_change(self):
        self.protocol['pair_orders'] = ['AB']
        self.assertNotEqual(self.run_fixture().returncode, 0)
        self.assertFalse((self.root / 'output').exists())

    def test_reject_modified_input(self):
        (self.frames / '00000.jpg').write_bytes(b'changed')
        self.assertNotEqual(self.run_fixture().returncode, 0)
        self.assertFalse((self.root / 'output').exists())

    def test_failed_child_is_not_a_sample(self):
        self.child.write_text('#!' + sys.executable + '\nraise SystemExit(7)\n')
        self.assertNotEqual(self.run_fixture().returncode, 0)
        self.assertFalse((self.root / 'output/summary.json').exists())

    def test_output_is_never_overwritten(self):
        (self.root / 'output').mkdir()
        marker = self.root / 'output/keep'
        marker.write_text('keep')
        self.assertNotEqual(self.run_fixture().returncode, 0)
        self.assertEqual(marker.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
