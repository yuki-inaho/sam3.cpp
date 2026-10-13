#!/usr/bin/env python3
"""Synthetic scheduler/resume contracts, no model weights or inference needed."""
import copy
import importlib.util
import json
import sys
import time
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
from pathlib import Path

spec = importlib.util.spec_from_file_location('frame_major', Path(__file__).resolve().parents[1] / 'tools/annotator/tools/frame-major-track.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class FakeNative:
    commands = []
    fail_at = None
    def __init__(self, *args):
        self.ready = dict(ok=True); self.objects = {}; self.frame = -1
    def invoke(self, command):
        self.commands.append(command)
        words = command.split(); op = words[0]
        if op == 'frame':
            self.frame = int(words[1]); return dict(ok=True)
        if op == 'stats': return dict(ok=True, ram_hits=1, full_encoder=dict(enabled=True,hits=2,misses=1,evictions=0,resident_bytes=84))
        fid = int(words[1])
        if op == 'drop': del self.objects[fid]; return dict(ok=True)
        if op == 'seed': self.objects[fid] = self.frame
        if (fid, self.frame) == self.fail_at: raise RuntimeError('synthetic interruption')
        return dict(frame_index=self.frame, local_frame_index=self.frame-self.objects[fid], object_id=fid, width=2, height=2, seconds=1,
                    detections=[dict(object_id=fid, mask_size=[2, 2], score=0.9, counts=[0, 4])])
    def close(self): pass


class BinaryReplies(unittest.TestCase):
    def test_partial_line_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            start=time.monotonic()
            script = "import sys,time;sys.stdout.write(" + repr('{"ok":') + ");sys.stdout.flush();time.sleep(20)"
            with self.assertRaises(ValueError):
                m.Native([sys.executable, '-u', '-c', script], Path(directory)/'log', .1)
            self.assertLess(time.monotonic()-start, 2)
    def test_multiple_lines_in_one_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            script = "import time;print(" + repr('{"ok":true,"n":1}\n{"ok":true,"n":2}') + ",flush=True);time.sleep(20)"
            native=m.Native([sys.executable, '-u', '-c', script], Path(directory)/'log', .2)
            try:
                self.assertEqual(native.ready['n'],1)
                self.assertEqual(native.read()['n'],2)
            finally: native.close()


class Provenance(unittest.TestCase):
    def test_missing_runtime_fails_closed(self):
        with mock.patch.object(m.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr='')):
            with self.assertRaises(ValueError): m.native_runtime(Path('/fake'))
    def test_native_runtime_failure_fails_closed(self):
        with mock.patch.object(m.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='', stderr='unsupported')):
            with self.assertRaises(ValueError): m.native_runtime(Path('/fake'))
    def test_complete_runtime_descriptor(self):
        reply=dict(ok=True, runtime_identity=dict(schema='test',objects=[]),native_build_sha256='abc')
        with mock.patch.object(m.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(reply), stderr='')):
            self.assertEqual(m.native_runtime(Path('/fake'))['native_build_sha256'],'abc')


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.original = m.Native; m.Native = FakeNative
        FakeNative.commands = []; FakeNative.fail_at = None
        self.manifest = dict(sequence_id='synthetic', model='fake', model_sha256='m', executable_sha256='e', experimental_video=True, threads=1, max_frames=None,
            frames=[dict(path=f'/frames/{i}.jpg', fileName=f'{i}.jpg', width=2, height=2, sha256=f'h{i}') for i in range(4)],
            seeds=[dict(proposed_fruit_id=7, frame_index=0, bbox_xyxy=[0, 0, 2, 2]), dict(proposed_fruit_id=19, frame_index=1, bbox_xyxy=[0, 0, 1, 1])])
    def tearDown(self):
        m.Native = self.original; self.temp.cleanup()
    def run_batch(self): m.run(self.manifest, self.root, [], dict(driver='fake'), 1, True)
    def track(self, fid): return json.loads((self.root/'tracks'/f'{fid:04}.json').read_text())
    def test_frame_major_independent_and_completed_skip(self):
        self.run_batch()
        self.assertEqual(FakeNative.commands[:6], ['frame 0 /frames/0.jpg', 'seed 7 0 0 2 2', 'stats', 'frame 1 /frames/1.jpg', 'step 7', 'seed 19 0 0 1 1'])
        self.assertEqual([r['frame_index'] for r in self.track(19)['results']], [1,2,3])
        self.assertEqual(self.track(7)['status'], 'complete')
        before = (self.root/'tracks/0007.json').read_bytes()
        FakeNative.commands=[]; self.run_batch()
        self.assertEqual(FakeNative.commands, [])
        self.assertEqual(before, (self.root/'tracks/0007.json').read_bytes())
    def test_interrupt_replay_preserves_output_prefix(self):
        FakeNative.fail_at=(19,2)
        with self.assertRaises(RuntimeError): self.run_batch()
        saved=copy.deepcopy(self.track(7)['results'])
        FakeNative.fail_at=None; FakeNative.commands=[]; self.run_batch()
        self.assertEqual(self.track(7)['results'][:len(saved)], saved)
        self.assertEqual(self.track(19)['status'], 'complete')
        self.assertIn('seed 7 0 0 2 2', FakeNative.commands)
    def test_shortened_track_does_not_export_old_suffix(self):
        self.run_batch()
        self.manifest['max_frames']=2
        self.run_batch()
        self.assertEqual([r['frame_index'] for r in self.track(7)['results']], [0,1])
        self.assertEqual([r['frame_index'] for r in self.track(19)['results']], [1,2])
        project=json.loads((self.root/'annotation-project.json').read_text())
        self.assertEqual(project['frames'][3]['instances'],[])
    def test_prompt_invalidation_is_object_specific(self):
        self.run_batch(); old=self.track(19)
        self.manifest['seeds'][0]['bbox_xyxy']=[0,0,1,2]
        FakeNative.commands=[]; self.run_batch()
        self.assertEqual(self.track(19), old)
        self.assertFalse(any(c.startswith('seed 19') for c in FakeNative.commands))
    def test_frame_change_invalidates_only_causal_suffix(self):
        self.run_batch(); old=self.track(7)['results']
        self.manifest['frames'][2]['sha256']='changed'
        expected=m.dependencies(self.manifest,self.manifest['seeds'][0],dict(driver='fake'))
        self.assertEqual(len(m.retained_prefix(self.track(7),expected,0)),2)
        self.run_batch(); new=self.track(7)['results']
        self.assertEqual(old[:2],new[:2]); self.assertNotEqual(old[2]['dependency_sha256'],new[2]['dependency_sha256'])
    def test_postload_runtime_change_blocks_inference(self):
        with self.assertRaises(ValueError):
            m.run(self.manifest,self.root,[],dict(native_runtime=dict(runtime_identity={'schema':'different'})),1,True)
        self.assertEqual(FakeNative.commands, [])
    def test_full_sequence_length_invalidates_temporal_prefix(self):
        self.run_batch()
        self.manifest['frames'].append(copy.deepcopy(self.manifest['frames'][-1]))
        expected=m.dependencies(self.manifest,self.manifest['seeds'][0],dict(driver='fake'))
        self.assertEqual(m.retained_prefix(self.track(7),expected,0),[])
    def test_full_encoder_cache_flag_invalidates_prefix(self):
        m.run(self.manifest,self.root,[],dict(driver='fake',efficient_fpn_cache=False),1,True)
        expected=m.dependencies(self.manifest,self.manifest['seeds'][0],dict(driver='fake',efficient_fpn_cache=True))
        self.assertEqual(m.retained_prefix(self.track(7),expected,0),[])
    def test_full_encoder_statistics_preserved(self):
        self.run_batch()
        stats=json.loads((self.root/'cache-statistics.json').read_text())
        self.assertEqual(stats['full_encoder'],dict(enabled=True,hits=2,misses=1,evictions=0,resident_bytes=84))
    def test_runtime_change_invalidates_prefix(self):
        self.run_batch()
        expected=m.dependencies(self.manifest,self.manifest['seeds'][0],dict(driver='changed'))
        self.assertEqual(m.retained_prefix(self.track(7),expected,0),[])
    def test_duplicate_ids_rejected_before_native(self):
        self.manifest['seeds'][1]['proposed_fruit_id']=7
        with self.assertRaises(ValueError): self.run_batch()
        self.assertEqual(FakeNative.commands, [])
    def test_prepare_has_no_native_calls(self):
        m.run(self.manifest,self.root,[],{},1,False)
        self.assertEqual(FakeNative.commands,[])
        self.assertFalse((self.root/'tracks').exists())


if __name__ == '__main__': unittest.main()
