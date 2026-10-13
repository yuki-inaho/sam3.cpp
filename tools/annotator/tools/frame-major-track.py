#!/usr/bin/env python3
"""Run a prepared batch-track.py manifest with independent frame-major trackers.

Preparation is default. --run explicitly enables inference. No native temporal
checkpoint is claimed: interrupted/changed tracks replay causally from the seed.
Matching output prefixes are verified during replay and preserved; complete
matching tracks are skipped. Feature cache artifacts are independent of outputs.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location('batch_contract', Path(__file__).with_name('batch-track.py'))
batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(batch)
_bridge_spec = importlib.util.spec_from_file_location('batch_native_bridge', Path(__file__).with_name('serve-browser.py'))
bridge = importlib.util.module_from_spec(_bridge_spec)
_bridge_spec.loader.exec_module(bridge)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def native_runtime(driver):
    """Use the same loaded-ELF resolver as native persistent-feature caching."""
    result = subprocess.run([str(driver), '--runtime-provenance'], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError('Cannot verify native runtime provenance: ' + result.stderr.strip())
    provenance = json.loads(result.stdout)
    if provenance.get('ok') is not True or not provenance.get('runtime_identity') or not provenance.get('native_build_sha256'):
        raise ValueError('Incomplete native runtime provenance; persisted resume unsupported')
    return dict(architecture=platform.machine(), **provenance)


def dependencies(manifest, seed, runtime):
    """Each result depends on its own causal inputs, never another object's prompt."""
    start = seed['frame_index']
    end = min(len(manifest['frames']), start + (manifest.get('max_frames') or len(manifest['frames'])))
    previous = digest(dict(seed=seed, model=manifest['model_sha256'], initialized_sequence_length=len(manifest['frames']), runtime=runtime))
    result = []
    for index in range(start, end):
        f = manifest['frames'][index]
        previous = digest(dict(previous=previous, frame_index=index, sha256=f['sha256'], width=f['width'], height=f['height']))
        result.append(previous)
    return result


def retained_prefix(track, expected, start):
    result = []
    for offset, record in enumerate(track.get('results', [])):
        if offset >= len(expected) or record.get('dependency_sha256') != expected[offset] or record.get('frame_index') != start + offset:
            break
        result.append(record)
    return result


class Native:
    def __init__(self, command, log, timeout):
        self.log = open(log, 'ab')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self.timeout = timeout
        self.reply_buffer = b''
        try:
            self.ready = self.read()
        except BaseException:
            self.close()
            raise

    def read(self):
        # Reuse the bridge's bounded binary framing and single total deadline.
        return bridge.Native._read_reply(self, self.process)

    def stop(self):
        self.close()

    def invoke(self, command):
        if '\n' in command or '\r' in command:
            raise ValueError('Line breaks are not supported in paths')
        self.process.stdin.write((command + '\n').encode())
        self.process.stdin.flush()
        return self.read()

    def close(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()
        self.process = None
        self.log.close()


def validate(manifest):
    frames, seeds = manifest['frames'], manifest['seeds']
    if not frames or not seeds or len({s['proposed_fruit_id'] for s in seeds}) != len(seeds):
        raise ValueError('Empty manifest or duplicate object IDs')
    maximum = manifest.get('max_frames')
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError('Invalid maximum frame count')
    for seed in seeds:
        fid, start, box = seed['proposed_fruit_id'], seed['frame_index'], seed['bbox_xyxy']
        if type(fid) is not int or not 1 <= fid <= 2147483647 or type(start) is not int or not 0 <= start < len(frames):
            raise ValueError('Invalid object ID or seed frame')
        f = frames[start]
        if seed.get('source_filename', f['fileName']) != f['fileName']:
            raise ValueError('Seed/source frame mismatch')
        if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
            raise ValueError('Invalid seed box')
        if not (0 <= box[0] < box[2] <= f['width'] and 0 <= box[1] < box[3] <= f['height']):
            raise ValueError('Seed box outside frame')


def run(manifest, root, command, runtime, timeout, authorized):
    validate(manifest)
    plans = []
    for seed in manifest['seeds']:
        fid = seed['proposed_fruit_id']
        expected = dependencies(manifest, seed, runtime)
        path = root / 'tracks' / f'{fid:04}.json'
        old = json.loads(path.read_text()) if path.exists() else {}
        kept = retained_prefix(old, expected, seed['frame_index'])
        complete = old.get('status') == 'complete' and len(kept) == len(expected) and len(old.get('results', [])) == len(expected)
        plans.append(dict(seed=seed, expected=expected, path=path, kept=kept, complete=complete))
    batch.write(root / 'plan.json', dict(scheduler='frame-major', tracks=len(plans), skipped_complete=sum(p['complete'] for p in plans),
        frame_inferences=sum(len(p['expected']) for p in plans if not p['complete']), inference_authorized_by_cli=authorized,
        resume_policy='deterministic causal replay from seed; verify and preserve valid output prefix'))
    if not authorized:
        return
    active = [p for p in plans if not p['complete']]
    if not active:
        batch.assemble(manifest, root)
        batch.write(root / 'status.json', dict(status='complete', skipped_complete=len(plans)))
        return
    native = None
    started = time.monotonic()
    timing = dict(model_startup_seconds=0.0, image_load_seconds=0.0, object_command_seconds=0.0, native_inference_seconds=0.0)
    try:
        native = Native(command, root / 'native.stderr.log', timeout)
        timing['model_startup_seconds'] = time.monotonic() - started
        if 'native_runtime' in runtime and native.ready.get('runtime_identity') != runtime['native_runtime']['runtime_identity']:
            raise ValueError('Runtime dependencies changed after model loading; persisted resume is unsupported for this backend')
        batch.write(root / 'capabilities.json', dict(**native.ready, runtime=runtime))
        for p in active:
            p['track'] = dict(fruit_id=p['seed']['proposed_fruit_id'], status='running', results=p['kept'][:], replay_from_seed=True, manual_interventions=0)
            p['segment'] = f"{p['seed']['proposed_fruit_id']}:frame-{p['seed']['frame_index']}"
            p['missing'] = p['unresolved'] = False
            batch.write(p['path'], p['track'])
        for index, frame in enumerate(manifest['frames']):
            current = [p for p in active if p['seed']['frame_index'] <= index < p['seed']['frame_index'] + len(p['expected'])]
            if not current:
                continue
            load_started = time.monotonic()
            native.invoke(f"frame {index} {frame['path']}")
            timing['image_load_seconds'] += time.monotonic() - load_started
            for p in current:
                seed = p['seed']; fid = seed['proposed_fruit_id']; offset = index - seed['frame_index']
                prompt = dict(pos=[], neg=[], box=seed['bbox_xyxy']) if offset == 0 else None
                command_started = time.monotonic()
                res = native.invoke('seed ' + str(fid) + ' ' + ' '.join(map(str, seed['bbox_xyxy'])) if prompt else f'step {fid}')
                timing['object_command_seconds'] += time.monotonic() - command_started
                timing['native_inference_seconds'] += res['seconds']
                if (res.get('frame_index'), res.get('local_frame_index'), res.get('object_id'), res.get('width'), res.get('height')) != (index, offset, fid, frame['width'], frame['height']):
                    raise ValueError('Native identity/frame/dimensions mismatch')
                if len(res['detections']) != 1:
                    raise ValueError('Expected one independent tracker output')
                if p['missing']:
                    p['segment'] = f'{fid}:frame-{index}'; p['unresolved'] = True
                inst = batch.instance(res['detections'][0], fid, p['segment'], frame['width'], frame['height'], prompt, p['unresolved'])
                p['missing'] = inst is None
                record = dict(frame_index=index, seconds=res['seconds'], status='prediction' if inst else 'unresolved', instance=inst, dependency_sha256=p['expected'][offset], native_verification=res.get('verification', {}))
                if offset < len(p['kept']):
                    previous = p['kept'][offset]
                    if previous['instance'] != inst or previous['status'] != record['status'] or previous.get('native_verification', {}) != record['native_verification']:
                        raise ValueError(f'Causal replay differs for object {fid}, frame {index}; previous output preserved')
                else:
                    p['track']['results'].append(record)
                if offset + 1 == len(p['expected']):
                    p['track']['status'] = 'complete'
                    native.invoke(f'drop {fid}')
                batch.write(p['path'], p['track'])
                batch.write(root / 'status.json', dict(status='running', fruit_id=fid, frame_index=index, seconds=res['seconds'], elapsed_seconds=time.monotonic()-started))
                print(json.dumps(dict(fruit_id=fid, frame_index=index, seconds=res['seconds'], replay=offset<len(p['kept']))), flush=True)
            batch.write(root / 'cache-statistics.json', native.invoke('stats'))
        batch.assemble(manifest, root)
        batch.write(root / 'status.json', dict(status='complete', elapsed_seconds=time.monotonic()-started, timing=timing))
    except BaseException as error:
        batch.assemble(manifest, root)
        batch.write(root / 'status.json', dict(status='interrupted', error=str(error)))
        raise
    finally:
        if native is not None:
            native.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--driver', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--prefix-cache-dir', type=Path)
    ap.add_argument('--prefix-cache-budget-mib', type=int, default=0)
    ap.add_argument('--efficient-prefix-cache', action='store_true')
    ap.add_argument('--efficient-fpn-cache', action='store_true')
    ap.add_argument('--experimental-efficient-q8-0', action='store_true')
    ap.add_argument('--timeout', type=int, default=600)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    root = args.output.resolve(); repo = Path(__file__).resolve().parents[3]
    if root.is_relative_to(repo):
        ap.error('Output must be outside the repository')
    if args.timeout <= 0 or args.prefix_cache_budget_mib < 0 or args.prefix_cache_budget_mib > 1048576 or (args.prefix_cache_dir and not args.prefix_cache_budget_mib):
        ap.error('Invalid timeout/cache budget; disk cache requires explicit positive budget')
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root / '.runner.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads(args.manifest.read_text())
    validate(manifest)
    # Verify source bytes rather than trusting stale preparation-time hashes.
    if batch.sha(manifest['model']) != manifest['model_sha256']:
        raise ValueError('Model changed: prepare a new manifest')
    for frame in manifest['frames']:
        if batch.sha(frame['path']) != frame['sha256']:
            raise ValueError('Frame changed: prepare a new manifest')
    command = [str(args.driver.resolve(strict=True)), '--model', manifest['model'], '--n-threads', str(manifest['threads']), '--sequence-length', str(len(manifest['frames']))]
    if manifest['experimental_video']:
        command.append('--experimental-efficient-video')
    if args.experimental_efficient_q8_0:
        command.append('--experimental-efficient-q8-0')
    if args.efficient_fpn_cache:
        command.append('--efficient-fpn-cache')
    cache = args.efficient_prefix_cache or args.prefix_cache_dir is not None
    if cache:
        command.append('--efficient-prefix-cache')
    if args.prefix_cache_dir:
        cache_root = args.prefix_cache_dir.resolve()
        if cache_root.is_relative_to(repo):
            ap.error('Feature cache must be outside the repository')
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_lock = open(cache_root / '.writer.lock', 'a')
        fcntl.flock(cache_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command += ['--prefix-cache-dir', str(args.prefix_cache_dir.resolve()), '--prefix-cache-budget-mib', str(args.prefix_cache_budget_mib)]
    runtime = dict(driver_sha256=batch.sha(args.driver), wrapper_sha256=batch.sha(Path(__file__)), annotation_contract_sha256=batch.sha(Path(batch.__file__)), bridge_sha256=batch.sha(Path(bridge.__file__)), native_runtime=native_runtime(args.driver.resolve()), threads=manifest['threads'], experimental_video=manifest['experimental_video'], experimental_q8_0=args.experimental_efficient_q8_0,
        efficient_video_preprocessing=True, efficient_fpn_cache=args.efficient_fpn_cache, initialized_sequence_length=len(manifest['frames']), scheduler='independent-frame-major-v1')
    manifest['executable'] = str(args.driver.resolve()); manifest['executable_sha256'] = runtime['driver_sha256']
    # Never mix projects that happen to use the same integer object IDs.
    existing = root / 'manifest.json'
    if existing.exists() and json.loads(existing.read_text())['sequence_id'] != manifest['sequence_id']:
        raise ValueError('Different sequence: use a different output directory')
    batch.write(existing, manifest)
    batch.write(root / 'runtime.json', dict(**runtime, exact_prefix_cache=cache, persistent_cache=bool(args.prefix_cache_dir)))
    run(manifest, root, command, runtime, args.timeout, args.run)


if __name__ == '__main__':
    main()
