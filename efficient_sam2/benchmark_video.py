"""Controlled paired end-to-end CPU benchmark, separate from quality acceptance.

Consumes a preregistered protocol JSON. No private images or trained weights are
copied into the repository. Run without competing heavy jobs. Timing is not
quality acceptance: retain failed fidelity/accuracy checks and reassess the
candidate on the target CPU before deployment. Nested phases overlap.
"""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def host_snapshot():
    result = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'logical_cpus': os.cpu_count()}
    if hasattr(os, 'sched_getaffinity'):
        result['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    for path in ['/proc/loadavg', '/proc/pressure/cpu', '/sys/fs/cgroup/cpu.max']:
        try:
            result[path] = Path(path).read_text().strip()
        except OSError:
            pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['executable', 'model', 'frames', 'protocol', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol['threads'] != 4 or protocol['initialized_sequence_length'] != 17:
        raise ValueError('This runner implements the frozen CPU4/full17 protocol only')
    if protocol['pair_orders'] != ['AB', 'BA', 'AB'] or protocol['consumed_frames'] != [0, 1, 2, 3]:
        raise ValueError('Expected three preregistered AB/BA/AB four-frame pairs')
    if protocol['variants'] != {'A': 'restored_fp32_cpu', 'B': 'experimental_ggml_q8_0_cpu'}:
        raise ValueError('Unexpected compute variants')
    if digest(args.model) != protocol['model_sha256']:
        raise ValueError('Model contents differ from protocol')
    for name, expected in protocol['input_files'].items():
        if digest(args.frames / Path(name).name) != expected:
            raise ValueError('Input contents differ from protocol: ' + name)
    source_frames = sorted(x for x in args.frames.iterdir() if x.suffix in ('.jpg', '.png'))
    if len(source_frames) != 17 or [x.name for x in source_frames[:4]] != [Path(x).name for x in protocol['input_files']]:
        raise ValueError('Initialized source sequence/order differs from protocol')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty output directory to avoid mixing samples')
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {'protocol_sha256': digest(args.protocol), 'executable_sha256': digest(args.executable),
                  'protocol': protocol, 'start_host': host_snapshot(), 'samples': []}

    def run(mode, label, count, measured):
        output = args.output / label
        output.mkdir()
        environment = {k: v for k, v in os.environ.items() if not k.startswith('EFFICIENT_SAM2_')}
        environment['EFFICIENT_SAM2_PROFILE'] = '1'
        if mode == 'B':
            environment['EFFICIENT_SAM2_Q8_0'] = '1'
        command = [str(args.executable.resolve()), str(args.model.resolve()), str(args.frames.resolve()),
                   str(output.resolve()), *map(str, protocol['box_xyxy']), str(count)]
        before = host_snapshot()
        with (output / 'stdout.jsonl').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
            start = time.perf_counter()
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment)
            usage = None
            try:
                if hasattr(os, 'wait4'):
                    _, status, usage = os.wait4(process.pid, 0)
                    process.returncode = os.waitstatus_to_exitcode(status)
                else:
                    process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            elapsed = time.perf_counter() - start
        if process.returncode:
            raise RuntimeError(f'{label} failed with {process.returncode}; see stderr.log')
        frames = [json.loads(line) for line in (output / 'stdout.jsonl').read_text().splitlines()]
        if len(frames) != count or [x['frame_index'] for x in frames] != list(range(count)):
            raise ValueError('Incomplete or unordered measured run')
        runtime = json.loads((output / 'runtime.json').read_text())
        expected_compute = protocol['variants'][mode]
        if runtime['compute'] != expected_compute or runtime['model_sha256'] != protocol['model_sha256']:
            raise ValueError('Measured runtime/model identity mismatch')
        if runtime['frames'] != count:
            raise ValueError('Measured frame count mismatch')
        if runtime.get('threads') != protocol['threads'] or not runtime.get('runtime_provenance'):
            raise ValueError('Runtime thread count or loaded-library provenance missing')
        for value in [runtime['model_load_seconds'], runtime['setup_seconds'],
                      *(frame['seconds'] for frame in frames)]:
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid elapsed measurement')
        if any(not math.isfinite(frame[key]) for frame in frames for key in ('score', 'object_logit')):
            raise ValueError('Nonfinite prediction in measured run')
        for previous in provenance['samples']:
            if runtime['runtime_provenance'] != previous['runtime']['runtime_provenance']:
                raise ValueError('Loaded executable or library identity changed between samples')
        events = []
        for line in (output / 'stderr.log').read_text().splitlines():
            if line.startswith('{'):
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('event') == 'efficient_profile':
                    events.append(event)
        if not events:
            raise ValueError('Missing opt-in phase measurements')
        weight_buffer = re.search(r'sam3_load_model: buffer size = ([0-9.]+) MB',
                                  (output / 'stderr.log').read_text())
        peak = None
        if usage is not None and sys.platform in ('linux', 'darwin'):
            peak = usage.ru_maxrss * (1024 if sys.platform == 'linux' else 1)
        sample = {'label': label, 'variant': mode, 'measured': measured, 'wall_seconds': elapsed,
                  'inference_seconds': sum(x['seconds'] for x in frames),
                  'model_load_seconds': runtime['model_load_seconds'],
                  'setup_seconds': runtime['setup_seconds'], 'peak_rss_bytes': peak,
                  'weight_buffer_mib_rounded': float(weight_buffer.group(1)) if weight_buffer else None,
                  'frames': frames, 'runtime': runtime, 'profile_events': events,
                  'mask_sha256': [digest(output / f'{i:05d}.png') for i in range(count)],
                  'host_before': before, 'host_after': host_snapshot()}
        provenance['samples'].append(sample)
        (args.output / 'raw_samples.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: sample[k] for k in ['label', 'variant', 'wall_seconds', 'inference_seconds',
                                               'model_load_seconds', 'peak_rss_bytes']}), flush=True)
        return sample

    for mode in ['A', 'B']:
        run(mode, 'warmup_' + mode, 2, False)
    pairs = []
    for index, order in enumerate(protocol['pair_orders']):
        if sorted(order) != ['A', 'B']:
            raise ValueError('Each pair must contain A and B exactly once')
        samples = {mode: run(mode, f'pair_{index}_{mode}', 4, True) for mode in order}
        pairs.append({'pair': index, 'order': order,
                      'inference_ratio_A_over_B': samples['A']['inference_seconds'] / samples['B']['inference_seconds'],
                      'wall_ratio_A_over_B': samples['A']['wall_seconds'] / samples['B']['wall_seconds']})
    summary = {'pairs': pairs,
               'median_paired_inference_ratio_A_over_B': statistics.median(p['inference_ratio_A_over_B'] for p in pairs),
               'median_paired_wall_ratio_A_over_B': statistics.median(p['wall_ratio_A_over_B'] for p in pairs),
               'quality_claim': False, 'profile_note': 'Nested phases overlap; do not sum totals across phases',
               'end_host': host_snapshot()}
    summary['variants'] = {}
    for mode in ['A', 'B']:
        samples = [s for s in provenance['samples'] if s['measured'] and s['variant'] == mode]
        summary['variants'][mode] = {}
        for key in ['wall_seconds', 'inference_seconds', 'model_load_seconds', 'setup_seconds', 'peak_rss_bytes']:
            values = [s[key] for s in samples if s[key] is not None]
            summary['variants'][mode][key] = ({'median': statistics.median(values), 'min': min(values),
                                              'max': max(values)} if values else None)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
