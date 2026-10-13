#!/usr/bin/env python3
"""Compare exact frame-major outputs and compact native verification metadata.

Reports observed equivalence and timings only, not deployment quality or a
controlled speedup. Pass completed output directories in baseline-first order.
"""
import argparse
import json
import math
import statistics
from pathlib import Path


def read_run(root):
    status = json.loads((root / 'status.json').read_text())
    if status['status'] != 'complete':
        raise ValueError(f'Run is not complete: {root.name}')
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = {}
    for seed in manifest['seeds']:
        fid = seed['proposed_fruit_id']
        track = json.loads((root / 'tracks' / f'{fid:04}.json').read_text())
        if track['status'] != 'complete':
            raise ValueError('Incomplete object')
        for r in track['results']:
            key = (fid, r['frame_index'])
            if key in rows:
                raise ValueError('Duplicate object/frame')
            rows[key] = r
    latencies = [r['seconds'] for r in rows.values()]
    verified = all(r.get('native_verification', {}).get('low_res_logits_count', 0) > 0 and
                   math.isfinite(r['native_verification'].get('object_logit', float('nan'))) for r in rows.values())
    diagnostic = [line for line in (root / 'native.stderr.log').read_text().splitlines()
                  if line.startswith(('Efficient-SAM2 packed Hiera:', 'Efficient-SAM2 SMR '))]
    summary = dict(name=root.name, cases=len(rows), all_native_logit_fingerprints_present=verified,
                   elapsed_seconds=status.get('elapsed_seconds'), sum_native_seconds=sum(latencies),
                   median_native_seconds=statistics.median(latencies), max_native_seconds=max(latencies),
                   cache=json.loads((root / 'cache-statistics.json').read_text()))
    return rows, diagnostic, summary


def compare(roots):
    baseline, base_diagnostic, first = read_run(roots[0])
    report = dict(baseline=first, candidates=[], claims='Same-build exact cache equivalence only; no tracking-quality or controlled-speedup claim')
    for root in roots[1:]:
        rows, diagnostic, summary = read_run(root)
        mismatches = []
        for key in sorted(set(baseline) | set(rows)):
            if key not in rows or key not in baseline:
                mismatches.append(dict(object_id=key[0], frame_index=key[1], field='missing_case'))
                continue
            for field in ('instance', 'native_verification', 'status'):
                if baseline[key].get(field) != rows[key].get(field):
                    mismatches.append(dict(object_id=key[0], frame_index=key[1], field=field))
        summary.update(exact_outputs_equal=not mismatches, diagnostic_window_and_memory_counts_equal=diagnostic==base_diagnostic, mismatches=mismatches)
        report['candidates'].append(summary)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error('Provide baseline and at least one candidate')
    result = compare(args.runs)
    print(json.dumps(result, indent=2, allow_nan=False))
    if any(not c['exact_outputs_equal'] or not c['diagnostic_window_and_memory_counts_equal'] for c in result['candidates']):
        raise SystemExit(1)
