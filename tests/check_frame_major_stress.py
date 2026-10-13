#!/usr/bin/env python3
"""Read-only validation of an eight-tracker duplicate-prompt memory stress run.

Checks all expected causal outputs, exact matching predictions for duplicated
prompts under different stable IDs, resident cache bound, and memory statistics.
Duplicated prompts are intentionally not accuracy or distinct-identity evidence.
"""
import argparse
import json
from pathlib import Path


def check(root):
    manifest=json.loads((root/'manifest.json').read_text())
    status=json.loads((root/'status.json').read_text())
    if status['status']!='complete': raise ValueError('Stress run is incomplete')
    seeds=manifest['seeds']; groups={}; cases=0; pairs=0; failures=[]
    for seed in seeds:
        fid=seed['proposed_fruit_id']
        track=json.loads((root/'tracks'/f'{fid:04}.json').read_text())
        expected=list(range(seed['frame_index'],len(manifest['frames'])))
        if track['status']!='complete' or [r['frame_index'] for r in track['results']]!=expected:
            failures.append(f'Object {fid}: incomplete/noncausal frames')
        normalized=[]
        for r in track['results']:
            inst=r['instance']
            normalized.append(dict(frame_index=r['frame_index'],native_verification=r.get('native_verification'),
                counts=inst['counts'] if inst else None,score=inst['score'] if inst else None,status=r['status']))
        cases+=len(normalized)
        key=(seed['frame_index'],tuple(seed['bbox_xyxy']))
        if key in groups:
            pairs+=1
            if groups[key]!=normalized: failures.append(f'Object {fid}: duplicate prompt differs')
        else: groups[key]=normalized
    cache=json.loads((root/'cache-statistics.json').read_text())
    if cache['resident_bytes']>49*1024*1024: failures.append('Resident prefix cache exceeds one49MiB entry')
    if cache['active_trackers']!=0: failures.append('Completed trackers were not released')
    return dict(tracker_count=len(seeds),distinct_prompts=len(groups),duplicate_pairs=pairs,cases=cases,
        duplicate_predictions_exact=not failures,failures=failures,peak_rss_bytes=cache['peak_rss_bytes'],cache=cache,
        elapsed_seconds=status['elapsed_seconds'],timing=status.get('timing'),
        scope='Duplicate-prompt load test, not distinct-identity or accuracy evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    result=check(a.output);print(json.dumps(result,indent=2))
    if result['failures']: raise SystemExit(1)
