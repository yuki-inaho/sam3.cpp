#!/usr/bin/env python3
"""Check a reduced-window late seed against an isolated fresh-state seed.

Read-only regression over completed frame-major output directories. The mixed
run must end with an existing object's reduced-window propagation followed by
the new seed. The isolated run contains only that seed. No inference is started.
"""
import argparse
import json
import re
from pathlib import Path


def check(combined, isolated, fruit_id, frame_index):
    def read(root):
        if json.loads((root/'status.json').read_text())['status'] != 'complete':
            raise ValueError('Run is not complete')
        records=json.loads((root/'tracks'/f'{fruit_id:04}.json').read_text())['results']
        record=next(r for r in records if r['frame_index']==frame_index)
        windows=[int(n) for n in re.findall(r'Efficient-SAM2 packed Hiera: (\d+)/25 windows', (root/'native.stderr.log').read_text())]
        return record,windows
    mixed,mixed_windows=read(combined); alone,alone_windows=read(isolated)
    equal=all(mixed.get(k)==alone.get(k) for k in ('instance','status','native_verification'))
    reset=(len(mixed_windows)>=2 and mixed_windows[-2]<25 and mixed_windows[-1]==25 and alone_windows==[25])
    return dict(exact_seed_outputs_equal=equal, reduced_selection_reset_observed=reset,
                combined_windows=mixed_windows, isolated_windows=alone_windows,
                fruit_id=fruit_id,frame_index=frame_index)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--combined',type=Path,required=True);p.add_argument('--isolated',type=Path,required=True)
    p.add_argument('--fruit-id',type=int,required=True);p.add_argument('--frame-index',type=int,required=True)
    a=p.parse_args();result=check(a.combined,a.isolated,a.fruit_id,a.frame_index)
    print(json.dumps(result,indent=2))
    if not result['exact_seed_outputs_equal'] or not result['reduced_selection_reset_observed']:
        raise SystemExit(1)
