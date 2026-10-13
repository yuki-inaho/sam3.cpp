#!/usr/bin/env python3
"""Fixed partial-reference box/identity audit. This is not mask or human GT accuracy.

Primary: frozen seeded-ID cohort per audit window; unmatched known-cohort
predictions count as errors, unannotated fruit are not negatives. Per-frame
all-admissible trajectory-pair IoU>=0.5 counts feed global ID assignment.
Per-frame maximum-cardinality matching is only for coverage/IDSW/diagnostics.
Explicit frozen cross-window links are supplemental, unknown joins unscored.
Dependencies: scipy. No inferred identity merges or manual corrections.
"""
import argparse, collections, hashlib, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=x*y
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0

def xyxy(box):
    x,y,w,h=box;return [x,y,x+w,y+h]

def match(gt,pred):
    if not gt or not pred:return []
    sim=np.array([[iou(g['bbox_xyxy'],xyxy(p['bbox'])) for p in pred] for g in gt])
    eligible=sim>=.5
    # One eligible edge outweighs every possible IoU tie-break combined.
    score=eligible*(min(len(gt),len(pred))+1+sim)
    rows,cols=linear_sum_assignment(score,maximize=True)
    return [(int(r),int(c),float(sim[r,c])) for r,c in zip(rows,cols) if eligible[r,c]]

def validate_inputs(project, audit):
    """Reject malformed/ambiguous evidence rather than silently last-write-wins.

    Duplicate prediction fruit IDs remain measurable errors: count every output
    in IDFP's denominator, but at most one admissible pair per frame/trajectory.
    Observation/exclusion records instead define eligibility and must be unique.
    """
    def integer(x, minimum=0):
        return type(x) is int and x >= minimum
    def box(values, xywh=False, dimensions=None):
        if not isinstance(values, list) or len(values) != 4 or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError('Invalid/nonfinite bbox')
        bounds = xyxy(values) if xywh else values
        x0, y0, x1, y1 = bounds
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError('Invalid bbox extent')
        if dimensions and (x1 > dimensions[0] or y1 > dimensions[1]):
            raise ValueError('BBox outside image dimensions')
    dimensions = audit.get('image_dimensions')
    if dimensions is not None and (not isinstance(dimensions, list) or len(dimensions) != 2 or any(not integer(n, 1) for n in dimensions)):
        raise ValueError('Invalid audit image dimensions')
    known = set()
    for target in audit['targets']:
        fid = target['stable_fruit_id']
        if not integer(fid, 1) or fid in known:
            raise ValueError('Duplicate/invalid audit target ID')
        known.add(fid)
    windows, frame_windows = {}, {}
    for window in audit['windows']:
        wid, frames = window['window_id'], window['frame_indices']
        if not isinstance(wid, str) or not wid or wid in windows or not frames:
            raise ValueError('Duplicate/invalid audit window')
        if any(not integer(f) or f in frame_windows for f in frames) or len(set(frames)) != len(frames):
            raise ValueError('Duplicate/invalid audit window frame')
        windows[wid] = set(frames)
        frame_windows.update({f: wid for f in frames})
    observed, excluded = set(), set()
    for label, records, seen in [('observation', audit['observations'], observed),
                                  ('exclusion', audit.get('exclusions', []), excluded)]:
        for row in records:
            wid, fi, fid = row['window_id'], row['frame_index'], row['stable_fruit_id']
            if (wid not in windows or not integer(fi) or fi not in windows[wid]
                    or not integer(fid, 1) or fid not in known):
                raise ValueError('Invalid window/frame/known ID in audit ' + label)
            key = (wid, fi, fid)
            if key in seen:
                raise ValueError('Duplicate audit ' + label)
            seen.add(key)
            if label == 'observation':
                box(row['bbox_xyxy'], dimensions=audit.get('image_dimensions'))
            elif row.get('bbox_xyxy') is not None:
                raise ValueError('Exclusion cannot supply scored geometry')
    if observed & excluded:
        raise ValueError('Observation/exclusion overlap')
    for cross, rows in [(False, audit['associations']),
                         (True, audit.get('cross_window_links', []))]:
        seen = set()
        for row in rows:
            fid, start, end = row['stable_fruit_id'], row['from_frame'], row['to_frame']
            if not integer(fid, 1) or fid not in known or not integer(start) or not integer(end) or end <= start:
                raise ValueError('Invalid association identity/frame')
            sw, ew = frame_windows.get(start), frame_windows.get(end)
            if ((sw, start, fid) not in observed or (ew, end, fid) not in observed
                    or (not cross and (sw != row['window_id'] or ew != sw or end != start + 1))
                    or (cross and sw == ew)):
                raise ValueError('Association endpoints are not eligible reviewed observations')
            key = (fid, start, end)
            if key in seen:
                raise ValueError('Duplicate association')
            seen.add(key)
    seen_frames = set()
    for frame in project['frames']:
        fi = frame['sourceFrameIndex']
        if not integer(fi) or fi in seen_frames:
            raise ValueError('Duplicate/invalid project sourceFrameIndex')
        seen_frames.add(fi)
        dims = None
        if 'width' in frame or 'height' in frame:
            dims = [frame.get('width'), frame.get('height')]
            unloaded = all(type(n) is int and n == 0 for n in dims) and not frame.get('instances', [])
            if not unloaded and any(not integer(n, 1) for n in dims):
                raise ValueError('Invalid project frame dimensions')
        for pred in frame['instances']:
            if not integer(pred['fruitId'], 1):
                raise ValueError('Invalid prediction fruit ID')
            box(pred['bbox'], xywh=True, dimensions=dims)


def assign_counts(counts, gids):
    pids = sorted({p for g, p in counts})
    if not pids:
        return 0, []
    matrix = np.array([[counts[g, p] for p in pids] for g in gids])
    rows, cols = linear_sum_assignment(matrix, maximize=True)
    assigned = [dict(audit_fruit_id=gids[r], predicted_fruit_id=pids[c],
                     matched_observations=int(matrix[r, c]))
                for r, c in zip(rows, cols) if matrix[r, c] > 0]
    return sum(a['matched_observations'] for a in assigned), assigned


def evaluate(project,audit,kind='raw'):
    validate_inputs(project, audit)
    frames={f['sourceFrameIndex']:f for f in project['frames']}
    totalgt=totalpred=idtp=matches=diagnostic_idtp=0;windows=[];mapped={};overlap=[]
    for window in audit['windows']:
        wid=window['window_id'];obs=[o for o in audit['observations'] if o['window_id']==wid]
        cohort={o['stable_fruit_id'] for o in obs};counts=collections.Counter();two_stage_counts=collections.Counter();ng=npred=nm=0
        for fi in window['frame_indices']:
            gt=[o for o in obs if o['frame_index']==fi];active={o['stable_fruit_id'] for o in gt}
            allpred=frames.get(fi,{}).get('instances',[])
            pred=[p for p in allpred if p['fruitId'] in active]
            ng+=len(gt);npred+=len(pred)
            admissible={(g['stable_fruit_id'],p['fruitId']) for g in gt for p in pred
                        if iou(g['bbox_xyxy'],xyxy(p['bbox']))>=.5}
            counts.update(admissible)
            for r,c,v in match(gt,pred):
                g,p=gt[r],pred[c];two_stage_counts[(g['stable_fruit_id'],p['fruitId'])]+=1;nm+=1
                mapped[(wid,fi,g['stable_fruit_id'])]=p['fruitId']
            for g in gt:
                candidates=[dict(fruit_id=p['fruitId'],iou=iou(g['bbox_xyxy'],xyxy(p['bbox']))) for p in allpred if iou(g['bbox_xyxy'],xyxy(p['bbox']))>=.5]
                if len(candidates)>1:overlap.append(dict(window_id=wid,frame_index=fi,audit_fruit_id=g['stable_fruit_id'],candidates=candidates))
        tp,assigned=assign_counts(counts,sorted(cohort))
        dtp,diagnostic_assignment=assign_counts(two_stage_counts,sorted(cohort))
        diagnostic_idtp+=dtp
        windows.append(dict(window_id=wid,reference_observations=ng,predicted_cohort_observations=npred,geometry_matches=nm,IDTP=tp,IDFP=npred-tp,IDFN=ng-tp,global_assignment=assigned,two_stage_diagnostic_IDTP=dtp,two_stage_diagnostic_assignment=diagnostic_assignment))
        totalgt+=ng;totalpred+=npred;idtp+=tp;matches+=nm
    def association(a,cross=False):
        fid=a['stable_fruit_id'];f,t=a['from_frame'],a['to_frame']
        if cross:
            ws=[w['window_id'] for w in audit['windows'] if f in w['frame_indices']];wt=[w['window_id'] for w in audit['windows'] if t in w['frame_indices']]
            if not ws or not wt:return None
            left=mapped.get((ws[0],f,fid));right=mapped.get((wt[0],t,fid))
        else:left=mapped.get((a['window_id'],f,fid));right=mapped.get((a['window_id'],t,fid))
        return dict(stable_fruit_id=fid,from_frame=f,to_frame=t,from_prediction_id=left,to_prediction_id=right,status='unobserved' if left is None or right is None else ('switch' if left!=right else 'consistent'))
    adjacent=[association(a) for a in audit['associations']];links=[association(a,True) for a in audit.get('cross_window_links',[])];links=[a for a in links if a]
    reviewed=sum(a['status']!='unobserved' for a in adjacent);switches=sum(a['status']=='switch' for a in adjacent)
    idfp=totalpred-idtp;idfn=totalgt-idtp;idf1=2*idtp/(2*idtp+idfp+idfn) if totalgt+totalpred else 0;idp=idtp/totalpred if totalpred else 0;idr=idtp/totalgt if totalgt else 0
    gates=dict(at_least_100_reviewed_adjacent_associations=reviewed>=100,eligible_observation_coverage_at_least_80pct=matches/totalgt>=.8 if totalgt else False,eligible_association_coverage_at_least_80pct=reviewed/len(adjacent)>=.8 if adjacent else False,IDF1_at_least_095=idf1>=.95,IDP_at_least_095=idp>=.95,IDR_at_least_095=idr>=.95,zero_observed_adjacent_switches=switches==0,zero_observed_explicit_cross_window_switches=all(a['status']!='switch' for a in links))
    return dict(schema='partial-reference-tracking-audit-v2',result_kind=kind,reference_type=audit['reference_type'],is_human_ground_truth=False,mask_accuracy_evaluated=False,exhaustive_coverage_claimed=False,matching='primary standard IDF1: all admissible trajectory-pair bbox IoU >= 0.5 counts then per-window global assignment; per-frame maximum-cardinality matching only for coverage/IDSW and named diagnostic; frame-active predeclared seeded IDs; excluded/unlisted pairs unknown and unscored',manual_interventions=project.get('manualInterventions', (project.get('model') or {}).get('manual_interventions', None)),reference_observations=totalgt,predicted_cohort_observations=totalpred,geometry_matches=matches,eligible_observation_coverage=matches/totalgt if totalgt else 0,IDTP=idtp,IDFP=idfp,IDFN=idfn,IDF1=idf1,IDP=idp,IDR=idr,two_stage_diagnostic=dict(IDTP=diagnostic_idtp,IDFP=totalpred-diagnostic_idtp,IDFN=totalgt-diagnostic_idtp,IDF1=2*diagnostic_idtp/(totalgt+totalpred) if totalgt+totalpred else 0,IDP=diagnostic_idtp/totalpred if totalpred else 0,IDR=diagnostic_idtp/totalgt if totalgt else 0),eligible_adjacent_associations=len(adjacent),reviewed_adjacent_associations=reviewed,adjacent_association_coverage=reviewed/len(adjacent) if adjacent else 0,observed_adjacent_switches=switches,windows=windows,adjacent_checks=adjacent,explicit_cross_window_checks=links,all_track_duplicate_overlap_candidates=overlap,gates=gates,all_gates_pass=all(gates.values()))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--audit-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--kind',choices=['raw','manually-corrected'],default='raw');a=p.parse_args()
    ah=hashlib.sha256(a.audit.read_bytes()).hexdigest()
    if ah!=a.audit_sha256:raise ValueError('Frozen audit SHA256 mismatch')
    project=json.loads(a.project.read_text());audit=json.loads(a.audit.read_text());result=evaluate(project,audit,a.kind)
    result.update(audit_sha256=ah,project_sha256=hashlib.sha256(a.project.read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ['IDF1','IDP','IDR','reviewed_adjacent_associations','observed_adjacent_switches','all_gates_pass']}))
if __name__=='__main__':main()
