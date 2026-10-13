#!/usr/bin/env python3
"""Private, resumable single-object native video sweeps through the frontend bridge.

Preparation is default. --run is an explicit inference opt-in. Completed tracks
are reused; interrupted tracks replay from their seed to reconstruct native
memory. No tensor dumps, model substitutions, identity merges, or corrections.
Dependencies: Pillow; bridge is imported from the adjacent serve-browser.py.
"""
import argparse, fcntl, hashlib, importlib.util, json, math, os, tempfile, time
from pathlib import Path
from PIL import Image

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.checkpoint-')
    try:
        with os.fdopen(fd,'w') as f: json.dump(data,f,allow_nan=False,separators=(',',':'));f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def geometry(counts,w,h):
    if not isinstance(counts,list) or any(type(n)!=int or n<0 for n in counts) or sum(counts)!=w*h:raise ValueError('Invalid RLE')
    p=area=0; x0,y0,x1,y1=w,h,-1,-1
    for i,n in enumerate(counts):
        if i%2 and n:
            area+=n;a,b=p,p+n-1; xa,ya=divmod(a,h);xb,yb=divmod(b,h)
            x0=min(x0,xa);x1=max(x1,xb)
            y0=min(y0,ya if xa==xb else 0);y1=max(y1,yb if xa==xb else h-1)
        p+=n
    return area,([x0,y0,x1-x0+1,y1-y0+1] if area else [0,0,0,0])

def instance(d,fid,seg,w,h,prompt=None,unresolved=False):
    if d.get('object_id',d.get('instance_id'))!=fid:raise ValueError('Native identity mismatch')
    if d.get('mask_size')!=[w,h] or not math.isfinite(d.get('score',float('nan'))):raise ValueError('Native dimensions/score mismatch')
    area,bbox=geometry(d['counts'],w,h)
    if not area:return None
    return dict(id=fid,fruitId=fid,segmentId=seg,label='tomato',counts=d['counts'],maskWidth=w,maskHeight=h,score=d['score'],source='native-video-seed' if prompt else 'native-video-propagation',reviewStatus='draft',identityStatus='unresolved' if unresolved else 'unverified',visible=True,colourIndex=(fid-1)%8,area=area,bbox=bbox,prompt=prompt)

def assemble(manifest,root):
    frames=[dict(key=f['fileName'],fileName=f['fileName'],index=i,sourceFrameIndex=i,width=f['width'],height=f['height'],instances=[],unresolvedFruitIds=[]) for i,f in enumerate(manifest['frames'])]
    for seed in manifest['seeds']:
        p=root/'tracks'/f"{seed['proposed_fruit_id']:04}.json"
        if not p.exists():continue
        track=json.loads(p.read_text())
        for r in track['results']:
            f=frames[r['frame_index']]
            if r['instance']:f['instances'].append(r['instance'])
            else:f['unresolvedFruitIds'].append(seed['proposed_fruit_id'])
    for f in frames:
        if len({i['fruitId'] for i in f['instances']})!=len(f['instances']):raise ValueError('Duplicate ID')
    project=dict(schema='sam3-annotation-project',version=1,sequenceId=manifest['sequence_id'],nextFruitId=max(s['proposed_fruit_id'] for s in manifest['seeds'])+1,model=dict(name=Path(manifest['model']).name,sha256=manifest['model_sha256'],device='cpu',native_executable_sha256=manifest['executable_sha256'],quantization_note='Model storage type does not assert native INT8 execution',raw_prediction=True,experimentalVideo=manifest['experimental_video'],manual_interventions=0),frames=frames)
    write(root/'annotation-project.json',project)
    return project

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['serve','model','images','seeds','output']:ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--first',type=int,required=True);ap.add_argument('--last',type=int,required=True)
    ap.add_argument('--seed-ids',help='comma-separated proposed IDs; default all')
    ap.add_argument('--max-frames',type=int,help='maximum frames per sweep, including seed')
    ap.add_argument('--threads',type=int,default=4);ap.add_argument('--timeout',type=int,default=600)
    ap.add_argument('--experimental-efficient-video',action='store_true',help='Explicit unvalidated research capability opt-in')
    ap.add_argument('--run',action='store_true');args=ap.parse_args()
    root=args.output.resolve();repo=Path(__file__).resolve().parents[3]
    if root.is_relative_to(repo):ap.error('Private output must be outside repository')
    if args.max_frames is not None and args.max_frames<1:ap.error('--max-frames must be positive')
    if not 1<=args.threads<=32 or args.timeout<=0:ap.error('Invalid thread count or timeout')
    if args.first<0 or args.last<args.first:ap.error('Invalid source frame range')
    root.mkdir(parents=True,exist_ok=True)
    output_lock=open(root/'.runner.lock','a')
    try:fcntl.flock(output_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('Another runner owns this output directory')
    seeds=json.loads(args.seeds.read_text())['seeds'];ids={int(i) for i in args.seed_ids.split(',')} if args.seed_ids else None
    if ids is not None:seeds=[s for s in seeds if s['proposed_fruit_id'] in ids]
    if not seeds or (ids and {s['proposed_fruit_id'] for s in seeds}!=ids):ap.error('Unknown/empty seed selection')
    frames=[]
    for n in range(args.first,args.last+1):
        p=(args.images/f'{n:08}.jpg').resolve(strict=True)
        with Image.open(p) as im:w,h=im.size
        frames.append(dict(path=str(p),fileName=p.name,width=w,height=h,sha256=sha(p)))
    if len({s['proposed_fruit_id'] for s in seeds})!=len(seeds):raise ValueError('Duplicate seed fruit IDs')
    for s in seeds:
        if type(s['proposed_fruit_id']) is not int or not 1<=s['proposed_fruit_id']<=2147483647:raise ValueError('Invalid seed ID')
        if type(s['frame_index']) is not int or not 0<=s['frame_index']<len(frames) or frames[s['frame_index']]['fileName']!=s['source_filename']:raise ValueError('Seed/frame mismatch')
        box=s['bbox_xyxy'];f=frames[s['frame_index']]
        if not isinstance(box,list) or len(box)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in box):raise ValueError('Invalid seed geometry')
        if not 0<=box[0]<box[2]<=f['width'] or not 0<=box[1]<box[3]<=f['height']:raise ValueError('Out-of-frame or inverted seed box')
    manifest=dict(version=1,sequence_id=f'frames_{args.first:08}_{args.last:08}',frames=frames,seeds=seeds,seed_file_sha256=sha(args.seeds),model=str(args.model.resolve(strict=True)),model_sha256=sha(args.model),executable=str(args.serve.resolve(strict=True)),executable_sha256=sha(args.serve),threads=args.threads,experimental_video=args.experimental_efficient_video,max_frames=args.max_frames,resume_policy='replay incomplete tracks from original seed; skip completed tracks')
    mp=root/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Manifest differs: choose a new output directory')
    write(mp,manifest)
    work=sum(min(len(frames)-s['frame_index'],args.max_frames or len(frames)) for s in seeds)
    write(root/'plan.json',dict(tracks=len(seeds),frame_inferences=work,seed_inferences=len(seeds),propagated_frames=work-len(seeds),full_sequence_frames=len(frames),inference_authorized_by_cli=args.run,exhaustive=False))
    print(json.dumps(dict(status='prepared',tracks=len(seeds),frame_inferences=work)),flush=True)
    if not args.run:return
    spec=importlib.util.spec_from_file_location('native_bridge',Path(__file__).with_name('serve-browser.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    native=mod.Native(manifest['executable'],[manifest['model']],[f['path'] for f in frames],timeout=args.timeout,experimental_video=args.experimental_efficient_video)
    try:
        started=time.monotonic();caps=native.invoke('sam_start',dict(modelPath=manifest['model'],cpu=True,nThreads=args.threads))
        write(root/'capabilities.json',caps)
        if caps.get('video_tracking') is not True:raise ValueError('Native video capability unavailable; explicit research opt-in may be required')
        for seed in seeds:
            fid=seed['proposed_fruit_id'];tp=root/'tracks'/f'{fid:04}.json'
            if tp.exists() and json.loads(tp.read_text()).get('status')=='complete':continue
            start=seed['frame_index'];end=min(len(frames),start+(args.max_frames or len(frames)))
            track=dict(fruit_id=fid,status='running',results=[],replay_from_seed=True,manual_interventions=0)
            native.invoke('sam_track_reset',{});t=time.monotonic();native.invoke('sam_load_image',dict(path=frames[start]['path']))
            prompt=dict(pos=[],neg=[],box=seed['bbox_xyxy']);res=native.invoke('sam_track_add',dict(objectId=fid,**prompt));seed_seconds=time.monotonic()-t
            seg=f'{fid}:frame-{start}';missing=False;unresolved=False
            for index in range(start,end):
                if index>start:
                    t=time.monotonic();res=native.invoke('sam_track_frame',dict(path=frames[index]['path']));elapsed=time.monotonic()-t
                else:elapsed=seed_seconds
                w,h=frames[index]['width'],frames[index]['height']
                if res.get('width')!=w or res.get('height')!=h:raise ValueError('Frame dimensions mismatch')
                if res.get('frame_index',index-start)!=index-start:raise ValueError('Native frame order mismatch')
                ds=res.get('detections',[])
                if not isinstance(ds,list) or len(ds)>1:raise ValueError('Expected single-object sweep')
                if missing:seg=f'{fid}:frame-{index}';unresolved=True
                inst=instance(ds[0],fid,seg,w,h,prompt if index==start else None,unresolved) if ds else None
                missing=inst is None
                track['results'].append(dict(frame_index=index,seconds=elapsed,status='prediction' if inst else 'unresolved',instance=inst))
                write(tp,track);write(root/'status.json',dict(status='running',fruit_id=fid,frame_index=index,seconds=elapsed,elapsed_seconds=time.monotonic()-started))
                print(json.dumps(dict(fruit_id=fid,frame_index=index,seconds=round(elapsed,3),status=track['results'][-1]['status'])),flush=True)
            track['status']='complete';write(tp,track);assemble(manifest,root)
        assemble(manifest,root);write(root/'status.json',dict(status='complete',elapsed_seconds=time.monotonic()-started))
    except BaseException as e:
        assemble(manifest,root);write(root/'status.json',dict(status='interrupted',error=str(e)));raise
    finally:native.stop()
if __name__=='__main__':main()
