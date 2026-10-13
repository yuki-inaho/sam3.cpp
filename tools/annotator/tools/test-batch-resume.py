#!/usr/bin/env python3
"""Deterministic fake-native lifecycle test; never starts a model."""
import json,subprocess,tempfile,unittest,sys
from pathlib import Path
from PIL import Image
class Resume(unittest.TestCase):
    def test_completed_and_partial_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);images=r/'images';images.mkdir();out=r/'results';model=r/'model';model.write_text('test model placeholder')
            for f in range(3):Image.new('RGB',(4,3)).save(images/f'{f:08}.jpg')
            seeds=r/'seeds.json';seeds.write_text(json.dumps({'seeds':[dict(proposed_fruit_id=1,frame_index=0,source_filename='00000000.jpg',bbox_xyxy=[0,0,2,2])]}))
            native=r/'native';native.write_text('''#!/usr/bin/env python3
import json,sys
print(json.dumps({'ok':True,'video_tracking':True}),flush=True)
i=0;fid=1
for line in sys.stdin:
 cmd=line.strip().split(' ',1);r={'ok':True}
 if cmd[0]=='track_add':fid=int(cmd[1]);i=0
 if cmd[0]=='track_frame':i+=1
 if cmd[0] in ('track_add','track_frame'):
  r.update(width=4,height=3,frame_index=i,detections=[{'object_id':fid,'score':.9,'mask_size':[4,3],'counts':[0,2,1,2,7]}])
 print(json.dumps(r),flush=True)
''');native.chmod(0o700)
            script=Path(__file__).with_name('batch-track.py')
            args=[sys.executable,str(script),'--serve',str(native),'--model',str(model),'--images',str(images),'--seeds',str(seeds),'--output',str(out),'--first','0','--last','2','--run']
            subprocess.run(args,check=True,capture_output=True);tp=out/'tracks/0001.json';complete=tp.read_bytes()
            subprocess.run(args,check=True,capture_output=True);self.assertEqual(tp.read_bytes(),complete)
            track=json.loads(complete);track['status']='running';track['results']=track['results'][:1];tp.write_text(json.dumps(track))
            subprocess.run(args,check=True,capture_output=True);replayed=json.loads(tp.read_text());self.assertEqual(replayed['status'],'complete');self.assertEqual(len(replayed['results']),3)
            p=json.loads((out/'annotation-project.json').read_text());self.assertEqual([f['instances'][0]['fruitId'] for f in p['frames']],[1,1,1])
            for x,y in zip(json.loads(complete)['results'],replayed['results']):self.assertEqual(x['instance'],y['instance'])
            # A changed model cannot reuse existing checkpoints.
            model.write_text('changed model');bad=subprocess.run(args,check=False,capture_output=True);self.assertNotEqual(bad.returncode,0);self.assertIn(b'Manifest differs',bad.stderr)
            # Unknown seed IDs fail during preparation.
            bad=subprocess.run(args+['--seed-ids','99'],check=False,capture_output=True);self.assertNotEqual(bad.returncode,0);self.assertIn(b'Unknown/empty seed',bad.stderr)
            subprocess.run(['node',str(Path(__file__).with_name('verify-batch-project.mjs')),str(out/'annotation-project.json')],check=True,capture_output=True)
if __name__=='__main__':unittest.main()
