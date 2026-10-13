#!/usr/bin/env python3
import copy,importlib.util,unittest
from pathlib import Path

def module(name,file):
    s=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=module('batch','batch-track.py');e=module('evaluation','evaluate-tracking.py')
class Tests(unittest.TestCase):
    def test_rle(self):
        self.assertEqual(b.geometry([1,2,1,2],2,3),(4,[0,1,2,2]))
        self.assertEqual(b.geometry([6],2,3),(0,[0,0,0,0]))
        for counts in [[7],[-1,7],[1.5,4.5]]:
            with self.assertRaises(ValueError):b.geometry(counts,2,3)
    def test_cardinality(self):
        gt=[{'bbox_xyxy':[0,0,10,10]},{'bbox_xyxy':[4,0,14,10]}]
        pred=[{'bbox':[2,0,10,10]},{'bbox':[0,0,10,10]}]
        self.assertEqual(len(e.match(gt,pred)),2)
    def fixture(self,swap=False,missing=False,unknown=False):
        obs=[dict(window_id='w',frame_index=f,stable_fruit_id=i,bbox_xyxy=[20*i,0,20*i+10,10]) for f in range(2) for i in [1,2]]
        audit=dict(targets=[dict(stable_fruit_id=i) for i in [1,2]],reference_type='assistant_visual_audit',windows=[dict(window_id='w',frame_indices=[0,1])],observations=obs,associations=[dict(window_id='w',stable_fruit_id=i,from_frame=0,to_frame=1) for i in [1,2]])
        fs=[]
        for f in range(2):
            instances=[dict(fruitId=3-i if swap and f else i,bbox=[20*i,0,10,10]) for i in [1,2] if not(missing and i==2 and f==1)]
            if unknown:instances.append(dict(fruitId=99,bbox=[100,0,10,10]))
            fs.append(dict(sourceFrameIndex=f,instances=instances))
        return dict(frames=fs,model={'manual_interventions':0}),audit
    def test_perfect_and_unknown(self):
        p,a=self.fixture(unknown=True);r=e.evaluate(p,a);self.assertEqual(r['IDF1'],1);self.assertEqual(r['predicted_cohort_observations'],4)
    def test_switch(self):
        p,a=self.fixture(swap=True);r=e.evaluate(p,a);self.assertEqual(r['observed_adjacent_switches'],2);self.assertEqual(r['IDF1'],.5)
    def test_missing(self):
        p,a=self.fixture(missing=True);r=e.evaluate(p,a);self.assertEqual(r['IDFN'],1);self.assertEqual(r['reviewed_adjacent_associations'],1)
    def test_duplicate(self):
        p,a=self.fixture();p['frames'][0]['instances'].append(dict(p['frames'][0]['instances'][0]));r=e.evaluate(p,a);self.assertEqual(r['IDFP'],1);self.assertEqual(len(r['all_track_duplicate_overlap_candidates']),1)
    def test_all_admissible_primary_not_two_stage(self):
        p,a=self.fixture()
        for o in a['observations']:
            x=2*(o['stable_fruit_id']-1);o['bbox_xyxy']=[x,0,x+10,10]
        for f in p['frames']:
            for d in f['instances']:
                x=2*((3-d['fruitId'] if f['sourceFrameIndex'] else d['fruitId'])-1)
                d['bbox']=[x,0,10,10]
        r=e.evaluate(p,a)
        self.assertEqual(r['IDF1'],1)
        self.assertEqual(r['two_stage_diagnostic']['IDF1'],.5)
        self.assertEqual(r['observed_adjacent_switches'],2)
    def test_unknown_is_not_known_absence(self):
        p,a=self.fixture()
        a['observations']=[o for o in a['observations'] if not(o['frame_index']==1 and o['stable_fruit_id']==2)]
        a['associations']=[x for x in a['associations'] if x['stable_fruit_id']!=2]
        unlisted=e.evaluate(p,a)
        self.assertEqual(unlisted['reference_observations'],3)
        self.assertEqual(unlisted['predicted_cohort_observations'],3)
        self.assertEqual(unlisted['IDF1'],1)
        a['exclusions']=[dict(window_id='w',frame_index=1,stable_fruit_id=2,bbox_xyxy=None)]
        excluded=e.evaluate(p,a)
        self.assertEqual(excluded['IDF1'],1)
        self.assertEqual(excluded['predicted_cohort_observations'],3)
    def test_duplicate_prediction_not_extra_idtp(self):
        p,a=self.fixture()
        p['frames'][0]['instances']*=2
        r=e.evaluate(p,a)
        self.assertEqual(r['IDTP'],4)
        self.assertEqual(r['IDFP'],2)
    def test_ambiguous_reference_rejected(self):
        cases=[]
        p,a=self.fixture();a['observations'].append(dict(a['observations'][0]));cases.append((p,a))
        p,a=self.fixture();a['exclusions']=[dict(window_id='w',frame_index=0,stable_fruit_id=1,bbox_xyxy=None)];cases.append((p,a))
        p,a=self.fixture();a['observations'][0]['stable_fruit_id']=99;cases.append((p,a))
        p,a=self.fixture();a['observations'][0]['window_id']='unknown';cases.append((p,a))
        p,a=self.fixture();a['observations'][0]['frame_index']=99;cases.append((p,a))
        p,a=self.fixture();a['windows'][0]['frame_indices']=[0,0,1];cases.append((p,a))
        p,a=self.fixture();a['associations'].append(dict(a['associations'][0]));cases.append((p,a))
        p,a=self.fixture();a['targets'].append(dict(a['targets'][0]));cases.append((p,a))
        p,a=self.fixture();p['frames'].append(copy.deepcopy(p['frames'][0]));cases.append((p,a))
        p,a=self.fixture();a['exclusions']=[dict(window_id='w',frame_index=0,stable_fruit_id=2,bbox_xyxy=None)]*2;cases.append((p,a))
        for p,a in cases:
            with self.subTest(audit=a):
                with self.assertRaises(ValueError):e.evaluate(p,a)
    def test_invalid_geometry_rejected(self):
        for bbox in [[0,0,float('nan'),10],[0,0,float('inf'),10],[0,0,0,10],[-1,0,10,10],[0,0,-1,10],[True,0,10,10],[0,0,10]]:
            p,a=self.fixture();p['frames'][0]['instances'][0]['bbox']=bbox
            with self.subTest(bbox=bbox):
                with self.assertRaises(ValueError):e.evaluate(p,a)
        for bbox in [[0,0,float('nan'),10],[5,0,4,10],[0,0,10,float('inf')]]:
            p,a=self.fixture();a['observations'][0]['bbox_xyxy']=bbox
            with self.assertRaises(ValueError):e.evaluate(p,a)
        p,a=self.fixture();p['frames'][0].update(width=10,height=10)
        with self.assertRaises(ValueError):e.evaluate(p,a)
    def test_frontend_manual_intervention_counter(self):
        p,a=self.fixture();p.update(model=None,manualInterventions=3)
        self.assertEqual(e.evaluate(p,a)['manual_interventions'],3)
        p['model']={'manual_interventions':1}
        self.assertEqual(e.evaluate(p,a)['manual_interventions'],3)
        del p['manualInterventions']
        self.assertEqual(e.evaluate(p,a)['manual_interventions'],1)
    def test_frontend_unloaded_frame(self):
        p,a=self.fixture()
        p['frames'].append(dict(sourceFrameIndex=2,width=0,height=0,instances=[]))
        self.assertEqual(e.evaluate(p,a)['IDF1'],1)
        for width,height in [(0,10),(False,False),(-1,0)]:
            p['frames'][-1].update(width=width,height=height)
            with self.assertRaises(ValueError):e.evaluate(p,a)
        p['frames'][-1].update(width=0,height=0,instances=[dict(fruitId=1,bbox=[0,0,1,1])])
        with self.assertRaises(ValueError):e.evaluate(p,a)
if __name__=='__main__':unittest.main()
