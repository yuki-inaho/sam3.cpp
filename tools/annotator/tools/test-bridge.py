"""Synthetic bridge contract/security tests; never load model weights or private data."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

spec = importlib.util.spec_from_file_location('bridge', Path(__file__).with_name('serve-browser.py'))
bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)

class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.model = self.root / 'model.gguf'; self.model.write_bytes(b'synthetic-not-model')
        self.image = self.root / 'frame.png'; self.image.write_bytes(b'synthetic-image')
        self.exe = self.root / 'native-fixture'
        self.exe.write_text('#!' + sys.executable + '''
import json,sys,time
print(json.dumps({'ok':True,'event':'ready','video_tracking':True,'compute':'test-fixture','preprocessing':'synthetic','build_identity':'fixture-build','runtime_provenance':{'objects':[{'path':'/synthetic/library.so','sha256':'fixture-sha'}]}}),flush=True)
for line in sys.stdin:
 c=line.strip()
 if c=='partial':
  sys.stdout.write('{"ok":');sys.stdout.flush();time.sleep(30)
 elif c=='hang':time.sleep(30)
 elif c.startswith('track_add '): print(json.dumps({'ok':True,'object_id':int(c.split()[1])}),flush=True)
 elif c.startswith('track_config '): print(json.dumps({'ok':True,'event':'track_configured','sequence_length':int(c.split()[1]),'source_start_frame':int(c.split()[2])}),flush=True)
 elif c=='cache_stats': print(json.dumps({'ok':True,'event':'cache_stats','enabled':'--efficient-prefix-cache' in sys.argv,'ram_hits':1,'disk_hits':2,'misses':3,'invalid_entries':0,'writes':2,'evictions':0,'resident_bytes':51380224,'disk_bytes':102760448}),flush=True)
 else: print(json.dumps({'ok':True,'command':c}),flush=True)
''')
        self.exe.chmod(0o700)
        self.native = bridge.Native(self.exe, [str(self.model)], [str(self.image)], timeout=.15, annotation_root=self.root)
    def tearDown(self): self.native.stop(); self.temp.cleanup()
    def start(self): return self.native.invoke('sam_start', {'modelPath':str(self.model),'cpu':True})
    def test_explicit_allowlist_and_symlink_change(self):
        for path in (str(self.root/'../secret'), '/etc/passwd', str(self.root/'frame.png/../frame.png')):
            with self.assertRaises(ValueError): self.native.invoke('read_image', {'path':path})
        self.image.unlink(); self.image.symlink_to(self.model)
        with self.assertRaises(ValueError): self.native.invoke('read_image', {'path':str(self.image)})
        self.model.unlink(); self.model.symlink_to(self.exe)
        with self.assertRaises(ValueError): self.start()
    def test_cpu_and_prompt_validation(self):
        with self.assertRaises(ValueError): self.native.invoke('sam_start', {'modelPath':str(self.model),'cpu':False})
        for args in ({'pos':[[float('nan'),1]]},{'bbox':[2,2,1,1]},{'neg':[[1,1]]},{'pos':[[True,1]]}):
            with self.assertRaises(ValueError): bridge.prompts(args)
    def test_tracking_capability_and_stable_id(self):
        self.start(); self.native.capabilities['video_tracking'] = False
        with self.assertRaises(ValueError): self.native.invoke('sam_track_frame', {'path':str(self.image)})
        self.native.capabilities['video_tracking'] = True
        result = self.native.invoke('sam_track_add', {'objectId':121,'pos':[[1,1]]})
        self.assertEqual(result['object_id'],121)
        for oid in (0,-1,True,2147483648,'1'):
            with self.assertRaises(ValueError): self.native.invoke('sam_track_add', {'objectId':oid,'pos':[[1,1]]})
    def test_partial_line_deadline(self):
        self.start(); started=time.monotonic()
        with self.assertRaisesRegex(ValueError,'timed out'): self.native.command('partial')
        self.assertLess(time.monotonic()-started, 2); self.assertIsNone(self.native.process)
    def test_track_config_bounds_and_streaming_default(self):
        self.start()
        for length, start in ((2,1),(121,120),(0,0),(0,17)):
            result=self.native.invoke('sam_track_config',{'sequenceLength':length,'sourceStartFrame':start})
            self.assertEqual(result['sequence_length'],length);self.assertEqual(result['source_start_frame'],start)
        for length, start in ((-1,0),(2,-1),(2,2),(2,401),(2,True),(True,0),(2,1.5),('2',1),(2147483648,0),(0,2147483648)):
            with self.assertRaisesRegex(ValueError,'bounds'):
                self.native.invoke('sam_track_config',{'sequenceLength':length,'sourceStartFrame':start})
        self.native.capabilities['video_tracking']=False
        with self.assertRaisesRegex(ValueError,'video tracking'):
            self.native.invoke('sam_track_config',{'sequenceLength':2,'sourceStartFrame':1})
    def test_track_config_requires_exact_acknowledgement(self):
        self.start();self.native.command=lambda command:{'ok':True,'event':'track_configured','sequence_length':99,'source_start_frame':1}
        with self.assertRaisesRegex(ValueError,'not acknowledged'):
            self.native.invoke('sam_track_config',{'sequenceLength':2,'sourceStartFrame':1})
    def test_ready_retains_runtime_provenance(self):
        ready=self.start()
        self.assertEqual(ready['preprocessing'],'synthetic');self.assertEqual(ready['build_identity'],'fixture-build')
        self.assertEqual(ready['runtime_provenance']['objects'][0]['sha256'],'fixture-sha')
    def test_cache_and_q8_disabled_by_default(self):
        ready=self.start()
        for flag in ('--efficient-prefix-cache','--prefix-cache-dir','--prefix-cache-budget-mib','--experimental-efficient-q8-0'):
            self.assertNotIn(flag,self.native.process.args)
        self.assertEqual(ready['cache_config'],{'requested':False,'persistent':False,'budget_mib':None})
        self.assertFalse(self.native.invoke('sam_cache_stats',{})['enabled'])
    def test_cli_only_ram_cache_and_q8_flags(self):
        self.native=bridge.Native(self.exe,[str(self.model)],[str(self.image)],prefix_cache=True,experimental_q8=True)
        ready=self.start()
        self.assertIn('--efficient-prefix-cache',self.native.process.args);self.assertIn('--experimental-efficient-q8-0',self.native.process.args)
        self.assertNotIn('--prefix-cache-dir',self.native.process.args)
        self.assertEqual(ready['cache_config'],{'requested':True,'persistent':False,'budget_mib':None})
        stats=self.native.invoke('sam_cache_stats',{});self.assertTrue(stats['enabled']);self.assertEqual(stats['disk_hits'],2)
        for args in ({'prefixCacheDir':'/tmp'},{'prefixCacheBudgetMiB':999},{'experimentalQ8':True}):
            with self.assertRaisesRegex(ValueError,'CLI-only'): self.native.invoke('sam_start',{'modelPath':str(self.model),'cpu':True,**args})
        with self.assertRaisesRegex(ValueError,'no arguments'): self.native.invoke('sam_cache_stats',{'command':'quit'})
        with self.assertRaisesRegex(ValueError,'unsupported'): self.native.invoke('cache_clear',{})
    def test_persistent_cache_argv_budget_and_no_path_in_provenance(self):
        cache=self.root/'cache';cache.mkdir()
        self.native=bridge.Native(self.exe,[str(self.model)],[str(self.image)],prefix_cache_dir=cache,prefix_cache_budget_mib=8192)
        ready=self.start();argv=self.native.process.args
        self.assertEqual(argv[argv.index('--prefix-cache-dir')+1],str(cache))
        self.assertEqual(argv[argv.index('--prefix-cache-budget-mib')+1],'8192')
        self.assertEqual(ready['cache_config'],{'requested':True,'persistent':True,'budget_mib':8192})
        self.assertNotIn(str(cache),json.dumps(ready['cache_config']))
    def test_cache_bounds_git_and_symlink_rejected(self):
        cache=self.root/'cache';cache.mkdir()
        for budget in (None,0,-1,True,1.5,'8192',1048577):
            with self.assertRaises(ValueError): bridge.cache_options(False,cache,budget)
        with self.assertRaises(ValueError): bridge.cache_options(True,None,8192)
        with self.assertRaises(ValueError): bridge.cache_options('yes')
        with self.assertRaises(ValueError): bridge.Native(self.exe,[],[],experimental_q8='yes')
        with self.assertRaises(FileNotFoundError): bridge.cache_options(False,self.root/'missing',8192)
        alias=self.root/'alias';alias.symlink_to(cache,target_is_directory=True)
        with self.assertRaises(ValueError): bridge.cache_options(False,alias,8192)
        with self.assertRaises(ValueError): bridge.cache_options(False,cache/'..'/'cache',8192)
        repo=self.root/'repo';repo.mkdir();(repo/'.git').mkdir();(repo/'.git'/'HEAD').write_text('ref: refs/heads/test\n');inside=repo/'cache';inside.mkdir()
        with self.assertRaisesRegex(ValueError,'outside Git'): bridge.cache_options(False,inside,8192)
        self.assertEqual(bridge.cache_options(False,cache,1048576)[2],1048576)
    def test_cache_directory_swap_rejected_before_restart(self):
        cache=self.root/'cache';cache.mkdir();replacement=self.root/'other';replacement.mkdir()
        self.native.prefix_cache,self.native.prefix_cache_dir,self.native.prefix_cache_budget_mib=bridge.cache_options(False,cache,8192)
        cache.rmdir();cache.symlink_to(replacement,target_is_directory=True)
        with self.assertRaises(ValueError): self.start()
        self.assertIsNone(self.native.process)
    def test_cache_stats_rejects_invalid_or_unsupported_native_response(self):
        self.start();stats=self.native.invoke('sam_cache_stats',{})
        for key,value in (('enabled',1),('ram_hits',-1),('disk_bytes',9007199254740992),('writes',True),('event','segment')):
            self.native.command=lambda command,key=key,value=value:{**stats,key:value}
            with self.assertRaisesRegex(ValueError,'invalid native cache'): self.native.invoke('sam_cache_stats',{})
        def unsupported(command): raise ValueError('unsupported legacy command')
        self.native.command=unsupported
        with self.assertRaisesRegex(ValueError,'unsupported legacy'): self.native.invoke('sam_cache_stats',{})
        self.assertTrue(self.native.invoke('sam_running',{}))
    def test_cancel_hung_then_restart_no_stale_reply(self):
        self.native.timeout=30
        for _ in range(3):
            self.start(); errors=[]
            worker=threading.Thread(target=lambda:self.capture(errors,lambda:self.native.command('hang')))
            worker.start(); time.sleep(.05); started=time.monotonic(); self.native.invoke('sam_stop',{})
            worker.join(2); self.assertFalse(worker.is_alive()); self.assertTrue(errors)
            self.assertLess(time.monotonic()-started,2)
            self.start(); self.assertEqual(self.native.command('reset')['command'],'reset')
    @staticmethod
    def capture(errors, fn):
        try: fn()
        except Exception as error: errors.append(error)
    def test_catalog_excludes_symlinks(self):
        (self.root/'alias.png').symlink_to(self.image)
        images,sequences=bridge.catalog([self.root])
        self.assertEqual(images,[str(self.image)]); self.assertEqual(sequences[0]['frames'][0]['key'],'frame.png')
    def test_save_atomic_no_follow_client_path(self):
        target=self.root/'annotation-project.json'; target.symlink_to(self.model)
        result=self.native.invoke('save_project',{'project':{'schema':'sam3-annotation-project','version':1},'path':'/etc/passwd'})
        self.assertTrue(result['saved']); self.assertFalse(target.is_symlink()); self.assertEqual(self.model.read_bytes(),b'synthetic-not-model')
    def test_saved_project_read_uses_only_fixed_private_file(self):
        project={'schema':'sam3-annotation-project','version':1,'frames':[]}
        self.native.invoke('save_project',{'project':project})
        self.assertEqual(self.native.invoke('load_saved_project',{}),project)
        target=self.root/'annotation-project.json'; target.unlink(); target.symlink_to(self.model)
        with self.assertRaises(ValueError): self.native.invoke('load_saved_project',{})
    def test_research_video_flag_is_explicit_only(self):
        self.start(); self.assertNotIn('--experimental-efficient-video',self.native.process.args)
        self.native.stop(); self.native.experimental_video=True; self.start()
        self.assertIn('--experimental-efficient-video',self.native.process.args)
    def test_http_boundaries_assets_and_origin(self):
        server=bridge.ThreadingHTTPServer(('127.0.0.1',0),bridge.Handler)
        port=server.server_address[1]; origin=f'http://127.0.0.1:{port}'
        server.allowed_hosts={f'127.0.0.1:{port}'};server.origins={origin};server.native=self.native;server.config={}
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            with urlopen(origin+'/project.js') as response: self.assertEqual(response.status,200)
            for path in ('/../model.gguf','/%2e%2e/model.gguf','/api/invoke'):
                with self.assertRaises(HTTPError): urlopen(origin+path)
            payload=json.dumps({'command':'sam_running','args':{}}).encode()
            for bad_origin in (None,'http://evil.test','null'):
                headers={'Content-Type':'application/json'}
                if bad_origin: headers['Origin']=bad_origin
                with self.assertRaises(HTTPError) as err: urlopen(Request(origin+'/api/invoke',data=payload,headers=headers))
                self.assertEqual(err.exception.code,403)
            with urlopen(Request(origin+'/api/invoke',data=payload,headers={'Content-Type':'application/json','Origin':origin})) as response:
                self.assertEqual(json.load(response),{'result':False})
            self.start()
            for start in (1,401):
                config=json.dumps({'command':'sam_track_config','args':{'sequenceLength':2,'sourceStartFrame':start}}).encode()
                request=Request(origin+'/api/invoke',data=config,headers={'Content-Type':'application/json','Origin':origin})
                if start==1:
                    with urlopen(request) as response: self.assertEqual(json.load(response)['result']['source_start_frame'],1)
                else:
                    with self.assertRaises(HTTPError) as err: urlopen(request)
                    self.assertEqual(err.exception.code,400)
            with self.assertRaises(HTTPError) as err: urlopen(Request(origin+'/',headers={'Host':'evil.test'}))
            self.assertEqual(err.exception.code,403)
        finally: server.shutdown();server.server_close();worker.join()

if __name__ == '__main__': unittest.main(verbosity=2)
