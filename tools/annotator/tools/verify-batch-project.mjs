#!/usr/bin/env node
/** Validate persistence and actual frontend COCO round-trip without inference. */
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import assert from 'node:assert/strict';
const root=path.resolve(path.dirname(new URL(import.meta.url).pathname),'../src');
const c=vm.createContext({window:{}});
for(const f of ['rle.js','project.js','coco.js'])vm.runInContext(fs.readFileSync(path.join(root,f),'utf8'),c);
const input=process.argv[2];if(!input)throw new Error('Provide annotation-project.json');
const raw=JSON.parse(fs.readFileSync(input,'utf8'));
const project=c.window.AnnotationProject.validate(raw);
const reload=c.window.AnnotationProject.validate(JSON.parse(JSON.stringify(c.window.AnnotationProject.serialise(project))));
const coco=c.window.COCO.build(project);
const back=c.window.AnnotationProject.fromCoco(JSON.parse(JSON.stringify(coco)));
const plain=x=>JSON.parse(JSON.stringify(c.window.AnnotationProject.serialise(x)));
assert.deepEqual(plain(project),plain(reload));assert.deepEqual(plain(project),plain(back));
const out=input.replace(/\.json$/,'.coco.json');fs.writeFileSync(out,JSON.stringify(coco));
console.log(JSON.stringify({status:'pass',frames:project.frames.length,instances:project.frames.reduce((a,f)=>a+f.instances.length,0),unique_ids_per_frame:true,geometry_rle_valid:true,persistence_roundtrip:true,frontend_coco_roundtrip:true,coco_path:out}));
