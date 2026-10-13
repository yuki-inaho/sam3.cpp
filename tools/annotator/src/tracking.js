/** One native object per sweep. Every frame commit is atomic and IDs come from
 * the explicit seed, never a nearest-box match. A missing detection stays unresolved. */
(function(global){
  'use strict';
  async function run({project,sequence,seeds,startFrame,bridge,backend,onFrame,onProgress,isCancelled}) {
    const sequenceLength=sequence.frames.length;
    if(!Number.isSafeInteger(sequenceLength)||sequenceLength<1||sequenceLength>2147483647||project.frames.length!==sequenceLength||!Number.isSafeInteger(startFrame)||startFrame<0||startFrame>=sequenceLength)throw new Error('Invalid tracking sequence bounds');
    // Native temporal PE uses the ordinal in this initialized sequence, not an
    // external sourceFrameIndex (which may start at e.g. 400).
    const trackingConfig=typeof bridge.samTrackConfig==='function'?{sequenceLength,sourceStartFrame:startFrame}:null;
    const seedPath=sequence.frames[startFrame].path;
    for (const seed of seeds) {
      if(isCancelled())return;
      onProgress(seed.fruitId,startFrame);
      await bridge.samTrackReset();
      if(isCancelled())return;
      await bridge.samLoadImage(seedPath);
      if(isCancelled())return;
      if(trackingConfig) {
        const configured=await bridge.samTrackConfig(sequenceLength,startFrame);
        if(isCancelled())return;
        if(configured?.event!=='track_configured'||configured.sequence_length!==sequenceLength||configured.source_start_frame!==startFrame)throw new Error('Native tracking configuration was not acknowledged');
      }
      const added=await bridge.samTrackAdd(seed.fruitId,seed.prompt);
      if(isCancelled())return;
      if(added.object_id!==undefined && added.object_id!==seed.fruitId)throw new Error('Native seed returned a different Fruit ID');
      let segmentId=seed.segmentId, currentlyMissing=false, hadUnresolvedGap=seed.identityStatus==='unresolved';
      for(let index=startFrame+1;index<sequence.frames.length;index++) {
        if(isCancelled())return;
        onProgress(seed.fruitId,index);
        const res=await bridge.samTrackFrame(sequence.frames[index].path);
        if(isCancelled())return;
        const frame=project.frames[index], w=res.width, h=res.height;
        if(!Number.isSafeInteger(w)||!Number.isSafeInteger(h)||w<1||h<1||w*h>64*1024*1024)throw new Error('Invalid tracking image dimensions');
        if(frame.instances.length&&(frame.width!==w||frame.height!==h))throw new Error('Tracking dimensions differ from saved annotations');
        if(!Array.isArray(res.detections)||res.detections.length>1)throw new Error('Expected one native object per sweep');
        if(res.frame_index!==undefined&&res.frame_index!==index-startFrame)throw new Error('Native frame index is out of sequence');
        let result=null;
        if(res.detections.length) {
          const d=res.detections[0],id=d.object_id??d.instance_id;
          if(id!==seed.fruitId)throw new Error('Native tracking returned a different Fruit ID');
          if(!Array.isArray(d.mask_size)||d.mask_size[0]!==w||d.mask_size[1]!==h||!Number.isFinite(d.score))throw new Error('Invalid native detection');
          const mask=global.RLE.decode(d.counts,w,h);
          if(!mask)throw new Error('Invalid native RLE');
          const area=global.RLE.area(mask);
          if(area) {
            if(currentlyMissing)segmentId=`${id}:frame-${frame.sourceFrameIndex??index}`;
            result={id,fruitId:id,segmentId,label:seed.label,counts:[...d.counts],maskWidth:w,maskHeight:h,score:d.score,
              source:'native-video-propagation',provenance:{model:backend?JSON.parse(JSON.stringify(backend)):null,method:'native-video-propagation',seedFruitId:seed.fruitId,seedFrameKey:project.frames[startFrame].key,trackingConfig:trackingConfig&&{...trackingConfig}},reviewStatus:'draft',identityStatus:hadUnresolvedGap?'unresolved':'unverified',
              visible:true,colourIndex:seed.colourIndex,area,bbox:global.RLE.bbox(mask,w,h),prompt:null};
          }
        }
        // Commit only after the complete response passes validation.
        frame.width=w;frame.height=h;
        const unresolved=new Set(frame.unresolvedFruitIds||[]);
        if(result) {
          if(!frame.instances.some(i=>i.fruitId===seed.fruitId))frame.instances.push(result);
          unresolved.delete(seed.fruitId);currentlyMissing=false;
        } else { unresolved.add(seed.fruitId);currentlyMissing=true;hadUnresolvedGap=true; }
        frame.unresolvedFruitIds=[...unresolved];
        await onFrame(index,seed.fruitId);
      }
    }
  }
  global.AnnotationTracking={run};
})(window);
