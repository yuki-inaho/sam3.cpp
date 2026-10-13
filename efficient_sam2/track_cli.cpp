// Explicit experimental temporal-parity runner. This program is not a claim
// that a model meets quality gates; it records the actual native predictions.
#include "sam3.h"
#include "efficient_sam2/prefix_cache/digest.h"
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
namespace fs=std::filesystem;
int main(int argc,char**argv) {
    if(argc!=9) {
        fprintf(stderr,"Usage: %s model.gguf frames_dir output_dir x0 y0 x1 y1 max_frames\n",argv[0]);
        return 2;
    }
    sam3_params params;
    params.model_path=argv[1]; params.use_gpu=false; params.n_threads=4;
    params.experimental_efficient_video=true;
    params.efficient_video_preprocessing=true;
    params.experimental_efficient_q8_0=std::getenv("EFFICIENT_SAM2_Q8_0")!=nullptr;
    params.efficient_prefix_cache=std::getenv("EFFICIENT_SAM2_PREFIX_CACHE")!=nullptr;
    params.efficient_fpn_cache=std::getenv("EFFICIENT_SAM2_FPN_CACHE")!=nullptr;
    if(const char* directory=std::getenv("EFFICIENT_SAM2_PREFIX_CACHE_DIR")) {
        params.efficient_prefix_cache=true; params.efficient_cache_directory=directory;
    }
    if(const char* budget=std::getenv("EFFICIENT_SAM2_PREFIX_CACHE_BUDGET_MIB")) {
        const uint64_t mib=std::strtoull(budget,nullptr,10);
        if(mib>1024*1024) return 2;
        params.efficient_cache_disk_budget=mib*1024*1024;
    }
    const auto load_started=std::chrono::steady_clock::now();
    auto model=sam3_load_model(params);
    const double load_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-load_started).count();
    const auto setup_started=std::chrono::steady_clock::now();
    if(!model) return 1;
    auto state=sam3_create_state(*model,params);
    sam3_visual_track_params vp; vp.fill_hole_area=0; vp.recondition_every=0;
    if(!state) return 1;
    std::vector<fs::path> frames;
    for(const auto& p:fs::directory_iterator(argv[2])) {
        auto extension=p.path().extension().string();
        if(extension==".jpg" || extension==".png") frames.push_back(p.path());
    }
    std::sort(frames.begin(),frames.end());
    const size_t limit=std::min(frames.size(),size_t(std::atoi(argv[8])));
    if(!limit) return 2;
    vp.efficient_sequence_length=frames.size();
    auto tracker=sam3_create_visual_tracker(*model,vp);
    if(!tracker) return 1;
    const fs::path output=argv[3];
    fs::create_directories(output);
    std::string digest_error;
    const auto model_digest=efficient_sam2::prefix_cache::sha256_file(argv[1],digest_error);
    if(model_digest.empty()) { fprintf(stderr,"Cannot fingerprint model: %s\n",digest_error.c_str()); return 1; }
    const double setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup_started).count();
    auto write_metadata=[&]() {
    const auto pc=sam3_get_prefix_cache_statistics(*model);
    const auto fc=sam3_get_fpn_cache_statistics(*model);
    std::ofstream metadata(output/"runtime.json");
    metadata << "{\"model_sha256\":\"" << model_digest
        << "\",\"compute\":\"" << (params.experimental_efficient_q8_0?"experimental_ggml_q8_0_cpu":"restored_fp32_cpu")
        << "\",\"restoration\":\"sam31-regular-h4-f32-v1\",\"preprocessing\":\"" << sam3_get_preprocessing(*model)
        << "\",\"build_identity\":\"" << sam3_get_build_identity()
        << "\",\"model_load_seconds\":" << std::setprecision(9) << load_seconds
        << ",\"setup_seconds\":" << setup_seconds
        << ",\"threads\":4,\"frames\":" << limit
        << ",\"prefix_cache\":" << (params.efficient_prefix_cache?"true":"false")
        << ",\"full_encoder_cache\":" << (params.efficient_fpn_cache?"true":"false")
        << ",\"cache_statistics\":{\"prefix_ram_hits\":" << pc.ram_hits << ",\"prefix_disk_hits\":" << pc.disk_hits
        << ",\"prefix_misses\":" << pc.misses << ",\"fpn_hits\":" << fc.hits << ",\"fpn_misses\":" << fc.misses
        << ",\"fpn_evictions\":" << fc.evictions << ",\"fpn_resident_bytes\":" << fc.resident_bytes << "}"
        << ",\"runtime_provenance\":" << sam3_get_model_runtime_identity(*model) << "}\n";
    return bool(metadata);
    };
    if(!write_metadata()) return 1;
    const bool dump_logits=std::getenv("EFFICIENT_SAM2_DUMP_LOGITS")!=nullptr;
    int instance_id=-1;
    sam3_pvs_params prompt; prompt.use_box=true; prompt.multimask=false;
    prompt.box={float(std::atof(argv[4])),float(std::atof(argv[5])),float(std::atof(argv[6])),float(std::atof(argv[7]))};
    for(size_t f=0;f<limit;++f) {
        const auto started=std::chrono::steady_clock::now();
        auto image=sam3_load_image_for_model(frames[f].string(),*model);
        sam3_result result;
        if(f==0) {
            if(!sam3_encode_conditioning_image(*state,*model,image)) return 1;
            result=sam3_segment_pvs(*state,*model,prompt);
            instance_id=sam3_tracker_add_instance(*tracker,*state,*model,prompt);
            if(instance_id<0) return 1;
            for(auto& detection:result.detections) detection.instance_id=instance_id;
        } else result=sam3_propagate_frame(*tracker,*state,*model,image);
        if(result.detections.empty()) { fprintf(stderr,"No result at frame %zu\n",f); return 1; }
        const auto& det=result.detections[0];
        char name[64]; std::snprintf(name,sizeof(name),"%05zu.png",f);
        if(!sam3_save_mask(det.mask,(output/name).string())) return 1;
        if(dump_logits && !det.low_res_logits.empty()) {
            std::snprintf(name,sizeof(name),"%05zu.f32",f);
            std::ofstream raw(output/name,std::ios::binary);
            raw.write(reinterpret_cast<const char*>(det.low_res_logits.data()),det.low_res_logits.size()*sizeof(float));
        }
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        printf("{\"frame_index\":%zu,\"instance_id\":%d,\"score\":%.9g,\"object_logit\":%.9g,\"seconds\":%.6f}\n",f,det.instance_id,det.score,det.object_logit,seconds);
        fflush(stdout);
    }
    if(!write_metadata()) return 1;
    return 0;
}
