// Sequential frame-major CPU driver. Every object owns an independent tracker.
// Protocol: frame <global-index> <path>, seed <id> <x0> <y0> <x1> <y1>,
// step <id>, drop <id>, stats, quit. stdout is JSONL; diagnostics use stderr.
#include "sam3.h"
#include "efficient_sam2/prefix_cache/digest.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

struct Track { sam3_tracker_ptr tracker; int seed, last, native_id; };
static uint64_t peak_rss_bytes() {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage usage{};
    if(getrusage(RUSAGE_SELF,&usage)!=0) return 0;
#if defined(__APPLE__)
    return uint64_t(usage.ru_maxrss);
#else
    return uint64_t(usage.ru_maxrss)*1024;
#endif
#else
    return 0; // Unsupported; do not infer a memory bound from this value.
#endif
}
static void ok() { std::cout << "{\"ok\":true}" << std::endl; }
static void reply(const sam3_result& result, int id, int global, int local,
                  const sam3_image& image, double seconds) {
    if(result.detections.size()!=1) throw std::runtime_error("expected one native object");
    const auto& d=result.detections.front(); const auto& m=d.mask;
    if(m.width!=image.width || m.height!=image.height || m.data.size()!=size_t(m.width)*m.height || !std::isfinite(d.score))
        throw std::runtime_error("invalid native output");
    std::cout << "{\"ok\":true,\"object_id\":" << id << ",\"frame_index\":" << global
              << ",\"local_frame_index\":" << local << ",\"width\":" << image.width
              << ",\"height\":" << image.height << ",\"seconds\":" << seconds
              << ",\"verification\":{\"low_res_logits_sha256\":\""
              << efficient_sam2::prefix_cache::sha256_bytes(d.low_res_logits.data(), d.low_res_logits.size()*sizeof(float))
              << "\",\"low_res_logits_count\":" << d.low_res_logits.size()
              << ",\"decoder_saliency_sha256\":\""
              << efficient_sam2::prefix_cache::sha256_bytes(d.decoder_saliency.data(), d.decoder_saliency.size()*sizeof(float))
              << "\",\"decoder_saliency_count\":" << d.decoder_saliency.size() << ",\"object_logit\":" << d.object_logit << "}"
              << ",\"detections\":[{\"object_id\":" << id << ",\"instance_id\":" << id
              << ",\"score\":" << d.score << ",\"mask_size\":[" << m.width << ',' << m.height << "],\"counts\":[";
    size_t run=0; bool value=false, first=true;
    for(int x=0;x<m.width;++x) for(int y=0;y<m.height;++y) {
        bool next=m.data[size_t(y)*m.width+x]!=0;
        if(next==value) ++run;
        else { if(!first) std::cout << ','; std::cout << run; first=false; run=1; value=next; }
    }
    if(!first) std::cout << ',';
    std::cout << run << "]}]}" << std::endl;
}
int main(int argc,char** argv) {
    int sequence_length=0;
    sam3_params p; p.use_gpu=false; p.efficient_video_preprocessing=true; std::cout << std::setprecision(8);
    try {
        for(int i=1;i<argc;++i) {
            std::string a=argv[i];
            auto arg=[&]() { if(++i>=argc) throw std::runtime_error("missing option value"); return std::string(argv[i]); };
            if(a=="--runtime-provenance") {
                std::string identity,error;
                if(!sam3_get_runtime_identity(identity,error)) throw std::runtime_error("runtime provenance unavailable: "+error);
                const auto build=sam3_get_build_identity();
                std::cout << "{\"ok\":true,\"runtime_identity\":" << identity << ",\"native_build_sha256\":\""
                    << efficient_sam2::prefix_cache::sha256_bytes(build.data(),build.size()) << "\"}" << std::endl;
                return 0;
            }
            else if(a=="--model") p.model_path=arg();
            else if(a=="--sequence-length") sequence_length=std::stoi(arg());
            else if(a=="--n-threads") p.n_threads=std::stoi(arg());
            else if(a=="--experimental-efficient-video") { p.experimental_efficient_video=true; p.efficient_video_preprocessing=true; }
            else if(a=="--experimental-efficient-q8-0") p.experimental_efficient_q8_0=true;
            else if(a=="--efficient-fpn-cache") p.efficient_fpn_cache=true;
            else if(a=="--efficient-prefix-cache") p.efficient_prefix_cache=true;
            else if(a=="--prefix-cache-dir") { p.efficient_cache_directory=arg(); p.efficient_prefix_cache=true; }
            else if(a=="--prefix-cache-budget-mib") { auto n=std::stoull(arg()); if(n>1048576) throw std::runtime_error("cache budget too large"); p.efficient_cache_disk_budget=n*1024*1024; }
            else if(a=="--help") { std::cerr << "sam3_track_batch --model PATH [--n-threads N] [--sequence-length N] [--experimental-efficient-video] [--experimental-efficient-q8-0] [--efficient-fpn-cache] [--efficient-prefix-cache] [--prefix-cache-dir PATH --prefix-cache-budget-mib N]\n       sam3_track_batch --runtime-provenance\n"; return 0; }
            else throw std::runtime_error("unknown option");
        }
        if(p.model_path.empty() || p.n_threads<1 || p.n_threads>32 || sequence_length<0) throw std::runtime_error("invalid model/threads");
        auto model=sam3_load_model(p);
        if(!model || !sam3_supports_video_tracking(*model)) throw std::runtime_error("video model unavailable or gate disabled");
        auto state=sam3_create_state(*model,p); if(!state) throw std::runtime_error("state allocation failed");
        std::map<int,Track> tracks; sam3_image image; int global=-1;
        const auto identity=sam3_get_model_runtime_identity(*model);
        std::cout << "{\"ok\":true,\"event\":\"ready\",\"video_tracking\":true,\"independent_trackers\":true,\"runtime_identity\":"
            << (identity.empty()?"null":identity) << ",\"preprocessing\":\"" << sam3_get_preprocessing(*model) << "\"}" << std::endl;
        for(std::string line;std::getline(std::cin,line);) {
            std::istringstream in(line); std::string cmd; in>>cmd;
            if(cmd=="quit") { ok(); break; }
            if(cmd=="frame") {
                int index; std::string path;
                if(!(in>>index) || index<=global || (sequence_length && index>=sequence_length)) throw std::runtime_error("frame order must increase");
                std::getline(in>>std::ws,path); image=sam3_load_image_for_model(path,*model);
                if(image.data.empty()) throw std::runtime_error("cannot read frame");
                global=index; ok(); continue;
            }
            if(cmd=="stats") {
                auto s=sam3_get_prefix_cache_statistics(*model);
                auto f=sam3_get_fpn_cache_statistics(*model);
                std::cout << "{\"ok\":true,\"active_trackers\":" << tracks.size() << ",\"ram_hits\":" << s.ram_hits
                    << ",\"disk_hits\":" << s.disk_hits << ",\"misses\":" << s.misses << ",\"writes\":" << s.writes
                    << ",\"invalid_entries\":" << s.invalid_entries << ",\"evictions\":" << s.evictions
                    << ",\"peak_rss_bytes\":" << peak_rss_bytes() << ",\"resident_bytes\":" << s.resident_bytes << ",\"disk_bytes\":" << s.disk_bytes
                    << ",\"full_encoder\":{\"enabled\":" << (f.enabled ? "true" : "false")
                    << ",\"hits\":" << f.hits << ",\"misses\":" << f.misses << ",\"evictions\":" << f.evictions
                    << ",\"resident_bytes\":" << f.resident_bytes << "}}" << std::endl; continue;
            }
            int id; if(!(in>>id) || id<1) throw std::runtime_error("expected positive object ID");
            if(cmd=="drop") { if(!tracks.erase(id)) throw std::runtime_error("unknown object"); ok(); continue; }
            if(global<0) throw std::runtime_error("load frame first");
            auto begin=std::chrono::steady_clock::now(); sam3_result result;
            if(cmd=="seed") {
                sam3_pvs_params prompt; prompt.use_box=true; auto& b=prompt.box;
                if(tracks.count(id) || !(in>>b.x0>>b.y0>>b.x1>>b.y1) ||
                    !(0<=b.x0 && b.x0<b.x1 && b.x1<=image.width && 0<=b.y0 && b.y0<b.y1 && b.y1<=image.height))
                    throw std::runtime_error("duplicate object or invalid seed box");
                // Conditioning must reset object-specific window selection before encoding.
                if(!sam3_encode_conditioning_image(*state,*model,image)) throw std::runtime_error("conditioning encode failed");
                sam3_visual_track_params vp; vp.fill_hole_area=0; vp.recondition_every=0;
                vp.efficient_sequence_length=sequence_length; vp.efficient_source_start_frame=global;
                auto tracker=sam3_create_visual_tracker(*model,vp); if(!tracker) throw std::runtime_error("tracker allocation failed");
                result=sam3_segment_pvs(*state,*model,prompt);
                const int native_id=sam3_tracker_add_instance(*tracker,*state,*model,prompt);
                if(result.detections.empty() || native_id<0) throw std::runtime_error("seed failed");
                auto best=std::max_element(result.detections.begin(),result.detections.end(),[](const sam3_detection& a,const sam3_detection& b){return a.score<b.score;});
                sam3_detection selected=*best; selected.instance_id=native_id; result.detections={std::move(selected)};
                tracks.emplace(id,Track{std::move(tracker),global,global,native_id});
            } else if(cmd=="step") {
                auto it=tracks.find(id); if(it==tracks.end() || global!=it->second.last+1) throw std::runtime_error("noncausal object frame");
                auto& t=it->second;
                if(sam3_tracker_frame_index(*t.tracker)!=global-t.seed) throw std::runtime_error("native frame mismatch");
                result=sam3_propagate_frame(*t.tracker,*state,*model,image); t.last=global;
            } else throw std::runtime_error("unknown command");
            for(const auto& detection:result.detections)
                if(detection.instance_id!=tracks.at(id).native_id) throw std::runtime_error("native object identity mismatch");
            reply(result,id,global,global-tracks.at(id).seed,image,std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());
        }
    } catch(const std::exception& e) { std::cerr << "sam3_track_batch: " << e.what() << '\n'; std::cout << "{\"ok\":false,\"error\":\"batch stopped; see stderr\"}" << std::endl; return 1; }
}
