#include "sam31_native.h"
#include "runtime.h"
#include "synthetic.h"
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
namespace fs=std::filesystem;
using namespace sam31::native;
namespace {
    void usage() {
        std::cout<< "SAM 3.1 native C++ CPU inference (GGUF / ConvRot INT8 storage, FP32 compute)\n" "  sam31 image --model MODEL.gguf --image INPUT --point ID:X:Y[:LABEL] --output DIR\n" "  sam31 video --model MODEL.gguf --frames DIR --point ID:X:Y[:LABEL] --output DIR\n" "  sam31 inspect MODEL.gguf [--allow-test-fixture] [--read-all]\n" "  sam31 manifest [--smoke]\n" "  sam31 synthetic --output DIR [--frames COUNT]\n" "Options: --threads N --cache-mb N --allow-test-fixture --ablate-memory\n" "Coordinates are normalized [0,1]. LABEL is 1 (positive, default) or 0 (negative).\n" "One bucket supports 1..16 objects. Multiple points per ID are allowed.\n" "No Python, PyTorch, ONNX runtime or child process is used during inference.\n" "Test fixtures are UNTRAINED; smoke execution is not a segmentation accuracy test.\n";
    }
    size_t number(const std::string&s,size_t max,const std::string&what) {
        require(!s.empty()&&std::all_of(s.begin(),s.end(),[](unsigned char c) {
            return std::isdigit(c);
        }
        ),"invalid "+what);
        size_t used=0;
        auto n=std::stoull(s,&used);
        require(used==s.size()&&n>0&&n<=max,"out-of-range "+what);
        return size_t(n);
    }
    float coordinate(const std::string&s) {
        size_t used=0;
        float v=std::stof(s,&used);
        require(used==s.size()&&std::isfinite(v)&&v>=0&&v<=1,"point coordinate must be finite and in [0,1]");
        return v;
    }
    Point point(const std::string&s) {
        std::vector<std::string>v;
        std::stringstream stream(s);
        std::string item;
        while(std::getline(stream,item,':'))v.push_back(item);
        require((v.size()==3||v.size()==4)&&s.back()!=':',"point must be ID:X:Y[:LABEL]");
        int label=1;
        if(v.size()==4) {
            require(v[3]=="0"||v[3]=="1","point label must be 0 or 1");
            label=v[3]=="1"?1:0;
        }
        return {
            int64_t(number(v[0],size_t(std::numeric_limits<int64_t>::max()),"object ID")),coordinate(v[1]),coordinate(v[2]),label
        };
    }
    void text_file(const fs::path&p,const std::string&content) {
        std::ofstream out(p,std::ios::binary);
        require(bool(out),"cannot create "+p.string());
        out<<content<<'\n';
        out.flush();
        require(bool(out),"cannot finish "+p.string());
    }
    std::string padded(size_t n) {
        std::ostringstream out;
        out<<std::setw(6)<<std::setfill('0')<<n;
        return out.str();
    }
    struct OutputDirectory {
        fs::path destination,staging;
        bool committed=false;
        explicit OutputDirectory(const std::string&path) {
            require(!path.empty(),"output directory required");
            destination=fs::absolute(path).lexically_normal();
            require(!fs::exists(fs::symlink_status(destination)),"output exists; refusing to overwrite: "+destination.string());
            fs::create_directories(destination.parent_path());
            auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
            for(int trial=0; trial<100; ++trial) {
                staging=destination.parent_path()/("."+destination.filename().string()+".sam31-"+std::to_string(stamp)+"-"+std::to_string(trial));
                if(fs::create_directory(staging))return;
            }
            throw std::runtime_error("cannot allocate output staging directory");
        }
        void commit() {
            require(!fs::exists(fs::symlink_status(destination)),"output appeared during inference");
            fs::rename(staging,destination);
            committed=true;
        }
        ~OutputDirectory() {
            if(!committed) {
                std::error_code ignored;
                fs::remove_all(staging,ignored);
            }
        }
    };
    bool natural_less(const fs::path&a,const fs::path&b) {
        std::string x=a.filename().string(),y=b.filename().string();
        size_t i=0,j=0;
        while(i<x.size()&&j<y.size()) {
            if(std::isdigit((unsigned char)x[i])&&std::isdigit((unsigned char)y[j])) {
                size_t ie=i,je=j;
                while(ie<x.size()&&std::isdigit((unsigned char)x[ie]))++ie;
                while(je<y.size()&&std::isdigit((unsigned char)y[je]))++je;
                size_t ib=i,jb=j;
                while(ib+1<ie&&x[ib]=='0')++ib;
                while(jb+1<je&&y[jb]=='0')++jb;
                if(ie-ib!=je-jb)return ie-ib<je-jb;
                int cmp=x.compare(ib,ie-ib,y,jb,je-jb);
                if(cmp)return cmp<0;
                i=ie;
                j=je;
            }
            else {
                if(x[i]!=y[j])return x[i]<y[j];
                ++i;
                ++j;
            }
        }
        if(i!=x.size()||j!=y.size())return i==x.size();
        return x<y;
    }
    Json save_result(const fs::path&dir,const Result&r,const std::string&input) {
        Json j;
        j["frame_index"]=r.frame_index;
        j["width"]=r.width;
        j["height"]=r.height;
        j["input"]=input;
        Json::array objects;
        for(const auto&o:r.objects) {
            auto name="frame_"+padded(r.frame_index)+"_obj_"+std::to_string(o.object_id)+".png";
            write_mask_png((dir/name).string(),o,r.width,r.height);
            Json d;
            d["object_id"]=std::to_string(o.object_id);
            d["score_logit"]=double(o.score);
            d["quality"]=double(o.quality);
            d["mask"]=name;
            auto mm=std::minmax_element(o.logits.begin(),o.logits.end());
            d["logit_min"]=double(*mm.first);
            d["logit_max"]=double(*mm.second);
            d["foreground_pixels"]=size_t(std::count_if(o.logits.begin(),o.logits.end(),[](float v) {
                return v>0;
            }
            ));
            objects.push_back(d);
        }
        std::string name="frame_"+padded(r.frame_index)+".f32";
        write_logits((dir/name).string(),r);
        j["logits_file"]=name;
        j["logits_layout"]="little-endian float32 [objects,height,width]";
        j["objects"]=objects;
        return j;
    }
    int inspect(const std::vector<std::string>&args) {
        require(!args.empty(),"inspect requires MODEL.gguf");
        bool fixture=false,all=false;
        for(size_t i=1; i<args.size(); ++i) {
            if(args[i]=="--allow-test-fixture")fixture=true;
            else if(args[i]=="--read-all")all=true;
            else throw std::runtime_error("unknown inspect argument: "+args[i]);
        }
        sam31::gguf_file f(args[0],fixture);
        size_t bytes=0;
        for(const auto&t:f.tensors()) {
            bytes+=t.size;
            if(all)(void)f.read_tensor(t.name);
        }
        Json j;
        j["architecture"]="sam3_1_multiplex";
        j["tensor_count"]=f.tensors().size();
        j["payload_bytes"]=bytes;
        j["source_sha256"]=f.source_sha256();
        j["test_fixture"]=f.is_test_fixture();
        j["all_payloads_read"]=all;
        j["data_offset"]=size_t(f.data_offset());
        j["native_backend_available"]=true;
        j["native_model_validated"]=false;
        std::cout<<j.dump()<<'\n';
        return 0;
    }
}
int main(int argc,char**argv) {
    try {
        std::vector<std::string>args;
#ifdef SAM31_FIXED_COMMAND
        std::string command=SAM31_FIXED_COMMAND;
        for(int i=1; i<argc; ++i)args.emplace_back(argv[i]);
#else
        if(argc<2) {
            usage();
            return 2;
        }
        std::string command=argv[1];
        for(int i=2; i<argc; ++i)args.emplace_back(argv[i]);
#endif
        if(command=="--help"||command=="-h"||std::find(args.begin(),args.end(),"--help")!=args.end()) {
            usage();
            return 0;
        }
        if(command=="inspect")return inspect(args);
        if(command=="manifest") {
            require(args.empty()||(args.size()==1&&args[0]=="--smoke"),"manifest accepts only --smoke");
            std::cout<<manifest_json(!args.empty())<<'\n';
            return 0;
        }
        if(command=="synthetic") {
            std::string dest;
            size_t count=6;
            for(size_t i=0; i<args.size(); ++i) {
                require(i+1<args.size(),"missing synthetic option value");
                if(args[i]=="--output")dest=args[++i];
                else if(args[i]=="--frames")count=number(args[++i],10000,"frame count");
                else throw std::runtime_error("unknown synthetic option");
            }
            OutputDirectory out(dest);
            fs::create_directory(out.staging/"frames");
            for(size_t i=0; i<count; ++i)write_image_png((out.staging/"frames"/(padded(i)+".png")).string(),synthetic_image(i));
            write_image_png((out.staging/"image.png").string(),synthetic_image(0));
            Json j;
            j["frames"]=count;
            j["points"]=Json::array {
                Json("1:0.25:0.32"),Json("2:0.73:0.68")
            };
            j["purpose"]="synthetic input images only, no predicted or ground-truth masks";
            text_file(out.staging/"scene.json",j.dump());
            out.commit();
            std::cout<<j.dump()<<'\n';
            return 0;
        }
        require(command=="image"||command=="video","unknown native command: "+command);
        Options options;
        std::string model_path,image_path,frames_path,destination;
        std::vector<Point>points;
        bool ablate=false;
        for(size_t i=0; i<args.size(); ++i) {
            const std::string key=args[i];
            if(key=="--allow-test-fixture") {
                options.allow_test_fixture=true;
                continue;
            }
            if(key=="--ablate-memory") {
                ablate=true;
                continue;
            }
            require(i+1<args.size(),"missing value for "+key);
            const auto value=args[++i];
            if(key=="--model")model_path=value;
            else if(key=="--image")image_path=value;
            else if(key=="--frames")frames_path=value;
            else if(key=="--output")destination=value;
            else if(key=="--point")points.push_back(point(value));
            else if(key=="--threads")options.threads=int(number(value,256,"thread count"));
            else if(key=="--cache-mb")options.weight_cache_mb=number(value,4096,"cache size");
            else throw std::runtime_error("unknown native option: "+key);
        }
        require(!model_path.empty()&&!destination.empty()&&!points.empty(),"--model, --point and --output are required");
        require(command=="video"||!ablate,"memory ablation requires video");
        std::vector<fs::path>inputs;
        if(command=="image") {
            require(!image_path.empty()&&frames_path.empty(),"image requires --image, not --frames");
            inputs.emplace_back(image_path);
        }
        else {
            require(!frames_path.empty()&&image_path.empty()&&fs::is_directory(frames_path),"video requires a frame directory");
            for(const auto&e:fs::directory_iterator(frames_path)) {
                if(!e.is_regular_file())continue;
                auto ext=e.path().extension().string();
                std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c) {
                    return char(std::tolower(c));
                }
                );
                if(ext==".png"||ext==".jpg"||ext==".jpeg"||ext==".bmp"||ext==".ppm"||ext==".pgm"||ext==".tga")inputs.push_back(e.path());
            }
            std::sort(inputs.begin(),inputs.end(),natural_less);
            require(inputs.size()>=2,"video requires at least two images");
        }
        options.total_frames=inputs.size();
        Model model(model_path,options);
        Session session(model);
        OutputDirectory output(destination);
        auto start=std::chrono::steady_clock::now();
        Json::array frames;
        for(size_t i=0; i<inputs.size(); ++i) {
            Image image=read_image(inputs[i].string());
            if(i==1&&ablate)session.zero_memory_for_test();
            Result result=i==0?session.begin(image,points):session.step(image);
            frames.push_back(save_result(output.staging,result,inputs[i].string()));
        }
        Json report;
        report["model"]=Json::parse(model.info_json());
        report["stats"]=Json::parse(session.stats_json());
        report["frames"]=frames;
        report["elapsed_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        report["python_inference"]=false;
        report["child_processes_required"]=false;
        report["accuracy_validation_performed"]=false;
        report["complete"]=true;
        text_file(output.staging/"report.json",report.dump());
        output.commit();
        std::cout<<report.dump()<<'\n';
        return 0;
    }
    catch(const std::exception&e) {
        std::cerr<<"sam31 native: "<<e.what()<<'\n';
        return 2;
    }
}
