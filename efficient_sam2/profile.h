#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

// Disabled by default, no tensor/graph changes, monotonic wall-clock timings.
// Nested scopes overlap; sum only disjoint phases when producing a profile.
struct sam2_profile_scope {
    using clock = std::chrono::steady_clock;
    bool enabled;
    std::string phase;
    clock::time_point start, previous;
    explicit sam2_profile_scope(std::string name, bool applicable=true)
        : enabled(applicable && std::getenv("EFFICIENT_SAM2_PROFILE")!=nullptr), phase(std::move(name)) {
        if(enabled) start=previous=clock::now();
    }
    void lap(const char* part) {
        if(!enabled) return;
        const auto now=clock::now();
        emit(part,std::chrono::duration<double,std::milli>(now-previous).count());
        previous=now;
    }
    void emit(const char* part,double ms) const {
        fprintf(stderr,"{\"event\":\"efficient_profile\",\"phase\":\"%s\",\"part\":\"%s\",\"ms\":%.6f}\n",
                phase.c_str(),part,ms);
    }
    ~sam2_profile_scope() {
        if(enabled) emit("total",std::chrono::duration<double,std::milli>(clock::now()-start).count());
    }
};

// Explicit public numerical diagnostic only. No dump is written by default.
inline void sam2_diag_dump(int frame, const char* label, const float* values, size_t count) {
    const char* directory=std::getenv("EFFICIENT_SAM2_DIAGNOSTIC_DIR");
    if(!directory || !*directory) return;
    char name[128]; std::snprintf(name,sizeof(name),"/frame_%05d_%s.f32",frame,label);
    FILE* file=std::fopen((std::string(directory)+name).c_str(),"wb");
    if(!file) { std::fprintf(stderr,"Cannot open requested Efficient-SAM2 diagnostic %s\n",label); return; }
    const auto written=std::fwrite(values,sizeof(float),count,file);
    const int status=std::fclose(file);
    if(written!=count || status) std::fprintf(stderr,"Incomplete Efficient-SAM2 diagnostic %s\n",label);
}
