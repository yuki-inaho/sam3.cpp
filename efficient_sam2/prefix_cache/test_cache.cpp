#include "cache.h"
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace efficient_sam2::prefix_cache;
namespace fs=std::filesystem;
static void check(bool ok,const char* reason) { if(!ok) throw std::runtime_error(reason); }
static Key key() {
    Key k; k.model_sha256=std::string(64,'a'); k.input_sha256=std::string(64,'b');
    k.policy_sha256=std::string(64,'c'); k.preprocessing="video-pil-bicubic-v1"; k.precision="restored-f32"; k.backend="cpu-avx2";
    k.build="test-build-v1"; k.threads=4; k.input_width=1920; k.input_height=1080; return k;
}
static Entry fixture() {
    Entry e; const int64_t dims[3][4]={{448,64,64,1},{112,256,256,1},{224,128,128,1}};
    for(int i=0;i<3;++i) {
        for(int j=0;j<4;++j) e.tensors[i].shape[j]=dims[i][j];
        e.tensors[i].values.resize(size_t(dims[i][0]*dims[i][1]*dims[i][2]),float(i)+.25f);
    }
    e.tensors[0].values[1]=-0.0f; return e;
}
static void same(const Entry& a,const Entry& b) {
    for(size_t i=0;i<3;++i) {
        check(a.tensors[i].shape==b.tensors[i].shape,"Shape roundtrip differs");
        check(a.tensors[i].values==b.tensors[i].values,"Values roundtrip differs");
    }
    check(std::signbit(b.tensors[0].values[1]),"Signed zero changed");
}
static void mutate(const fs::path& file,uint64_t offset,unsigned char value) {
    std::fstream f(file,std::ios::in|std::ios::out|std::ios::binary); f.seekp(offset); f.put(char(value)); check(bool(f),"Mutation failed");
}
int main(int argc,char**argv) {
    if(argc!=2) { std::cerr<<"Usage: test_cache EMPTY_TEST_DIRECTORY\n";return 2; }
    try {
        const fs::path root=argv[1];
        check(!fs::exists(root),"Use a new empty test directory"); fs::create_directories(root);
        Entry e=fixture(); Key k=key(); std::string error; Hit hit;
        auto ram=create(Options{},error); check(bool(ram),"RAM cache create failed");
        check(!lookup(*ram,k,hit,error) && hit==Hit::miss,"Expected cold RAM miss");
        check(store(*ram,k,e,error),"RAM store failed");
        auto found=lookup(*ram,k,hit,error); check(found && hit==Hit::ram,"Expected RAM hit"); same(e,*found); found.reset();
        for(int field=0;field<10;++field) {
            auto changed=k;
            if(field==0) changed.model_sha256[0]='d';
            if(field==1) changed.input_sha256[0]='d';
            if(field==2) changed.policy_sha256[0]='d';
            if(field==3) changed.precision+="-q8";
            if(field==4) changed.backend+="-other";
            if(field==5) changed.build+="-other";
            if(field==6) ++changed.threads;
            if(field==7) ++changed.input_width;
            if(field==8) ++changed.input_height;
            if(field==9) changed.preprocessing="image-bilinear-v1";
            check(!lookup(*ram,changed,hit,error),"Changed semantic key hit cache");
        }
        auto bad=k;bad.model_sha256="filename-only";
        check(!lookup(*ram,bad,hit,error) && hit==Hit::rejected,"Invalid content identity accepted");
        check(clear(*ram,false,error) && stats(*ram).resident_bytes==0,"RAM clear failed");
        const auto exported=root/"export.es2prefix";
        check(write_file(exported.string(),k,e,error),"Export failed");
        found=read_file(exported.string(),k,error); check(bool(found),"Import failed"); same(e,*found);found.reset();
        bad=k; bad.input_sha256[0]='d'; check(!read_file(exported.string(),bad,error),"Wrong full key imported");
        std::atomic<bool> read_failed{false};
        std::thread reader([&] {
            for(int attempt=0;attempt<3;++attempt) {std::string reason;if(!read_file(exported.string(),k,reason)) read_failed=true;}
        });
        const bool published=write_file(exported.string(),k,e,error);
        reader.join();check(published && !read_failed,"Atomic publish exposed a partial file");
        for(auto mutation:std::vector<std::pair<uint64_t,unsigned char>>{{0,'X'},{8,2},{40,0},{fs::file_size(exported)-1,0xff}}) {
            auto damaged=root/"damaged.es2prefix"; fs::copy_file(exported,damaged,fs::copy_options::overwrite_existing);
            mutate(damaged,mutation.first,mutation.second); check(!read_file(damaged.string(),k,error),"Corrupt/version/shape cache accepted");
        }
        auto short_file=root/"truncated.es2prefix";fs::copy_file(exported,short_file);
        fs::resize_file(short_file,100);check(!read_file(short_file.string(),k,error),"Truncated header accepted");
        fs::copy_file(exported,short_file,fs::copy_options::overwrite_existing);
        fs::resize_file(short_file,fs::file_size(short_file)-1);check(!read_file(short_file.string(),k,error),"Truncated payload accepted");
        {std::ofstream trailing(short_file,std::ios::binary|std::ios::app);trailing<<"xx";}
        check(!read_file(short_file.string(),k,error),"Trailing bytes accepted");
        e.tensors[0].shape[0]=1;check(!write_file((root/"invalid").string(),k,e,error),"Invalid tensor shape exported");e.tensors[0].shape[0]=448;
        e.tensors[0].values[0]=std::numeric_limits<float>::quiet_NaN();check(!write_file((root/"invalid").string(),k,e,error),"Nonfinite tensor exported");e.tensors[0].values[0]=.25f;
        Options disk_options;disk_options.ram_budget_bytes=0;disk_options.disk_directory=(root/"disk").string();
        const auto entry_bytes=fs::file_size(exported);disk_options.disk_budget_bytes=entry_bytes+2048;
        auto disk=create(disk_options,error);check(bool(disk),"Disk cache create failed");
        check(store(*disk,k,e,error),"Disk store failed");
        auto path=artifact_path(*disk,k,error);check(fs::exists(path),"Missing persistent artifact");
        const auto abandoned=path+".tmp-12345-0";fs::copy_file(path,abandoned);fs::resize_file(abandoned,1024);
        disk.reset();disk=create(disk_options,error);
        check(!fs::exists(abandoned),"Interrupted temporary not recovered");check(bool(disk),"Process-session reopen failed");
        found=lookup(*disk,k,hit,error);check(found && hit==Hit::disk,"No disk hit after reopen");same(e,*found);found.reset();
        auto other=k;other.input_sha256[0]='d';check(store(*disk,other,e,error),"Second disk store failed");
        check(!lookup(*disk,k,hit,error),"Budget did not evict old entry");
        check(lookup(*disk,other,hit,error)!=nullptr,"Newest disk entry missing");
        check(stats(*disk).disk_bytes<=disk_options.disk_budget_bytes && stats(*disk).evictions>0,"Disk budget exceeded");
        const auto protected_file=fs::path(path).parent_path()/"unrelated.txt";{std::ofstream f(protected_file);f<<"keep";}
        auto corrupt=artifact_path(*disk,other,error);mutate(corrupt,fs::file_size(corrupt)-1,0xff);
        check(!lookup(*disk,other,hit,error) && hit==Hit::rejected && stats(*disk).invalid_entries==1,"Corruption was not a safe miss");
        check(!store(*disk,other,e,error),"Corrupt entry was overwritten without validation");
        check(!clear(*disk,true,error) && !error.empty(),"Clear did not diagnose preserved corrupt entry");check(fs::exists(corrupt),"Clear removed an unvalidated corrupt entry");check(fs::exists(protected_file),"Clear removed unrelated file");
        Options tiny;tiny.disk_directory=(root/"tiny").string();tiny.disk_budget_bytes=1;tiny.ram_budget_bytes=0;
        auto disabled=create(tiny,error);check(disabled && !store(*disabled,k,e,error),"Oversize cache entry accepted");
        const auto impostor=fs::path(path).parent_path()/(std::string(64,'e')+".es2prefix");
        {std::ofstream f(impostor);f<<"not a cache entry";}
        (void)clear(*disk,true,error);check(fs::exists(impostor),"Clear deleted an unrelated cache-shaped filename");
        check(!write_file(protected_file.string(),k,e,error),"Export overwrote unrelated file");
        const auto linkdir=root/"linked-directory";fs::create_directory_symlink(root/"disk",linkdir);
        Options linked=disk_options;linked.disk_directory=linkdir.string();check(!create(linked,error),"Symlink directory accepted");
        check(!read_file((root/".."/root.filename()/"export.es2prefix").string(),k,error),"Parent path traversal accepted");
        auto symlink=root/"symlink.es2prefix";fs::create_symlink(exported,symlink);
        check(!read_file(symlink.string(),k,error),"Symlink import accepted");
        check(!write_file(symlink.string(),k,e,error),"Symlink export accepted");
        // Warm-filesystem helper timings only: not native end-to-end speedup.
        auto time_ms=[](auto&& fn) {
            const auto start=std::chrono::steady_clock::now();fn();
            return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        };
        double write_ms=0, read_ms=0;
        for(int i=0;i<3;++i) {
            write_ms+=time_ms([&] {check(write_file(exported.string(),k,e,error),"Timed export failed");});
            read_ms+=time_ms([&] {found=read_file(exported.string(),k,error);check(bool(found),"Timed import failed");});found.reset();
        }
        check(store(*ram,k,e,error),"Timed RAM store failed");
        const double ram_ms=time_ms([&] {for(int i=0;i<1000;++i) {found=lookup(*ram,k,hit,error);check(bool(found),"Timed RAM hit failed");}})/1000.;
        std::cout<<"helper_warm_filesystem_mean_ms,write,"<<write_ms/3<<",read,"<<read_ms/3<<",ram_shared_hit,"<<ram_ms<<'\n';
        fs::remove_all(root); // This test created the previously nonexistent directory.
        std::cout<<"PASS: RAM/disk/reopen/exactness/key invalidation/shape/checksum/version/partial/budget/clear/symlink tests\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
