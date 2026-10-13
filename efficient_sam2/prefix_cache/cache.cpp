#include "cache.h"
#include "digest.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace efficient_sam2 { namespace prefix_cache {
namespace fs = std::filesystem;
namespace {
constexpr char magic[8] = {'E','S','2','P','F','X','0','1'};
constexpr uint64_t header_bytes = 136;
constexpr uint32_t max_key_bytes = 4096;
const std::array<std::array<int64_t,4>,3> shapes{{
    {{448,64,64,1}}, {{112,256,256,1}}, {{224,128,128,1}}
}};
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
bool hex_digest(const std::string& value) {
    return value.size()==64 && std::all_of(value.begin(),value.end(),[](char c) {
        return (c>='0'&&c<='9') || (c>='a'&&c<='f');
    });
}
void validate(const Entry& entry) {
    static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,"IEEE FP32 required");
    for(size_t i=0;i<3;++i) {
        require(entry.tensors[i].shape==shapes[i],"Invalid prefix tensor shape");
        const auto& t=entry.tensors[i];
        const uint64_t count=uint64_t(shapes[i][0])*shapes[i][1]*shapes[i][2];
        require(t.values.size()==count,"Invalid prefix tensor size");
        for(float v:t.values) require(std::isfinite(v),"Non-finite prefix feature");
    }
}
// CRC32 detects accidental corruption only. This is not authentication/security.
uint32_t crc_update(uint32_t crc, const void* data, size_t bytes) {
    static const auto table=[] {
        std::array<uint32_t,256> t{};
        for(uint32_t i=0;i<256;++i) {
            uint32_t c=i;
            for(int bit=0;bit<8;++bit) c=(c>>1)^(0xedb88320U&uint32_t(-int32_t(c&1)));
            t[i]=c;
        }
        return t;
    }();
    auto p=static_cast<const unsigned char*>(data);
    for(size_t i=0;i<bytes;++i) crc=table[(crc^p[i])&255]^(crc>>8);
    return crc;
}
uint32_t checksum(const std::string& key, const Entry& entry) {
    uint32_t crc=crc_update(0xffffffffU,key.data(),key.size());
    for(const auto& t:entry.tensors) crc=crc_update(crc,t.values.data(),t.values.size()*4);
    return crc^0xffffffffU;
}
bool little_endian() { const uint32_t x=1; return *reinterpret_cast<const uint8_t*>(&x)==1; }
void number(FILE* out,uint64_t value,size_t bytes) {
    for(size_t i=0;i<bytes;++i) require(std::fputc(int((value>>(8*i))&255),out)!=EOF,"Cache header write failed");
}
void write(FILE* out,const void* data,size_t bytes) {
    require(std::fwrite(data,1,bytes,out)==bytes,"Cache data write failed");
}
void safe_path(const fs::path& input) {
    require(!input.empty(),"Empty cache path");
    for(const auto& part:input) require(part!="..","Parent traversal is not allowed in cache paths");
    fs::path current;
    for(const auto& part:fs::absolute(input)) {
        current/=part;
        require(!fs::is_symlink(fs::symlink_status(current)),"Symlink component is not allowed in cache paths");
    }
}
uint64_t number(std::istream& in,size_t bytes) {
    uint64_t value=0;
    for(size_t i=0;i<bytes;++i) {
        const int c=in.get(); require(c!=EOF,"Truncated cache header");
        value|=uint64_t(uint8_t(c))<<(8*i);
    }
    return value;
}
std::string filename(const std::string& key) {
    return sha256_bytes(key.data(),key.size())+".es2prefix";
}
bool owned_name(const fs::path& path) {
    const auto s=path.filename().string();
    return s.size()==74 && s.substr(64)==".es2prefix" && hex_digest(s.substr(0,64));
}
// Establish ownership before deleting: valid format/key/shape header whose
// full key hashes to the filename. Complete entries additionally need CRC32.
bool owned_file(const fs::path& path,const std::string& expected_name,bool partial) {
    try {
        if(!fs::is_regular_file(fs::symlink_status(path))) return false;
        const auto bytes=fs::file_size(path);
        if(bytes<header_bytes || bytes>header_bytes+max_key_bytes+payload_bytes) return false;
        std::ifstream in(path,std::ios::binary); char signature[8];in.read(signature,8);
        if(!in || std::memcmp(signature,magic,8) || number(in,4)!=1 || number(in,4)!=0x01020304) return false;
        const auto key_bytes=number(in,4);
        if(!key_bytes || key_bytes>max_key_bytes || number(in,4)!=3 || number(in,8)!=payload_bytes) return false;
        const auto expected_crc=number(in,4);if(number(in,4)!=0) return false;
        for(const auto& shape:shapes) for(auto d:shape) if(number(in,8)!=uint64_t(d)) return false;
        if(bytes<header_bytes+key_bytes || (!partial && bytes!=header_bytes+key_bytes+payload_bytes)) return false;
        std::string key(size_t(key_bytes),'\0');in.read(&key[0],key.size());
        if(!in || key.find("efficient-sam2-prefix-block5-fp32-v1\n")!=0 || filename(key)!=expected_name) return false;
        if(partial) return true;
        uint32_t crc=crc_update(0xffffffffU,key.data(),key.size());
        std::array<char,65536> buffer{};
        uint64_t remaining=payload_bytes;
        while(remaining) {
            const auto n=size_t(std::min<uint64_t>(remaining,buffer.size()));in.read(buffer.data(),n);
            if(!in) return false;
            crc=crc_update(crc,buffer.data(),n);remaining-=n;
        }
        return (crc^0xffffffffU)==expected_crc;
    } catch(...) {return false;}
}
void cleanup_temporaries(const fs::path& directory) {
    safe_path(directory);
    for(const auto& f:fs::directory_iterator(directory)) {
        const auto name=f.path().filename().string();const auto marker=name.find(".tmp-");
        if(marker==std::string::npos || !owned_name(fs::path(name.substr(0,marker)))) continue;
        const auto suffix=name.substr(marker+5);
        if(suffix.empty() || !std::all_of(suffix.begin(),suffix.end(),[](char c) { return (c>='0'&&c<='9')||c=='-'; })) continue;
        if(owned_file(f.path(),name.substr(0,marker),true)) fs::remove(f.path());
    }
}
struct DiskFile { fs::path path; uint64_t bytes; fs::file_time_type modified; bool removable; };
std::vector<DiskFile> scan(const fs::path& directory,uint64_t& total) {
    safe_path(directory);
    total=0; std::vector<DiskFile> files;
    for(const auto& f:fs::directory_iterator(directory)) {
        if(!owned_name(f.path()) || !fs::is_regular_file(f.symlink_status())) continue;
        const auto bytes=f.file_size();
        require(bytes<=std::numeric_limits<uint64_t>::max()-total,"Cache size overflow");
        total+=bytes; files.push_back({f.path(),bytes,f.last_write_time(),owned_file(f.path(),f.path().filename().string(),true)});
    }
    std::sort(files.begin(),files.end(),[](const DiskFile& a,const DiskFile& b) {
        return a.modified==b.modified ? a.path.string()<b.path.string() : a.modified<b.modified;
    });
    return files;
}
}

struct Cache {
    Options options;
    Stats counters;
    std::mutex mutex;
    std::string resident_key;
    std::shared_ptr<const Entry> resident;
    fs::path directory;
};

std::string canonical_key(const Key& k) {
    if(!hex_digest(k.model_sha256)||!hex_digest(k.input_sha256)||!hex_digest(k.policy_sha256)||
       k.threads<1||k.threads>4096||k.input_width<1||k.input_height<1)
        throw std::invalid_argument("Cache requires verified SHA256 identities and positive geometry/thread count");
    std::string out="efficient-sam2-prefix-block5-fp32-v1\n";
    for(const auto& field:{k.model_sha256,k.input_sha256,k.policy_sha256,k.preprocessing,k.precision,k.backend,k.build}) {
        if(field.empty()||field.size()>512||field.find('\0')!=std::string::npos)
            throw std::invalid_argument("Invalid/empty cache identity field");
        out+=std::to_string(field.size())+":"+field+"\n";
    }
    out+=std::to_string(k.threads)+":"+std::to_string(k.input_width)+":"+std::to_string(k.input_height)+"\n";
    if(out.size()>max_key_bytes) throw std::invalid_argument("Cache key too long");
    return out;
}

bool write_file(const std::string& path,const Key& key,const Entry& entry,std::string& error) {
    error.clear(); fs::path temporary; bool created=false;
    try {
        require(little_endian(),"Cache format currently requires little-endian IEEE FP32");
        const auto encoded=canonical_key(key); validate(entry);
        const fs::path target(path); safe_path(target);
        require(!path.empty() && fs::is_directory(target.parent_path().empty()?fs::path("."):target.parent_path()),"Cache export parent directory must exist");
        require(!fs::is_symlink(fs::symlink_status(target)),"Cache target cannot be a symlink");
        if(fs::exists(target)) {
            std::string ignored;
            require(bool(read_file(target.string(),key,ignored)),"Refusing to overwrite an unvalidated cache/export file");
        }
        static std::atomic<uint64_t> sequence{0};
        const auto tick=std::chrono::steady_clock::now().time_since_epoch().count();
        temporary=target.string()+".tmp-"+std::to_string(tick)+"-"+std::to_string(sequence++);
        require(!fs::exists(temporary),"Cache temporary file collision");
        // C11 exclusive-create mode never truncates an existing file/symlink.
        const auto close_file=[](FILE* f) { if(f) std::fclose(f); };
        std::unique_ptr<FILE,decltype(close_file)> file(std::fopen(temporary.string().c_str(),"wbx"),close_file);
        require(bool(file),"Cannot exclusively create cache temporary file"); created=true;
        FILE* out=file.get();
        write(out,magic,8); number(out,1,4); number(out,0x01020304,4);
        number(out,encoded.size(),4); number(out,3,4); number(out,payload_bytes,8);
        number(out,checksum(encoded,entry),4); number(out,0,4);
        for(const auto& t:entry.tensors) for(auto d:t.shape) number(out,uint64_t(d),8);
        write(out,encoded.data(),encoded.size());
        for(const auto& t:entry.tensors) write(out,t.values.data(),t.values.size()*4);
        require(std::fflush(out)==0,"Cache flush failed");
        FILE* closing=file.release();require(std::fclose(closing)==0,"Cache close failed");
        // Same-directory rename publishes a complete immutable file. No claim of
        // power-loss durability (fsync) or multi-writer coordination is made.
        fs::rename(temporary,target); temporary.clear();
        return true;
    } catch(const std::exception& e) {
        if(created && !temporary.empty()) { std::error_code ignored; fs::remove(temporary,ignored); }
        error=e.what(); return false;
    }
}

std::shared_ptr<const Entry> read_file(const std::string& path,const Key& key,std::string& error) {
    error.clear();
    try {
        require(little_endian(),"Cache format currently requires little-endian IEEE FP32");
        safe_path(fs::path(path));
        const auto expected=canonical_key(key);
        require(fs::is_regular_file(fs::symlink_status(path)),"Missing/nonregular cache file");
        const auto bytes=fs::file_size(path);
        require(bytes>=header_bytes && bytes<=header_bytes+max_key_bytes+payload_bytes,"Invalid cache file size");
        std::ifstream in(path,std::ios::binary); char signature[8]; in.read(signature,8);
        require(bool(in)&&!std::memcmp(signature,magic,8),"Invalid cache magic");
        require(number(in,4)==1,"Unsupported cache representation version");
        require(number(in,4)==0x01020304,"Unsupported cache endian/representation");
        const auto key_bytes=number(in,4);
        require(key_bytes>0 && key_bytes<=max_key_bytes,"Invalid cache key length");
        require(number(in,4)==3 && number(in,8)==payload_bytes,"Invalid cache payload dimensions");
        const auto crc=number(in,4); require(number(in,4)==0,"Invalid cache reserved field");
        require(bytes==header_bytes+key_bytes+payload_bytes,"Truncated/trailing cache data");
        for(const auto& shape:shapes) for(auto d:shape) require(number(in,8)==uint64_t(d),"Cache shape mismatch");
        std::string encoded(size_t(key_bytes),'\0'); in.read(&encoded[0],encoded.size());
        require(bool(in)&&encoded==expected,"Cache identity mismatch");
        auto entry=std::make_shared<Entry>();
        for(size_t i=0;i<3;++i) {
            auto& t=entry->tensors[i]; t.shape=shapes[i];
            t.values.resize(size_t(shapes[i][0]*shapes[i][1]*shapes[i][2]));
            in.read(reinterpret_cast<char*>(t.values.data()),t.values.size()*4);
        }
        require(bool(in),"Truncated cache payload");
        require(checksum(encoded,*entry)==crc,"Cache checksum mismatch"); validate(*entry);
        return entry;
    } catch(const std::exception& e) { error=e.what(); return {}; }
}

std::shared_ptr<Cache> create(const Options& options,std::string& error) {
    error.clear();
    try {
        auto cache=std::make_shared<Cache>(); cache->options=options;
        require(options.disk_directory.empty()==(options.disk_budget_bytes==0),"Disk cache requires both directory and explicit byte budget");
        if(!options.disk_directory.empty()) {
            fs::path parent(options.disk_directory); safe_path(parent);
            require(!fs::is_symlink(fs::symlink_status(parent)),"Cache directory cannot be a symlink");
            fs::create_directories(parent);
            cache->directory=parent/"efficient-sam2-prefix-v1";
            require(!fs::is_symlink(fs::symlink_status(cache->directory)),"Cache namespace cannot be a symlink");
            fs::create_directories(cache->directory);
            cleanup_temporaries(cache->directory);
            const auto files=scan(cache->directory,cache->counters.disk_bytes);
            for(const auto& file:files) {
                if(cache->counters.disk_bytes<=options.disk_budget_bytes) break;
                if(!file.removable || !owned_file(file.path,file.path.filename().string(),false)) continue;
                fs::remove(file.path); cache->counters.disk_bytes-=file.bytes; ++cache->counters.evictions;
            }
        }
        return cache;
    } catch(const std::exception& e) { error=e.what(); return {}; }
}

std::string artifact_path(Cache& cache,const Key& key,std::string& error) {
    error.clear();
    try {
        const auto encoded=canonical_key(key);
        require(!cache.directory.empty(),"Disk cache is disabled");
        return (cache.directory/filename(encoded)).string();
    } catch(const std::exception& e) { error=e.what(); return {}; }
}

std::shared_ptr<const Entry> lookup(Cache& cache,const Key& key,Hit& hit,std::string& error) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    error.clear(); hit=Hit::miss;
    try {
        const auto encoded=canonical_key(key);
        if(cache.resident && cache.resident_key==encoded) { ++cache.counters.ram_hits; hit=Hit::ram; return cache.resident; }
        if(!cache.directory.empty()) {
            const auto path=cache.directory/filename(encoded);
            if(fs::exists(path)) {
                auto entry=read_file(path.string(),key,error);
                if(entry) {
                    ++cache.counters.disk_hits; hit=Hit::disk;
                    if(cache.options.ram_budget_bytes>=payload_bytes) {
                        cache.resident=entry; cache.resident_key=encoded; cache.counters.resident_bytes=payload_bytes;
                    }
                    std::error_code ignored; fs::last_write_time(path,fs::file_time_type::clock::now(),ignored);
                    return entry;
                }
                ++cache.counters.invalid_entries; hit=Hit::rejected;
            }
        }
        ++cache.counters.misses; return {};
    } catch(const std::exception& e) { error=e.what(); hit=Hit::rejected; ++cache.counters.invalid_entries; ++cache.counters.misses; return {}; }
}

bool store(Cache& cache,const Key& key,const Entry& entry,std::string& error) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    error.clear();
    try {
        const auto encoded=canonical_key(key); validate(entry);
        if(cache.options.ram_budget_bytes>=payload_bytes) {
            cache.resident=std::make_shared<Entry>(entry); cache.resident_key=encoded;
            cache.counters.resident_bytes=payload_bytes;
        }
        if(cache.directory.empty()) return true;
        const uint64_t bytes=header_bytes+encoded.size()+payload_bytes;
        require(bytes<=cache.options.disk_budget_bytes,"Entry exceeds disk cache byte budget");
        const auto target=cache.directory/filename(encoded);
        cleanup_temporaries(cache.directory);
        auto files=scan(cache.directory,cache.counters.disk_bytes);
        // Evict before writing so the complete temporary file also fits budget.
        // Overwriting an entry first removes it; readers may observe a miss but
        // never partially-written tensors. Validity is independent of cache hits.
        for(const auto& file:files) {
            if(file.path!=target && cache.counters.disk_bytes<=cache.options.disk_budget_bytes-bytes) continue;
            if(!file.removable || !owned_file(file.path,file.path.filename().string(),false)) continue;
            fs::remove(file.path); cache.counters.disk_bytes-=file.bytes; ++cache.counters.evictions;
        }
        require(cache.counters.disk_bytes<=cache.options.disk_budget_bytes-bytes,"Cannot satisfy disk cache budget");
        if(!write_file(target.string(),key,entry,error)) return false;
        cache.counters.disk_bytes+=bytes; ++cache.counters.writes; return true;
    } catch(const std::exception& e) { error=e.what(); return false; }
}

bool clear(Cache& cache,bool include_disk,std::string& error) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    error.clear(); cache.resident.reset(); cache.resident_key.clear(); cache.counters.resident_bytes=0;
    try {
        if(include_disk && !cache.directory.empty()) {
            cleanup_temporaries(cache.directory);
            const auto files=scan(cache.directory,cache.counters.disk_bytes);
            for(const auto& file:files) if(file.removable && owned_file(file.path,file.path.filename().string(),false)) { fs::remove(file.path); cache.counters.disk_bytes-=file.bytes; }
            if(cache.counters.disk_bytes) {
                error="Unvalidated/corrupt entries were preserved; clear removed only validated cache files";
                return false;
            }
        }
        return true;
    } catch(const std::exception& e) { error=e.what(); return false; }
}
Stats stats(Cache& cache) { std::lock_guard<std::mutex> lock(cache.mutex); return cache.counters; }
} }
