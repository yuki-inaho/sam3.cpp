#include "digest.h"
extern "C" {
#include "third_party/sha256/sha256.h"
}
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
namespace efficient_sam2 { namespace prefix_cache {
namespace {
std::string finish(sha256_t& hash) {
    unsigned char digest[SHA256_DIGEST_SIZE]; sha256_final(&hash,digest);
    static const char hex[]="0123456789abcdef";
    std::string out;out.reserve(64);
    for(auto v:digest) {out.push_back(hex[v>>4]);out.push_back(hex[v&15]);} return out;
}
}
std::string sha256_bytes(const void* data,size_t bytes) {
    if(!data && bytes) throw std::invalid_argument("Null nonempty SHA256 input");
    sha256_t hash;sha256_init(&hash);
    if(bytes) sha256_update(&hash,static_cast<const unsigned char*>(data),bytes);
    return finish(hash);
}
std::string sha256_file(const std::string& path,std::string& error) {
    error.clear();
    std::ifstream input(path,std::ios::binary);
    if(!input) {error="Cannot open content-hash input";return {};}
    sha256_t hash;sha256_init(&hash);std::array<unsigned char,65536> buffer{};
    while(input) {
        input.read(reinterpret_cast<char*>(buffer.data()),buffer.size());
        const auto n=input.gcount();if(n>0) sha256_update(&hash,buffer.data(),size_t(n));
    }
    if(!input.eof()) {error="Content-hash input read failed";return {};}
    return finish(hash);
}
std::string sha256_rgb(const uint8_t* rgb,size_t bytes,int width,int height) {
    if(width<=0 || height<=0 || uint64_t(width)>std::numeric_limits<size_t>::max()/uint64_t(height)/3 ||
       bytes!=size_t(width)*size_t(height)*3 || !rgb)
        throw std::invalid_argument("Invalid contiguous RGB geometry/byte count");
    sha256_t hash;sha256_init(&hash);
    static const unsigned char domain[]="efficient-sam2-rgb-v1\n";
    sha256_update(&hash,domain,sizeof(domain)-1);
    unsigned char geometry[16];
    for(int i=0;i<8;++i) {geometry[i]=uint8_t(uint64_t(width)>>(8*i));geometry[8+i]=uint8_t(uint64_t(height)>>(8*i));}
    sha256_update(&hash,geometry,sizeof(geometry));sha256_update(&hash,rgb,bytes);
    return finish(hash);
}
} }
