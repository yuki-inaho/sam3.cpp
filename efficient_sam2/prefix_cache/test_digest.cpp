#include "digest.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace efficient_sam2::prefix_cache;
static void check(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
int main(int argc,char**argv) {
    try {
        check(argc==2,"Usage: test_digest TEMPORARY_FILE");
        check(sha256_bytes(nullptr,0)=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","Empty SHA256 failed");
        check(sha256_bytes("abc",3)=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","abc SHA256 failed");
        const std::string longer="abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        check(sha256_bytes(longer.data(),longer.size())=="248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1","Multi-block SHA256 failed");
        const std::string million(1000000,'a');
        check(sha256_bytes(million.data(),million.size())=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0","Million-a SHA256 failed");
        std::vector<uint8_t> bytes(256);for(size_t i=0;i<bytes.size();++i)bytes[i]=uint8_t(i);
        std::cout<<"bytes256="<<sha256_bytes(bytes.data(),bytes.size())<<'\n';
        std::string error;
        {std::ofstream out(argv[1],std::ios::binary);out.write(million.data(),million.size());check(bool(out),"Fixture file write failed");}
        const auto file=sha256_file(argv[1],error);check(error.empty()&&file==sha256_bytes(million.data(),million.size()),"Streaming file SHA256 differs");
        std::cout<<"file_million_a="<<file<<'\n';std::remove(argv[1]);
        std::cout<<"rgb2x1="<<sha256_rgb(bytes.data(),6,2,1)<<'\n';
        check(sha256_rgb(bytes.data(),6,2,1)!=sha256_rgb(bytes.data(),6,1,2),"RGB geometry not included");
        check(sha256_file(std::string(argv[1])+"-missing",error).empty()&&!error.empty(),"Missing hash input accepted");
        bool rejected=false;try{sha256_rgb(bytes.data(),5,2,1);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"Invalid RGB count accepted");
        rejected=false;try{sha256_bytes(nullptr,1);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"Null byte input accepted");
        std::cout<<"PASS: SHA256 known vectors, streamed file, RGB geometry and input checks\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
