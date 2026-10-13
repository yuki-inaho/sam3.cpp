#include "resize.h"
#include <fstream>
#include <cstdlib>
#include <iterator>
int main(int argc,char** argv) {
    if(argc!=7) return 2;
    std::ifstream input(argv[1],std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(input)),{});
    const int w=std::atoi(argv[3]),h=std::atoi(argv[4]),ow=std::atoi(argv[5]),oh=std::atoi(argv[6]);
    if(w<=0 || h<=0 || data.size()!=size_t(w)*h*3) return 2;
    const auto result=efficient_sam2::resize_video_rgb(data.data(),w,h,ow,oh);
    std::ofstream out(argv[2],std::ios::binary);
    out.write(reinterpret_cast<const char*>(result.data()),result.size());
    return out?0:1;
}
