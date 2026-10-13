// Pillow-compatible resampling adaptation. See NOTICE.md and licenses/Pillow.txt.
#include "resize.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace efficient_sam2 { namespace {
struct Coefficients { int start; std::vector<int32_t> weights; };
double cubic(double x) {
    x=std::abs(x);
    if(x<1) return (1.5*x-2.5)*x*x+1;
    if(x<2) return ((x-5)*x+8)*x*(-.5)+2;
    return 0;
}
std::vector<Coefficients> coefficients(int input,int output) {
    const double scale=double(input)/output, filter_scale=std::max(scale,1.0), support=2*filter_scale;
    std::vector<Coefficients> result(output);
    for(int i=0;i<output;++i) {
        const double center=(i+.5)*scale;
        const int begin=std::max(0,int(center-support+.5));
        const int end=std::min(input,int(center+support+.5));
        std::vector<double> weights(end-begin);
        double sum=0;
        for(int j=begin;j<end;++j) { weights[j-begin]=cubic((j-center+.5)/filter_scale); sum+=weights[j-begin]; }
        auto& row=result[i]; row.start=begin;
        for(double w:weights) {
            const double scaled=w/sum*(1<<22);
            row.weights.push_back(int32_t(scaled+(scaled<0?-.5:.5)));
        }
    }
    return result;
}
uint8_t rounded(int64_t value) {
    // Avoid implementation-defined right shift of a negative signed integer.
    if(value<=0) return 0;
    return uint8_t(std::min<int64_t>(255,value/(1<<22)));
}
} // namespace
std::vector<uint8_t> resize_video_rgb(const uint8_t* rgb,int width,int height,int ow,int oh) {
    if(!rgb || width<=0 || height<=0 || ow<=0 || oh<=0 || width>32768 || height>32768 || ow>32768 || oh>32768)
        throw std::invalid_argument("Invalid RGB resize geometry");
    std::vector<uint8_t> horizontal(size_t(ow)*height*3),output(size_t(ow)*oh*3);
    const auto x_coeff=coefficients(width,ow),y_coeff=coefficients(height,oh);
    for(int y=0;y<height;++y) for(int x=0;x<ow;++x) for(int c=0;c<3;++c) {
        const auto& row=x_coeff[x]; int64_t sum=1<<21;
        for(size_t j=0;j<row.weights.size();++j) sum+=int64_t(rgb[(size_t(y)*width+row.start+j)*3+c])*row.weights[j];
        horizontal[(size_t(y)*ow+x)*3+c]=rounded(sum);
    }
    for(int y=0;y<oh;++y) for(int x=0;x<ow;++x) for(int c=0;c<3;++c) {
        const auto& row=y_coeff[y]; int64_t sum=1<<21;
        for(size_t j=0;j<row.weights.size();++j) sum+=int64_t(horizontal[((row.start+j)*ow+x)*3+c])*row.weights[j];
        output[(size_t(y)*ow+x)*3+c]=rounded(sum);
    }
    return output;
}
} // namespace
