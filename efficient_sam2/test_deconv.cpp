// Verify the F32 ConvTranspose primitive used by the Efficient decoder against
// a direct scalar oracle; the legacy F16 primitive is kept as a control.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <vector>

static bool run(ggml_type type, double & error) {
    auto * backend=ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend,1);
    ggml_init_params params{ggml_tensor_overhead()*32+ggml_graph_overhead(),nullptr,true};
    auto * ctx=ggml_init(params);
    auto * weights=ggml_new_tensor_4d(ctx,type,2,2,3,2);
    auto * input=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,2,3,2,1);
    auto * output=ggml_conv_transpose_2d_p0(ctx,weights,input,2);
    auto * graph=ggml_new_graph(ctx);
    ggml_build_forward_expand(graph,output);
    auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    std::vector<float> w(24),x(12),actual(72),expected(72);
    for(size_t i=0;i<w.size();++i) w[i]=float(int(i%9)-4)*.12347f+.00031f;
    for(size_t i=0;i<x.size();++i) x[i]=float(int(i%7)-3)*.73119f+.00043f;
    if(type==GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(w.size());
        ggml_fp32_to_fp16_row(w.data(),half.data(),w.size());
        ggml_backend_tensor_set(weights,half.data(),0,half.size()*sizeof(ggml_fp16_t));
    } else ggml_backend_tensor_set(weights,w.data(),0,w.size()*sizeof(float));
    ggml_backend_tensor_set(input,x.data(),0,x.size()*sizeof(float));
    const bool computed=ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS;
    if(computed) ggml_backend_tensor_get(output,actual.data(),0,actual.size()*sizeof(float));
    for(int ci=0;ci<2;++ci) for(int y=0;y<3;++y) for(int x0=0;x0<2;++x0)
        for(int co=0;co<3;++co) for(int ky=0;ky<2;++ky) for(int kx=0;kx<2;++kx)
            expected[(2*x0+kx)+4*((2*y+ky)+6*co)]+=x[x0+2*(y+3*ci)]*w[kx+2*(ky+2*(co+3*ci))];
    double squared=0,norm=0;
    for(size_t i=0;i<actual.size();++i) {
        squared+=std::pow(double(actual[i])-expected[i],2); norm+=std::pow(double(expected[i]),2);
    }
    error=std::sqrt(squared/norm);
    ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(backend);
    return computed && std::isfinite(error);
}
int main() {
    double f32=0,f16=0;
    if(!run(GGML_TYPE_F32,f32) || !run(GGML_TYPE_F16,f16) || f32>1e-6 || f16<1e-5) return 1;
    std::printf("ConvTranspose scalar oracle: F32 relative L2 %.9g; inherited F16 control %.9g\n",f32,f16);
    return 0;
}
