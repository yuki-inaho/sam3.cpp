#include "weights.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sys/resource.h>

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
static void check(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
struct run {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{nullptr, ggml_free};
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cplan plan{};
    std::vector<uint8_t> workspace;
};
static run make_run(const sam2_cpu_quant::packed_weight & w, const std::vector<float> & x,
                    int64_t tokens, int threads, ggml_threadpool * pool) {
    run r;
    const size_t size = w.bytes.size() + x.size()*4 + size_t(w.rows*tokens)*4 + 4*1024*1024;
    r.ctx.reset(ggml_init({size, nullptr, false})); check(bool(r.ctx), "ggml context allocation failed");
    auto * weight = ggml_new_tensor_2d(r.ctx.get(), w.type, w.cols, w.rows);
    auto * input = ggml_new_tensor_2d(r.ctx.get(), GGML_TYPE_F32, w.cols, tokens);
    std::memcpy(weight->data, w.bytes.data(), w.bytes.size());
    std::memcpy(input->data, x.data(), x.size()*4);
    r.output = ggml_mul_mat(r.ctx.get(), weight, input);
    r.graph = ggml_new_graph(r.ctx.get()); ggml_build_forward_expand(r.graph, r.output);
    r.plan = ggml_graph_plan(r.graph, threads, pool);
    r.workspace.resize(r.plan.work_size); r.plan.work_data = r.workspace.data();
    return r;
}
static double compute(run & r) {
    auto start = Clock::now();
    check(ggml_graph_compute(r.graph, &r.plan) == GGML_STATUS_SUCCESS, "ggml compute failed");
    return elapsed(start);
}
static double relative_error(run & r, const std::vector<double> & oracle) {
    auto * y = static_cast<const float *>(r.output->data);
    double error=0, norm=0;
    for (size_t i=0;i<oracle.size();++i) {
        check(std::isfinite(y[i]), "Non-finite output");
        error += (y[i]-oracle[i])*(y[i]-oracle[i]); norm += oracle[i]*oracle[i];
    }
    return std::sqrt(error/std::max(norm,1e-30));
}
static void dump(const std::string & path, const void * data, size_t size) {
    std::ofstream f(path, std::ios::binary); f.write(static_cast<const char *>(data), size);
    check(bool(f), "Fixture write failed");
}
static double median(std::vector<double> a) { std::sort(a.begin(),a.end()); return a[a.size()/2]; }
static void test_pack() {
    std::vector<float> w(112*3, 0.25f);
    auto p = sam2_cpu_quant::pack(w.data(),w.size(),112,3,true);
    check(p.type==GGML_TYPE_F32 && p.bytes.size()==w.size()*4, "112 must fallback without padding");
    check(!std::memcmp(w.data(),p.bytes.data(),p.bytes.size()), "Fallback changed weights");
    w.assign(64,0);
    check(sam2_cpu_quant::pack(w.data(),64,32,2).type==GGML_TYPE_F32,"Default must remain FP32");
    check(sam2_cpu_quant::pack(w.data(),64,32,2,true).type==GGML_TYPE_Q8_0,"Eligible shape rejected");
    w[0]=1e-30f;
    check(sam2_cpu_quant::pack(w.data(),64,32,2,true).type==GGML_TYPE_F32,"Scale underflow must fallback");
    w[0]=1e20f;
    check(sam2_cpu_quant::pack(w.data(),64,32,2,true).type==GGML_TYPE_F32,"Scale overflow must fallback");
    auto rejects=[&](const float * data,size_t n,int64_t k,int64_t rows) {
        try { (void)sam2_cpu_quant::pack(data,n,k,rows,true); } catch(const std::invalid_argument &) { return true; }
        return false;
    };
    check(rejects(nullptr,64,32,2),"Null accepted");
    check(rejects(w.data(),63,32,2),"Wrong count accepted");
    check(rejects(w.data(),64,0,2),"Zero shape accepted");
    check(rejects(w.data(),64,INT64_MAX,INT64_MAX),"Overflow shape accepted");
    w[0]=std::numeric_limits<float>::quiet_NaN(); check(rejects(w.data(),64,32,2),"NaN accepted");
    w[0]=std::numeric_limits<float>::infinity(); check(rejects(w.data(),64,32,2),"Infinity accepted");
}
int main(int argc,char ** argv) {
    try {
        test_pack();
        const int threads = argc>1 ? std::atoi(argv[1]) : 1;
        const int repeats = argc>2 ? std::atoi(argv[2]) : 11;
        check(threads>0 && threads<=64 && repeats>=3,"Usage: benchmark [threads1..64] [repeats>=3]");
        auto params=ggml_threadpool_params_default(threads);
        std::unique_ptr<ggml_threadpool,decltype(&ggml_threadpool_free)> pool(ggml_threadpool_new(&params),ggml_threadpool_free);
        check(bool(pool),"Threadpool allocation failed");
        const int shapes[][3]={{112,336,64},{224,672,64},{448,1344,64},{896,2688,64},{448,1792,256},{1792,448,256}};
        std::cout << "K,N,M,threads,repeats,candidate,fp32_bytes,candidate_bytes,pack_ms,setup_ms,fp32_ms,candidate_ms,paired_speedup,fp32_rel_l2,candidate_rel_l2,peak_rss_kib\n";
        for (auto & s : shapes) {
            const int k=s[0],n=s[1],m=s[2];
            std::mt19937 rng(20261013); std::normal_distribution<float> normal(0,1);
            std::vector<float> w(k*n),x(k*m);
            for (auto & v:w) v=normal(rng)/std::sqrt(float(k));
            for(auto &v:x) v=normal(rng);
            auto start=Clock::now(); auto q=sam2_cpu_quant::pack(w.data(),w.size(),k,n,true); double pack_ms=elapsed(start);
            auto f=sam2_cpu_quant::pack(w.data(),w.size(),k,n,false);
            start=Clock::now(); auto a=make_run(f,x,m,threads,pool.get()); auto b=make_run(q,x,m,threads,pool.get()); double setup_ms=elapsed(start);
            std::vector<double> oracle(size_t(n)*m,0);
            for(int t=0;t<m;++t) for(int o=0;o<n;++o) for(int j=0;j<k;++j) oracle[t*n+o]+=double(x[t*k+j])*w[o*k+j];
            for(int i=0;i<3;++i) { compute(a); compute(b); }
            std::vector<double> at,bt,ratio;
            for(int i=0;i<repeats;++i) {
                double av,bv; if(i%2) { bv=compute(b); av=compute(a); } else { av=compute(a); bv=compute(b); }
                at.push_back(av); bt.push_back(bv); ratio.push_back(av/bv);
                std::cerr << "sample,"<<k<<','<<n<<','<<m<<','<<threads<<','<<i<<','<<av<<','<<bv<<'\n';
            }
            const double ae=relative_error(a,oracle),be=relative_error(b,oracle);
            check(ae<=1e-5 && be<=.02,"Numerical microbenchmark tolerance exceeded");
            if(k==112) check(!std::memcmp(a.output->data,b.output->data,oracle.size()*4),"112 fallback output differs");
            if (argc>3) {
                const std::string prefix=std::string(argv[3])+"/"+std::to_string(k)+"_"+std::to_string(n)+"_"+std::to_string(m);
                dump(prefix+"_w.f32",w.data(),w.size()*4); dump(prefix+"_x.f32",x.data(),x.size()*4);
                dump(prefix+"_y.f32",b.output->data,oracle.size()*4);
                dump(prefix+"_w.packed",q.bytes.data(),q.bytes.size());
                if(q.type==GGML_TYPE_Q8_0) {
                    std::vector<uint8_t> packed_x(ggml_row_size(q.type,k)*m);
                    ggml_get_type_traits_cpu(q.type)->from_float(x.data(),packed_x.data(),int64_t(x.size()));
                    dump(prefix+"_x.packed",packed_x.data(),packed_x.size());
                }
            }
            struct rusage usage{}; getrusage(RUSAGE_SELF,&usage);
            std::cout<<std::setprecision(8)<<k<<','<<n<<','<<m<<','<<threads<<','<<repeats<<','<<ggml_type_name(q.type)<<','<<f.bytes.size()<<','<<q.bytes.size()<<','<<pack_ms<<','<<setup_ms<<','<<median(at)<<','<<median(bt)<<','<<median(ratio)<<','<<ae<<','<<be<<','<<usage.ru_maxrss<<std::endl;
        }
        ggml_quantize_free();
    } catch(const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
}
