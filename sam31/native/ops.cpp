#include "runtime.h"
#include <atomic>
#ifdef SAM31_USE_BLAS
#include <cblas.h>
#endif
namespace sam31 {
    namespace native {
        namespace {
            // Row-major A[m,k], B[n,k] -> C[m,n], or ordinary A[m,k] B[k,n].
            void gemm(const float*a,const float*b,float*c,size_t m,size_t n,size_t k,size_t lda,size_t ldb,size_t ldc,bool bt) {
                require(std::max( {
                    m,n,k,lda,ldb,ldc
                }
                )<=size_t(std::numeric_limits<int>::max()),"BLAS dimension overflow");
#ifdef SAM31_USE_BLAS
                cblas_sgemm(CblasRowMajor,CblasNoTrans,bt?CblasTrans:CblasNoTrans,int(m),int(n),int(k),1,a,int(lda),b,int(ldb),0,c,int(ldc));
#else
                for(size_t i=0; i<m; ++i)for(size_t j=0; j<n; ++j) {
                    float s=0;
                    for(size_t l=0; l<k; ++l)s+=a[i*lda+l]*b[bt?j*ldb+l:l*ldb+j];
                    c[i*ldc+j]=s;
                }
#endif
            }
        }
        Tensor add(Tensor x,const Tensor&y,float scale) {
            require(x.rows()==y.rows()&&x.c==y.c,"add shape mismatch");
            for(size_t i=0; i<x.v.size(); ++i)x.v[i]+=scale*y.v[i];
            return x;
        }
        Tensor add_vector(Tensor x,const std::vector<float>&v,float scale) {
            require(v.size()==x.c,"channel broadcast mismatch");
            for(size_t n=0; n<x.rows(); ++n)for(size_t c=0; c<x.c; ++c)x.row(n)[c]+=scale*v[c];
            return x;
        }
        Tensor slice_channels(const Tensor&x,size_t begin,size_t count) {
            require(count&&begin<=x.c&&count<=x.c-begin,"invalid channel slice");
            Tensor out(x.h,x.w,count);
            for(size_t n=0; n<x.rows(); ++n)std::copy_n(x.row(n)+begin,count,out.row(n));
            return out;
        }
        Tensor slice_rows(const Tensor&x,size_t begin,size_t count) {
            require(count&&begin<=x.rows()&&count<=x.rows()-begin,"invalid row slice");
            Tensor out(count,1,x.c);
            std::copy_n(x.row(begin),count*x.c,out.v.data());
            return out;
        }
        Tensor concat(const std::vector<Tensor>&xs) {
            require(!xs.empty(),"empty concatenation");
            size_t rows=0;
            for(const auto&x:xs) {
                require(x.c==xs[0].c,"concatenation channels");
                require(x.rows()<=std::numeric_limits<size_t>::max()-rows,"concatenation overflow");
                rows+=x.rows();
            }
            Tensor out(rows,1,xs[0].c);
            size_t at=0;
            for(const auto&x:xs) {
                std::copy(x.v.begin(),x.v.end(),out.v.begin()+at);
                at+=x.v.size();
            }
            return out;
        }
        Tensor gelu(Tensor x) {
            for(float&v:x.v)v=0.5f*v*(1.0f+std::erf(v*0.7071067811865475244f));
            return x;
        }
        Tensor relu(Tensor x) {
            for(float&v:x.v)v=std::max(0.0f,v);
            return x;
        }
        Tensor resize(const Tensor&x,size_t h,size_t w,bool antialias) {
            require(h&&w&&h<=16384&&w<=16384,"invalid resize dimensions");
            if(x.h==h&&x.w==w)return x;
            // Separable triangle filter. Pixel-center mapping matches align_corners=False.
            // Downsampling widens the triangle only when antialias is requested.
            struct Tap {
                size_t i;
                float weight;
            };
            auto taps=[](size_t in,size_t out,bool aa) {
                std::vector<std::vector<Tap>>t(out);
                double scale=double(in)/out, support=aa?std::max(1.0,scale):1.0;
                for(size_t j=0; j<out; ++j) {
                    double center=(j+0.5)*scale-0.5;
                    long lo=long(std::floor(center-support+1)),hi=long(std::ceil(center+support));
                    double sum=0;
                    for(long i=lo; i<hi; ++i) {
                        double weight=std::max(0.0,1.0-std::abs(double(i)-center)/support);
                        if(weight==0)continue;
                        if(aa&&support>1&&(i<0||i>=long(in)))continue;
                        size_t idx=size_t(std::max(0L,std::min(long(in)-1,i)));
                        t[j].push_back( {
                            idx,float(weight)
                        }
                        );
                        sum+=weight;
                    }
                    require(sum>0,"invalid resampling weights");
                    for(auto&a:t[j])a.weight=float(a.weight/sum);
                }
                return t;
            };
            auto tx=taps(x.w,w,antialias),ty=taps(x.h,h,antialias);
            Tensor tmp(x.h,w,x.c),out(h,w,x.c);
            for(size_t y=0; y<x.h; ++y)for(size_t z=0; z<w; ++z)for(const auto&t:tx[z])for(size_t c=0; c<x.c; ++c)tmp.row(y*w+z)[c]+=t.weight*x.row(y*x.w+t.i)[c];
            for(size_t y=0; y<h; ++y)for(size_t z=0; z<w; ++z)for(const auto&t:ty[y])for(size_t c=0; c<x.c; ++c)out.row(y*w+z)[c]+=t.weight*tmp.row(t.i*w+z)[c];
            return out;
        }
        Tensor sine_position(size_t h,size_t w,size_t c) {
            require(c%4==0,"position channels must be divisible by four");
            Tensor out(h,w,c);
            const size_t f=c/2;
            const double pi=3.14159265358979323846;
            for(size_t y=0; y<h; ++y)for(size_t x=0; x<w; ++x)for(size_t j=0; j<f; ++j) {
                double den=std::pow(10000.0,2.0*(j/2)/f),ay=(y+1.0)/(h+1e-6)*2*pi/den,ax=(x+1.0)/(w+1e-6)*2*pi/den;
                out.row(y*w+x)[j]=float(j%2?std::cos(ay):std::sin(ay));
                out.row(y*w+x)[j+f]=float(j%2?std::cos(ax):std::sin(ax));
            }
            return out;
        }
        Tensor random_position(const Weight&g,size_t h,size_t w) {
            require(g.shape.size()==2&&g.shape[0]==2,"Gaussian PE shape");
            size_t f=g.shape[1];
            Tensor out(h,w,2*f);
            const float tau=6.283185307179586f;
            for(size_t y=0; y<h; ++y)for(size_t x=0; x<w; ++x)for(size_t i=0; i<f; ++i) {
                float a=tau*((2.0f*(x+0.5f)/w-1)*g.v[i]+(2.0f*(y+0.5f)/h-1)*g.v[f+i]);
                out.row(y*w+x)[i]=std::sin(a);
                out.row(y*w+x)[f+i]=std::cos(a);
            }
            return out;
        }
        void rope(Tensor&x,size_t heads,size_t grid,float scale,size_t exclude) {
            require(heads&&x.c%heads==0&&x.c/heads%4==0&&exclude<=x.rows()&&grid>0,"invalid rotary dimensions");
            size_t d=x.c/heads,n=x.rows()-exclude;
            require(n%(grid*grid)==0,"rotary spatial sequence mismatch");
            for(size_t i=0; i<n; ++i) {
                size_t pos=i%(grid*grid);
                for(size_t head=0; head<heads; ++head)for(size_t pair=0; pair<d/2; ++pair) {
                    size_t j=pair%(d/4);
                    float axis=float(pair<d/4?pos%grid:pos/grid);
                    float a=axis*scale/std::pow(10000.0f,float(4*j)/d),cs=std::cos(a),sn=std::sin(a);
                    float*r=x.row(i)+head*d+2*pair,real=r[0],imag=r[1];
                    r[0]=real*cs-imag*sn;
                    r[1]=real*sn+imag*cs;
                }
            }
        }
        Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, size_t heads) {
            require(heads && q.c == k.c && q.c % heads == 0 && v.c % heads == 0 && k.rows() == v.rows(), "attention shape mismatch");
            const size_t nq = q.rows(), nk = k.rows(), d = q.c / heads;
            const size_t dv = v.c / heads, block = 32;
            require(nq > 0 && nk > 0, "empty attention sequence");
            Tensor out(q.h, q.w, v.c);
            const float scale = 1.0f / std::sqrt(float(d));
            std::atomic<bool> invalid {
                false
            };
            // Heads write disjoint output channels. Each head owns a bounded score buffer.
            // No N-by-N attention matrix is allocated, even with production dimensions.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(heads > 1)
#endif
            for (long hi = 0; hi < long(heads); ++hi) {
                const size_t head = size_t(hi);
                std::vector<float> scores(std::min(nq, block) * nk);
                for (size_t start = 0; start < nq; start += block) {
                    const size_t count = std::min(block, nq - start);
                    gemm(q.row(start) + head * d, k.v.data() + head * d, scores.data(), count, nk, d, q.c, k.c, nk, true);
                    for (size_t i = 0; i < count; ++i) {
                        float *row = scores.data() + i * nk;
                        float maxv = -std::numeric_limits<float>::infinity();
                        for (size_t j = 0; j < nk; ++j) {
                            row[j] *= scale;
                            maxv = std::max(maxv, row[j]);
                        }
                        double total = 0;
                        for (size_t j = 0; j < nk; ++j) {
                            row[j] = std::exp(row[j] - maxv);
                            total += row[j];
                        }
                        if (!std::isfinite(total) || total <= 0) {
                            invalid.store(true, std::memory_order_relaxed);
                            total = 1;
                        }
                        for (size_t j = 0; j < nk; ++j) row[j] = float(row[j] / total);
                    }
                    gemm(scores.data(), v.v.data() + head * dv, out.row(start) + head * dv, count, dv, nk, nk, v.c, out.c, false);
                }
            }
            require(!invalid.load(std::memory_order_relaxed), "non-finite attention");
            return out;
        }
        Tensor Ops::linear(const Tensor&x,const std::string&p,bool bias) {
            ++calls.linear;
            auto w=weights.get(p+".weight");
            require(w->shape.size()==2&&w->shape[1]==x.c,"linear shape: "+p);
            Tensor out(x.h,x.w,w->shape[0]);
            gemm(x.v.data(),w->v.data(),out.v.data(),x.rows(),out.c,x.c,x.c,x.c,out.c,true);
            if(bias)out=add_vector(std::move(out),weights.get(p+".bias")->v);
            return out;
        }
        Tensor Ops::norm(const Tensor&x,const std::string&p,float eps) {
            auto w=weights.get(p+".weight"),b=weights.get(p+".bias");
            require(w->v.size()==x.c&&b->v.size()==x.c,"norm shape: "+p);
            Tensor out(x.h,x.w,x.c);
            for(size_t n=0; n<x.rows(); ++n) {
                double mean=0,var=0;
                for(size_t c=0; c<x.c; ++c)mean+=x.row(n)[c];
                mean/=x.c;
                for(size_t c=0; c<x.c; ++c) {
                    double d=x.row(n)[c]-mean;
                    var+=d*d;
                }
                float inv=float(1/std::sqrt(var/x.c+eps));
                for(size_t c=0; c<x.c; ++c)out.row(n)[c]=float(x.row(n)[c]-mean)*inv*w->v[c]+b->v[c];
            }
            return out;
        }
        Tensor Ops::conv(const Tensor&x,const std::string&p,size_t stride,size_t pad,bool bias,bool depthwise) {
            ++calls.conv;
            auto w=weights.get(p+".weight");
            require(w->shape.size()==4,"conv rank: "+p);
            size_t oc=w->shape[0],ic=w->shape[1],kh=w->shape[2],kw=w->shape[3];
            require(stride>0&&kh==kw&&x.h+2*pad>=kh&&x.w+2*pad>=kw,"conv dimensions: "+p);
            require(depthwise?(ic==1&&oc==x.c):ic==x.c,"conv channels: "+p);
            Tensor out((x.h+2*pad-kh)/stride+1,(x.w+2*pad-kw)/stride+1,oc);
            if(depthwise) {
                for(size_t oy=0; oy<out.h; ++oy)for(size_t ox=0; ox<out.w; ++ox)for(size_t ky=0; ky<kh; ++ky)for(size_t kx=0; kx<kw; ++kx) {
                    long iy=long(oy*stride+ky)-long(pad),ix=long(ox*stride+kx)-long(pad);
                    if(iy<0||ix<0||iy>=long(x.h)||ix>=long(x.w))continue;
                    const float*r=x.row(size_t(iy)*x.w+size_t(ix));
                    float*z=out.row(oy*out.w+ox);
                    for(size_t c=0; c<oc; ++c)z[c]+=r[c]*w->v[(c*kh+ky)*kw+kx];
                }
            }
            else if(kh==1&&stride==1&&pad==0)gemm(x.v.data(),w->v.data(),out.v.data(),x.rows(),oc,ic,ic,ic,oc,true);
            else {
                size_t inner=ic*kh*kw,block=128;
                std::vector<float>col(std::min(block,out.rows())*inner);
                for(size_t start=0; start<out.rows(); start+=block) {
                    size_t count=std::min(block,out.rows()-start);
                    std::fill(col.begin(),col.end(),0);
                    for(size_t n=0; n<count; ++n) {
                        size_t oy=(start+n)/out.w,ox=(start+n)%out.w;
                        for(size_t ky=0; ky<kh; ++ky)for(size_t kx=0; kx<kw; ++kx) {
                            long iy=long(oy*stride+ky)-long(pad),ix=long(ox*stride+kx)-long(pad);
                            if(iy<0||ix<0||iy>=long(x.h)||ix>=long(x.w))continue;
                            const float*r=x.row(size_t(iy)*x.w+size_t(ix));
                            for(size_t c=0; c<ic; ++c)col[n*inner+(c*kh+ky)*kw+kx]=r[c];
                        }
                    }
                    gemm(col.data(),w->v.data(),out.row(start),count,oc,inner,inner,inner,oc,true);
                }
            }
            if(bias)out=add_vector(std::move(out),weights.get(p+".bias")->v);
            return out;
        }
        Tensor Ops::deconv(const Tensor&x,const std::string&p) {
            ++calls.deconv;
            auto w=weights.get(p+".weight");
            require(w->shape.size()==4&&w->shape[0]==x.c&&w->shape[2]==2&&w->shape[3]==2,"stride-two deconv shape: "+p);
            size_t oc=w->shape[1];
            Tensor out(x.h*2,x.w*2,oc);
            std::vector<float>matrix(oc*x.c),tmp(x.rows()*oc);
            for(size_t dy=0; dy<2; ++dy)for(size_t dx=0; dx<2; ++dx) {
                for(size_t o=0; o<oc; ++o)for(size_t i=0; i<x.c; ++i)matrix[o*x.c+i]=w->v[((i*oc+o)*2+dy)*2+dx];
                gemm(x.v.data(),matrix.data(),tmp.data(),x.rows(),oc,x.c,x.c,x.c,oc,true);
                for(size_t y=0; y<x.h; ++y)for(size_t z=0; z<x.w; ++z)std::copy_n(tmp.data()+(y*x.w+z)*oc,oc,out.row((2*y+dy)*out.w+2*z+dx));
            }
            return add_vector(std::move(out),weights.get(p+".bias")->v);
        }
        Tensor Ops::mlp(Tensor x,const std::string&p,size_t layers) {
            for(size_t i=0; i<layers; ++i) {
                x=linear(x,p+".layers."+std::to_string(i));
                if(i+1<layers)x=relu(std::move(x));
            }
            return x;
        }
        Tensor Ops::projected_attention(const Tensor&q,const Tensor&k,const Tensor&v,const std::string&p,size_t heads) {
            auto qp=linear(q,p+".q_proj"),kp=linear(k,p+".k_proj"),vp=linear(v,p+".v_proj");
            return linear(sdpa(qp,kp,vp,heads),p+".out_proj");
        }
    }
}
