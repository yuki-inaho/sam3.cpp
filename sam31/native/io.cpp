#include "sam31_native.h"
#include "runtime.h"
#include <fstream>
#include <memory>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
namespace sam31 {
    namespace native {
        Image read_image(const std::string&path) {
            int w=0,h=0,c=0;
            require(stbi_info(path.c_str(),&w,&h,&c)!=0,"cannot inspect image: "+path);
            require(w>0&&h>0&&w<=16384&&h<=16384&&size_t(w)*size_t(h)<=(size_t(1)<<26),"image size limit");
            using Ptr=std::unique_ptr<unsigned char,decltype(&stbi_image_free)>;
            Ptr data(stbi_load(path.c_str(),&w,&h,&c,3),&stbi_image_free);
            require(bool(data),"cannot decode image: "+path);
            Image image;
            image.width=size_t(w);
            image.height=size_t(h);
            image.rgb.assign(data.get(),data.get()+size_t(w)*size_t(h)*3);
            return image;
        }
        void write_image_png(const std::string&path,const Image&image) {
            require(image.rgb.size()==product( {
                image.width,image.height,3
            }
            ),"RGB shape mismatch");
            require(stbi_write_png(path.c_str(),int(image.width),int(image.height),3,image.rgb.data(),int(image.width*3))!=0,"cannot write RGB image");
        }
        void write_mask_png(const std::string&path,const ObjectMask&mask,size_t width,size_t height) {
            require(mask.logits.size()==product( {
                width,height
            }
            ),"mask shape mismatch");
            std::vector<uint8_t>pixels(mask.logits.size());
            for(size_t i=0; i<pixels.size(); ++i) {
                require(std::isfinite(mask.logits[i]),"non-finite mask");
                pixels[i]=mask.logits[i]>0?255:0;
            }
            require(stbi_write_png(path.c_str(),int(width),int(height),1,pixels.data(),int(width))!=0,"cannot write mask: "+path);
        }
        // Portable raw float output (little endian), accompanied by JSON shapes and IDs.
        void write_logits(const std::string&path,const Result&result) {
            std::ofstream out(path,std::ios::binary);
            require(bool(out),"cannot write logits: "+path);
            for(const auto&object:result.objects) {
                require(object.logits.size()==product( {
                    result.height,result.width
                }
                ),"logit shape mismatch");
                for(float value:object.logits) {
                    require(std::isfinite(value),"non-finite logits");
                    uint32_t bits;
                    std::memcpy(&bits,&value,4);
                    char b[4]= {
                        char(bits),char(bits>>8),char(bits>>16),char(bits>>24)
                    };
                    out.write(b,4);
                }
            }
            out.flush();
            require(bool(out),"logits write failed");
        }
    }
}
