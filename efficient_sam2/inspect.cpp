#include "gguf_loader.h"
#include <cstdio>
#include <fstream>
int main(int argc, char** argv) {
    if(argc!=2 && argc!=3 && argc!=4) {
        fprintf(stderr,"Usage: %s model.gguf [--native-profile | tensor.name output.f32]\n",argv[0]);
        return 2;
    }
    sam2_gguf_file file; std::string error;
    if(!sam2_gguf_open(argv[1],file,error)) { fprintf(stderr,"%s\n",error.c_str()); return 1; }
    if(argc==3) {
        if(std::string(argv[2])!="--native-profile" || !sam2_gguf_validate_native_profile(file,error)) {
            fprintf(stderr,"%s\n",error.c_str()); return 1;
        }
        puts("Native architecture/policy metadata is compatible; numerical validation is separate.");
    } else if(argc==4) {
        std::vector<float> values;
        if(!sam2_gguf_restore(file,argv[2],values,error)) { fprintf(stderr,"%s\n",error.c_str()); return 1; }
        std::ofstream output(argv[3],std::ios::binary);
        output.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(float));
        if(!output) return 1;
        printf("restored %zu values (FP32)\n",values.size());
    } else printf("Efficient-SAM2 storage v1: %zu tensors; variant=%s; FP32-restoration reference; inspect does not execute video\n",file.tensors.size(),file.variant.c_str());
    return 0;
}
