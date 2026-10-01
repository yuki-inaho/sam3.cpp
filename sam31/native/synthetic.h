#pragma once
#include "sam31_native.h"
#include <cmath>
namespace sam31 {
    namespace native {
        inline Image synthetic_image(size_t frame,size_t width=80,size_t height=64) {
            Image im;
            im.width=width;
            im.height=height;
            im.rgb.resize(width*height*3);
            for(size_t y=0; y<height; ++y)for(size_t x=0; x<width; ++x) {
                size_t p=(y*width+x)*3;
                im.rgb[p]=uint8_t(20+x%17);
                im.rgb[p+1]=uint8_t(27+y%13);
                im.rgb[p+2]=35;
                float cx=width*(0.25f+0.015f*float(frame%10)),cy=height*0.32f;
                float dx=(float(x)-cx)/(width*0.13f),dy=(float(y)-cy)/(height*0.18f);
                if(dx*dx+dy*dy<1) {
                    im.rgb[p]=225;
                    im.rgb[p+1]=45;
                    im.rgb[p+2]=30;
                }
                float left=width*(0.62f-0.01f*float(frame%10)),top=height*(0.55f+0.007f*float(frame%10));
                if(float(x)>=left&&float(x)<left+width*0.23f&&float(y)>=top&&float(y)<top+height*0.25f) {
                    im.rgb[p]=40;
                    im.rgb[p+1]=195;
                    im.rgb[p+2]=225;
                }
            }
            return im;
        }
    }
}
