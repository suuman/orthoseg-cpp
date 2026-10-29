#include "segmentor.hpp"
#include <cstdio>
int main(int argc,char** argv) {
    if (argc!=2) return 2;
    try {
        auto model=xray_ocv::XRaySegmentor::create(argv[1],"cpu",0.4f,false,true);
        // Narrow input covers padding; wide input covers weighted multi-tile merging.
        for (int width : {128,256}) {
            cv::Mat image(512,width,CV_8UC3,cv::Scalar(50,50,50));
            auto mask=model->predict(image).label_mask;
            if(mask.size()!=image.size() || cv::countNonZero(mask!=3)) return 1;
        }
        bool rejected=false;
        try {
            auto old=xray_ocv::XRaySegmentor::create(argv[1],"cpu",0.4f,false);
            old->predict(cv::Mat(512,128,CV_8UC3,cv::Scalar(0,0,0)));
        } catch (const std::exception&) { rejected=true; }
        if(!rejected) return 1;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
    return 0;
}
