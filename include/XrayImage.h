#pragma once
#include <opencv2/imgproc.hpp>
namespace orthoseg {
inline cv::Mat displayXray(const cv::Mat& raw) {
    if(raw.empty())return {};
    cv::Mat display;
    if(raw.depth()==CV_8U)display=raw;
    else {
        double low,high;cv::minMaxLoc(raw.reshape(1),&low,&high);
        raw.convertTo(display,CV_8U,high>low?255.0/(high-low):0,high>low?-low*255.0/(high-low):0);
    }
    cv::Mat color;
    if(display.channels()==1)cv::cvtColor(display,color,cv::COLOR_GRAY2BGR);
    else if(display.channels()==4)cv::cvtColor(display,color,cv::COLOR_BGRA2BGR);
    else if(display.channels()==3)color=display;
    return color;
}
}
