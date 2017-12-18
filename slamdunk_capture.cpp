#include "slamdunk_capture.h"

#include <functional>
#include <iostream>
#include <memory>

#include <kalamos_context.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>



void SlamdunkCapture::Init (void) {
	return;
}

void SlamdunkCapture::CBstereo(kalamos::StereoYuvData const& data) {
	// callback function for stereo recording
	// only the left image is used

	//prepare left frame:
	std::vector<cv::Mat> channels_left_uv,channels_left_yuv;
	channels_left_uv.push_back(*(data.leftYuv[2]));	// BGR order for OpenCV
	channels_left_uv.push_back(*(data.leftYuv[1]));
	cv::Mat channel_y = *(data.leftYuv[0]);
	cv::Mat frame_left_uv,frame_left_yuv;
	cv::merge(channels_left_uv,frame_left_uv);
	cv::resize(frame_left_uv,frame_left_uv,output_image_size_,0,0);
	cv::resize(channel_y,channel_y,output_image_size_,0,0);
	cv::Mat tmp_left_uv[2];
	cv::split(frame_left_uv,tmp_left_uv); 
	channels_left_yuv.push_back(channel_y);
	channels_left_yuv.push_back(tmp_left_uv[0]);
	channels_left_yuv.push_back(tmp_left_uv[1]);
	cv::merge(channels_left_yuv,frame_left_yuv);  // all channels the same size

	mutex_image_1_.lock();
	cv::cvtColor(frame_left_yuv,frame_left_,CV_YUV2RGB);	// copy in RGB
	mutex_image_2_.unlock();

}


void SlamdunkCapture::Start (void) {
    std::cout << "Starting camera!" << std::endl;
    cam_running_=true;
    cam_thread_ = std::thread(&SlamdunkCapture::WorkerThread,this);
    WaitForImage();
    WaitForImage();
    im_width_ = frame_left_.cols;
    im_height_ = frame_left_.rows;
    std::cout << "Cam started! " << im_width_ << "x" << im_height_ << std::endl;

}

void SlamdunkCapture::WorkerThread(void) {
	cbs_.stereoYuvCallback =
		std::bind(&SlamdunkCapture::CBstereo, this, std::placeholders::_1);
	opt_.streamingOptions.fileEnableRight = false;

	if (std::unique_ptr<kalamos::Context> kalamos_context
			= kalamos::init(cbs_, opt_)) {	

	    std::unique_ptr<kalamos::ServiceHandle> captureHandle =
	    	kalamos_context->startService(kalamos::ServiceType::CAPTURE);	
	    std::cout << "Kalamos init succes!" << std::endl;

	    kalamos_context->run();
  	}

}


void SlamdunkCapture::WaitForImage() {
    mutex_image_1_.unlock();
    actual_frame_++;
    mutex_image_2_.lock();
}


void SlamdunkCapture::Close(void) {    
    cam_running_ = false;
    mutex_image_1_.unlock();
    mutex_image_2_.unlock();
}

cv::Mat SlamdunkCapture::GetLatestFrame(void){
	return frame_left_;
}

