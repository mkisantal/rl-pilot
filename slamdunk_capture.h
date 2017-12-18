#ifndef SLAMDUNK_CAPTURE_H_
#define SLAMDUNK_CAPTURE_H_

#include <mutex>
#include <thread>

#include <kalamos_context.hpp>

class SlamdunkCapture{

public:
	
	void stopcam (void) {cam_running_=false;}
    void WaitForImage(void);
    void Start (void);
    void Close (void);
    void Init (void);
    cv::Mat GetLatestFrame(void);

private:
	kalamos::Callbacks cbs_;
	kalamos::Options opt_;
	std::thread cam_thread_;

	std::mutex mutex_image_1_;
	std::mutex mutex_image_2_;

	bool cam_running_;
	int im_width_;
    int im_height_;

    int actual_frame_ = 0;

	cv::Size output_image_size_ = cv::Size(84, 84);
    cv::Mat frame_left_ = cv::Mat::zeros(output_image_size_,CV_8UC3);

    void CBstereo(kalamos::StereoYuvData const& data);
    void WorkerThread(void);
};

#endif // SLAMDUNK_CAPTURE_H_