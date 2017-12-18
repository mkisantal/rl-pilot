#include "rlpilot.h"

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <ctime>

int main(void){

	RLPilot pilot;

	pilot.cam.Init();
	std::cout << "--- cam init done" << std::endl;
	pilot.cam.Start();
	std::cout << "--- pilot cam start" << std::endl;

	std::unique_ptr<tensorflow::Session> session;
	pilot.inference.Init(&session);
	std::cout << "--- inference initialized  "<< std::endl;
	cv::namedWindow("Results", CV_WINDOW_AUTOSIZE);
	std::clock_t start;

	int counter = 0;
	double avg_fps = 0.0;
	double fps;
    
	bool loop = true;
	while (loop) {

		start = std::clock();

		//cv::namedWindow("Results", CV_WINDOW_AUTOSIZE);
		pilot.cam.WaitForImage();
		cv::imshow("Results", pilot.cam.GetLatestFrame());
		pilot.inference.NewImageInput(pilot.cam.GetLatestFrame());
		pilot.inference.Run(session);

		loop = (cv::waitKey(1) == -1);

		double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "Done in " << duration  << " seconds." << std::endl;
		counter += 1;
		fps = 1.0/duration;
		avg_fps = (fps-avg_fps)/((double)counter) + avg_fps;
		std::cout << fps << " FPS.    Average FPS is  " <<  avg_fps << std::endl;


	}

	std::cout << "all done!" << std::endl;

	cv::destroyAllWindows();
	pilot.cam.Close();

	return 0;
}