// FaceTracking.cpp : Defines the entry point for the application.
//

#include <opencv2/opencv.hpp>

using namespace std;

int main()
{
	// open camera
	cv::VideoCapture cap(0);

	// current frame container
	cv::Mat currentFrame;

	// get frames from camera
	while (1) {

		// read current frame
		cap.read(currentFrame);

		// Display current frame
		cv::imshow("Frame", currentFrame);

		// Press ESC on keyboard to exit
		char c = (char)cv::waitKey(25);
		if (c == 27)
			break;
	}

	// When everything done, release the video capture object
	cap.release();

	// closes all the frames
	cv::destroyAllWindows();

	return 0;
}
