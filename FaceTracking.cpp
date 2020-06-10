#include <chrono> 
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main()
{
	// get start time
	auto start = std::chrono::high_resolution_clock::now();

	// open camera
	cv::VideoCapture cap(0);

	// current frame container
	cv::Mat currentFrame;

	// variible for storing video information
	float frameCounter = 0;

	// get frames from camera
	while (1) {

		// read current frame
		cap.read(currentFrame);


		// update video information
		frameCounter++;
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
		float fps = frameCounter / duration.count();

		// write fps on frame
		std::string textToDisplay = "Duration: " + std::to_string(duration.count()) + " seconds, Frame Processed: " + std::to_string(int(frameCounter)) + ", Average FPS: " + std::to_string(fps);
		cv::putText(currentFrame, textToDisplay, cv::Point(20, currentFrame.rows - 20), cv::FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(0, 255, 0), 2);

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
