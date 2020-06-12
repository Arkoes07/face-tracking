#include <iostream>
#include <chrono> 
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>  // FAST
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_transforms.h>

#include "LandmarkPredictor.h"

float pointEuclideanDist(cv::Point2f p, cv::Point2f q);
float eyeAspectRatio(std::vector<cv::Point2f> coordinates);
void checkEyeClosed(std::vector<cv::Point2f> points);
void drawLandmarks(cv::Mat& frame, std::vector<cv::Point2f> points);

float EYE_CLOSED_THRESHOLD = 0.2;
int eyeClosedCounter = 0;


int main()
{
	// Setup Face Detector and Facial Landmark Predictor
	LP::initializePredictor();

	// get start time
	auto start = std::chrono::high_resolution_clock::now();

	// open camera
	// cv::VideoCapture cap(0);

	// open video file
	cv::VideoCapture cap("D:\\datasets\\ngantuk\\01\\0.mp4");

	if (!cap.isOpened())  // isOpened() returns true if capturing has been initialized.
	{
		std::cout << "Cannot open the video file. \n";
		return -1;
	}

	// landmaarks container
	dlib::full_object_detection landmarks;

	// current frame container
	cv::Mat currentFrame;

	// variable for storing video information
	float frameCounter = 0;

	// get frames from camera
	while (1) {

		// read current frame
		cap.read(currentFrame);

		// container for points from all areas of interest
		std::vector<cv::Point2f> coordinates;

		// Check if the program is in tracking mode
		if (!LK::isTracking()) {
			// detect landmarks
			try {
				LP::predictLandmarks(landmarks, currentFrame);

				// get points from all areas of interest
				coordinates = LP::getCoordinatesFromLandmarks(landmarks);

				// comment to disable tracking for the next frames
				LK::start(currentFrame, landmarks);
			}
			catch (int errorCode) {
				if (errorCode == 1) {
					std::cout << "no face detected" << std::endl;
					LK::setTracking(false);
				}
			}
		}
		else {
			// get points from track
			coordinates = LK::track(currentFrame);
		}

		if (!coordinates.empty()) {
			// check eye closed
			checkEyeClosed(coordinates);
		}

		// draw landmark points on frame
		drawLandmarks(currentFrame, coordinates);

		// update video information
		frameCounter++;
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
		float fps = frameCounter / duration.count();

		// write fps on frame
		std::string textToDisplay = "Duration: " + std::to_string(duration.count()) + " seconds, Frame Processed: " + std::to_string(int(frameCounter)) + ", Average FPS: " + std::to_string(fps);
		cv::putText(currentFrame, textToDisplay, cv::Point(20, currentFrame.rows - 20), cv::FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(0, 255, 0), 2);

		// write blink on frame
		cv::putText(currentFrame, "eye closed counter: "+std::to_string(eyeClosedCounter), cv::Point(20, 20), cv::FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(0, 255, 0), 2);

		// Display current frame
		cv::imshow("Frame", currentFrame);

		// Press ESC on keyboard to exit
		char c = (char)cv::waitKey(1);
		if (c == 27)			
			break;
	}

	// When everything done, release the video capture object
	cap.release();

	// closes all the frames
	cv::destroyAllWindows();

	return 0;
}


float pointEuclideanDist(cv::Point2f p, cv::Point2f q) {
	float a = q.x - p.x;
	float b = q.y - p.y;
	return std::sqrt(a * a + b * b);
}

float eyeAspectRatio(std::vector<cv::Point2f> coordinates) {
	// compute the euclidean distances between the two sets of vertical eye landmarks(x, y) - coordinates
	float a = pointEuclideanDist(coordinates[1], coordinates[5]);
	float b = pointEuclideanDist(coordinates[2], coordinates[4]);
	// compute the euclidean distance between the horizontal eye landmark(x, y) - coordinates
	float c = pointEuclideanDist(coordinates[0], coordinates[3]);
	// compute eye aspect ratio
	return (a + b) / (2 * c);
}

void checkEyeClosed(std::vector<cv::Point2f> points) {
	const int part[3][2] = { {0,5}, {6,11}, {12,19} };

	float aspectRatio[2];
	for (int i = 0; i < 2; ++i) {
		std::vector<cv::Point2f> coordinates;
		// for each part (right eye, left eye, mouth)
		for (int pointIdx = part[i][0]; pointIdx <= part[i][1]; ++pointIdx) {
			coordinates.emplace_back(std::forward<cv::Point2f>(points[pointIdx]));
		}
		aspectRatio[i] = eyeAspectRatio(coordinates);
	}
	if (aspectRatio[0] < EYE_CLOSED_THRESHOLD && aspectRatio[1] < EYE_CLOSED_THRESHOLD)
		eyeClosedCounter++;
}

void drawLandmarks(cv::Mat& frame, std::vector<cv::Point2f> points) {
	for (auto const& point : points) {
		cv::circle(frame, point, 1, CV_RGB(0, 255, 0), 2);
	}
}