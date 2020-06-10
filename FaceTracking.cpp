#include <iostream>
#include <chrono> 
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_transforms.h>

using namespace std;

// model path
const std::string CAFFE_CONFIG_FILE = "D:\\models\\cvDnn\\deploy.prototxt";
const std::string CAFFE_WEIGHT_FILE = "D:\\models\\cvDnn\\res10_300x300_ssd_iter_140000_fp16.caffemodel";
const std::string DLIB_LANDMARKS_MODEL = "D:\\resource\\shape_predictor_68_face_landmarks.dat";
// part of eyes and mouth on predicted landmark
const int BOUNDARY[3][2] = { {36,41}, {42,47}, {60,67} };

// OpenCV DNN
cv::dnn::Net DNN_NET;
const size_t DNN_INWIDTH = 640;
const size_t DNN_INHEIGHT = 480; 
const double DNN_INSCALE_FACTOR = 1.0; 
const float DNN_CONFIDENCE_THRESHOLD = 0.7; 
const cv::Scalar DNN_MEAN_VAL = cv::Scalar(104.0, 177.0, 123.0);

// Dlib facial landmark predictor
dlib::shape_predictor DLIB_PREDICTOR;

// function declaration
void predictLandmarks(dlib::full_object_detection& container, cv::Mat& inFrame);
float pointEuclideanDist(dlib::point p, dlib::point q);
float eyeAspectRatio(std::vector<dlib::point> coordinates);
void drawLandmarks(cv::Mat& frame, dlib::full_object_detection& landmarks);

int blinkCounter = 0;

int main()
{
	// Setup Face Detector using cv::Dnn
	DNN_NET = cv::dnn::readNetFromCaffe(CAFFE_CONFIG_FILE, CAFFE_WEIGHT_FILE);

	// Dlib facial landmark predictor
	dlib::deserialize(DLIB_LANDMARKS_MODEL) >> DLIB_PREDICTOR;
	dlib::full_object_detection landmarks;

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

		try {
			// predict landmarks
			predictLandmarks(landmarks, currentFrame);

			// draw landmarks
			drawLandmarks(currentFrame, landmarks);
		}
		catch (int errorCode) {
			if (errorCode == 1)
				std::cout << "no face detected" << std::endl;
		}

		// update video information
		frameCounter++;
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
		float fps = frameCounter / duration.count();

		// write fps on frame
		std::string textToDisplay = "Duration: " + std::to_string(duration.count()) + " seconds, Frame Processed: " + std::to_string(int(frameCounter)) + ", Average FPS: " + std::to_string(fps);
		cv::putText(currentFrame, textToDisplay, cv::Point(20, currentFrame.rows - 20), cv::FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(0, 255, 0), 2);

		// write blink on frame
		cv::putText(currentFrame, "Blink: "+std::to_string(blinkCounter), cv::Point(20, 20), cv::FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(0, 255, 0), 2);

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

void predictLandmarks(dlib::full_object_detection& container, cv::Mat& inFrame) {
	cv::Mat frame = inFrame;

	int frameHeight = frame.rows;
	int frameWidth = frame.cols;
	
	// detect face using DNN
	cv::Mat inputBlob = cv::dnn::blobFromImage(frame, DNN_INSCALE_FACTOR, cv::Size(DNN_INWIDTH, DNN_INHEIGHT), DNN_MEAN_VAL, false, false);
	DNN_NET.setInput(inputBlob, "data");
	cv::Mat detection = DNN_NET.forward("detection_out");
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	// container to store all faces
	std::vector<dlib::rectangle> rectangles;
	// get faces
	for (int i = 0; i < detectionMat.rows; ++i) {
		// get the confidence of current face
		float confidence = detectionMat.at<float>(i, 2);
		// accept as face if confidence bigger than confidenceThreshold
		if (confidence > DNN_CONFIDENCE_THRESHOLD) {
			// get coordinates for top-left corner and bottom-right corner
			long x1 = static_cast<long>(detectionMat.at<float>(i, 3) * frameWidth);
			long y1 = static_cast<long>(detectionMat.at<float>(i, 4) * frameHeight);
			long x2 = static_cast<long>(detectionMat.at<float>(i, 5) * frameWidth);
			long y2 = static_cast<long>(detectionMat.at<float>(i, 6) * frameHeight);
			// push to the container
			rectangles.emplace_back(dlib::rectangle(x1, y1, x2 - 1, y2 - 1));
		}
	}

	// check if no faces detected
	if (rectangles.size() == 0) {
		throw 1; // no face detected (1)
	} 

	// get biggest detected face
	int biggestAreaIdx = 0;
	unsigned long biggestArea = 0;
	for (int i = 0; i < rectangles.size(); ++i) {
		// calculate the area of current rectangle
		unsigned long area = rectangles[i].area();
		// check if the current area is bigger than current biggest rectangle
		if (area > biggestArea) {
			// if yes, change the current biggest rectangle
			biggestAreaIdx = i;
			biggestArea = area;
		}
	}

	// return the biggest rectangle
	dlib::rectangle face = rectangles[biggestAreaIdx];

	cv::rectangle(inFrame, cv::Rect(face.tl_corner().x(), face.tl_corner().y(), face.width(), face.height()), CV_RGB(0, 255, 0), 2);

	// convert frame into grayscale
	cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

	// convert into dlib format
	dlib::cv_image<unsigned char>&& image(frame);

	// get landmarks using dlib
	container = DLIB_PREDICTOR(image, face);
}

float pointEuclideanDist(dlib::point p, dlib::point q) {
	float a = (float)q.x() - (float)p.x();
	float b = (float)q.y() - (float)p.y();
	return std::sqrt(a * a + b * b);
}

float eyeAspectRatio(std::vector<dlib::point> coordinates) {
	// compute the euclidean distances between the two sets of vertical eye landmarks(x, y) - coordinates
	float a = pointEuclideanDist(coordinates[1], coordinates[5]);
	float b = pointEuclideanDist(coordinates[2], coordinates[4]);
	// compute the euclidean distance between the horizontal eye landmark(x, y) - coordinates
	float c = pointEuclideanDist(coordinates[0], coordinates[3]);
	// compute eye aspect ratio
	return (a + b) / (2 * c);
}

void drawLandmarks(cv::Mat& frame, dlib::full_object_detection& landmarks) {
	// right eye, left eye, mouth
	float aspectRatio[2];
	for (int i = 0; i < 3; ++i) {
		std::vector<dlib::point> coordinates;
		// for each part (right eye, left eye, mouth)
		for (int pointIdx = BOUNDARY[i][0]; pointIdx <= BOUNDARY[i][1]; ++pointIdx) {
			dlib::point point = landmarks.part(pointIdx);
			coordinates.emplace_back(std::forward<dlib::point>(point));
			cv::circle(frame, cv::Point(point.x(), point.y()), 1, CV_RGB(0, 255, 0), 2);
		}
		if (i != 2) 
			aspectRatio[i] = eyeAspectRatio(coordinates);
	}
	if (aspectRatio[0] < 0.3 && aspectRatio[1] < 0.3)
		blinkCounter++;
}