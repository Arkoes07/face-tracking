#include "LandmarkPredictor.h"

// model path
const std::string CAFFE_CONFIG_FILE = "D:\\models\\cvDnn\\deploy.prototxt";
const std::string CAFFE_WEIGHT_FILE = "D:\\models\\cvDnn\\res10_300x300_ssd_iter_140000_fp16.caffemodel";
const std::string DLIB_LANDMARKS_MODEL = "D:\\resource\\shape_predictor_68_face_landmarks.dat";

// part of eyes and mouth on predicted landmark
const int BOUNDARY[3][2] = { {36,41}, {42,47}, {60,67} };
const int PART[3][2] = { {0,5}, {6,11}, {12,19} };

// OpenCV DNN
cv::dnn::Net DNN_NET;
const size_t DNN_INWIDTH = 640;
const size_t DNN_INHEIGHT = 480;
const double DNN_INSCALE_FACTOR = 1.0;
const float DNN_CONFIDENCE_THRESHOLD = 0.7;
const cv::Scalar DNN_MEAN_VAL = cv::Scalar(104.0, 177.0, 123.0);

// Dlib facial landmark predictor
dlib::shape_predictor DLIB_PREDICTOR;

// Lucas-Kanade Constant
const int MAX_FRAME_COUNT = 10;
const int PYRAMIDS = 3;
const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 25, 0.01);
const cv::Size ROI(20, 20);

// Lucas-Kanade variables
bool tracking = false;
int frameCount = 0;
cv::Mat prev_img;
std::vector<cv::Point2f> prev_pts;
std::vector<cv::Point2f> next_pts;

// -------------------------------------------------------------------------------------------------
// -- Lucas-Kanade Optical Flow Tracker
// -------------------------------------------------------------------------------------------------

void LK::start(cv::Mat mat, dlib::full_object_detection& pts) {
	// convert to grayscale
	cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);

	// release stuff..
	prev_img.release();
	prev_img = mat;
	prev_pts.clear();
	next_pts.clear();

	// get points
	prev_pts = LP::getCoordinatesFromLandmarks(pts);

	// reset count
	frameCount = 0;
	tracking = true;
}

std::vector<cv::Point2f> LK::track(cv::Mat frame) {
	std::vector<uchar> status;
	std::vector<float> err;
	std::vector<cv::Point2f> tracked;

	// convert to grayscale
	cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

	// get the new points from the old one
	cv::calcOpticalFlowPyrLK(prev_img, frame, prev_pts, next_pts, status, err, ROI, PYRAMIDS, criteria);

	for (int i = 0; i < status.size(); ++i) {
		if (status[i] == 0) {
			// flow not found: take the old point
			tracked.push_back(prev_pts[i]);
		}
		else {
			// flow found: take the new point
			tracked.push_back(next_pts[i]);
		}
	}

	// switch the previous points and image with the current
	swap(prev_img, frame);
	swap(prev_pts, tracked);
	next_pts.clear();

	// increase tracking frame count
	if (frameCount++ > MAX_FRAME_COUNT) {
		tracking = false;
	}

	return prev_pts;
}

bool LK::isTracking() {
	return tracking;
}

void LK::setTracking(bool _tracking) {
	tracking = _tracking;
}

// -------------------------------------------------------------------------------------------------
// -- Landmark Predictor
// -------------------------------------------------------------------------------------------------

void LP::initializePredictor() {
	// Setup Face Detector using cv::Dnn
	DNN_NET = cv::dnn::readNetFromCaffe(CAFFE_CONFIG_FILE, CAFFE_WEIGHT_FILE);

	// Dlib facial landmark predictor
	dlib::deserialize(DLIB_LANDMARKS_MODEL) >> DLIB_PREDICTOR;
}

void LP::predictLandmarks(dlib::full_object_detection& container, cv::Mat& inFrame) {
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

std::vector<cv::Point2f> LP::getCoordinatesFromLandmarks(dlib::full_object_detection shape) {
	std::vector<cv::Point2f> coordinates;
	for (int idx = 0; idx < 3; ++idx) {
		for (unsigned long pointIdx = BOUNDARY[idx][0]; pointIdx <= BOUNDARY[idx][1]; ++pointIdx) {
			// push to container
			dlib::point pt = shape.part(pointIdx);
			coordinates.emplace_back(std::forward<cv::Point2f>(cv::Point2f(pt.x(), pt.y())));
		}
	}
	return coordinates;
}
