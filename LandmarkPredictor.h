#pragma once

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

// -------------------------------------------------------------------------------------------------
// -- Lucas-Kanade Optical Flow Tracker
// -------------------------------------------------------------------------------------------------
namespace LK {
	/** Initialize tracking with the current frame and detected landmarks */
	void start(cv::Mat mat, dlib::full_object_detection& pts);
	/** tracking points in the next captured frame */
	std::vector<cv::Point2f> track(cv::Mat frame);
	/** Getter for tracking variable */
	bool isTracking();
	/** Setter for tracking variable */
	void setTracking(bool _tracking);
}

// -------------------------------------------------------------------------------------------------
// -- Landmark Predictor
// -------------------------------------------------------------------------------------------------
namespace LP {
	void initializePredictor();
	void predictLandmarks(dlib::full_object_detection& container, cv::Mat& inFrame);
	std::vector<cv::Point2f> getCoordinatesFromLandmarks(dlib::full_object_detection shape);
}