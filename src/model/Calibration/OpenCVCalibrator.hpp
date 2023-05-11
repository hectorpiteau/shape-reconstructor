#pragma once

// #include <opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/features2d.hpp>

class OpenCVCalibrator {
public:
    OpenCVCalibrator() {}

    void Calibrate(){
        /** Extract SIFT features. */
        // const cv::Mat input = cv::imread("input.jpg", 0); //Load as grayscale
        // cv::SiftFeatureDetector detector;
        // std::vector<cv::KeyPoint> keypoints;
        // detector.detect(input, keypoints);

        // // Add results to image and save.
        // cv::Mat output;
        // cv::drawKeypoints(input, keypoints, output);
        // cv::imwrite("sift_result.jpg", output);
    }
};