#include "ros/ros.h"
#include "std_msgs/String.h"
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <bits/stdc++.h>
#include <iostream>
 #include <opencv2/opencv_modules.hpp>


using namespace std;
using namespace cv;

int main(){
	namedWindow("output",1);
  	// VideoCapture cap("/home/vinit/Desktop/lane_detection/out1.avi");
  	// if(!cap.isOpened()){
   //      return -1;
  	// }
  	while(1){
  		// Mat inpt;
  		// cap>>inpt;
  		Mat lines;
		//Mat out(inpt.rows,inpt.cols,CV_8UC3,Scalar(0,0,0));
		cv::Mat gray;
		Mat temp=imread("lanes.jpg",1);
    	cv::cvtColor(temp, gray, CV_BGR2GRAY);


	    cv::Ptr<cv::LineSegmentDetector> det;
	    det = cv::createLineSegmentDetector();


		Mat out(temp.rows,temp.cols,CV_8UC3,Scalar(0,0,0));
	    det->detect(gray, lines);

	    det->drawSegments(out, lines);
		imshow("output", out);
		waitKey(10);
  	}
  	return 0;
}