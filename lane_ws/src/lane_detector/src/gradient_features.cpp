#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<cmath>
#include<opencv2/highgui/highgui.hpp>
#include<bits/stdc++.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;

class Intensity_features{
	Mat frame;
	Mat inv_perspective;
	Mat h_inv;
	Mat detected_lanes;
	int width_thresh;
	int diff_thresh=50;
public:
	Intensity_features(Mat image){
		frame=image;
	}
	void perspective_change(){
		Mat im_src=frame;
		Size size(300,400);
	    Mat im_dst = Mat::zeros(size,CV_8UC3);

	    
	    // Create a vector of destination points.
	    vector<Point2f> pts_dst;
	    
	    pts_dst.push_back(Point2f(0,0));
	    pts_dst.push_back(Point2f(size.width - 1, 0));
	    pts_dst.push_back(Point2f(size.width - 1, size.height -1));
	    pts_dst.push_back(Point2f(0, size.height - 1 ));
	    
	    int tlx=(1.0/2.0)*(im_src.cols);
	    int trx=(3.0/4.0)*(im_src.cols);
	    int brx=im_src.cols-2;
	    int blx=10;
	    int tly=(1.0/2.0)*(im_src.rows);
	    int ttry=(1.0/2.0)*(im_src.rows);
	    int bry=(2.0/3.0)*(im_src.rows);
	    int bly=(2.0/3.0)*(im_src.rows);

	    cout<<tlx<<" "<<tly<<endl;
	    cout<<trx<<" "<<ttry<<endl;
	    cout<<brx<<" "<<bry<<endl;
	    cout<<blx<<" "<<bly<<endl;

	    vector<Point2f> pts_src;
	    pts_src.push_back(Point2f(tlx, tly));
	    pts_src.push_back(Point2f(trx,ttry));
	    pts_src.push_back(Point2f(brx, bry));
	    pts_src.push_back(Point2f(blx, bly));
	    // Set data for mouse event
	    // Mat im_temp = im_src.clone();
	    // userdata data;
	    // data.im = im_temp;
	    // cout << "Click on the four corners of the book -- top left first and" << endl
	    // << "bottom left last -- and then hit ENTER" << endl;
	    
	    // // Show image and wait for 4 clicks. 
	    // imshow("Image", im_temp);
	    // // Set the callback function for any mouse event
	    // setMouseCallback("Image", mouseHandler, &data);
	    // waitKey(0);
	    
	    // Calculate the homography
	    Mat h = findHomography(pts_src, pts_dst);
	    
	    // Warp source image to destination
	    warpPerspective(im_src, im_dst, h, size);
	    inv_perspective=im_dst.clone();

	    h_inv=h.inv();
	    for(int i=0; i<h_inv.rows; i++){
	        for(int j=0; j<h_inv.cols; j++){
	            cout<<h_inv.at<double>(i,j)<<" ";
	        }
	        cout<<endl;
	    }
	    Mat prev=Mat::zeros(3,1,DataType<double>::type);
	    Mat newy=Mat::zeros(3,1,DataType<double>::type);
	    
	    newy.at<double>(0,0)=200;
	    newy.at<double>(1,0)=0;
	    newy.at<double>(2,0)=1;
	    prev=h_inv*newy;
	    cout<<prev.at<double>(0,0)/prev.at<double>(2,0)<<" "<<prev.at<double>(1,0)/prev.at<double>(2,0)<<endl;
	    // Show image
	    imshow("Image", im_dst);
	}
	vector<Point> find_width(Mat inv, int x, int &y){
		vector <Point> roi;
		int i=y+2;
		while(i<inv.cols-1){
			//cout<<i<<" ";
			if(abs(inv.at<uchar>(x,i+1)-inv.at<uchar>(x,i))>60){
				break;
			}
			roi.push_back(Point(x,i));
			i++;
		}
		if(i-y<20){
			y=i;
			return roi;
		}
		else{
			y=i;
			roi.clear();
			roi.push_back(Point(-1,-1));
			return roi;
		}
	}
	Mat find_lanes(){
		int r=inv_perspective.rows;
		int c=inv_perspective.cols;
		Mat v_smooth=inv_perspective.clone();
		
		//vertical smoothening
		for(int i=0; i<c; i++){
			for(int j=2; j<r-2; j++){
				v_smooth.at<uchar>(j,i)=(inv_perspective.at<uchar>(j-2,i)+inv_perspective.at<uchar>(j-1,i)+inv_perspective.at<uchar>(j,i)+inv_perspective.at<uchar>(j+1,i)+inv_perspective.at<uchar>(j+2,i))/5;
			}
		}
		cout<<v_smooth.rows<<" "<<v_smooth.cols<<endl;
		imwrite("smooth.jpeg", v_smooth);
		Mat temp_lanes(frame.rows, frame.cols, CV_8UC1, Scalar(0));
		for(int i=0; i<r; i++){
			for(int j=0; j<c-3; j++){
				if(v_smooth.at<uchar>(i,j+3)-v_smooth.at<uchar>(i,j)>10){
					vector<Point> roi=find_width(v_smooth, i, j);
					if(roi.size()!=0 && roi[0].x!=-1){
						for(auto k : roi){
							cout<<k.x<<" "<<k.y<<endl;
							// Mat prev=Mat::zeros(3,1,DataType<double>::type);
	    		// 			Mat newy=Mat::zeros(3,1,DataType<double>::type);
	    		// 			newy.at<double>(0,0)=k.x;
	    		// 			newy.at<double>(1,0)=k.y;
	    		// 			newy.at<double>(2,0)=1;
	    		// 			prev=h_inv*newy;
	    		// 			prev.at<double>(0,0)=prev.at<double>(0,0)/prev.at<double>(2,0);
	    		// 			prev.at<double>(1,0)=prev.at<double>(1,0)/prev.at<double>(2,0);
	    		// 			int xx=prev.at<double>(0,0);
	    		// 			int yy=prev.at<double>(1,0);
	    		// 			//cout<<xx<<" "<<yy<<endl;
	    		// 			temp_lanes.at<uchar>(yy,xx)=255;
						}
					}
				}
			}
		}
		namedWindow("smooth",1);
		imshow("smooth",v_smooth);
		namedWindow("lanes1",1);
		imshow("lanes1",temp_lanes);
		return temp_lanes;
	}
};

class Visual_Features
{
	Mat frame;
	VideoWriter output;

public:

	Visual_Features(Mat img)
	{
		frame = img;
	}

	class Gradient_Features
	{
		Mat frame;
	private:

		bool find_intersection_point(Vec4i v1, Vec4i v2, Point* intersection)
		{
			float a1, b1, c1, a2, b2, c2;

			a1 = v1[1] - v1[3];
			b1 = v1[0] - v1[2];
			c1 = b1 * (frame.rows - v1[1]) + a1 *  v1[0];

			a2 = v2[1] - v2[3];
			b2 = v2[0] - v2[2];
			c2 = b2 * (frame.rows - v2[1]) + a2 *  v2[0];

			float det = a1 * b2 - b1 * a2;

			if(det == 0)
				return false;
			else
			{
				//Return intersection point considering botton left corner as origin
				(*intersection).x = (b2 * c1 - b1 * c2)/det;	
				(*intersection).y = (a1 * c2 - c1 * a2)/det;
				//cout << (*intersection).x << ", " << (*intersection).y << endl;
				return true;
			}
		}

		/*Need to improve the efficiency of find_horizon()*/
		//Return horizon considering bottom-left corner as origin.
		int find_horizon(vector<Vec4i> lsd_lines, int tolerance)
		{
			unordered_map<int, int> hash;

			for(int i=0; i<lsd_lines.size(); i++)
			{
				for(int j=0; j<lsd_lines.size(); j++)
				{
					if(i != j)
					{
						Point p;
						if(find_intersection_point(lsd_lines.at(i), lsd_lines.at(j), &p))
						{
							hash[p.y]++;
							for(int k=-tolerance; k<=tolerance; k++)
							{
								if(k!=0)
								{
									if(hash[p.y + k] != 0)
										hash[p.y + k]++;
								}
							}	
						}
					}
				}
			}
			int horizon=0;
			int max_count = 0;
			for(auto i : hash)
			{
				if(max_count < i.second)
				{
					horizon = i.first;
					max_count = i.second;
				}
			}

			return horizon;
		}

		vector<float> find_scores(vector<Vec4i> lsd_lines, int tolerance, int horizon, float* total_score)
		{
			vector<float> scores;
			(*total_score) = 0;

			for(int i=0; i<lsd_lines.size(); i++)
			{
				(*total_score) += sqrt(pow(lsd_lines.at(i)[0] - lsd_lines.at(i)[2], 2) + pow(lsd_lines.at(i)[1] - lsd_lines.at(i)[3], 2));
				float curr_score = 0;
				for(int j=0; j<lsd_lines.size(); j++)
				{
					if(i != j)
					{
						Point p;
						if(find_intersection_point(lsd_lines.at(i), lsd_lines.at(j), &p));
						{
							if(abs(p.y - horizon) <= tolerance)
							{
								curr_score += sqrt(pow(lsd_lines.at(j)[0] - lsd_lines.at(j)[2], 2) + pow(lsd_lines.at(j)[1] - lsd_lines.at(j)[3], 2));
							}
						}
					}
				}
				//cout << curr_score << endl;
				scores.push_back(curr_score);
			}

			return scores;
		}


	public:

		Gradient_Features(Visual_Features* vf)
		{
			frame = vf->frame;
		}

		Mat run()
		{
			vector<Vec4i> lsd_lines;
			vector<float> scores;
			vector<Vec4i> req_lsd_lines;
			
			Ptr<LineSegmentDetector> lsd = createLineSegmentDetector();
			lsd->detect(frame, lsd_lines);
			Mat lsd_plot(frame.rows, frame.cols, CV_8UC1, Scalar(0));

			int tolerance = 10;
			float total_score;

			int horizon = find_horizon(lsd_lines, tolerance);
			scores = find_scores(lsd_lines, tolerance, horizon, &total_score);

			for(int i=0; i<scores.size(); i++)
			{
				if(scores.at(i) >= total_score*0.05)
				{
					req_lsd_lines.push_back(lsd_lines.at(i));
				}
			}

			for(int i=0; i<req_lsd_lines.size(); i++)
			{
				line(lsd_plot, Point(req_lsd_lines.at(i)[0], req_lsd_lines.at(i)[1]), Point(req_lsd_lines.at(i)[2], req_lsd_lines.at(i)[3]), Scalar(255), 2);
			}

			namedWindow("Input", WINDOW_NORMAL);
			imshow("Input", frame);
			namedWindow("LSD plot", WINDOW_NORMAL);
			imshow("LSD plot", lsd_plot);

			return lsd_plot;
		}
	};
};

int main()
{
	//VideoCapture cap("out1.avi");
	//VideoWriter video("outcpp.avi",CV_FOURCC('M','J','P','G'),10, Size(480, 360));

	//if(!(cap.isOpened()))
		//return -1;

	//while(1)
	//{
		Mat img = imread("lanes.jpg", 1);
		//Mat img; 
		Mat gradient_features;
		//cap >> img;
		
		cvtColor(img, img, CV_BGR2GRAY);
		resize(img, img, Size(480, 360));

		Visual_Features vf(img);
				
		gradient_features = Visual_Features::Gradient_Features(&vf).run();

		//cvtColor(gradient_features, gradient_features, CV_GRAY2BGR);
		//video.write(gradient_features);
		Intensity_features If(img);
		If.perspective_change();
		Mat feature2=If.find_lanes();
		waitKey(0);

		//Breaks is ESC key is pressed.
		//char c = (char)waitKey(1);
		//if(c == 27)
			//break;
	//}

	return 0;
}