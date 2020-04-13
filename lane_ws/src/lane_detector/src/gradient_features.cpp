#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<cmath>
#include<opencv2/highgui/highgui.hpp>
#include<bits/stdc++.h>

using namespace std;
using namespace cv;

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
		Mat img = imread("raw_images/25.png", 1);
		//Mat img; 
		Mat gradient_features;
		//cap >> img;
		
		cvtColor(img, img, CV_BGR2GRAY);
		resize(img, img, Size(480, 360));

		Visual_Features vf(img);
				
		gradient_features = Visual_Features::Gradient_Features(&vf).run();

		//cvtColor(gradient_features, gradient_features, CV_GRAY2BGR);
		//video.write(gradient_features);

		waitKey(0);

		//Breaks is ESC key is pressed.
		//char c = (char)waitKey(1);
		//if(c == 27)
			//break;
	//}

	return 0;
}