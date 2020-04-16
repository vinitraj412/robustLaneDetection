#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<cmath>
#include<opencv2/highgui/highgui.hpp>
#include<bits/stdc++.h>
#include<queue>
#include<opencv2/calib3d/calib3d.hpp>
#define pi 3.1415

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


	class Intensity_Features
	{
		Mat frame;
		Mat inverse_homography;
		int th, f;	//Theta and Focal Length
		int offset_y;
		float sigma;	//Sigma for 2nd order Gaussian
		int upper_th, lower_th;	//Threshold values for hysthersis.

	private:

		struct graph_node
		{
			Point current, parent;
			float cost_value;
		};

		Point in_top_view(Point p)
		{
			Point new_p;
			float x1 = p.x, y1 = p.y, x2, y2;

			float theta =  (-th) * pi / 180;

			y2 = ( f * sin(theta) + y1 * cos(theta) ) / (cos(theta) - (y1 * sin(theta) / f)) * -1;
			x2 = x1 * (y2 * sin(theta) / f + cos(theta)) * -1;

			new_p.x = (int)x2;
			new_p.y = (int)y2;

			return new_p;
		}

		Mat top_view()
		{
			//Find corresponding point in top view
			//Considering center of image as origin.
			Point p1 = in_top_view(Point(-frame.cols/2, frame.rows/2));
			Point p2 = in_top_view(Point(frame.cols/2, frame.rows/2));
			Point p3 = in_top_view(Point(frame.cols/2, -frame.rows/2));
			Point p4 = in_top_view(Point(-frame.cols/2, -frame.rows/2));
			
			p1.x = p1.x + frame.cols/2;
			p1.y = frame.rows - (p1.y + frame.rows/2);

			p2.x = p2.x + frame.cols/2;
			p2.y = frame.rows - (p2.y + frame.rows/2);

			p3.x = p3.x + frame.cols/2;
			p3.y = frame.rows - (p3.y + frame.rows/2);

			p4.x = p4.x + frame.cols/2;
			p4.y = frame.rows - (p4.y + frame.rows/2);

			p1.y += offset_y;
			p2.y += offset_y;
			p3.y += offset_y;
			p4.y += offset_y;

			Mat bird_eye(1000, 600, CV_8UC3, Scalar(0, 0, 0));

			vector<Point2f> src;
			src.push_back(Point(0, 0));
			src.push_back(Point(frame.cols, 0));
			src.push_back(Point(frame.cols, frame.rows));
			src.push_back(Point(0, frame.rows));

			vector<Point2f> dst;
			dst.push_back(p1);
			dst.push_back(p2);
			dst.push_back(p3);
			dst.push_back(p4);

			Mat homography = findHomography(src, dst);
			warpPerspective(frame, bird_eye, homography, bird_eye.size());

			inverse_homography = homography.inv();	//Inverse homography to convert from top view to normal view.

			return bird_eye;
		}

		//Extract vertical lines by applying 2nd order Gaussian
		Mat extract_lines(Mat bird_eye)
		{

			Mat extracted1(bird_eye.rows, bird_eye.cols, CV_8UC1, Scalar(0));
			Mat extracted2(bird_eye.rows, bird_eye.cols, CV_8UC1, Scalar(0));
			Mat extracted(bird_eye.rows, bird_eye.cols, CV_8UC1, Scalar(0));
			//Sobel(bird_eye, extracted, CV_8U, 2, 0, 3);
			
			GaussianBlur(bird_eye, extracted1, Size(-1, -1), sigma/sqrt(2));
			GaussianBlur(bird_eye, extracted2, Size(-1, -1), sigma*sqrt(2));
			extracted = extracted1 - extracted2;

			return extracted;
		}

		//Find the diameter of graph using bfs.
		//Assumption: Non weighted edges. {To reduce processing otherwise need to use dijkstra}
		float find_diameter(Mat* temp, Point p1)
		{
			queue<graph_node> OPEN_list;
			struct graph_node g1;
			g1.current = p1;
			g1.cost_value = 0;
			float diameter = 0;

			OPEN_list.push(g1);
			temp->at<uchar>(p1) = 0;

			while(OPEN_list.size())
			{
				struct graph_node temp_node = OPEN_list.front();
				OPEN_list.pop();
				if(temp_node.cost_value > diameter)
					diameter = temp_node.cost_value;

				for(int i1=-1; i1<=1; i1++)
				{
					for(int j1=-1; j1<=1; j1++)
					{
						int i2 = temp_node.current.y + i1, j2 = temp_node.current.x + j1;
						if(i2 >= 0 && i2 < temp->rows && j2 >= 0 && j2 < temp->cols)
						{
							if(temp->at<uchar>(i2, j2) != 0)
							{
								struct graph_node g2;
								g2.current = Point(j2, i2);
								g2.parent = temp_node.current;
								g2.cost_value = temp_node.cost_value + 1;
								temp->at<uchar>(g2.current) = 0;
								OPEN_list.push(g2);
							}
						}
					}
				}
			}

			return diameter;
		}

		//Copy the patch in Mat temp to Mat threshold_extracted which contains a point p.
		//Uses BFS
		void copy_this_extract(Mat temp, Mat* threshold_extracted, Point p)
		{
			queue<Point> OPEN_list;
			OPEN_list.push(p);
			threshold_extracted->at<uchar>(p.y, p.x) = temp.at<uchar>(p.y, p.x);
			temp.at<uchar>(p.y, p.x) = 0;

			while(OPEN_list.size())
			{
				Point p1 = OPEN_list.front();
				OPEN_list.pop();
				for(int i1=-1; i1<=1; i1++)
				{
					for(int j1=-1; j1<=1; j1++)
					{
						int i2 = p1.y + i1, j2 = p1.x + j1;
						if(i2>=0 && i2<temp.rows && j2>=0 && j2<temp.cols)
						{
							if(temp.at<uchar>(i2, j2)!=0)
							{
								threshold_extracted->at<uchar>(i2, j2) = temp.at<uchar>(i2, j2);
								temp.at<uchar>(i2, j2) = 0;
								OPEN_list.push(Point(j2, i2));
							}
						}
					}
				}
			}
		}


		Mat hysthersis(Mat extracted)
		{

			Mat after_hysthersis(extracted.rows, extracted.cols, CV_8UC1, Scalar(0));

			Mat temp = extracted.clone();
			int upper_th = 15, lower_th = 5;

			float max_diameter = 0;

			for(int i=0; i<temp.rows; i++)
			{
				for(int j=0; j<temp.cols; j++)
				{
					if(temp.at<uchar>(i, j) >= upper_th)
					{
						after_hysthersis.at<uchar>(i, j) = temp.at<uchar>(i, j);
						temp.at<uchar>(i, j) = 0;
					}
					else if(temp.at<uchar>(i, j) < lower_th)
					{
						temp.at<uchar>(i, j) = 0;
					}
				}
			}	

			Mat extracted_temp = after_hysthersis.clone();

			//Finding max-diameter from diameters of graphs with intensity value > upper threshold.
			for(int i=0; i<temp.rows; i++)
			{
				for(int j=0; j<temp.cols; j++)
				{
					if(extracted_temp.at<uchar>(i, j) != 0)
					{
						float diameter = find_diameter(&extracted_temp, Point(j, i));
						if(diameter > max_diameter)
							max_diameter = diameter;
					}
				}
			}

			//Processing region with intensity between upper and lower threshold.
			extracted_temp = temp.clone();
			for(int i=0; i<temp.rows; i++)
			{
				for(int j=0; j<temp.cols; j++)
				{
					if(extracted_temp.at<uchar>(i, j)!=0)
					{
						float diameter = find_diameter(&extracted_temp, Point(j, i));
						if(diameter >= max_diameter/2)
						{
							//cout << diameter << endl;
							copy_this_extract(temp, &after_hysthersis, Point(j, i));
						}
					}
				}
			}

			return after_hysthersis;
		}

	public:
		Intensity_Features(Visual_Features* vf)
		{
			frame = vf->frame;
			upper_th = 15;
			lower_th = 5;
			th = 135;
			f = 395;
			offset_y = 700;
			sigma = 3;
		}

		Mat run()
		{
			Mat bird_eye = top_view();

			namedWindow("Bird_Eye", WINDOW_NORMAL);
			//namedWindow("Bird_Eye_Smooth", WINDOW_NORMAL);
			//namedWindow("Extracted", WINDOW_NORMAL);
			namedWindow("Final Extracted", WINDOW_NORMAL);
			
			//Vertical smoothning.
			Mat bird_eye_smooth;
			GaussianBlur(bird_eye, bird_eye_smooth, Size(1, 3), 0);

			//imshow("Input", img);
			imshow("Bird_Eye", bird_eye);
			//imshow("Bird_Eye_Smooth", bird_eye_smooth);

			Mat extracted = extract_lines(bird_eye_smooth);
			//imshow("Extracted", extracted);

			Mat after_hysthersis = hysthersis(extracted);
			imshow("Final Extracted", after_hysthersis);

			Mat final_img(360, 480, CV_8UC1, Scalar(0));

			for(int i=0; i<after_hysthersis.rows; i++)
			{
				for(int j=0; j<after_hysthersis.cols; j++)
				{
					if(after_hysthersis.at<uchar>(i, j) != 0)
					{
						Mat prev=Mat::zeros(3,1,DataType<double>::type);
			    		Mat newy=Mat::zeros(3,1,DataType<double>::type);
						newy.at<double>(0,0)=j;
			    		newy.at<double>(1,0)=i;
			    		newy.at<double>(2,0)=1;
			    		prev=inverse_homography*newy;
			    		prev.at<double>(0,0)=prev.at<double>(0,0)/prev.at<double>(2,0);
			    		prev.at<double>(1,0)=prev.at<double>(1,0)/prev.at<double>(2,0);
			    		int xx=prev.at<double>(0,0);
			    		int yy=prev.at<double>(1,0);
			    		final_img.at<uchar>(yy, xx) = after_hysthersis.at<uchar>(i, j);
		    		}	
				}
			}

			namedWindow("Final Lanes", WINDOW_NORMAL);
			imshow("Final Lanes", final_img);


			return final_img;
			//waitKey(0);
			
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
		Mat gradient_features, intensity_features;
		//cap >> img;
		
		cvtColor(img, img, CV_BGR2GRAY);
		resize(img, img, Size(480, 360));

		Visual_Features vf(img);
				
		gradient_features = Visual_Features::Gradient_Features(&vf).run();

		//cvtColor(gradient_features, gradient_features, CV_GRAY2BGR);
		//video.write(gradient_features);

		intensity_features = Visual_Features::Intensity_Features(&vf).run();

		waitKey(0);

		//Breaks is ESC key is pressed.
		//char c = (char)waitKey(1);
		//if(c == 27)
			//break;
	//}

	return 0;
}