/*------------------------------------------------------
				2021. 04. 02. Fri
	   Image Proccessing with Deep Learning
		  LAP #1: Tempurature Measurement
			 21600372  Yoonkyoung Song
------------------------------------------------------*/

#include "myOpenCV.h"

using namespace std;
using namespace cv;

float rho_left=0, theta_left= CV_PI;
float rho_right = 0, theta_rigth = 0;
float rho_mid = 0, theta_mid = 0;
bool flag = true;
int fps = 0;
int ffps = 0;
double bias = 0;
double bcent = 0;

cv::Point interP;

int main()
{
	Mat image, image_gray, image_disp, mask, dst;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;


	VideoCapture cap("Part1_lanechange1.mp4");

	bool bSuccess = cap.read(image);

	if (!bSuccess)
	{
		cout << "Cannot find a frame from video stream/n";
	}

	/// 영상 저장
	//double fps = 30; // 영상 프레임
	//int  fourcc = VideoWriter::fourcc('x', 'v', 'i', 'd'); // 코덱 설정
	//VideoWriter outputVideo;
	////save.open(입력 영상, 코덱, 프레임, 영상 크기, 컬러)
	//outputVideo.open("LAB2_DEMO.avi", fourcc, fps, image.size(), 1);

	clock_t start = clock();

	for (;;)
	{
		clock_t fstart = clock();
		cap.read(image);
		image.copyTo(image_disp);

		if (image.empty()) {
			break;
		}

		cvtColor(image, image_gray, COLOR_RGB2GRAY);
		Mat cdstP = image.clone();


		imFilter(&image, &image, 3, GAUSSIAN);
		threshold(image_gray, image_gray, 0, 255, CV_THRESH_OTSU);
		imMorphology(&image_gray, &image_gray, MORPH_RECT, 5, CLOSING);
		Canny(image_gray, image_gray, 100, 150, 3); //canny의 값 조절

		// set ROI
		Point pts[1][4];
		pts[0][0] = { image.cols * 1 / 5 + 50, image.rows * 5 / 8 };
		pts[0][1] = { 0, image.rows - 70 };
		pts[0][2] = { image.cols, image.rows - 70 };
		pts[0][3] = { image.cols * 4 / 5 - 50, image.rows * 5 / 8 };

		const Point* ppt[1] = { pts[0] };
		int npt[] = { 4 };

		mask = Mat::zeros(image.size(), CV_8UC1);
		fillPoly(mask, ppt, npt, 1, Scalar(255, 255, 255), 8);
		//imshow("mask", mask);

		Mat roiImg;
		bitwise_and(image_gray, mask, roiImg);

		vector<Vec2f> lines;
		HoughLines(roiImg, lines, 1, CV_PI / 180, 65, 0, 0, 0, CV_PI);
		
		bool leftflag = false;
		bool rightflag = false;
		bool midflag = false;
		
		//detecting lines
		if (lines.size() > 0) {
			for (size_t i = 0; i < lines.size(); i++) {

				float rho = lines[i][0], theta = lines[i][1];

				if (theta < CV_PI * 145 / 180 && theta >= CV_PI * 120 / 180) {
					rho_right = 0, theta_rigth = CV_PI * 2 / 3;
					if (theta > theta_rigth) {
						theta_rigth = theta;
						rho_right = rho;
						rightflag = true;
						flag = true;
					}
				}

				if (theta > CV_PI * 35 / 180 && theta <= CV_PI * 60 / 180) {
					rho_left = 0, theta_left = CV_PI / 3;
					if (theta < theta_left) {
						theta_left = theta;
						rho_left = rho;
						leftflag = true;
						flag = true;

					}
				}

				if (theta < CV_PI * 35 / 180 || theta > CV_PI * 145 / 180) {
					rho_right = 0, theta_rigth = CV_PI * 120 / 180;
					rho_left = 0, theta_left = CV_PI * 60 / 180;

					theta_mid = theta;
					rho_mid = rho;
					midflag = true;
					leftflag = false;
					rightflag = false;
					flag = false;
				}
			}
		}


		/* draw line according to each flag */
		if (flag && (!midflag)) {
			
			drawinterpoint(&cdstP, &interP, rho_left, theta_left, rho_right, theta_rigth, RED);

			if (leftflag && rightflag) {
				drawline(&cdstP, interP, rho_left, theta_left, GREEN);
				drawline(&cdstP, interP, rho_right, theta_rigth, BLUE);
			}
			else if (leftflag && (!rightflag)) {
				drawline(&cdstP, interP, rho_left, theta_left, GREEN);
				drawline(&cdstP, interP, rho_right, theta_rigth, YELLOW);
			}
			else if ((!leftflag) && rightflag) {
				drawline(&cdstP, interP, rho_left, theta_left, YELLOW);
				drawline(&cdstP, interP, rho_right, theta_rigth, BLUE);

			}
			else if ((!leftflag) && (!rightflag)) {
				drawline(&cdstP, interP, rho_left, theta_left, YELLOW);
				drawline(&cdstP, interP, rho_right, theta_rigth, YELLOW);
			}

			calculateBias(&cdstP, &bcent, &bias, rho_left, theta_left, rho_right, theta_rigth);

			/*draw bias line*/
			Point pt1(cdstP.cols / 2, cdstP.rows - 70);
			Point pt2(cdstP.cols / 2, cdstP.rows);

			line(cdstP, pt1, pt2, WHITE, 3);

			Point bi(bcent, cdstP.rows - 70);
			Point as(bcent, cdstP.rows);
			line(cdstP, bi, as, PINK, 3);

		}
		else {
			drawinterpointCenterline(&cdstP, interP, rho_mid, theta_mid, RED);
		}



		/*write information in frame*/
		///bais
		char Massage_bias[50] = "bias = ";
		char sbias[20];
		sprintf(sbias, "%.2lf", bias);
		strcat(Massage_bias, sbias);


		if (bias > 0) {
			char per[20] = " % right";
			strcat(Massage_bias, per);
			putText(cdstP, Massage_bias, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 2);
		}
		else {
			char per[20] = " % left";
			strcat(Massage_bias, per);
			putText(cdstP, Massage_bias, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 2);
		}

		///line change
		if(!flag){
			char Massage_safe[50] = "line? change line";
			if (bias > 0) {
				putText(cdstP, Massage_safe, Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.8, RED, 2);
			}
			else {
				putText(cdstP, Massage_safe, Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.8, RED, 2);
			}
		}else {
			char Massage_safe[50] = "line? safe in line";
			putText(cdstP, Massage_safe, Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2);
		}

		/// fps
		char sfps[20];
		char Massage_fps[50] = "fps =  ";
		sprintf(sfps, "%d", ffps);
		strcat(Massage_fps, sfps);

		putText(cdstP, Massage_fps, Point(10, 80), FONT_HERSHEY_SIMPLEX, 0.8, BLUE, 2);


		/* frame check */
		clock_t finsh = clock();
		clock_t del = finsh-start;
		if (del > 1000) {
			ffps = fps;
			fps = 0;
			start = clock();
		}
		else {
			fps++;
		}

		imshow("roiImg", roiImg);
		imshow("image", cdstP);
		/// 영상 저장
		//outputVideo << cdstP;

		char c = (char)waitKey(10);
		if (c == 27)
			break;

	}

	return 0;
}
