#include "myOpenCV.h"

/* [scale]
0: GRAY
1: COLOR
*/
void imRead(cv::Mat* src, const char filename[], int scale) {
	
	cv::String imname = filename;

	*src = imread(imname, scale);

	if ((*src).empty())	
	{
		std::cout << "File Read Failed : src is empty" << std::endl;
		cv::waitKey(0);
	}

}


void imWrite(cv::Mat* src, const char filename[]) {

	cv::String imname = filename;

	imwrite(imname, *src);

	if ((*src).empty())
	{
		std::cout << "File Write Failed : src is empty" << std::endl;
		cv::waitKey(0);
	}


}



void imNormalHist1D(cv::Mat* src, cv::Mat* dst, int histSize) {

	// Set the ranges for B,G,R
	float range[] = { 0, histSize }; //the upper boundary is exclusive
	const float* histRange = { range };

	// Set histogram param 
	bool uniform = true, accumulate = false;

	// Compute the histograms
	cv::Mat hist1D;
	cv::calcHist(src, 1, 0, cv::Mat(), hist1D, 1, &histSize, &histRange, uniform, accumulate);


	// Draw the histograms for B, G and R
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);


	cv::Mat histImage1D(hist_h, hist_w, CV_8UC1, cv::Scalar(255));
	normalize(hist1D, hist1D, 0, histImage1D.rows, cv::NORM_MINMAX, -1, cv::Mat());

	*dst = histImage1D;

	// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage1D, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist1D.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(hist1D.at<float>(i))),
			cv::Scalar(0, 0, 0), 2, 8, 0);
	}

}

/* [threshold_type]
0: THRESH_BINARY
1: THRESH_BINARY_INV 
2: THRESH_TRUNC 
3: THRESH_TOZERO 
4: THRESH_TOZERO_INV
7: THRESH_MASK
8: THRESH_OTSU
16: THRESH_TRIANGLE
*/
void imThreshold(cv::Mat* src, cv::Mat* dst, double threshold_value, double ifBinaryMax, int threshold_type) {
	threshold(*src, *dst, threshold_value, ifBinaryMax, threshold_type);
}

/* [threshold_type]
0: THRESH_BINARY
1: THRESH_BINARY_INV
*/
void imAdpthreshold(cv::Mat* src, cv::Mat* dst, double BinaryMax, int threshold_value, int size, double constant) {
	adaptiveThreshold(*src, *dst, BinaryMax, cv::BORDER_REPLICATE, threshold_value, size, constant);
}

/* [ element shape ]
0: MORPH_RECT
1: MORPH_CROSS
2: MORPH_ELLIPSE
天天天天天天天天天天天天天天天天天天
[ Mopology type ]
0: DILATE
1: ERODE
2: OPENING (erode -> dilate)
3: CLOSING (dilate -> erode)
4: OPTOCL (opening -> closing)
5: CLTOOP (clsoing -> opening)
*/
void imMorphology(cv::Mat* src, cv::Mat* dst, int elementshape, int elementsize, int type){
	
	cv::Mat element = cv::getStructuringElement(elementshape, cv::Size(elementsize, elementsize));

	switch (type)
	{
	case DILATE:
		dilate(*src, *dst, element);
		break;

	case ERODE:
		erode(*src, *dst, element);
		break;

	case OPENING:
		erode(*src, *dst, element);
		dilate(*dst, *dst, element);
		break;

	case CLOSING:
		dilate(*src, *dst, element);
		erode(*dst, *dst, element);
		break;

	case OPTOCL:
		erode(*src, *dst, element);
		dilate(*dst, *dst, element);
		dilate(*dst, *dst, element);
		erode(*dst, *dst, element);
		break;

	case CLTOOP:
		dilate(*src, *dst, element);
		erode(*dst, *dst, element);
		erode(*dst, *dst, element);
		dilate(*dst, *dst, element);
		break;

	}

}



/* [ filter type ]
0: FILTER2D
1: BLUR
2: GAUSSIAN
3: MEDIAN
4: BILATERAL
5: LAPLACIAN (SHARPENING)
*/
void imFilter(cv::Mat* src, cv::Mat* dst, int size, int type) {

	cv::Size kernelSize = cv::Size(size, size);
	int delta = 0;
	int ddepth;

	switch (type) {
	
	case FILTER2D:
		ddepth = -1;
	
		(*src).convertTo(*src, CV_8UC1);

		filter2D(*src, *dst, ddepth, size);
		break;
	
	case BLUR:
		cv::blur(*src, *dst, cv::Size(size, size), cv::Point(-1, -1));
		break;
	
	case GAUSSIAN:
		cv::GaussianBlur(*src, *dst, cv::Size(size, size), 0, 0);
		break;
	
	case MEDIAN:
		cv::medianBlur(*src, *dst, size);
		break;

	case BILATERAL:
		cv::bilateralFilter(*src, *dst, size, size * 2, size / 2);
		break;	
	
	case LAPLACIAN:
		int scale = 1;
		ddepth = CV_16S;

		cv::Mat Laplacian = *dst;

		cv::Laplacian(*src, Laplacian, ddepth, size, scale, delta, cv::BORDER_DEFAULT);
		(*src).convertTo(*src, CV_16S);
		
		*dst = *src - Laplacian;
		(*dst).convertTo(*dst, CV_8U);

		break;	
	}




}


/* [ filter type ]
0: SQDIFF
1: SQDIFF NORMED
2: TM CCORR
3: TM CCORR NORMED
4: TM COEFF
5: TM COEFF NORMED
*/
void imTemplateMaching(cv::Mat img, cv::Mat templ , cv::Mat* result, cv::Mat *img_display, int match_method) {

	img.copyTo(*img_display);

	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	(*result).create(result_rows, result_cols, CV_32FC1);

	cv::matchTemplate(img, templ, *result, match_method);
	cv::normalize(*result, *result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

	double minVal, maxVal;
	cv::Point minLoc, maxLoc, matchLoc;
	cv::minMaxLoc(*result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED) {
		matchLoc = minLoc;
	}
	else {
		matchLoc = maxLoc;
	}

	cv::rectangle(*img_display, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(0), 2, 8, 0);
	cv::rectangle(*result, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(0), 2, 8, 0);


}


void drawHoughLines(cv::Mat* src, std::vector<cv::Vec2f> lines, cv::Scalar color) {

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 2000 * (-b));
		pt1.y = cvRound(y0 + 2000 * (a));
		pt2.x = cvRound(x0 - 2000 * (-b));
		pt2.y = cvRound(y0 - 2000 * (a));
		line(*src, pt1, pt2, color, 3, cv::LINE_AA);
	}

}

void drawHoughLine(cv::Mat* src, float rho, float theta, cv::Scalar color) {
	cv::Point pt1, pt2;
	double a = cos(theta), b = sin(theta);
	double x0 = a * rho, y0 = b * rho;
	pt1.x = cvRound(x0 + 2000 * (-b));
	pt1.y = cvRound(y0 + 2000 * (a));
	pt2.x = cvRound(x0 - 2000 * (-b));
	pt2.y = cvRound(y0 - 2000 * (a));
	line(*src, pt1, pt2, color, 3, cv::LINE_AA);
}

void drawinterpoint(cv::Mat* src, cv::Point* interP, float rho1, float theta1, float rho2, float theta2, cv::Scalar color) {
	
	double x1, y1, a1, b1;

	x1 = cvRound(rho1 * cos(theta1) + 2000 * (-sin(theta1)));
	y1 = cvRound(rho1 * sin(theta1) + 2000 * (cos(theta1)));

	a1 = 1 / tan(theta1);
	b1 = -y1 - x1 * a1;

	double x2, y2, a2, b2;

	x2 = cvRound(rho2 * cos(theta2) + 2000 * (-sin(theta2)));
	y2 = cvRound(rho2 * sin(theta2) + 2000 * (cos(theta2)));

	a2 = 1/ tan(theta2);
	b2 = -y2 - x2 * a2;

	double x, y;

	x = (b2 - b1) / (a1 - a2);
	y = -(a2 * (b2 - b1) / (a1 - a2) + b2);

	if (x > (*src).cols / 3 && x < (*src).cols * 2 / 3) {
		interP->x = (b2 - b1) / (a1 - a2);
		interP->y = -(a2 * (b2 - b1) / (a1 - a2) + b2);
	}
	cv::circle(*src, *interP, 5, color, 3);

	//std::cout << "inter point (x, y) = (" << interP->x << ", " << interP->y << ")\n";
	

}



void drawinterpointCenterline(cv::Mat* src, cv::Point interPC, float rho, float theta, cv::Scalar color) {

	double x1, y1, a1, b1;

	if (theta < CV_PI * 40 / 180) {
		x1 = cvRound(rho * cos(theta) + 2000 * (-sin(theta)));
		y1 = cvRound(rho * sin(theta) + 2000 * (cos(theta)));
	}
	else {
		x1 = cvRound(rho * cos(theta) - 2000 * (-sin(theta)));
		y1 = cvRound(rho * sin(theta) - 2000 * (cos(theta)));
	}

	a1 = -1 / tan(theta);
	b1 = -y1 - x1 * a1;


	cv::Point pt;

	pt.x = x1;
	pt.y = y1;

	double x2, y2, a2, b2;


	//std::cout << "inter point (x, y) = (" << interPC->x << ", " << interPC->y << ")\n";

	cv::circle(*src, interPC, 5, color, 3);


	line(*src, pt, interPC, color, 3, cv::LINE_AA);

}


void drawline(cv::Mat* src, cv::Point interPC, float rho, float theta, cv::Scalar color) {
	
	double x1, y1, a1, b1;
	cv::Point pt;


	if (theta <= CV_PI * 60 / 180) {
		x1 = cvRound(rho * cos(theta) + 2000 * (-sin(theta)));
		y1 = cvRound(rho * sin(theta) + 2000 * (cos(theta)));

	}
	else {
		x1 = cvRound(rho * cos(theta) - 2000 * (-sin(theta)));
		y1 = cvRound(rho * sin(theta) - 2000 * (cos(theta)));
		
	}

	pt.x = x1;
	pt.y = y1;

	line(*src, pt, interPC, color, 3, cv::LINE_AA);

}

void calculateBias(cv::Mat* src, double* bcent, double* bias, float rho1, float theta1, float rho2, float theta2) {
	double x1, y1, a1, b1;

	x1 = cvRound(rho1 * cos(theta1) + 2000 * (-sin(theta1)));
	y1 = cvRound(rho1 * sin(theta1) + 2000 * (cos(theta1)));

	a1 = 1 / tan(theta1);
	b1 = y1 + x1 * a1;

	double x2, y2, a2, b2;

	x2 = cvRound(rho2 * cos(theta2) + 2000 * (-sin(theta2)));
	y2 = cvRound(rho2 * sin(theta2) + 2000 * (cos(theta2)));

	a2 = 1 / tan(theta2);
	b2 = y2 + x2 * a2;

	double bwidth;

	bwidth = -((*src).rows - b1) / a1- ((*src).rows - b2) / a2;
	*bcent = -(((*src).rows - b1) / a1 + ((*src).rows - b2) / a2) / 2;


	*bias = (*bcent - (*src).cols / 2) / bwidth * 100;

}

void drawHoughLinesP(cv::Mat* src, std::vector<cv::Vec4i> lines, cv::Scalar color) {

	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
		line(*src, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), color, 3, cv::LINE_AA);
	}

}

void drawHoughLineP(cv::Mat* src, cv::Vec4i lines, cv::Scalar color) {
	line(*src, cv::Point(lines[0], lines[1]), cv::Point(lines[2], lines[3]), color, 3, cv::LINE_AA);

}