#include <opencv.hpp>

using namespace cv;

int main() {
	Mat image(100, 100, CV_8UC1, Scalar(255));
	imshow("1", image);
	waitKey(0);
}