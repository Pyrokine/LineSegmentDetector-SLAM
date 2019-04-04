#include <opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

int main() {
	clock_t time_start, time_end;
	time_start = clock();
	FILE *fp;
	fp = fopen("../data/mapCache.txt", "r");
	Mat mapCache(1377, 428, CV_64F);
	for (unsigned i = 0; i < 1377; i++)
		for (unsigned j = 0; j < 428; j++)
			fscanf(fp, "%lf", &mapCache.ptr<double>(i)[j]);
	fclose(fp);
	time_end = clock();
	printf("time = %lf\n", (double)(time_end - time_start) / CLOCKS_PER_SEC);

	//mapCache
	ifstream File;
	File.open("..\\data\\mapCache.txt");
	for (unsigned i = 0; i < 1377; i++)
		for (unsigned j = 0; j < 428; j++)
			File >> *mapCache.ptr<double>(i, j);
	File.close();
	time_end = clock();
	printf("time = %lf\n", (double)(time_end - time_start) / CLOCKS_PER_SEC);
}