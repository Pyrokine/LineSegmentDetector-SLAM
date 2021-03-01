#include <time.h>
#include <stdio.h>
#include <opencv.hpp>
#include <fstream>
#include <myLSD.h>
#include <myRDP.h>
#include <myFA.h>
#include <baseFunc.h>

using namespace cv;
using namespace std;

myfa::structFAInput trans2FA(myrdp::structFeatureScan FS, mylsd::LSD::structLSD LSD, Mat mapCache, structPosition lastPose,\
	Eigen::Matrix<double, 9, 1> kalman_x, Eigen::Matrix<double, 9, 9> kalman_P, structPosition ScanPose, Mat Display);

int main() {
	clock_t time_start, time_end;
	time_start = clock();
	//路径
	//string path1 = "../line_data/data0/";
	//string path1 = "../data_20190523/data/";
	//string path1 = "../data_20190514/data_f4key/data10/";
	//string path1 = "../data_20190513/data_f3key/data9/";
	//string path1 = "../line_data/data9/";
	string path1 = "../data_20210223/3236/";
	string path2;
	const char *path;
	//读取mapParam 地图信息
	path2 = path1 + "mapParam.txt";
	path = path2.data();
	FILE *fp = fopen(path, "r");
	structMapParam mapParam;
	fscanf(fp, "%d %d %lf %lf %lf", &mapParam.oriMapCol, &mapParam.oriMapRow, &mapParam.mapResol, &mapParam.mapOriX, &mapParam.mapOriY);
	fclose(fp);
	int oriMapCol = mapParam.oriMapCol, oriMapRow = mapParam.oriMapRow;
	
	//读取mapValue 地图像素数据
	int cnt_row, cnt_col;
	path2 = path1 + "mapValue.txt";
	path = path2.data();
	fp = fopen(path, "r");
	Mat mapValue = Mat::zeros(oriMapRow, oriMapCol, CV_8UC1);
	int max = 0;
	for (cnt_row = 0; cnt_row < oriMapRow; cnt_row++)
		for (cnt_col = 0; cnt_col < oriMapCol; cnt_col++)
			fscanf(fp, "%d", &mapValue.ptr<uint8_t>(cnt_row)[cnt_col]);
	fclose(fp);
	imshow("1", mapValue);
	//waitKey(1);

	//读取Odometry 里程计数据
	vector<structPosition> Odom;
	path2 = path1 + "Odom.txt";
	path = path2.data();
	fp = fopen(path, "r");
	while (!feof(fp)) {
		structPosition tempOdom;
		fscanf(fp, "%lf %lf %lf", &tempOdom.x, &tempOdom.y, &tempOdom.ang);
		Odom.push_back(tempOdom);
	}
	fclose(fp);
	Odom[0].x = 0;
	//for (cnt_row = 0; cnt_row < Odom.size(); cnt_row++) {
	//	printf("%d : %f %f %f\n", cnt_row, Odom[cnt_row].x, Odom[cnt_row].y, Odom[cnt_row].ang);
	//}

	//计算mapCache，用于特征匹配的先验概率
	mylsd::LSD lsd = mylsd::LSD();
	//Mat mapCache = lsd.createMapCache(mapValue, mapParam.mapResol);
	Mat mapCache;

	//LineSegmentDetector 提取地图边界直线信息
	double last_time = clock();
	mylsd::LSD::structLSD LSD = lsd.myLineSegmentDetector(mapValue, oriMapCol, oriMapRow, lsd_sca, lsd_sig, lsd_angThre, lsd_denThre, pseBin);
	printf("%lf\n", (clock() - last_time) / CLOCKS_PER_SEC);
	Mat Display = mapValue.clone();
	resize(Display, Display, Size(0, 0), 0.5, 0.5);

#ifdef drawPicture
	imshow("temp", LSD.lineIm);
#endif
	waitKey(0);

	//printf("%d %d\n", Display.size[0], Display.size[1]);
	//imshow("mapCache", mapCache);
	//imshow("Display", Display);
	//waitKey(0);

	//初始化运动过程
	structPosition lastPose;
	lastPose.x = -1;
	lastPose.y = -1;
	lastPose.ang = 0;
	Eigen::Matrix<double, 9, 1> kalman_x;
	Eigen::Matrix<double, 9, 9> kalman_P;
	kalman_x << -1, -1, 0, 0, 0, 0, 0, 0, 0;
	kalman_P << 100, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 100, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 100, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 1, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 1, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 1, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0.1, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0.1, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0.1;

	//读取雷达信息
	path2 = path1 + "Lidar.txt";
	path = path2.data();
	fp = fopen(path, "r");
	int i = 0, len_lp = 0, cnt_frame = 0;
	bool is_offset = false;
	myrdp::structLidarPointPolar lidarPointPolar[360];
	vector<double> angRotate;
	while (!feof(fp)) {
		len_lp = 0;
		bool is_EOF = false;
		printf("第%d帧:\n", ++cnt_frame);
		//每帧最多360帧数据 循环读取（输入）
		for (i = 0; i < pointPerLoop; i++) {
			double val1, val2;
			if (feof(fp)) {
				is_EOF = true;
				break;
			}
			fscanf(fp, "%lf%lf", &val1, &val2);

			if (val1 != INFINITY) {
				lidarPointPolar[len_lp].range = val1;
				lidarPointPolar[len_lp].angle = val2;
				lidarPointPolar[len_lp].split = false;
				len_lp++;
			}
		}
		if (is_EOF == false) {
			//匹配雷达特征到地图特征 返回像素坐标和真实坐标
			myrdp::structFeatureScan FS = FeatureScan(mapParam, lidarPointPolar, len_lp, rdp_leastPoint, rdp_threLine, rdp_leastDist);
			//imshow("RDP", FS.lineIm);
			
			//Mat poseAll;

			structPosition ScanPose;
			double theta = 0;
			if (abs(kalman_x(0) + 1) < 0.0001) {
				ScanPose.x = 0;
				ScanPose.y = 0;
				ScanPose.ang = 0;
			}
			else {
				int cnt;
				for (cnt = 0; cnt < angRotate.size(); cnt++) {
					theta += angRotate[cnt];
				}
				theta /= angRotate.size();

				structPosition tempScanPose;
				tempScanPose.x = (Odom[cnt_frame].x - Odom[cnt_frame - 1].x) / mapParam.mapResol;
				tempScanPose.y = (Odom[cnt_frame].y - Odom[cnt_frame - 1].y) / mapParam.mapResol;
				tempScanPose.ang = atand(Odom[cnt_frame].ang - Odom[cnt_frame - 1].ang);
				ScanPose.x = tempScanPose.x * cosd(theta) - tempScanPose.y * sind(theta);
				ScanPose.y = tempScanPose.y * sind(theta) + tempScanPose.y * cosd(theta);
				ScanPose.ang = tempScanPose.ang;
			}

			myfa::structFAInput FAInput = trans2FA(FS, LSD, mapCache, lastPose, kalman_x, kalman_P, ScanPose, Display);
			myfa::structFAOutput FA = myfa::FeatureAssociation(&FAInput);

			kalman_x = FA.kalman_x;
			kalman_P = FA.kalman_P;
			lastPose.x = FA.kalman_x(0);
			lastPose.y = FA.kalman_x(1);
			lastPose.ang = FA.kalman_x(2);
			printf("x:%f y:%f theta:%f\n\n", kalman_x(0), kalman_x(1), theta);

			double angDiff = FA.kalman_x(2) - atand(Odom[cnt_frame].ang);
			if (abs(angDiff) > 90 && cnt_frame == 1)
				is_offset = true;
			if (is_offset == true) {
				if (angDiff < 0)
					angDiff += 360;
			}
			angRotate.push_back(angDiff);

			//将图像坐标加入地图中
			circle(Display, Point((int)kalman_x(0) / 2, (int)kalman_x(1) / 2), 2, Scalar(255, 255, 255));
			//line(Display, Point((int)kalman_x(0), (int)kalman_x(1)), Point((int)kalman_x(0) + ScanPose.x, (int)kalman_x(1) + ScanPose.y), Scalar(255, 255, 255));
			imshow("Display", Display);
			waitKey(1);
		}
		else
			destroyWindow("Display");

		if (cnt_frame >= Odom.size() - 1)
			break;
	}
	fclose(fp);

	//imshow("MapGray", mapValue);
	time_end = clock();
	printf("time = %lf\n", (double)(time_end - time_start) / CLOCKS_PER_SEC);
	//imshow("lineIm", LSD.lineIm);
	waitKey(0);
	destroyAllWindows();
	return 0;
}

myfa::structFAInput trans2FA(myrdp::structFeatureScan FS, mylsd::LSD::structLSD LSD, Mat mapCache, structPosition lastPose,\
	Eigen::Matrix<double, 9, 1> kalman_x, Eigen::Matrix<double, 9, 9> kalman_P, structPosition ScanPose, Mat Display) {
	//将数据格式转为FeatureAssociation格式
	myfa::structFAInput FA;
	int i;
	//scanLinesInfo
	FA.scanLinesInfo.resize(FS.len_linesInfo);
	for (i = 0; i < FS.len_linesInfo; i++) {
		FA.scanLinesInfo[i].k = FS.linesInfo[i].k;
		FA.scanLinesInfo[i].b = FS.linesInfo[i].b;
		FA.scanLinesInfo[i].dx = FS.linesInfo[i].dx;
		FA.scanLinesInfo[i].dy = FS.linesInfo[i].dy;
		FA.scanLinesInfo[i].x1 = FS.linesInfo[i].x1;
		FA.scanLinesInfo[i].y1 = FS.linesInfo[i].y1;
		FA.scanLinesInfo[i].x2 = FS.linesInfo[i].x2;
		FA.scanLinesInfo[i].y2 = FS.linesInfo[i].y2;
		FA.scanLinesInfo[i].len = FS.linesInfo[i].len;
	}
	//mapLinesInfo
	FA.mapLinesInfo.resize(LSD.len_linesInfo);
	for (i = 0; i < LSD.len_linesInfo; i++) {
		FA.mapLinesInfo[i].k = LSD.linesInfo[i].k;
		FA.mapLinesInfo[i].b = LSD.linesInfo[i].b;
		FA.mapLinesInfo[i].dx = LSD.linesInfo[i].dx;
		FA.mapLinesInfo[i].dy = LSD.linesInfo[i].dy;
		FA.mapLinesInfo[i].x1 = LSD.linesInfo[i].x1;
		FA.mapLinesInfo[i].y1 = LSD.linesInfo[i].y1;
		FA.mapLinesInfo[i].x2 = LSD.linesInfo[i].x2;
		FA.mapLinesInfo[i].y2 = LSD.linesInfo[i].y2;
		FA.mapLinesInfo[i].len = LSD.linesInfo[i].len;
	}
	//lidarPos
	FA.lidarPose.x = (int)round(FS.lidarPos.x);
	FA.lidarPose.y = (int)round(FS.lidarPos.y);
	//printf("x:%lf y:%lf\n", FS.lidarPos.x, FS.lidarPos.y);
	FA.scanImPoint = FS.scanImPoint;
	FA.mapCache = mapCache;
	FA.lastPose = lastPose;
	FA.kalman_x = kalman_x;
	FA.kalman_P = kalman_P;
	FA.ScanPose = ScanPose;
	FA.Display = Display;

	return FA;
}

