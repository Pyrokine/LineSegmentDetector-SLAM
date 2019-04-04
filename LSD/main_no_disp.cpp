#include <time.h>
#include <stdio.h>
#include <opencv.hpp>
#include <fstream>
#include <myLSD.h>
#include <myRDP.h>
#include <FeatureAssociation.h>
#include <baseFunc.h>

using namespace cv;
using namespace std;

typedef struct _structFA {
	vector<myfa::SCANLINES_INFO> scanLinesInfo;
	vector<myfa::LINES_INFO> mapLinesInfo;
	myfa::MAP_PARAM mapParam;
	int lidarPos[2];
	vector<double> ScanRanges;
	vector<double> ScanAngles;
} structFA;

structFA trans2FA(myrdp::structFeatureScan FS, mylsd::structLSD LSD, structMapParam oriMapParam, myrdp::structLidarPointPolar *lidarPointPolar, int len_lp);

int nframe = 99, pointPerLoop = 360;

int main() {
	clock_t time_start, time_end;
	time_start = clock();
	//读取mapParam 地图信息
	FILE *fp = fopen("../data/mapParam.txt", "r");
	structMapParam mapParam;
	fscanf(fp, "%d %d %lf %lf %lf", &mapParam.oriMapCol, &mapParam.oriMapRow, &mapParam.mapResol, &mapParam.mapOriX, &mapParam.mapOriY);
	fclose(fp);
	int oriMapCol = mapParam.oriMapCol, oriMapRow = mapParam.oriMapRow;
	
	//读取mapValue 地图像素数据
	int cnt_row, cnt_col;
	fp = fopen("../data/mapValue.txt", "r");
	Mat mapValue = Mat::zeros(oriMapRow, oriMapCol, CV_8UC1);
	int max = 0;
	for (cnt_row = 0; cnt_row < oriMapRow; cnt_row++)
		for (cnt_col = 0; cnt_col < oriMapCol; cnt_col++)
			fscanf(fp, "%d", &mapValue.ptr<uint8_t>(cnt_row)[cnt_col]);
	fclose(fp);

	//计算mapCache，用于特征匹配的先验概率
	double z_occ_max_dis = 2;
	Mat mapCache = mylsd::createMapCache(mapValue, mapParam.mapResol, z_occ_max_dis);

	//LineSegmentDetector 提取地图边界直线信息
	mylsd::structLSD LSD = mylsd::myLineSegmentDetector(mapValue, oriMapCol, oriMapRow, 0.3, 0.6, 22.5, 0.7, 1024);

	//读取雷达信息
	fp = fopen("../data/Lidar.txt", "r");
	int i = 0, len_lp = 0;
	myrdp::structLidarPointPolar lidarPointPolar[360];
	while (!feof(fp)) {
		len_lp = 0;
		bool is_EOF = false;
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
			myrdp::structFeatureScan FS = FeatureScan(mapParam, lidarPointPolar, len_lp, 3, 0.04, 0.5);
			
			double estimatePose_realworld[3];
			double estimatePose[3];
			Mat poseAll;
			structFA FA = trans2FA(FS, LSD, mapParam, lidarPointPolar, len_lp);
			myfa::FeatureAssociation(FS.lineIm, FA.scanLinesInfo, FA.mapLinesInfo, FA.mapParam, FA.lidarPos, LSD.lineIm, \
				mapCache, mapValue, FA.ScanRanges, FA.ScanAngles, estimatePose_realworld, estimatePose, poseAll);
			for (int i = 0; i < 3; i++)
				cout << estimatePose_realworld[i] << '\t';
			cout << endl;
			for (int i = 0; i < 3; i++)
				cout << estimatePose[i] << '\t';
			cout << endl << endl;

			//将图像坐标加入地图中
			LSD.lineIm.ptr<uint8_t>((int)estimatePose[1])[(int)estimatePose[0]] = 255;
			imshow("lineIm", LSD.lineIm);
			waitKey(1);
		}
		else
			destroyWindow("lineIm");
	}
	fclose(fp);

	imshow("MapGray", mapValue);
	time_end = clock();
	printf("time = %lf\n", (double)(time_end - time_start) / CLOCKS_PER_SEC);
	imshow("lineIm", LSD.lineIm);
	waitKey(0);
	destroyAllWindows();
	return 0;
}

structFA trans2FA(myrdp::structFeatureScan FS, mylsd::structLSD LSD, structMapParam oriMapParam, myrdp::structLidarPointPolar *lidarPointPolar, int len_lp) {
	//将数据格式转为FeatureAssociation格式
	structFA FA;
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
	//mapParam
	FA.mapParam.mapHeigh = oriMapParam.oriMapCol;
	FA.mapParam.mapWidth = oriMapParam.oriMapRow;
	FA.mapParam.mapResol = oriMapParam.mapResol;
	FA.mapParam.mapOrigin[0] = oriMapParam.mapOriX;
	FA.mapParam.mapOrigin[1] = oriMapParam.mapOriY;
	//lidarPos
	FA.lidarPos[0] = FS.lidarPos.x;
	FA.lidarPos[1] = FS.lidarPos.y;
	printf("x:%lf y:%lf\n", FS.lidarPos.x, FS.lidarPos.y);
	//ScanRanges ScanAngles
	for (i = 0; i < len_lp; i++) {
		FA.ScanRanges.push_back(lidarPointPolar[i].range);
		FA.ScanAngles.push_back(lidarPointPolar[i].angle);
	}
	return FA;
}

