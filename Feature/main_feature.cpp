#define _CRT_SECURE_NO_WARNINGS
#include"../LSD/myLSD.h"
#include"../LSD/myRDP.h"
#include"../LSD/FeatureAssociation.h"
#include<stdio.h>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<time.h>

int main()
{
	using namespace std;
	using namespace cv;
	using namespace myfa;
	clock_t starttime, endtime;
	//readtxt()
	structMapParam mapParam;
	Mat mapValue;
	vector<Mat> Scan;
	Mat Odom;
	Mat Lidar;

	FILE* fp;
	// mapParam
	fp = fopen("..\\data\\mapParam.txt", "r");
	fscanf(fp, "%u%u%lf%lf%lf", &mapParam.oriMapCol, &mapParam.oriMapRow, &mapParam.mapResol, &mapParam.mapOriX, &mapParam.mapOriY);
	fclose(fp);
	// mapValue
	fp = fopen("..\\data\\mapValue.txt", "r");
	mapValue.create(mapParam.oriMapRow, mapParam.oriMapCol, CV_64F);
	for (int i = 0; i < mapParam.oriMapRow; i++)
		for (int j = 0; j < mapParam.oriMapCol; j++)
			fscanf(fp, "%lf", mapValue.ptr<double>(i, j));
	fclose(fp);
	// Odom
	fp = fopen("..\\data\\Odom.txt", "r");
	int ch;
	fpos_t fpos = 0;
	int Cols = 0;
	while ((ch = fgetc(fp)) != EOF)
	{
		if (ch != '\n')
			Cols++;
		while ((ch = fgetc(fp)) != EOF && ch != '\n');
	}
	fsetpos(fp, &fpos);
	Odom.create(3, Cols, CV_64F);
	for (int c = 0; c < Cols; c++)
		for (int r = 0; r < 3; r++)
			fscanf(fp, "%lf", Odom.ptr<double>(r, c));
	fclose(fp);
	// Lidar
	fp = fopen("..\\data\\Lidar.txt", "r");
	char str[50];
	Mat mypair(1, 2, CV_64F);
	while (fscanf(fp, "%s%lf", str, mypair.ptr<double>(0, 1)) != EOF)
	{
		if (strcmp(str, "inf") == 0)
			*mypair.ptr<double>(0, 0) = INFINITY;
		else
			*mypair.ptr<double>(0, 0) = stod(str);
		Lidar.push_back(mypair);
	}
	fclose(fp);
	// Scan
	int nframe = Odom.cols;
	int scanPoint_count = Lidar.rows / nframe;
	Scan.resize(nframe);
	for (int i = 0; i < nframe; i++)
	{
		for (int j = 0; j < scanPoint_count; j++)
		{
			if (*Lidar.ptr<double>(i * scanPoint_count + j, 0) != INFINITY)
				Scan[i].push_back(Lidar.row(i * scanPoint_count + j));
		}
	}
	// 真实位置
	Mat recored_Odom = (Mat_<int>(1, 10) << 1, 10, 20, 28, 37, 47, 58, 69, 80, 90);
	Mat realPos = (Mat_<double>(2, 10) << 2.4, 3.402, 4.442, 5.193, 6.642, 7.185, 8.249, 9.484, 10.516, 11.588, 0, 0.03, 0.12, 0.15, 0.19, 0.3, 0.42, 0.62, 0.79, 0.98);
	Mat sampleRealPos;
	samplePos(realPos, recored_Odom, sampleRealPos);
	//mapCache
	fp = fopen("..\\data\\mapCache.txt", "r");
	Mat mapCache(mapParam.oriMapRow, mapParam.oriMapCol, CV_64F);
	for (int i = 0; i < mapParam.oriMapRow; i++)
		for (int j = 0; j < mapParam.oriMapCol; j++)
			fscanf(fp, "%lf", mapCache.ptr<double>(i, j));
	fclose(fp);
	
	starttime = clock();
	// 将 未知区域 和 空闲区域变成相同区域，与占据点区分
	MatIterator_<double> it = mapValue.begin<double>(), it_end = mapValue.end<double>();
	for (; it < it_end; it++)
		if (*it == -1)
			*it = 0;
	// 转化为 灰度值
	it = mapValue.begin<double>();
	Mat mapGray(mapValue.rows, mapValue.cols, CV_8U);
	for (uint8_t& p : Mat_<uint8_t>(mapGray))
	{
		p = (int)*it * 255;
		it++;
	}
	// LSD 提取全局地图 线特征
	mylsd::structLSD sLSD = mylsd::myLineSegmentDetector(mapGray, mapGray.cols, mapGray.rows, 0.3, 0.6, 22.5, 0.7, 1024);
	Mat MaplineIm = sLSD.lineIm;
	vector<structLinesInfo> MaplinesInfo(sLSD.linesInfo, sLSD.linesInfo + sLSD.len_linesInfo);

	Mat Feature_estiPose(3, Odom.cols, CV_64F);
	Mat ScanPose;
	myrdp::structLidarPointPolar lidarPointPolar[360];
	int len_lp;
	double estimatePose[3], estimatePose_realworld[3];
	Mat poseAll;
	for (int i = 0; i < Feature_estiPose.cols; i++)
	{
		cout << "特征匹配 处理Scan 帧数：" << i + 1 << endl;
		// 处理激光点云图
		vector<double> ScanRanges = Scan[i].col(0), ScanAngles = Scan[i].col(1);
		if (i == 0)
			ScanPose = Odom.col(0);
		else
			ScanPose = Odom.col(0) + Odom.col(1);
		len_lp = Scan[i].rows;
		for (int r = 0; r < len_lp; r++)
		{
			lidarPointPolar[r].range = *Scan[i].ptr<double>(r, 0);
			lidarPointPolar[r].angle = *Scan[i].ptr<double>(r, 1);
			lidarPointPolar[r].split = false;
		}
		myrdp::structFeatureScan sFS = myrdp::FeatureScan(mapParam, lidarPointPolar, len_lp, 3, 0.04, 0.5);
		Mat ScanlineIm = sFS.lineIm;
		vector<structLinesInfo> ScanlinesInfo(sFS.linesInfo, sFS.linesInfo + sFS.len_linesInfo);
		int LidarPos[2] = { (int)sFS.lidarPos.x, (int)sFS.lidarPos.y };
		myfa::FeatureAssociation(
			ScanlineIm,
			ScanlinesInfo,
			MaplinesInfo,
			mapParam,
			LidarPos,
			MaplineIm,
			mapCache,
			mapValue,
			ScanRanges,
			ScanAngles,
			estimatePose_realworld,
			estimatePose,
			poseAll
		);
		*Feature_estiPose.ptr<double>(0, i) = estimatePose_realworld[0];
		*Feature_estiPose.ptr<double>(1, i) = estimatePose_realworld[1];
		*Feature_estiPose.ptr<double>(2, i) = estimatePose_realworld[2];
	}
	endtime = clock();
	cout << "特征匹配 运行时间：" << (endtime - starttime) / (double)CLOCKS_PER_SEC << endl;
	// 计算 特征匹配 误差
	Mat F_estiPos = Feature_estiPose.rowRange(0, 2).colRange(0, *(recored_Odom.end<int>() - 1));
	int esti_n = F_estiPos.cols < sampleRealPos.cols ? F_estiPos.cols : sampleRealPos.cols;
	Mat F_err(3, esti_n, CV_64F);
	double dx, dy, mean_s;
	for (int i = 0; i < esti_n; i++)
	{
		dx = abs(*F_estiPos.ptr<double>(0, i) - *sampleRealPos.ptr<double>(0, i));
		dy = abs(*F_estiPos.ptr<double>(1, i) - *sampleRealPos.ptr<double>(1, i));
		mean_s = sqrt(dx * dx + dy * dy);
		*F_err.ptr<double>(0, i) = dx;
		*F_err.ptr<double>(1, i) = dy;
		*F_err.ptr<double>(2, i) = mean_s;
	}
	// 绘制误差图
	Mat chart(750, 1000, CV_8UC3, Scalar(200, 200, 200));
	chart.rowRange(75, 675).colRange(100, 900).setTo(Scalar(255, 255, 255));
	rectangle(chart, Point(100, 75), Point(900, 675), Scalar(0, 0, 0));
	for (int i = 0; i < 10; i++)
	{
		line(chart, Point(100 + i * 800 / 9, 675), Point(100 + i * 800 / 9, 675 - 10), Scalar(0, 0, 0));
		putText(chart, to_string(i * 10), Point(100 + i * 800 / 9 - 5, 675 + 25), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_AA);
	}
	for (int i = 0; i < 7; i++)
	{
		line(chart, Point(100, 675 - i * 600 / 6), Point(100 + 10, 675 - i * 600 / 6), Scalar(0, 0, 0));
		string str = to_string(i * 0.5);
		str = string(str.data(), str.data() + 3);
		putText(chart, str, Point(100 - 35, 675 - i * 600 / 6), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_AA);
	}
	vector<Point> points[3];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < esti_n; j++)
		{
			Point pt;
			pt.x = 800 * (j + 1) / (esti_n) + 99;
			pt.y = 600 - int(600 * *F_err.ptr<double>(i, j) / 3) + 74;
			points[i].push_back(pt);
		}
	polylines(chart, points[0], false, Scalar(0, 0, 255), 1, LINE_AA, 0);
	polylines(chart, points[1], false, Scalar(0, 255, 0), 1, LINE_AA, 0);
	polylines(chart, points[2], false, Scalar(255, 0, 0), 1, LINE_AA, 0);
	imshow("", chart);
	waitKey(0);
	return 0;
}
