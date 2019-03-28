/////////////////////////////////////////////////////////////////////////
//@Copyright(C) Pyrokine
//All rights reserved
//博客 http://www.cnblogs.com/Pyrokine/
//Github https://github.com/Pyrokine
//创建日期 20190227
//版本 1.1
//**********************************************************************
//V1.0
//实现了RamerDouglasPeucker的基本算法
//
//V1.1
//增加注释量，将算法提取到单独文件并可以独立调用，增加命名空间myrdp
//
/////////////////////////////////////////////////////////////////////////

#ifndef _MYRDP_
#define _MYRDP_

#include <opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <baseFunc.h>
using namespace cv;

namespace myrdp {
	typedef struct _structLidarPointPolar {
		double range;
		double angle;
		bool split;
	} structLidarPointPolar;

	typedef struct _structLidarPointRec {
		double x;
		double y;
		int num;
	} structLidarPointRec;

	typedef struct _structPointCell {
		structLidarPointRec startPointLoc;
		structLidarPointRec endPointLoc;
		int startPointNum;
		int endPointNum;
	} structPointCell;

	typedef struct _structRegionSegmentation {
		structPointCell *pointCell;
		int cellNum;
	} structRegionSegmentation;

	typedef struct _structFeatureScan {
		Mat lineIm;
		structLinesInfo *linesInfo;
		structLidarPointRec lidarPos;
		int len_linesInfo;
	} structFeatureScan;

	structFeatureScan FeatureScan(structMapParam mapParam, structLidarPointPolar *lidarPointPolar, int len_lp, int RegionPointLimitNumber, double Threshold_line, double line_len_threshold_m);
	structRegionSegmentation RegionSegmentation(structLidarPointPolar *lidarPoint, int len_lp, double *scanPose, int RegionPointLimitNumber);
	void SplitMerge(structLidarPointPolar *lidarPointPolar, int len_lp, double *scanPose, structRegionSegmentation RS, double threLine);
	void SplitMergeAssistant(structLidarPointPolar *lidarPointPolar, structLidarPointRec *lidarPointRec, int len_lp, int startPoint, int endPoint, double threLine);
	double getThresholdDeltaDist(double val);
}

#endif // !_MYRDP_
