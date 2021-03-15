/////////////////////////////////////////////////////////////////////////
// @Copyright(C) Pyrokine
// All rights reserved
// 博客 http://www.cnblogs.com/Pyrokine/
// Github https://github.com/Pyrokine
// 创建日期 20190227
// 版本 1.2
//**********************************************************************
// V1.0
// 实现了RamerDouglasPeucker的基本算法
//
// V1.1
// 增加注释量，将算法提取到单独文件并可以独立调用，增加命名空间myrdp
//
// V1.2
// 代码重构
/////////////////////////////////////////////////////////////////////////

#ifndef _MYRDP_
#define _MYRDP_

#include <opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <baseFunc.h>

using namespace std;
using namespace cv;

namespace myrdp {
	class RDP {
	public:
		typedef struct _structLidarPointPolar {
			float radian = 0.0f;
			int angle = 0;
			float range = 0.0f;
			float originX = 0.0f;
			float originY = 0.0f;
			float globalX = 0.0f;
			float globalY = 0.0f;
			int intensity = 0;
			bool split = false;
			int sn;
		} structLidarPoint;

		typedef struct _structFeatureScan {
			vector<structLinesInfo> linesInfo;
			RDP::structLidarPoint lidarPos;
			vector<structPosition> scanImPoint;
			vector<vector<structLidarPoint>> pointGroup;
		} structFeatureScan;

		structFeatureScan FeatureScan(structMapParam& mapParam, vector<structLidarPoint>& lidarPoint, const int lenLidarPoint, const int regionPointLimitNumber, const float threLine, const double line_len_threshold_m);

	private:
		int lenLidarPoint, regionPointLimitNumber;
		float threLine;
		vector<structLidarPoint> lidarPoint;
		vector<double> scanPose = { 0, 0, 0 };

		typedef struct _structPointCell {
			structLidarPoint startPoint;
			structLidarPoint endPoint;
			int startPointNumber = 0;
			int endPointNumber = 0;
		} structPointCell;

		vector<structPointCell> RegionSegmentation();
		void SplitMerge(vector<structPointCell>& pointCell);
		void SplitMergeAssistant(const int startPointNumber, const int endPointNumber);
		float getThresholdDeltaDist(const float val);
	};
}

#endif // !_MYRDP_
