/////////////////////////////////////////////////////////////////////////
// @Copyright(C) Pyrokine
// All rights reserved
// 博客 http://www.cnblogs.com/Pyrokine/
// Github https://github.com/Pyrokine
// 创建日期 20190126
// 版本 2.2
//**********************************************************************
// V1.0
// 实现了LineSegmentDetector的基本算法， 并添加了基础的可视化演示
//
// V1.1
// 完成了算法优化，将定长数组用动态数组实现，将伪排序用快速排序实现，
// 优化了高斯降采样中计算高斯核的算法，改进后只需计算一次高斯核,但只适用于
// 0.3缩放比的地图（其他比例需要重新计算偏移值），将区域增长次数从多次减少
// 到1次
//
// V1.2
// 完成了Degree动画，并分离出main_with_disp.cpp，修正了UsedMap动画，增加了
// 梯度排名和区域内像素点显示，增加了虚警数的更新数据显示，提取出LSD算法为
// 单独函数
//
// V2.0
// 提取出LSD算法到单独文件，可以单独引用，增加命名空间mylsd
//
// V2.1
// 计算出FeatureAssociation需要的mapCache，用于表示先验概率
// 
// V2.2
// 代码重构
/////////////////////////////////////////////////////////////////////////

#ifndef _MYLSD_
#define _MYLSD_

#include <opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <baseFunc.h>
#include <unordered_map>

using namespace cv;
using namespace std;

namespace mylsd {
	class LSD {
	public:
		typedef struct _structLSD {
			Mat lineIm;
			vector<structLinesInfo> linesInfo;
			int len_linesInfo;
		} structLSD;

		structLSD myLineSegmentDetector(Mat& MapGray, const int _oriMapCol, const int _oriMapRow, const float _sca, const float _sig,
			const float _angThre, const float _denThre, const int _pseBin);
		Mat CreateMapCache(const Mat& MapGray, const float res);

	private:
		typedef struct _nodeBinCell {
			int x;
			int y;
		} nodeBinCell;

		typedef struct _structCache {
			int srcY;
			int srcX;
			int curY;
			int curX;
		} structCache;

		typedef struct _structReg {
			int x;
			int y;
			int num;
			float deg;
			vector<int> regPts_x;
			vector<int> regPts_y;
		} structReg;

		typedef struct _structRegionGrower {
			vector<unordered_map<int, int>> curMap;
			structReg reg;
		} structRegionGrower;

		typedef struct _structRectangleConverter {
			float x1;
			float y1;
			float x2;
			float y2;
			float wid;
			float cX;
			float cY;
			float deg;
			float dx;
			float dy;
			float p;
			float prec;
		} structRec;

		typedef struct _structCenterGetter {
			float cenX;
			float cenY;
		} structCenterGetter;

		typedef struct _structRefiner {
			bool boolean;
			vector<unordered_map<int, int>> curMap;
			structReg reg;
			structRec rec;
		} structRefiner;

		typedef struct _structRegionRadiusReducer {
			bool boolean;
			vector<unordered_map<int, int>> curMap;
			structReg reg;
			structRec rec;
		} structRegionRadiusReducer;

		typedef struct _structRectangleImprover {
			float logNFA;
			structRec rec;
		} structRectangleImprover;

		typedef struct _structRecVer {
			float verX[4] = { 0 };
			float verY[4] = { 0 };
		} structRecVer;
		
		const float pi = 4.0 * atan(1.0), pi1_5 = 1.5 * pi, pi2 = 2 * pi;
		int oriMapCol, oriMapRow, newMapCol, newMapRow, hSize;
		float sca, sig, pseBin, logNT, aliPro;
		float angThre, denThre, gradThre, regThre;
		const float coefA = 0.1066 * logNT + 2.6750;
		const float coefB = 0.004120 * logNT - 0.6223;
		const float coefC = -0.002607 * logNT + 0.1550;
		vector<unordered_map<int, int>> usedMap2, curMap2;
		vector<unordered_map<int, float>> degMap2, magMap2;

#ifdef drawPicture
		Mat lineIm, lineImColor;
#endif

		float FetchDegMapValue(const int y, const int x);
		float FetchMagMapValue(const int y, const int x);
		void GenerateLinesInfo(vector<structLinesInfo>& linesInfo, const int x1, const int y1, const int x2, const int y2);
		vector<float> GenerateGaussianKernel(unordered_map<float, vector<float>>& kernelCache, const float kerMean);
		Mat GaussianSampler(const Mat& image);
		structRegionGrower RegionGrower(const int x, const int y, float regDeg, const float degThre);
		structRec RectangleConverter(const structReg& reg, const float degThre);
		structCenterGetter CenterGetter(const int regNum, const vector<int>& regX, const vector<int>& regY);
		float OrientationGetter(const structReg& reg, const float cenX, const float cenY, const float degThre);
		structRefiner Refiner(structReg& reg, structRec& rec, vector<unordered_map<int, int>>& curMap);
		float RectangleNFACalculator(const structRec& rec);
		structRegionRadiusReducer RegionRadiusReducer(structReg& reg, structRec& rec, vector<unordered_map<int, int>>& curMap);
		structRectangleImprover RectangleImprover(structRec& rec);
	};
}

#endif // !_MYLSD_
