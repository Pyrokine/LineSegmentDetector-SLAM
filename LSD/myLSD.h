/////////////////////////////////////////////////////////////////////////
//@Copyright(C) Pyrokine
//All rights reserved
//博客 http://www.cnblogs.com/Pyrokine/
//Github https://github.com/Pyrokine
//创建日期 20190126
//版本 2.1
//**********************************************************************
//V1.0
//实现了LineSegmentDetector的基本算法， 并添加了基础的可视化演示
//
//V1.1
//完成了算法优化，将定长数组用动态数组实现，将伪排序用快速排序实现，
//优化了高斯降采样中计算高斯核的算法，改进后只需计算一次高斯核,但只适用于
//0.3缩放比的地图（其他比例需要重新计算偏移值），将区域增长次数从多次减少
//到1次
//
//V1.2
//完成了Degree动画，并分离出main_with_disp.cpp，修正了UsedMap动画，增加了
//梯度排名和区域内像素点显示，增加了虚警数的更新数据显示，提取出LSD算法为
//单独函数
//
//V2.0
//提取出LSD算法到单独文件，可以单独引用，增加命名空间mylsd
//
//V2.1
//计算出FeatureAssociation需要的mapCache，用于表示先验概率
/////////////////////////////////////////////////////////////////////////

#ifndef _MYLSD_
#define _MYLSD_

#include <opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <baseFunc.h>
using namespace cv;

namespace mylsd {
	class LSD {
	public:
		typedef struct _structLSD {
			Mat lineIm;
			structLinesInfo* linesInfo;
			int len_linesInfo;
		} structLSD;

		structLSD myLineSegmentDetector(Mat& MapGray, const int oriMapCol, const int oriMapRow, const double sca, const double sig,
			const double angThre, const double denThre, const int pseBin);
		Mat createMapCache(Mat MapGray, double res);
	private:
		typedef struct _nodeBinCell {
			int value;
			int x;
			int y;
		} nodeBinCell;

		typedef struct _structPts {
			int x;
			int y;
			struct _structPts* next;
		}structPts;

		typedef struct _structCache {
			int src_i;
			int src_j;
			int cur_i;
			int cur_j;
			struct _structCache* next;
		}structCache;

		typedef struct _structReg {
			int x;
			int y;
			int num;
			double deg;
			int* regPts_x;
			int* regPts_y;
			struct _structReg* next;
		}structReg;

		typedef struct _structRegionGrower {
			Mat curMap;
			structReg reg;
		}structRegionGrower;

		typedef struct _structRectangleConverter {
			double x1;
			double y1;
			double x2;
			double y2;
			double wid;
			double cX;
			double cY;
			double deg;
			double dx;
			double dy;
			double p;
			double prec;
			struct _structRectangleConverter* next;
		}structRec;

		typedef struct _structCenterGetter {
			double cenX;
			double cenY;
		}structCenterGetter;

		typedef struct _structRefiner {
			bool boolean;
			Mat curMap;
			structReg reg;
			structRec rec;
		}structRefiner;

		typedef struct _structRegionRadiusReducer {
			bool boolean;
			Mat curMap;
			structReg reg;
			structRec rec;
		} structRegionRadiusReducer;

		typedef struct _structRectangleImprover {
			double logNFA;
			structRec rec;
		} structRectangleImprover;

		typedef struct _structRecVer {
			double verX[4] = { 0 };
			double verY[4] = { 0 };
		} structRecVer;

		typedef struct _compVector {
			template <typename T>
			bool operator() (const T& a, const T& b) {
				return a.value > b.value;
			}
		} compVector;

		structRec* recSaveDisp;
		
		const double pi = 4.0 * atan(1.0), pi2 = 2 * pi;
		int oriMapCol, oriMapRow, newMapCol, newMapRow;
		double sca, sig, pseBin, logNT, aliPro;
		double angThre, denThre, gradThre, regThre;
		Mat usedMap, degMap, magMap;

		Mat GaussianSampler(Mat image, double sca, double sig);
		structRegionGrower RegionGrower(int x, int y, double regDeg, double degThre);
		structRec RectangleConverter(const structReg reg, const double degThre);
		structCenterGetter CenterGetter(const int regNum, const int* regX, const int* regY);
		double OrientationGetter(const structReg reg, const double cenX, const double cenY, const int* regX, const int* regY, const double degThre);
		structRefiner Refiner(structReg reg, structRec rec, Mat curMap);
		double RectangleNFACalculator(structRec rec);
		structRegionRadiusReducer RegionRadiusReducer(structReg reg, structRec rec, Mat curMap);
		structRectangleImprover RectangleImprover(structRec rec);
	};
}

#endif // !_MYLSD_
