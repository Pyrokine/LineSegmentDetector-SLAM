/////////////////////////////////////////////////////////////////////////
//@Copyright(C) Pyrokine
//All rights reserved
//博客 http://www.cnblogs.com/Pyrokine/
//Github https://github.com/Pyrokine
//创建日期 20190528
//版本 1.3
//**********************************************************************
//V1.0
//实现了FetureAssociation的基本算法
//
//V1.1
//增加注释量，将RDP点云的提取修改到RDP算法中提取，简化计算量
//
//V1.2
//对计算Score过程增加了基于pthread的线程池，引用自
//https://github.com/mbrossard/threadpool
//
//V1.3
//增加了基于Eigen的UKF滤波，融合了里程计信息
/////////////////////////////////////////////////////////////////////////
#ifndef _MYFA_
#define _MYFA_

#include <opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <threadpool.h>
#include <baseFunc.h>

using namespace cv;
using namespace std;

namespace myfa {
	typedef struct _structFAInput {
		vector<structLinesInfo> scanLinesInfo;
		vector<structLinesInfo> mapLinesInfo;
		vector<structPosition> scanImPoint;
		Mat mapCache;
		Mat Display;
		structPosition lidarPose;
		structPosition lastPose;
		structPosition ScanPose;
		Eigen::Matrix<double, 9, 1> kalman_x;
		Eigen::Matrix<double, 9, 9> kalman_P;
	} structFAInput;

	typedef struct _structScore {
		structPosition pos;
		structPosition* rotateScanImPoint;
		double score;
		//struct _structScore *next = NULL;
	} structScore;

	typedef struct _structFAOutput {
		Eigen::Matrix<double, 9, 1> kalman_x;
		Eigen::Matrix<double, 9, 9> kalman_P;
	} structFAOutput;

	typedef struct _structStaEnd {
		double staX;
		double staY;
		double endX;
		double endY;
	} structStaEnd;

	typedef struct _structRotateScanIm {
		structPosition rotateLidarPos;
		structPosition *rotateScanImPoint;
		int numScanImPoint;
		double angDiff;
	} structRotateScanIm;

	typedef struct _structThreadSTMM {
		structFAInput *FAInput;
		int cntMapLine;
		int cntScanLine;
		structPosition lastPose;
		vector<structScore> *Score;
	} structThreadSTMM;

	structFAOutput FeatureAssociation(structFAInput *FAInput);
	void thread_ScanToMapMatch(void *arg);
	double NormalizedLineDirection(structStaEnd lineStaEnd);
	structRotateScanIm rotateScanIm(structFAInput *FAInput, structPosition mapPose, structPosition scanPose, structPosition lastPose);
	double CalcScore(structFAInput *FAInput, structRotateScanIm RSI);
	int CompScore(const void *p1, const void *p2);
	structFAOutput ukf(structFAInput *FAInput, structScore poseEstimate);
}


#endif