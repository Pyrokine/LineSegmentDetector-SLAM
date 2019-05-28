/////////////////////////////////////////////////////////////////////////
//@Copyright(C) Pyrokine
//All rights reserved
//博客 http://www.cnblogs.com/Pyrokine/
//Github https://github.com/Pyrokine
//创建日期 20190528
//版本 1.2
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
//
/////////////////////////////////////////////////////////////////////////
#pragma comment(lib,"pthreadVC2.lib")  
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
		int lidarPos[2];
		Mat mapCache;
	} structFAInput;

	typedef struct _structScore {
		structPosition pos;
		double score;
		struct _structScore *next = NULL;
	} structScore;

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
		vector<structScore> *Score;
	} structThreadSTMM;

	structScore FeatureAssociation(structFAInput *FAInput);
	void thread_ScanToMapMatch(void *arg);
	void ScanToMapMatch(structFAInput *FAInput, int cntMapLine, int cntScanLine, vector<structScore> *Score);
	double NormalizedLineDirection(structStaEnd lineStaEnd);
	structRotateScanIm rotateScanIm(structFAInput *FAInput, structPosition mapPose, structPosition scanPose);
	double CalcScore(structFAInput *FAInput, structRotateScanIm RSI);
	int CompScore(const void *p1, const void *p2);
}


#endif