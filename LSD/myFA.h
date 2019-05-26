#pragma once
#ifndef _MYFA_
#define _MYFA_

#include <opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <baseFunc.h>

using namespace cv;
using namespace std;

namespace myfa {
	typedef struct _structFAInput {
		vector<structLinesInfo> scanLinesInfo;
		vector<structLinesInfo> mapLinesInfo;
		int lidarPos[2];
		Mat mapCache;
		Mat scanIm;
		Mat mapIm;
		//vector<double> ScanRanges;
		//vector<double> ScanAngles;
	} structFAInput;

	typedef struct _structScore {
		structPosition pos;
		double score;
		struct _structScore *next = NULL;
	} structScore;

	typedef struct _structSTMM {
		structScore *Score_head;
		structScore *Score_now;
	} structSTMM;

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

	structScore FeatureAssociation(structFAInput *FAInput);
	structSTMM ScanToMapMatch(structFAInput *FAInput, int cntMapLine, int cntScanLine);
	double NormalizedLineDirection(structStaEnd lineStaEnd);
	structRotateScanIm rotateScanIm(structFAInput *FAInput, structPosition mapPose, structPosition scanPose);
	double ScanToMapMatchScore(structFAInput *FAInput, structRotateScanIm RSI);
	int CompScore(const void *p1, const void *p2);
}


#endif