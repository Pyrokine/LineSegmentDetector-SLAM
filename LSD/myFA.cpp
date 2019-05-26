#include <myFA.h>

using namespace cv;
using namespace std;

namespace myfa {
	structScore FeatureAssociation(structFAInput *FAInput) {
		int sizeScanLine = (int)FAInput->scanLinesInfo.size();
		int sizeMapLine = (int)FAInput->mapLinesInfo.size();
		vector<structScore> Score;

		int cntScanLine = 0;
		for (cntScanLine = 0; cntScanLine < sizeScanLine; cntScanLine++) {
			double lenScanLine = FAInput->scanLinesInfo[cntScanLine].len;
			if (lenScanLine < 40)
				continue;

			double lenDiff = FAInput->scanLinesInfo[cntScanLine].len * 0.35;
			int cntMapLine = 0;
			for (cntMapLine = 0; cntMapLine < sizeMapLine; cntMapLine++) {
				double lenMapLine = FAInput->mapLinesInfo[cntMapLine].len;
				if (lenMapLine < lenScanLine - lenDiff || lenMapLine > lenScanLine + lenDiff)
					continue;

				ScanToMapMatch(FAInput, cntMapLine, cntScanLine, &Score);
			}
		}

		int lenScore = (int)Score.size();
		structScore *poseAll = (structScore*)malloc(lenScore * sizeof(structScore));
		int cnt;
		for (cnt = 0; cnt < lenScore; cnt++) {
			poseAll[cnt].pos = Score[cnt].pos;
			poseAll[cnt].score = Score[cnt].score;
		}
		qsort(poseAll, lenScore, sizeof(structScore), CompScore);
		structScore poseBase = poseAll[0];
		printf("Score:%f\n", poseBase.score);

		free(poseAll);
		return poseBase;
	}

	void ScanToMapMatch(structFAInput *FAInput, int cntMapLine, int cntScanLine, vector<structScore> *Score) {
		int lenScore = 0;
		int i = 0;
		for (i = 1; i <= 4; i++) {
			structStaEnd mapStaEndPoint, scanStaEndPoint;
			if (i == 1) {
				mapStaEndPoint.staX = FAInput->mapLinesInfo[cntMapLine].x1;
				mapStaEndPoint.staY = FAInput->mapLinesInfo[cntMapLine].y1;
				mapStaEndPoint.endX = FAInput->mapLinesInfo[cntMapLine].x2;
				mapStaEndPoint.endY = FAInput->mapLinesInfo[cntMapLine].y2;
				scanStaEndPoint.staX = FAInput->scanLinesInfo[cntScanLine].x1;
				scanStaEndPoint.staY = FAInput->scanLinesInfo[cntScanLine].y1;
				scanStaEndPoint.endX = FAInput->scanLinesInfo[cntScanLine].x2;
				scanStaEndPoint.endY = FAInput->scanLinesInfo[cntScanLine].y2;
			}
			else if (i == 2) {
				mapStaEndPoint.staX = FAInput->mapLinesInfo[cntMapLine].x1;
				mapStaEndPoint.staY = FAInput->mapLinesInfo[cntMapLine].y1;
				mapStaEndPoint.endX = FAInput->mapLinesInfo[cntMapLine].x2;
				mapStaEndPoint.endY = FAInput->mapLinesInfo[cntMapLine].y2;
				scanStaEndPoint.staX = FAInput->scanLinesInfo[cntScanLine].x2;
				scanStaEndPoint.staY = FAInput->scanLinesInfo[cntScanLine].y2;
				scanStaEndPoint.endX = FAInput->scanLinesInfo[cntScanLine].x1;
				scanStaEndPoint.endY = FAInput->scanLinesInfo[cntScanLine].y1;
			}
			else if (i == 3) {
				mapStaEndPoint.staX = FAInput->mapLinesInfo[cntMapLine].x2;
				mapStaEndPoint.staY = FAInput->mapLinesInfo[cntMapLine].y2;
				mapStaEndPoint.endX = FAInput->mapLinesInfo[cntMapLine].x1;
				mapStaEndPoint.endY = FAInput->mapLinesInfo[cntMapLine].y1;
				scanStaEndPoint.staX = FAInput->scanLinesInfo[cntScanLine].x1;
				scanStaEndPoint.staY = FAInput->scanLinesInfo[cntScanLine].y1;
				scanStaEndPoint.endX = FAInput->scanLinesInfo[cntScanLine].x2;
				scanStaEndPoint.endY = FAInput->scanLinesInfo[cntScanLine].y2;
			}
			else {
				mapStaEndPoint.staX = FAInput->mapLinesInfo[cntMapLine].x2;
				mapStaEndPoint.staY = FAInput->mapLinesInfo[cntMapLine].y2;
				mapStaEndPoint.endX = FAInput->mapLinesInfo[cntMapLine].x1;
				mapStaEndPoint.endY = FAInput->mapLinesInfo[cntMapLine].y1;
				scanStaEndPoint.staX = FAInput->scanLinesInfo[cntScanLine].x2;
				scanStaEndPoint.staY = FAInput->scanLinesInfo[cntScanLine].y2;
				scanStaEndPoint.endX = FAInput->scanLinesInfo[cntScanLine].x1;
				scanStaEndPoint.endY = FAInput->scanLinesInfo[cntScanLine].y1;
			}

			structPosition mapPose, scanPose;
			mapPose.x = mapStaEndPoint.staX;
			mapPose.y = mapStaEndPoint.staY;
			mapPose.ang = NormalizedLineDirection(mapStaEndPoint);
			scanPose.x = scanStaEndPoint.staX;
			scanPose.y = scanStaEndPoint.staY;
			scanPose.ang = NormalizedLineDirection(scanStaEndPoint);

			structRotateScanIm RSI = rotateScanIm(FAInput, mapPose, scanPose);
			
			structScore tempScore;
			tempScore.pos = RSI.rotateLidarPos;
			tempScore.score = CalcScore(FAInput, RSI);
			free(RSI.rotateScanImPoint);
			Score->push_back(tempScore);
		}
	}

	double NormalizedLineDirection(structStaEnd lineStaEnd) {
		//计算斜率， 归一化线段方向, 点（x1, y1） 为起始点
		//k ：为斜率，ang为线段角度, 单位为 度，大小为（ - 180，180]
		double angle;
		if (lineStaEnd.staX == lineStaEnd.endX && lineStaEnd.staY != lineStaEnd.endY) {
			if (lineStaEnd.staY < lineStaEnd.endY)
				angle = 90;
			else
				angle = -90;
		}
		else if (lineStaEnd.staX != lineStaEnd.endX && lineStaEnd.staY == lineStaEnd.endY) {
			if (lineStaEnd.staX < lineStaEnd.endX)
				angle = 0;
			else
				angle = 180;
		}
		else
			angle = (lineStaEnd.endY - lineStaEnd.staY) / (lineStaEnd.endX - lineStaEnd.staX);

		if (angle < 0 && lineStaEnd.staX > lineStaEnd.endX)
			angle += 180;

		if (angle > 0 && lineStaEnd.staX > lineStaEnd.endX)
			angle -= 180;

		return angle;
	}

	structRotateScanIm rotateScanIm(structFAInput *FAInput, structPosition mapPose, structPosition scanPose) {
		double angDiff = mapPose.ang - scanPose.ang;

		int numScanImPoint = (int)FAInput->scanImPoint.size();
		int cnt;
		structPosition *oriScanImPoint = (structPosition*)malloc(numScanImPoint * sizeof(structPosition));
		for (cnt = 0; cnt < numScanImPoint; cnt++) {
			oriScanImPoint[cnt].x = FAInput->scanImPoint[cnt].x - scanPose.x;
			oriScanImPoint[cnt].y = FAInput->scanImPoint[cnt].y - scanPose.y;
		}
		
		structPosition *rotateScanImPoint = (structPosition*)malloc(numScanImPoint * sizeof(structPosition));
		for (cnt = 0; cnt < numScanImPoint; cnt++) {
			rotateScanImPoint[cnt].x = oriScanImPoint[cnt].x * cosd(angDiff) - oriScanImPoint[cnt].y * sind(angDiff) + mapPose.x;
			rotateScanImPoint[cnt].y = oriScanImPoint[cnt].x * sind(angDiff) + oriScanImPoint[cnt].y * cosd(angDiff) + mapPose.y;
		}
		structPosition rotateLidarPos;
		rotateLidarPos.x = (FAInput->lidarPos[0] - scanPose.x) * cosd(angDiff) - (FAInput->lidarPos[1] - scanPose.y) * sind(angDiff) + mapPose.x;
		rotateLidarPos.y = (FAInput->lidarPos[0] - scanPose.x) * sind(angDiff) + (FAInput->lidarPos[1] - scanPose.y) * cosd(angDiff) + mapPose.y;
		rotateLidarPos.ang = scanPose.ang + angDiff;

		while (angDiff <= -180)
			angDiff += 360;
		while (angDiff > 180)
			angDiff -= 360;

		structRotateScanIm RSI;
		RSI.rotateScanImPoint = rotateScanImPoint;
		RSI.numScanImPoint = numScanImPoint;
		RSI.angDiff = angDiff;
		RSI.rotateLidarPos = rotateLidarPos;

		free(oriScanImPoint);
		return RSI;
	}

	double CalcScore(structFAInput *FAInput, structRotateScanIm RSI) {
		double sumValidDist = 0, sumMaxDist = 0, numValidDistPoint = 0, numMaxDistPoint = 0;
		double numAllPoint = 0, numValidPoint = 0;

		int cnt;
		for (cnt = 0; cnt < RSI.numScanImPoint; cnt++) {
			int y = (int)round(RSI.rotateScanImPoint[cnt].y);
			int x = (int)round(RSI.rotateScanImPoint[cnt].x);
			//printf("%d %d %d %d\n", FAInput->mapIm.size[0], FAInput->mapIm.size[1], y, x);
			//printf("%f %f\n", RSI.rotateScanImPoint[cnt].y, RSI.rotateScanImPoint[cnt].x);
			if (y >= 0 && y < FAInput->mapIm.size[0] && x >= 0 && x < FAInput->mapIm.size[1]) {
				numValidPoint += 1;
				if (FAInput->mapCache.ptr<double>(y)[x] >= z_occ_max_dis)
					numMaxDistPoint += 1;
				else {
					sumValidDist += FAInput->mapCache.ptr<double>(y)[x];
					numValidDistPoint += 1;
				}
			}
		}
		numAllPoint = RSI.numScanImPoint;
		sumMaxDist = 7 * numMaxDistPoint;
		//double Score = (sumValidDist + sumMaxDist) / (numValidPoint) + 10 * (numAllPoint - numValidPoint) / numValidPoint;
		double Score = (sumValidDist + sumMaxDist) / numValidPoint;
		if (numValidPoint == 0)
			Score = INFINITY;
		return Score;
	}

	int CompScore(const void *p1, const void *p2)
	{
		return(*(structScore*)p2).score < (*(structScore*)p1).score ? 1 : -1;
	}
}