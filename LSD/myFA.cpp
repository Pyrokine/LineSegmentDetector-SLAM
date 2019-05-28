#include <myFA.h>

using namespace cv;
using namespace std;

namespace myfa {
	//多线程的任务总数和完成数量
	int num_tasks = 0;
	int num_done = 0;
	//pthread的互斥锁
	pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

	structScore FeatureAssociation(structFAInput *FAInput) {
		//输入：LSD和RDP算法得到的线段数据以及距离地图
		//输出：定位结果
		int sizeScanLine = (int)FAInput->scanLinesInfo.size();
		int sizeMapLine = (int)FAInput->mapLinesInfo.size();
		//初始化任务数量
		num_tasks = 0;
		num_done = 0;
		//初始化线程池
		threadpool_t *pool = threadpool_create(numTHREAD, lenQUEUE, 0);
		//初始化Score
		vector<structScore> Score;

		int cntScanLine = 0;
		//以RDP线段作为基准进行匹配
		for (cntScanLine = 0; cntScanLine < sizeScanLine; cntScanLine++) {
			double lenScanLine = FAInput->scanLinesInfo[cntScanLine].len;
			//忽略过短的线段
			if (lenScanLine < ignoreScanLength)
				continue;

			double lenDiff = FAInput->scanLinesInfo[cntScanLine].len * scanToMapDiff;
			int cntMapLine = 0;
			for (cntMapLine = 0; cntMapLine < sizeMapLine; cntMapLine++) {
				double lenMapLine = FAInput->mapLinesInfo[cntMapLine].len;
				//对长度差在一定范围内的线段进行匹配
				if (lenMapLine < lenScanLine - lenDiff || lenMapLine > lenScanLine + lenDiff)
					continue;

				//单线程运算
				//ScanToMapMatch(FAInput, cntMapLine, cntScanLine, &Score);

				//多线程并行运算
				//等待队伍空出
				while (num_tasks - num_done > lenQUEUE);
				//传参
				structThreadSTMM *argSTMM = (structThreadSTMM*)malloc(sizeof(structThreadSTMM));
				argSTMM->cntMapLine = cntMapLine;
				argSTMM->cntScanLine = cntScanLine;
				argSTMM->FAInput = FAInput;
				argSTMM->Score = &Score;
				//添加任务
				threadpool_add(pool, &thread_ScanToMapMatch, argSTMM, 0);
				pthread_mutex_lock(&mutex);
				num_tasks++;
				pthread_mutex_unlock(&mutex);
			}
		}
		//等待所有任务结束并销毁线程（已知BUG：有一个任务可能不结束）
		//可以增加等待时间以结束等待
		while (num_tasks - num_done > 1);
		threadpool_destroy(pool, 0);

		int lenScore = 0;
		structScore *poseAll;
		structScore poseBase;
		//判断是否有匹配结果，不存在则创建新的马尔科夫链
		if (Score.empty()) {
			
			return poseBase;
		}
		else {
			lenScore = (int)Score.size();
			poseAll = new structScore[lenScore * sizeof(structScore)];
			memcpy(poseAll, &Score[0], lenScore * sizeof(structScore));
		}
		//确定Scorre最低的结果为第一次匹配的基准点
		qsort(poseAll, lenScore, sizeof(structScore), CompScore);
		poseBase = poseAll[0];
		printf("Score:%f\n", poseBase.score);

		free(poseAll);
		return poseBase;
	}

	void thread_ScanToMapMatch(void *arg) {
		//输入：待匹配的两条线段的序号
		//输出：所有匹配结果的激光雷达位置和得分
		structThreadSTMM *argSTMM = (structThreadSTMM*) arg;

		int lenScore = 0;
		int i = 0;
		//四种匹配方式
		for (i = 1; i <= 4; i++) {
			structStaEnd mapStaEndPoint, scanStaEndPoint;
			if (i == 1) {
				mapStaEndPoint.staX = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].x1;
				mapStaEndPoint.staY = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].y1;
				mapStaEndPoint.endX = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].x2;
				mapStaEndPoint.endY = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].y2;
				scanStaEndPoint.staX = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].x1;
				scanStaEndPoint.staY = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].y1;
				scanStaEndPoint.endX = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].x2;
				scanStaEndPoint.endY = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].y2;
			}
			else if (i == 2) {
				mapStaEndPoint.staX = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].x1;
				mapStaEndPoint.staY = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].y1;
				mapStaEndPoint.endX = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].x2;
				mapStaEndPoint.endY = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].y2;
				scanStaEndPoint.staX = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].x2;
				scanStaEndPoint.staY = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].y2;
				scanStaEndPoint.endX = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].x1;
				scanStaEndPoint.endY = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].y1;
			}
			else if (i == 3) {
				mapStaEndPoint.staX = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].x2;
				mapStaEndPoint.staY = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].y2;
				mapStaEndPoint.endX = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].x1;
				mapStaEndPoint.endY = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].y1;
				scanStaEndPoint.staX = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].x1;
				scanStaEndPoint.staY = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].y1;
				scanStaEndPoint.endX = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].x2;
				scanStaEndPoint.endY = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].y2;
			}
			else {
				mapStaEndPoint.staX = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].x2;
				mapStaEndPoint.staY = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].y2;
				mapStaEndPoint.endX = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].x1;
				mapStaEndPoint.endY = argSTMM->FAInput->mapLinesInfo[argSTMM->cntMapLine].y1;
				scanStaEndPoint.staX = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].x2;
				scanStaEndPoint.staY = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].y2;
				scanStaEndPoint.endX = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].x1;
				scanStaEndPoint.endY = argSTMM->FAInput->scanLinesInfo[argSTMM->cntScanLine].y1;
			}

			//生成匹配基准点和基准线段角度
			structPosition mapPose, scanPose;
			mapPose.x = mapStaEndPoint.staX;
			mapPose.y = mapStaEndPoint.staY;
			mapPose.ang = NormalizedLineDirection(mapStaEndPoint);
			scanPose.x = scanStaEndPoint.staX;
			scanPose.y = scanStaEndPoint.staY;
			scanPose.ang = NormalizedLineDirection(scanStaEndPoint);

			structRotateScanIm RSI = rotateScanIm(argSTMM->FAInput, mapPose, scanPose);

			structScore tempScore;
			tempScore.pos = RSI.rotateLidarPos;
			tempScore.score = CalcScore(argSTMM->FAInput, RSI);
			free(RSI.rotateScanImPoint);

			//写入结果
			pthread_mutex_lock(&mutex);
			argSTMM->Score->push_back(tempScore);
			pthread_mutex_unlock(&mutex);
		}
		//增加完成数量并释放参数
		pthread_mutex_lock(&mutex);
		num_done++;
		pthread_mutex_unlock(&mutex);
		free(argSTMM);
	}

	void ScanToMapMatch(structFAInput *FAInput, int cntMapLine, int cntScanLine, vector<structScore> *Score) {
		//单线程特征匹配（已弃用）
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
		//angle为线段角度, 单位为y度，大小为[-180，180]
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
		//旋转RDP点云以计算Score
		//LSD线段和RDP线段的角度差
		double angDiff = mapPose.ang - scanPose.ang;

		int numScanImPoint = (int)FAInput->scanImPoint.size();
		int cnt;
		//将点云原点平移到待匹配的RDP线段的基准点
		structPosition *oriScanImPoint = (structPosition*)malloc(numScanImPoint * sizeof(structPosition));
		for (cnt = 0; cnt < numScanImPoint; cnt++) {
			oriScanImPoint[cnt].x = FAInput->scanImPoint[cnt].x - scanPose.x;
			oriScanImPoint[cnt].y = FAInput->scanImPoint[cnt].y - scanPose.y;
		}
		//按角度差旋转图像后将原点平移到LSD线段基准点
		structPosition *rotateScanImPoint = (structPosition*)malloc(numScanImPoint * sizeof(structPosition));
		for (cnt = 0; cnt < numScanImPoint; cnt++) {
			rotateScanImPoint[cnt].x = oriScanImPoint[cnt].x * cosd(angDiff) - oriScanImPoint[cnt].y * sind(angDiff) + mapPose.x;
			rotateScanImPoint[cnt].y = oriScanImPoint[cnt].x * sind(angDiff) + oriScanImPoint[cnt].y * cosd(angDiff) + mapPose.y;
		}
		//将激光雷达坐标做上述相同变换
		structPosition rotateLidarPos;
		rotateLidarPos.x = (FAInput->lidarPos[0] - scanPose.x) * cosd(angDiff) - (FAInput->lidarPos[1] - scanPose.y) * sind(angDiff) + mapPose.x;
		rotateLidarPos.y = (FAInput->lidarPos[0] - scanPose.x) * sind(angDiff) + (FAInput->lidarPos[1] - scanPose.y) * cosd(angDiff) + mapPose.y;
		rotateLidarPos.ang = scanPose.ang + angDiff;

		//将角度差控制在[-180,180]
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
		//输入：旋转后的RDP点云和距离图mapCache
		//输出：距离评分Score，越小越好
		//sumDist为距离和，Valid为在mapCache中小于参数z_occ_max_dis的像素
		double sumValidDist = 0, sumMaxDist = 0, numValidDistPoint = 0, numMaxDistPoint = 0;
		//All为所有像素，Valid为在地图内的像素
		double numAllPoint = 0, numValidPoint = 0;

		int cnt;
		for (cnt = 0; cnt < RSI.numScanImPoint; cnt++) {
			int y = (int)round(RSI.rotateScanImPoint[cnt].y);
			int x = (int)round(RSI.rotateScanImPoint[cnt].x);
			//printf("%d %d %d %d\n", FAInput->mapIm.size[0], FAInput->mapIm.size[1], y, x);
			//printf("%f %f\n", RSI.rotateScanImPoint[cnt].y, RSI.rotateScanImPoint[cnt].x);
			if (y >= 0 && y < FAInput->mapCache.size[0] && x >= 0 && x < FAInput->mapCache.size[1]) {
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
		//提高权重，惩罚Max像素
		sumMaxDist = 7 * numMaxDistPoint;
		double Score;
		if (numValidPoint == 0)
			Score = INFINITY;
		else
			Score = (sumValidDist + sumMaxDist) / (numValidPoint) + 10 * (numAllPoint - numValidPoint) / numValidPoint;
		//double Score = (sumValidDist + sumMaxDist) / numValidPoint;
		
		return Score;
	}

	int CompScore(const void *p1, const void *p2)
	{
		//从小到大排序
		return(*(structScore*)p2).score < (*(structScore*)p1).score ? 1 : -1;
	}

}