#include <myFA.h>

using namespace cv;
using namespace std;

namespace myfa {
	//���̵߳������������������
	int num_tasks = 0;
	int num_done = 0;
	//pthread�Ļ�����
	pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

	structFAOutput FeatureAssociation(structFAInput *FAInput) {
		//���룺LSD��RDP�㷨�õ����߶������Լ������ͼ
		//�������λ���
		int sizeScanLine = (int)FAInput->scanLinesInfo.size();
		int sizeMapLine = (int)FAInput->mapLinesInfo.size();
		//��ʼ����������
		num_tasks = 0;
		num_done = 0;
		//��ʼ���̳߳�
		threadpool_t *pool = threadpool_create(numTHREAD, lenQUEUE, 0);
		//��ʼ��Score
		vector<structScore> Score;
		structFAOutput FAOutput;

		int cntScanLine = 0;
		//��RDP�߶���Ϊ��׼����ƥ��
		for (cntScanLine = 0; cntScanLine < sizeScanLine; cntScanLine++) {
			double lenScanLine = FAInput->scanLinesInfo[cntScanLine].len;
			//���Թ��̵��߶�
			if (lenScanLine < ignoreScanLength)
				continue;

			double lenDiff = FAInput->scanLinesInfo[cntScanLine].len * scanToMapDiff;
			int cntMapLine = 0;
			for (cntMapLine = 0; cntMapLine < sizeMapLine; cntMapLine++) {
				double lenMapLine = FAInput->mapLinesInfo[cntMapLine].len;
				//�Գ��Ȳ���һ����Χ�ڵ��߶ν���ƥ��
				if (lenMapLine < lenScanLine - lenDiff || lenMapLine > lenScanLine + lenDiff)
					continue;

				//���̲߳�������
				//�ȴ�����ճ�
				while (num_tasks - num_done > lenQUEUE);
				//����
				structThreadSTMM *argSTMM = (structThreadSTMM*)malloc(sizeof(structThreadSTMM));
				argSTMM->cntMapLine = cntMapLine;
				argSTMM->cntScanLine = cntScanLine;
				argSTMM->FAInput = FAInput;
				argSTMM->Score = &Score;
				argSTMM->lastPose = FAInput->lastPose;
				//�������
				threadpool_add(pool, &thread_ScanToMapMatch, argSTMM, 0);
				pthread_mutex_lock(&mutex);
				num_tasks++;
				pthread_mutex_unlock(&mutex);
			}
		}
		//�ȴ�������������������̣߳���֪BUG��������һ����������������
		//�������ӵȴ�ʱ���Խ����ȴ�
		while (num_tasks - num_done > 1);
		threadpool_destroy(pool, 0);

		int lenScore = 0;
		structScore *poseAll;
		structScore poseEstimate;
		//�ж��Ƿ���ƥ�������������򴴽��µ�������Ʒ���
		if (Score.empty()) {
			poseEstimate.pos.x = -1;
			poseEstimate.pos.y = -1;
			poseEstimate.pos.ang = 0;
			poseEstimate.score = INFINITY;
			Eigen::Matrix<double, 9, 1> kalman_x;
			Eigen::Matrix<double, 9, 9> kalman_P;
			kalman_x << -1, -1, 0, 0, 0, 0, 0, 0, 0;
			kalman_P << 100, 0, 0, 0, 0, 0, 0, 0, 0,
						0, 100, 0, 0, 0, 0, 0, 0, 0,
						0, 0, 100, 0, 0, 0, 0, 0, 0,
						0, 0, 0, 1, 0, 0, 0, 0, 0,
						0, 0, 0, 0, 1, 0, 0, 0, 0,
						0, 0, 0, 0, 0, 1, 0, 0, 0,
						0, 0, 0, 0, 0, 0, 0.1, 0, 0,
						0, 0, 0, 0, 0, 0, 0, 0.1, 0,
						0, 0, 0, 0, 0, 0, 0, 0, 0.1;
			FAOutput.kalman_x = kalman_x;
			FAOutput.kalman_P = kalman_P;
			return FAOutput;
		}
		else {
			lenScore = (int)Score.size();
			poseAll = new structScore[lenScore * sizeof(structScore)];
			memcpy(poseAll, &Score[0], lenScore * sizeof(structScore));
		}
		//ȷ��Scorre��͵Ľ��Ϊ��һ��ƥ��Ļ�׼��
		qsort(poseAll, lenScore, sizeof(structScore), CompScore);

		//����������Ʒ�����һ֡
		if (abs(FAInput->lastPose.x + 1) < 0.0001) {
			poseEstimate = poseAll[0];
			FAOutput.kalman_x = FAInput->kalman_x;
			FAOutput.kalman_P = FAInput->kalman_P;
			FAOutput.kalman_x(0) = poseEstimate.pos.x;
			FAOutput.kalman_x(1) = poseEstimate.pos.y;
			FAOutput.kalman_x(2) = poseEstimate.pos.ang;
			printf("Score:%lf\n", poseEstimate.score);
			return FAOutput;
		}

		//������Ʒ����м�֡
		double scaThre = 0;
		int cntPoseAll = 0, lenPoseAll = 0;


		//for (cntPoseAll = 0; cntPoseAll < lenScore; cntPoseAll++) {
		//	int cnt;
		//	Mat Display2 = FAInput->Display.clone();
		//	for (cnt = 0; cnt < FAInput->scanImPoint.size(); cnt++) {
		//		circle(Display2, Point((int)poseAll[cntPoseAll].rotateScanImPoint[cnt].x / 2, (int)poseAll[cntPoseAll].rotateScanImPoint[cnt].y / 2), 1, Scalar(255, 255, 255));
		//	}
		//	imshow("Display2", Display2);
		//	waitKey(0);
		//	free(poseAll[cntPoseAll].rotateScanImPoint);
		//	printf("%d : Score:%lf Ang:%lf\n", cntPoseAll, poseAll[cntPoseAll].score, poseAll[cntPoseAll].pos.ang);
		//}
		//vector<structScore> poseSimilar;

		//��ƥ�����������µ�������Ʒ���
		if (lenScore == 0) {
			poseEstimate.pos.x = -1;
			poseEstimate.pos.y = -1;
			poseEstimate.pos.ang = 0;
			poseEstimate.score = INFINITY;
			Eigen::Matrix<double, 9, 1> kalman_x;
			Eigen::Matrix<double, 9, 9> kalman_P;
			kalman_x << -1, -1, 0, 0, 0, 0, 0, 0, 0;
			kalman_P << 100, 0, 0, 0, 0, 0, 0, 0, 0,
						0, 100, 0, 0, 0, 0, 0, 0, 0,
						0, 0, 100, 0, 0, 0, 0, 0, 0,
						0, 0, 0, 1, 0, 0, 0, 0, 0,
						0, 0, 0, 0, 1, 0, 0, 0, 0,
						0, 0, 0, 0, 0, 1, 0, 0, 0,
						0, 0, 0, 0, 0, 0, 0.1, 0, 0,
						0, 0, 0, 0, 0, 0, 0, 0.1, 0,
						0, 0, 0, 0, 0, 0, 0, 0, 0.1;
			FAOutput.kalman_x = kalman_x;
			FAOutput.kalman_P = kalman_P;
			return FAOutput;
		}

		//Mat Display = Mat::zeros(FAInput->mapCache.size[0], FAInput->mapCache.size[1], CV_8UC1);
		//for (cntPoseAll = 0; cntPoseAll < lenScore; cntPoseAll++) {
		//	circle(Display, Point((int)poseAll[cntPoseAll].pos.x, (int)poseAll[cntPoseAll].pos.y), 1, Scalar(255, 255, 255));
		//}
		//imshow("Display", Display);
		//waitKey(0);

		//��ƥ������Ȩ���ֵ
		double sumX = 0, sumY = 0, sumAngle = 0, sumScore = 0;
		int cnt;
		for (cnt = 0; cnt < lenScore; cnt++) {
			double thisScore = 1 / pow(poseAll[cnt].score, 2);
			sumX += poseAll[cnt].pos.x * thisScore;
			sumY += poseAll[cnt].pos.y * thisScore;
			sumAngle += poseAll[cnt].pos.ang * thisScore;
			sumScore += thisScore;
		}
		poseEstimate.pos.x = sumX / sumScore;
		poseEstimate.pos.y = sumY / sumScore;
		poseEstimate.pos.ang = sumAngle / sumScore;
		poseEstimate.score = 1 / sqrt(sumScore / lenScore);

		printf("Score:%lf\n", poseEstimate.score);

		FAOutput = ukf(FAInput, poseEstimate);

		//poseEstimate.pos.x = FAOutput.kalman_x(0);
		//poseEstimate.pos.y = FAOutput.kalman_x(1);
		//poseEstimate.pos.ang = FAOutput.kalman_x(2);
		//FAOutput.poseEstimate = poseEstimate;

		free(poseAll);
		return FAOutput;
	}

	void thread_ScanToMapMatch(void *arg) {
		//���룺��ƥ��������߶ε����
		//���������ƥ�����ļ����״�λ�ú͵÷�
		structThreadSTMM *argSTMM = (structThreadSTMM*) arg;

		int lenScore = 0;
		int i = 0;
		//����ƥ�䷽ʽ
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

			//����ƥ���׼��ͻ�׼�߶νǶ�
			structPosition mapPose, scanPose;
			mapPose.x = mapStaEndPoint.staX;
			mapPose.y = mapStaEndPoint.staY;
			mapPose.ang = NormalizedLineDirection(mapStaEndPoint);
			scanPose.x = scanStaEndPoint.staX;
			scanPose.y = scanStaEndPoint.staY;
			scanPose.ang = NormalizedLineDirection(scanStaEndPoint);

			structRotateScanIm RSI = rotateScanIm(argSTMM->FAInput, mapPose, scanPose, argSTMM->lastPose);

			structScore tempScore;
			if (RSI.numScanImPoint != 0) {
				tempScore.pos = RSI.rotateLidarPos;
				tempScore.score = CalcScore(argSTMM->FAInput, RSI);
				tempScore.rotateScanImPoint = RSI.rotateScanImPoint;

				free(RSI.rotateScanImPoint);
			}
			else {
				tempScore.score = INFINITY;
			}

			//д����
			if (tempScore.score < 3) {
				pthread_mutex_lock(&mutex);
				argSTMM->Score->push_back(tempScore);
				pthread_mutex_unlock(&mutex);
			}
		}
		//��������������ͷŲ���
		pthread_mutex_lock(&mutex);
		num_done++;
		pthread_mutex_unlock(&mutex);
		free(argSTMM);
	}

	double NormalizedLineDirection(structStaEnd lineStaEnd) {
		//����б�ʣ� ��һ���߶η���, �㣨x1, y1�� Ϊ��ʼ��
		//angleΪ�߶νǶ�, ��λΪy�ȣ���СΪ[-180��180]
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
		else {
			angle = atand((lineStaEnd.endY - lineStaEnd.staY) / (lineStaEnd.endX - lineStaEnd.staX));
		}

		if (angle < 0 && lineStaEnd.staX > lineStaEnd.endX)
		{
			angle += 180;
			return angle;
		}
		if (angle > 0 && lineStaEnd.staX > lineStaEnd.endX) {
			angle -= 180;
			return angle;
		}
		
		return angle;
	}

	structRotateScanIm rotateScanIm(structFAInput *FAInput, structPosition mapPose, structPosition scanPose, structPosition lastPose) {
		//��תRDP�����Լ���Score
		//LSD�߶κ�RDP�߶εĽǶȲ�
		double angDiff = mapPose.ang - scanPose.ang;
		//printf("mapAng:%lf scanAng:%lf\n", mapPose.ang, scanPose.ang);

		int numScanImPoint = (int)FAInput->scanImPoint.size();
		int cnt;
		//������ԭ��ƽ�Ƶ���ƥ���RDP�߶εĻ�׼��
		structPosition *oriScanImPoint = (structPosition*)malloc(numScanImPoint * sizeof(structPosition));
		for (cnt = 0; cnt < numScanImPoint; cnt++) {
			oriScanImPoint[cnt].x = FAInput->scanImPoint[cnt].x - scanPose.x;
			oriScanImPoint[cnt].y = FAInput->scanImPoint[cnt].y - scanPose.y;
		}

		//�������״�������������ͬ�任
		structPosition rotateLidarPose;
		rotateLidarPose.x = (FAInput->lidarPose.x - scanPose.x) * cosd(angDiff) - (FAInput->lidarPose.y - scanPose.y) * sind(angDiff) + mapPose.x;
		rotateLidarPose.y = (FAInput->lidarPose.x - scanPose.x) * sind(angDiff) + (FAInput->lidarPose.y - scanPose.y) * cosd(angDiff) + mapPose.y;
		rotateLidarPose.ang = 0;

		structRotateScanIm RSI;
		//ȡmaxEstiDist�е����أ���Ϊ������Ʒ����ĵ�һ֡��������ͬ����
		if (sqrt(pow(rotateLidarPose.x - lastPose.x, 2) + pow(rotateLidarPose.y - lastPose.y, 2)) < maxEstiDist || lastPose.x == -1) {
			//���ǶȲ���תͼ���ԭ��ƽ�Ƶ�LSD�߶λ�׼��
			structPosition *rotateScanImPoint = (structPosition*)malloc(numScanImPoint * sizeof(structPosition));
			for (cnt = 0; cnt < numScanImPoint; cnt++) {
				rotateScanImPoint[cnt].x = oriScanImPoint[cnt].x * cosd(angDiff) - oriScanImPoint[cnt].y * sind(angDiff) + mapPose.x;
				rotateScanImPoint[cnt].y = oriScanImPoint[cnt].x * sind(angDiff) + oriScanImPoint[cnt].y * cosd(angDiff) + mapPose.y;
			}

			//���ǶȲ������[-180,180]
			while (angDiff <= -180)
				angDiff += 360;
			while (angDiff > 180)
				angDiff -= 360;
			rotateLidarPose.ang = angDiff;
			
			RSI.rotateScanImPoint = rotateScanImPoint;
			RSI.numScanImPoint = numScanImPoint;
			RSI.angDiff = angDiff;
			RSI.rotateLidarPos = rotateLidarPose;
		}
		else {
			RSI.numScanImPoint = 0;
		}
		free(oriScanImPoint);
		return RSI;
	}

	double CalcScore(structFAInput *FAInput, structRotateScanIm RSI) {
		//���룺��ת���RDP���ƺ;���ͼmapCache
		//�������������Score��ԽСԽ��
		//sumDistΪ����ͣ�ValidΪ��mapCache��С�ڲ���z_occ_max_dis������
		double sumValidDist = 0, sumMaxDist = 0, numValidDistPoint = 0, numMaxDistPoint = 0;
		//AllΪ�������أ�ValidΪ�ڵ�ͼ�ڵ�����
		double numAllPoint = 0, numValidPoint = 0;

		int cnt;
		for (cnt = 0; cnt < RSI.numScanImPoint; cnt++) {
			int x = round(RSI.rotateScanImPoint[cnt].x);
			int y = round(RSI.rotateScanImPoint[cnt].y);
			//printf("%d %d %d %d\n", FAInput->mapIm.size[0], FAInput->mapIm.size[1], y, x);
			//printf("%f %f\n", RSI.rotateScanImPoint[cnt].y, RSI.rotateScanImPoint[cnt].x);
			if (y >= 0 && y < FAInput->mapCache.size[0] && x >= 0 && x < FAInput->mapCache.size[1]) {
				numValidPoint += 1;
				if (FAInput->mapCache.ptr<float>(y)[x] >= z_occ_max_dis)
				{
					//���Ȩ�أ��ͷ�Max����
					sumMaxDist += 10;
					numMaxDistPoint += 1;
				}
				else {
					sumValidDist += FAInput->mapCache.ptr<float>(y)[x];
					numValidDistPoint += 1;
				}
			}
		}

		numAllPoint = RSI.numScanImPoint;

		double Score;
		if (numValidPoint < 0.7 * numAllPoint)
			Score = INFINITY;
		else
			Score = (sumValidDist + sumMaxDist) / (numValidPoint) + 10 * (numAllPoint - numValidPoint) / numAllPoint;
		//double Score = (sumValidDist + sumMaxDist) / numValidPoint;
		
		return Score;
	}

	int CompScore(const void *p1, const void *p2)
	{
		//��С��������
		return(*(structScore*)p2).score < (*(structScore*)p1).score ? 1 : -1;
	}

	structFAOutput ukf(structFAInput *FAInput, structScore poseEstimate) {
		Eigen::Matrix<double, 9, 9> kalman_Q;
		Eigen::Matrix<double, 3, 3> kalman_R;
		kalman_Q << 1, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 1, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 1, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0.01, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0.01, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0.01, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0.0001, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0.0001, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0.0001;// ��������Э�������
		kalman_R << 1, 0, 0,
					0, 1, 0,
					0, 0, 1;// ��������Э�������
		double kalman_t = 1; //���õ���
		
		Eigen::Matrix<double, 9, 1> kalman_x;
		Eigen::Matrix<double, 9, 9> kalman_P;
		kalman_x = FAInput->kalman_x;
		kalman_P = FAInput->kalman_P;
		kalman_x(0) += FAInput->ScanPose.x;
		kalman_x(1) += FAInput->ScanPose.y;
		kalman_x(2) += FAInput->ScanPose.ang;

		int L = 9; //kalman_x��ά��
		int m = 3; //poseEstimate.pos��ά��
		double alpha = 1e-2; //Ĭ��ϵ��
		double ki = 0; //Ĭ��ϵ��
		double beta = 2; //Ĭ��ϵ��
		double lambda = alpha * alpha * (L + ki) - L;
		double c = L + lambda;
		Eigen::Matrix<double, 1, 19> Wm;
		Eigen::Matrix<double, 1, 19> Wc;
		Wm(0) = lambda / c;
		Wc(0) = lambda / c;
		int cnt;
		for (cnt = 1; cnt < 2 * L + 1; cnt++) {
			Wm(cnt) = 0.5 / c;
			Wc(cnt) = 0.5 / c;
		}
		Wc(0) += 1 - alpha * alpha + beta;
		c = sqrt(c);

		// ��һ������ȡһ��sigma�㼯
		// sigma�㼯����״̬X�����ĵ㼯��X��6 * 13����ÿ��Ϊ1����
		Eigen::Matrix<double, 9, 9> A;
		Eigen::Matrix<double, 9, 9> A2;
		Eigen::Matrix<double, 9, 9> Y;
		Eigen::Matrix<double, 9, 19> Xset;
		//Eigen::LLT<Eigen::MatrixXd> lltOfA(kalman_P);
		//A = lltOfA.matrixL();
		A2 = kalman_P.llt().matrixL();
		A = c * A2.transpose();
		Y << kalman_x, kalman_x, kalman_x, kalman_x, kalman_x,\
			kalman_x, kalman_x, kalman_x, kalman_x;
		Xset << kalman_x, Y + A, Y - A;
		// �ڶ����������Ĳ�����sigma�㼯����һ��Ԥ�⣬�õ���ֵXImeans�ͷ���P1����sigma�㼯X1
		// ��״̬UT�任
		int LL = 2 * L + 1; //19
		Eigen::Matrix<double, 9, 1> Xmeans = Eigen::Matrix<double, 9, 1>::Zero();
		Eigen::Matrix<double, 9, 19> Xsigma_pre;
		Eigen::Matrix<double, 9, 19> Xdiv;
		for (cnt = 0; cnt < LL; cnt++) {
			Xsigma_pre(0, cnt) = Xset(0, cnt) + kalman_t * Xset(3, cnt) + 0.5 * kalman_t * kalman_t * Xset(6, cnt);
			Xsigma_pre(1, cnt) = Xset(1, cnt) + kalman_t * Xset(4, cnt) + 0.5 * kalman_t * kalman_t * Xset(7, cnt);
			Xsigma_pre(2, cnt) = Xset(2, cnt) + kalman_t * Xset(5, cnt) + 0.5 * kalman_t * kalman_t * Xset(8, cnt);
			Xsigma_pre(3, cnt) = Xset(3, cnt) + kalman_t * Xset(6, cnt);
			Xsigma_pre(4, cnt) = Xset(4, cnt) + kalman_t * Xset(7, cnt);
			Xsigma_pre(5, cnt) = Xset(5, cnt) + kalman_t * Xset(8, cnt);
			Xsigma_pre(6, cnt) = Xset(6, cnt);
			Xsigma_pre(7, cnt) = Xset(7, cnt);
			Xsigma_pre(8, cnt) = Xset(8, cnt);

			Xmeans(0) += Wm(cnt) * Xsigma_pre(0, cnt);
			Xmeans(1) += Wm(cnt) * Xsigma_pre(1, cnt);
			Xmeans(2) += Wm(cnt) * Xsigma_pre(2, cnt);
			Xmeans(3) += Wm(cnt) * Xsigma_pre(3, cnt);
			Xmeans(4) += Wm(cnt) * Xsigma_pre(4, cnt);
			Xmeans(5) += Wm(cnt) * Xsigma_pre(5, cnt);
			Xmeans(6) += Wm(cnt) * Xsigma_pre(6, cnt);
			Xmeans(7) += Wm(cnt) * Xsigma_pre(7, cnt);
			Xmeans(8) += Wm(cnt) * Xsigma_pre(8, cnt);
		}
		int cnt2;
		for (cnt = 0; cnt < LL; cnt++) {
			for (cnt2 = 0; cnt2 < L; cnt2++) {
				Xdiv(cnt2, cnt) = Xsigma_pre(cnt2, cnt) - Xmeans(cnt2);
			}
		}
		Eigen::Matrix<double, 9, 9> P1;
		P1 = Xdiv * Wc.asDiagonal() * Xdiv.transpose() + kalman_Q;

		// ���塢�������õ��۲�Ԥ�⣬Z1ΪX1���ϵ�Ԥ�⣬ZpreΪZ1�ľ�ֵ��
		// PzzΪЭ����
		Eigen::Matrix<double, 3, 1> Zmeans = Eigen::Matrix<double, 3, 1>::Zero();
		Eigen::Matrix<double, 3, 19> Zsigma_pre;
		Eigen::Matrix<double, 3, 19> Zdiv;
		for (cnt = 0; cnt < LL; cnt++) {
			Zsigma_pre(0, cnt) = Xsigma_pre(0, cnt);
			Zsigma_pre(1, cnt) = Xsigma_pre(1, cnt);
			Zsigma_pre(2, cnt) = Xsigma_pre(2, cnt);

			Zmeans(0) += Wm(cnt) * Zsigma_pre(0, cnt);
			Zmeans(1) += Wm(cnt) * Zsigma_pre(1, cnt);
			Zmeans(2) += Wm(cnt) * Zsigma_pre(2, cnt);
		}
		cnt2;
		for (cnt = 0; cnt < LL; cnt++) {
			for (cnt2 = 0; cnt2 < 3; cnt2++) {
				Zdiv(cnt2, cnt) = Xsigma_pre(cnt2, cnt) - Xmeans(cnt2);
			}
		}
		Eigen::Matrix<double, 3, 3> Pzz;
		Pzz = Zdiv * Wc.asDiagonal() * Zdiv.transpose() + kalman_R;

		// ���߲������㿨��������
		Eigen::Matrix<double, 9, 3> Pxz;
		Eigen::Matrix<double, 9, 3> K;
		Pxz = Xdiv * Wc.asDiagonal() * Zdiv.transpose();
		K = Pxz * Pzz.inverse();
		Eigen::Matrix<double, 3, 1> Zdiff;
		Zdiff(0) = poseEstimate.pos.x - Zmeans(0);
		Zdiff(1) = poseEstimate.pos.y - Zmeans(1);
		Zdiff(2) = poseEstimate.pos.ang - Zmeans(2);
		kalman_x = Xmeans + K * Zdiff;
		kalman_P = P1 - K * Pxz.transpose();
		structFAOutput FAOutput;
		FAOutput.kalman_x = kalman_x;
		FAOutput.kalman_P = kalman_P;

		return FAOutput;
	}
}