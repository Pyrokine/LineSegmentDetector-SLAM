#include <myLSD.h>

using namespace cv;
using namespace std;

namespace mylsd {
	// LineSegmentDetector
	LSD::structLSD LSD::myLineSegmentDetector(Mat& MapGray, const int _oriMapCol, const int _oriMapRow, const double _sca, const double _sig,
			const double _angThre, const double _denThre, const int _pseBin) {
		oriMapCol = _oriMapCol, oriMapRow = _oriMapRow;
		sca = _sca, sig = _sig, angThre = _angThre, denThre = _denThre, pseBin = _pseBin;

		double last_time, last_time2 = clock();
		newMapCol = floor(oriMapCol * sca);
		newMapRow = floor(oriMapRow * sca);
		// ��ʽ����ͼ
		for (int y = 0; y < oriMapRow; y++) {
			for (int x = 0; x < oriMapCol; x++) {
				if (MapGray.ptr<uint8_t>(y)[x] == 1)
					MapGray.ptr<uint8_t>(y)[x] = 255;
				else if (MapGray.ptr<uint8_t>(y)[x] == 255)
					MapGray.ptr<uint8_t>(y)[x] = 0;
			}
		}

#ifdef drawPicture
		imshow("MapGray", MapGray);
		//waitKey(0);
#endif

		// ͼ�����š�����˹������
		last_time = clock();
		//Mat GaussImage = GaussianSampler(MapGray, sca, sig); // ������ص���double�ͣ�ͬʱ���������趨���ű���
		Mat GaussImage;
		pyrDown(MapGray, GaussImage, Size(oriMapCol / 2.0, oriMapRow / 2.0)); // ͼ�������
		printf("0 %lf\n", (clock() - last_time) / CLOCKS_PER_SEC);

#ifdef drawPicture
		imshow("GaussImage", GaussImage);
		//waitKey(0);
#endif

		usedMap = Mat::zeros(newMapRow, newMapCol, CV_8UC1);//��¼���ص�״̬
		degMap = Mat::zeros(newMapRow, newMapCol, CV_64FC1);//level-line������
		magMap = Mat::zeros(newMapRow, newMapCol, CV_64FC1);//��¼ÿ����ݶ�
		double degThre = angThre / 180.0 * pi; // �Ƕ���ֵ
		gradThre = 2.0 / sin(degThre); // �ݶ���ֵ

		// �����ݶȺ�level-line�����򲢴����ݶȵ�����
		last_time = clock();
		vector<nodeBinCell> binCell;
		double maxGrad = 0, gradX, gradY, valueMagnitude, valueDegree, A, B, C, D;;
		for (int y = 1; y < newMapRow; y++) {
			for (int x = 1; x < newMapCol; x++) {
				A = GaussImage.ptr<uint8_t>(y)[x];
				B = GaussImage.ptr<uint8_t>(y)[x - 1];
				C = GaussImage.ptr<uint8_t>(y - 1)[x];
				D = GaussImage.ptr<uint8_t>(y - 1)[x - 1];
				gradX = (B + D - A - C) / 2.0;
				gradY = (C + D - A - B) / 2.0;

				valueMagnitude = sqrt(pow(gradX, 2) + pow(gradY, 2));
				magMap.ptr<double>(y)[x] = valueMagnitude;

				if (valueMagnitude < gradThre)
					usedMap.ptr<uint8_t>(y)[x] = 1;

				maxGrad = max(maxGrad, valueMagnitude);
				//valueDegMap = atan2(gradX, -gradY);
				valueDegree = atan2(abs(gradX), abs(gradY));
				if (abs(valueDegree - pi) < 0.000001)
					valueDegree = 0;
				degMap.ptr<double>(y)[x] = valueDegree;

				nodeBinCell tempNode;
				tempNode.value = valueMagnitude;
				tempNode.x = x;
				tempNode.y = y;
				binCell.push_back(tempNode);
			}
		}
		GaussImage.release();

#ifdef drawPicture
		imshow("degMap", degMap);
		imshow("magMap", magMap);
		//waitKey(0);
#endif 

		printf("0 %lf\n", (clock() - last_time) / CLOCKS_PER_SEC);

		// �ݶ�ֵ�Ӵ�С����
		last_time = clock();
		sort(binCell.begin(), binCell.end(), compVector());
		printf("0 %lf\n", (clock() - last_time) / CLOCKS_PER_SEC);

		//��������˳�� ����������������
		logNT = 5.0 * (log10(newMapRow) + log10(newMapCol)) / 2.0;//���������Ķ���ֵ
		regThre = -logNT / log10(angThre / 180.0); //С�������ֵ
		aliPro = angThre / 180.0;

		//��¼��������;���
		structRec* recSaveHead = (structRec*)malloc(sizeof(structRec));
		structRec* recSaveNow = recSaveHead;

#ifdef drawPicture
		Mat lineIm = Mat::zeros(oriMapRow, oriMapCol, CV_8UC1);//��¼ֱ��ͼ��
		Mat lineImColor = Mat::zeros(oriMapRow, oriMapCol, CV_8UC3);//��¼ֱ��ͼ��
#endif 
		
		printf("1 %lf\n", (clock() - last_time2) / CLOCKS_PER_SEC);
		last_time2 = clock();

		//������ݶȿ�ʼ����
		double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
		int regCnt = 0;
		for (int i = 0; i < binCell.size(); i++) {
			int yIdx = binCell[i].y;
			int xIdx = binCell[i].x;
			if (usedMap.ptr<uint8_t>(yIdx)[xIdx] != 0)
				continue;

			//�������� ����curMap��reg
			last_time = clock();
			structRegionGrower RG = RegionGrower(xIdx, yIdx, degMap.ptr<double>(yIdx)[xIdx], degThre);
			t1 += clock() - last_time;

			structReg reg = RG.reg;
			//ɾ��С����
			if (reg.num < regThre)
				continue;

			//������� ����rec
			last_time = clock();
			structRec rec = RectangleConverter(reg, degThre);
			t2 += clock() - last_time;

			//�����ܶ���ֵ���������� ����boolean, curMap, rec, reg
			last_time = clock();
			structRefiner RF = Refiner(reg, rec, RG.curMap);
			t3 += clock() - last_time;
			reg = RF.reg;
			rec = RF.rec;
			if (!RF.boolean)
				continue;

			//���ε��� ����logNFA, rec
			last_time = clock();
			structRectangleImprover RI = RectangleImprover(rec);
			rec = RI.rec;
			Mat nonZeroCoordinates;
			findNonZero(RF.curMap, nonZeroCoordinates);
			if (RI.logNFA <= 0) {
				for (int j = 0; j < nonZeroCoordinates.total(); j++) {
					usedMap.ptr<uint8_t>(nonZeroCoordinates.at<Point>(j).y)[nonZeroCoordinates.at<Point>(j).x] = 2;
				}
				t4 += clock() - last_time;
				continue;
			}
			t4 += clock() - last_time;

			last_time = clock();
			//�������ų߶����µ���ͼ�������ҵ���ֱ����Ϣ
			if (sca != 1) {
				rec.x1 = (rec.x1 - 1.0) / sca + 1.0;
				rec.y1 = (rec.y1 - 1.0) / sca + 1.0;
				rec.x2 = (rec.x2 - 1.0) / sca + 1.0;
				rec.y2 = (rec.y2 - 1.0) / sca + 1.0;
				rec.wid = (rec.wid - 1.0) / sca + 1.0;
			}
			for (int j = 0; j < nonZeroCoordinates.total(); j++) {
				usedMap.ptr<uint8_t>(nonZeroCoordinates.at<Point>(j).y)[nonZeroCoordinates.at<Point>(j).x] = 1;
			}
			//for (int y = 0; y < newMapRow; y++) {
			//	for (int x = 0; x < newMapCol; x++) {
			//		/*regIdx.ptr<uint8_t>(y)[x] += RF.curMap.ptr<uint8_t>(y)[x] * (regCnt + 1);*/
			//		if (RF.curMap.ptr<uint8_t>(y)[x] == 1)
			//			usedMap.ptr<uint8_t>(y)[x] = 1;
			//	}
			//}
			//�������ҵ���ֱ��֧���������Ͼ���
			structRec* tempRec = (structRec*)malloc(sizeof(structRec));
			recSaveNow[0] = rec;
			recSaveNow[0].next = tempRec;
			recSaveNow = tempRec;
			regCnt++;
			t5 += clock() - last_time;
		}
		printf("%lf, %lf, %lf, %lf, %lf\n", t1 / CLOCKS_PER_SEC, t2 / CLOCKS_PER_SEC, t3 / CLOCKS_PER_SEC, t4 / CLOCKS_PER_SEC, t5 / CLOCKS_PER_SEC);
		printf("2 %lf\n", (clock() - last_time2) / CLOCKS_PER_SEC);

		//������ȡ����ֱ�߰������ص�����ͼ�������
		Vec3b color;
		recSaveNow = recSaveHead;
		structLinesInfo* linesInfo = (structLinesInfo*)malloc(regCnt * sizeof(structLinesInfo));
		for (int i = 0; i < regCnt; i++) {
			//���ֱ�ߵĶ˵�����
			double x1 = recSaveNow[0].x1;
			double y1 = recSaveNow[0].y1;
			double x2 = recSaveNow[0].x2;
			double y2 = recSaveNow[0].y2;
			recSaveNow = recSaveNow[0].next;
			//��ȡֱ��б��
			double k = (y2 - y1) / (x2 - x1);
			double ang = atand(k);
			int orient = 1;
			if (ang < 0) {
				ang += 180;
				orient = -1;
			}

#ifdef drawPicture
			//ȷ��ֱ��X�������Y������Ŀ��
			int xLow, xHigh, yLow, yHigh;
			if (x1 > x2) {
				xLow = (int)floor(x2);
				xHigh = (int)ceil(x1);
			}
			else {
				xLow = (int)floor(x1);
				xHigh = (int)ceil(x2);
			}
			if (y1 > y2) {
				yLow = (int)floor(y2);
				yHigh = (int)ceil(y1);
			}
			else {
				yLow = (int)floor(y1);
				yHigh = (int)ceil(y2);
			}
			double xRang = abs(x2 - x1), yRang = abs(y2 - y1);
			//ȷ��ֱ�߿�Ƚϴ����������Ϊ�������Ტ����
			int xx_len = xHigh - xLow + 1, yy_len = yHigh - yLow + 1;
			int* xx, * yy;
			int j;
			if (xx_len > yy_len) {
				xx = (int*)malloc(xx_len * sizeof(int));
				yy = (int*)malloc(xx_len * sizeof(int));
				for (j = 0; j < xx_len; j++) {
					xx[j] = j + xLow;
					yy[j] = (int)round((xx[j] - x1) * k + y1);
					if (xx[j] < 0 || xx[j] >= oriMapCol || yy[j] < 0 || yy[j] >= oriMapRow) {
						xx[j] = 0;
						yy[j] = 0;
					}
				}
			}
			else {
				xx = (int*)malloc(yy_len * sizeof(int));
				yy = (int*)malloc(yy_len * sizeof(int));
				for (j = 0; j < yy_len; j++) {
					yy[j] = j + yLow;
					xx[j] = (int)round((yy[j] - y1) / k + x1);
					if (xx[j] < 0 || xx[j] >= oriMapCol || yy[j] < 0 || yy[j] >= oriMapRow) {
						xx[j] = 0;
						yy[j] = 0;
					}
				}
			}
			//���ֱ������
			color[0] = rand() % 255;
			color[1] = rand() % 255;
			color[2] = rand() % 255;
			for (j = 0; j < max(xx_len, yy_len); j++) {
				if (xx[j] != 0 && yy[j] != 0) {
					lineIm.ptr<uint8_t>(yy[j])[xx[j]] = 255;
					lineImColor.ptr<Vec3b>(yy[j])[xx[j]] = color;
				}
			}

			free(xx);
			free(yy);
#endif

			linesInfo[i].k = k;
			linesInfo[i].b = (y1 + y2) / 2.0 - k * (x1 + x2) / 2.0;
			linesInfo[i].dx = cosd(ang);
			linesInfo[i].dy = sind(ang);
			linesInfo[i].x1 = x1;
			linesInfo[i].y1 = y1;
			linesInfo[i].x2 = x2;
			linesInfo[i].y2 = y2;
			linesInfo[i].len = sqrt(pow(y2 - y1, 2) + pow(x2 - x1, 2));
			linesInfo[i].orient = orient;
		}
		structLSD returnLSD;
		returnLSD.linesInfo = linesInfo;
		returnLSD.len_linesInfo = regCnt;

#ifdef drawPicture
		returnLSD.lineIm = lineIm;
		imshow("lineImColor", lineImColor);
#endif

		return returnLSD;
	}

	Mat LSD::createMapCache(Mat MapGray, double res) {
		//����ͼ�е㵽��������С���룬������ƥ��ʱ�����������
		int cell_radius = floor(z_occ_max_dis / res);
		int height = MapGray.rows, width = MapGray.cols;
		Mat mapCache = Mat::zeros(height, width, CV_64FC1);
		Mat mapFlag = Mat::zeros(height, width, CV_64FC1);

		structCache* head = (structCache*)malloc(sizeof(structCache));
		structCache* now = head, * tail;

		int i, j;
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				if (MapGray.ptr<uint8_t>(i)[j] == 1) {
					structCache* temp = (structCache*)malloc(sizeof(structCache));
					temp->next = NULL;
					now->src_i = i;
					now->src_j = j;
					now->cur_i = i;
					now->cur_j = j;
					now->next = temp;
					now = temp;
					mapCache.ptr<double>(i)[j] = 0;
					mapFlag.ptr<double>(i)[j] = 1;
				}
				else {
					mapCache.ptr<double>(i)[j] = z_occ_max_dis;
				}
			}
		}

		tail = now;
		now = head;
		while (now->next != NULL) {
			int src_i = now->src_i, src_j = now->src_j;
			int cur_i = now->cur_i, cur_j = now->cur_j;

			if (cur_i >= 1 && mapFlag.ptr<double>(cur_i - 1)[cur_j] == 0) {
				double di = abs(cur_i - src_i);
				double dj = abs(cur_j - src_j);
				double distance = sqrt(di * di + dj * dj);

				if (distance <= cell_radius) {
					mapCache.ptr<double>(cur_i - 1)[cur_j] = distance * res;
					mapFlag.ptr<double>(cur_i - 1)[cur_j] = 1;
					structCache* temp = (structCache*)malloc(sizeof(structCache));
					temp->next = NULL;
					tail->src_i = src_i;
					tail->src_j = src_j;
					tail->cur_i = cur_i - 1;
					tail->cur_j = cur_j;
					tail->next = temp;
					tail = temp;
				}
			}

			if (cur_j >= 1 && mapFlag.ptr<double>(cur_i)[cur_j - 1] == 0) {
				double di = abs(cur_i - src_i);
				double dj = abs(cur_j - src_j);
				double distance = sqrt(di * di + dj * dj);

				if (distance <= cell_radius) {
					mapCache.ptr<double>(cur_i)[cur_j - 1] = distance * res;
					mapFlag.ptr<double>(cur_i)[cur_j - 1] = 1;
					structCache* temp = (structCache*)malloc(sizeof(structCache));
					temp->next = NULL;
					tail->src_i = src_i;
					tail->src_j = src_j;
					tail->cur_i = cur_i;
					tail->cur_j = cur_j - 1;
					tail->next = temp;
					tail = temp;
				}
			}

			if (cur_i < height - 1 && mapFlag.ptr<double>(cur_i + 1)[cur_j] == 0) {
				double di = abs(cur_i - src_i);
				double dj = abs(cur_j - src_j);
				double distance = sqrt(di * di + dj * dj);

				if (distance <= cell_radius) {
					mapCache.ptr<double>(cur_i + 1)[cur_j] = distance * res;
					mapFlag.ptr<double>(cur_i + 1)[cur_j] = 1;
					structCache* temp = (structCache*)malloc(sizeof(structCache));
					temp->next = NULL;
					tail->src_i = src_i;
					tail->src_j = src_j;
					tail->cur_i = cur_i + 1;
					tail->cur_j = cur_j;
					tail->next = temp;
					tail = temp;
				}
			}

			if (cur_j < width - 1 && mapFlag.ptr<double>(cur_i)[cur_j + 1] == 0) {
				double di = abs(cur_i - src_i);
				double dj = abs(cur_j - src_j);
				double distance = sqrt(di * di + dj * dj);

				if (distance <= cell_radius) {
					mapCache.ptr<double>(cur_i)[cur_j + 1] = distance * res;
					mapFlag.ptr<double>(cur_i)[cur_j + 1] = 1;
					structCache* temp = (structCache*)malloc(sizeof(structCache));
					temp->next = NULL;
					tail->src_i = src_i;
					tail->src_j = src_j;
					tail->cur_i = cur_i;
					tail->cur_j = cur_j + 1;
					tail->next = temp;
					tail = temp;
				}
			}
			now = now->next;
		}

		return mapCache;
	}

	Mat LSD::GaussianSampler(Mat image, double sca, double sig) {
		//����
		//sca; ���ų߶�
		//sig: ��˹ģ��ı�׼��
		//���
		//newIm: ������˹�������ź��ͼ��
		int prec = 3, xLim = image.cols, yLim = image.rows;
		int newXLim = (int)floor(xLim * sca);
		int newYLim = (int)floor(yLim * sca);
		//printf("newXLim=%d newYLim=%d\n", newXLim, newYLim);
		Mat auxImage = Mat::zeros(yLim, newXLim, CV_64FC1);
		Mat newImage = Mat::zeros(newYLim, newXLim, CV_64FC1);
		//�������Сͼ���������׼���ֵ
		if (sca < 1.0)
			sig = sig / sca;
		//printf("%f\n", sig);
		//��˹ģ���С
		int h = (int)ceil(sig * sqrt(2.0 * prec * log(10)));
		int hSize = 1 + 2 * h;
		int douXLim = xLim * 2;
		int douYLim = yLim * 2;
		//x�������
		int x;
		for (x = 0; x < newXLim; x++) {
			double xx = x / sca;
			int xc = (int)floor(xx + 0.5);
			//ȷ����˹������λ��
			double kerMean = h + xx - xc;
			double* kerVal = (double*)malloc(hSize * sizeof(double));
			double kerSum = 0;
			int k = 0;

			//��ǰ��˹�ˣ������й��ɿ�ѭ ���跴������ �������Ż���
			for (k = 0; k < hSize; k++) {
				kerVal[k] = (double)exp((-0.5) * pow((k - kerMean) / sig, 2));
				kerSum += kerVal[k];
			}
			//��˹�˹�һ��
			for (k = 0; k < hSize; k++) {
				kerVal[k] /= kerSum;
			}
			//�ñ�Ե�ԳƵķ�ʽ����X�����˹�˲�
			int y;
			for (y = 0; y < yLim; y++) {
				double newVal = 0;
				int i;
				structPosition* pixelCache = (structPosition*)malloc(hSize * sizeof(structPosition));
				for (i = 0; i < hSize; i++) {
				}

				for (i = 0; i < hSize; i++) {
					int j = xc - h + i;
					while (j < 0) {
						j += douXLim;
					}
					while (j >= douXLim) {
						j -= douXLim;
					}
					if (j >= xLim)
						j = douXLim - j - 1;
					newVal += image.ptr<uint8_t>(y)[j] * kerVal[i];
				}
				auxImage.ptr<double>(y)[x] = round(newVal);
			}
		}//end for��x���������

		//y�������
		int y;
		for (y = 0; y < newYLim; y++) {
			double yy = y / sca;
			int yc = (int)floor(yy + 0.5);
			//ȷ����˹������λ��
			double kerMean = h + yy - yc;
			double* kerVal = (double*)malloc(hSize * sizeof(double));
			double kerSum = 0;
			int k = 0;
			//��ǰ��˹��
			for (k = 0; k < hSize; k++) {
				kerVal[k] = exp((-0.5) * pow((k - kerMean) / sig, 2));
				kerSum += kerVal[k];
			}
			//��˹�˹�һ��
			for (k = 0; k < hSize; k++) {
				kerVal[k] /= kerSum;
			}
			//�ñ�Ե�ԳƵķ�ʽ����Y�����˹�˲�
			int x;
			for (x = 0; x < newXLim; x++)
			{
				double newVal = 0;
				int i;
				for (i = 0; i < hSize; i++)
				{
					int j = yc - h + i;
					while (j < 0) {
						j += douYLim;
					}
					while (j >= douYLim) {
						j -= douYLim;
					}
					if (j >= yLim)
						j = douYLim - j - 1;
					newVal += auxImage.ptr<double>(j)[x] * kerVal[i];
				}
				newImage.ptr<double>(y)[x] = newVal;
			}
		}//end for��y���������

		return newImage;
	}

	LSD::structRegionGrower LSD::RegionGrower(int x, int y, double regDeg, double degThre) {
		// ����
		// x��        ��ʼ��X������
		// y��        ��ʼ��Y������
		// regDeg��   ����level - line�������
		//
		// ���
		// curMap��   �����������ӵ�������������ָʾͼ
		// reg��      ��ǰ������������������Ϣ
		// .x��       ��ʼ��X���� 
		// .y��       ��ʼ��Y����
		// .num��     ��������������
		// .deg��     ����ƽ���ǻ���
		// .pts��     ����������������������ֵ
		//
		// �������ܣ� ͨ���ϲ���ͬ�����level - line��ʵ����������
		structPts *regPts_now, *regPts_head, *regPts_end;
		regPts_head = regPts_end = regPts_now = (structPts*)malloc(sizeof(structPts));
		regPts_head[0].x = x;
		regPts_head[0].y = y;
		regPts_head[0].next = NULL;
		double sinDeg = sin(regDeg), cosDeg = cos(regDeg);
		Mat curMap = Mat::zeros(newMapRow, newMapCol, CV_8UC1);
		curMap.ptr<uint8_t>(y)[x] = 1;
		int growNum = 1, exNum = 0, isFirstTime = 1, temp = 0, roi_x, roi_y;
		while (exNum != growNum) {
			exNum = growNum;
			regPts_now = regPts_head;
			for (int i = 0; i < growNum; i++) {
				// ����8���������Ƿ�����ǻ�����ֵ
				roi_x = regPts_now[0].x, roi_y = regPts_now[0].y;
				for (int m = roi_y - 1; m <= roi_y + 1; m++) {
					for (int n = roi_x - 1; n <= roi_x + 1; n++) {
						// �������ֵ��״̬
						if (m >= 0 && n >= 0 && m < newMapRow && n < newMapCol) {
							if (curMap.ptr<uint8_t>(m)[n] != 1 && usedMap.ptr<uint8_t>(m)[n] != 1) {
								// ����ǵ�ǰ����������ֵ ���� ��ǰ���ȼ�pi������ֵ
								double curDeg = degMap.ptr<double>(m)[n];
								double degDif = abs(regDeg - curDeg);
								if (degDif > pi * 3 / 2.0)
									degDif = abs(degDif - 2.0 * pi);
								if (degDif < degThre) {
									// ����ͳ�����û��ȵ����Һ�����ֵ
									cosDeg += cos(curDeg);
									sinDeg += sin(curDeg);
									regDeg = atan2(sinDeg, cosDeg);
									// ��¼��ǰ����
									curMap.ptr<uint8_t>(m)[n] = 1;
									growNum++;
									structPts* temp = (structPts*)malloc(sizeof(structPts));
									temp->x = n;
									temp->y = m;
									temp->next = NULL;
									regPts_end->next = temp;
									regPts_end = temp;
								}
							}
						}
					}
				}
				if (regPts_now != regPts_end)
					regPts_now = regPts_now[0].next;
			}
		}

		int* rePts_x = (int*)malloc(growNum * sizeof(int));
		int* rePts_y = (int*)malloc(growNum * sizeof(int));
		regPts_now = regPts_head;
		for (int i = 0; i < growNum; i++) {
			rePts_y[i] = regPts_now[0].y;
			rePts_x[i] = regPts_now[0].x;
			regPts_now = regPts_now[0].next;
		}

		structReg reg;
		reg.x = x;
		reg.y = y;
		reg.num = growNum;
		reg.deg = regDeg;
		reg.regPts_x = rePts_x;
		reg.regPts_y = rePts_y;

		structRegionGrower RG;
		RG.curMap = curMap;
		RG.reg = reg;

		return RG;
	}

	LSD::structCenterGetter LSD::CenterGetter(const int regNum, const int* regX, const int* regY) {
		// ���룺
		// regNum������������ص���
		// regX��  ���������ص�x��������
		// regY��  ���������ص�y��������
		//
		// �����
		// cenX��  ��������x����
		// cenY��  ��������y����
		//
		// �������ܣ�����������������ص��Ȩ�أ��ҵ���������
		double cenX = 0, cenY = 0, weiSum = 0, pixWei;
		for (int k = 0; k < regNum; k++) {
			pixWei = magMap.ptr<double>(regY[k])[regX[k]];
			cenX += pixWei * regX[k];
			cenY += pixWei * regY[k];
			weiSum += pixWei;
		}
		structCenterGetter CG;
		CG.cenX = cenX / weiSum;
		CG.cenY = cenY / weiSum;

		return CG;
	}

	double LSD::OrientationGetter(const structReg reg, const double cenX, const double cenY, const int* regX, const int* regY, const double degThre) {
		// ���룺
		// reg�� ����ṹ��
		// cenX����������X����
		// cenY����������Y����
		// regX��ֱ��֧�������и����X����
		// regY��ֱ��֧�������и����Y����
		//
		// �������ܣ���ȡ�����������᷽��Ľǻ���ֵ��

		double Ixx = 0, Iyy = 0, Ixy = 0, weiSum = 0, pixWei;
		// ��������������Ϊ���η���
		for (int i = 0; i < reg.num; i++) {
			pixWei = magMap.ptr<double>(reg.regPts_y[i])[reg.regPts_x[i]];
			Ixx += pixWei * pow(reg.regPts_y[i] - cenY, 2);
			Iyy += pixWei * pow(reg.regPts_x[i] - cenX, 2);
			Ixy -= pixWei * (reg.regPts_x[i] - cenX) * (reg.regPts_y[i] - cenY);
			weiSum += pixWei;
		}
		Ixx /= weiSum;
		Iyy /= weiSum;
		Ixy /= weiSum;

		const double lamb = (Ixx + Iyy - sqrt(pow(Ixx - Iyy, 2) + 4 * Ixy * Ixy)) / 2.0;
		double inertiaDeg;
		if (abs(Ixx) > abs(Iyy))
			inertiaDeg = atan2(lamb - Ixx, Ixy);
		else
			inertiaDeg = atan2(Ixy, lamb - Iyy);

		// ����һ��pi�����
		double regDif = inertiaDeg - reg.deg;
		while (regDif <= -pi) {
			regDif += pi2;
		}
		while (regDif > pi) {
			regDif -= pi2;
		}
		regDif = abs(regDif);
		if (regDif > degThre)
			inertiaDeg += pi;

		return inertiaDeg;
	}

	LSD::structRec LSD::RectangleConverter(const structReg reg, const double degThre) {
		// ����
		// reg��  ָʾ����Ľṹ��
		//
		// ���
		// rec��  ��õľ��νṹ��
		// .x1��  ���ζ̱�ĳһ���е�X����
		// .y1��  ���ζ̱�ĳһ���е�Y����
		// .x2��  ���ζ̱���һ���е�X����
		// .y2��  ���ζ̱�ĳһ���е�Y����
		// .wid�� ���ζ̱߳���
		// .cX��  ��������X����
		// .cY��  ��������Y����
		// .deg�� ����������ǻ���
		// .dx��  ����������ǻ�������ֵ
		// .dy��  ����������ǻ�������ֵ
		// .p��   �����ڵ�level - line���ǻ��������������ǻ���������ʣ�
		// .prec���жϾ����ڵ�level - line���ǻ��������������ǻ��ȵ���ֵ���Ƕ����̶ȣ�
		//
		// �������ܣ������߶�֧����Ѱ����С�ڽӾ���

		// �����߶�֧���������
		structCenterGetter CG = CenterGetter(reg.num, reg.regPts_x, reg.regPts_y);
		// ȷ������������
		const double inertiaDeg = OrientationGetter(reg, CG.cenX, CG.cenY, reg.regPts_x, reg.regPts_y, degThre);

		// ȷ�����γ��Ϳ�
		double dx = cos(inertiaDeg);
		double dy = sin(inertiaDeg);
		double lenMin = 0, lenMax = 0, widMin = 0, widMax = 0, len, wid;
		
		for (int m = 0; m < reg.num; m++) {
			len = (reg.regPts_x[m] - CG.cenX) * dx + (reg.regPts_y[m] - CG.cenY) * dy;
			wid = -(reg.regPts_x[m] - CG.cenX) * dy + (reg.regPts_y[m] - CG.cenY) * dx;
			lenMin = min(len, lenMin);
			lenMax = max(len, lenMax);
			widMin = min(wid, widMin);
			widMax = max(wid, widMax);
		}
		// ���������Ϣ���ṹ��
		structRec rec;
		rec.x1 = CG.cenX + lenMin * dx;
		rec.y1 = CG.cenY + lenMin * dy;
		rec.x2 = CG.cenX + lenMax * dx;
		rec.y2 = CG.cenY + lenMax * dy;
		rec.wid = max(widMax - widMin, 1.0);
		rec.cX = CG.cenX;
		rec.cY = CG.cenY;
		rec.deg = inertiaDeg;
		rec.dx = dx;
		rec.dy = dy;
		rec.p = aliPro;
		rec.prec = degThre;

		return rec;
	}

	LSD::structRegionRadiusReducer LSD::RegionRadiusReducer(structReg reg, structRec rec, Mat curMap) {
		// ���룺
		// reg��    ��ǰ����Ľṹ��
		// rec��    ��ǰ�������С��Ӿ��εĽṹ��
		// denThre�������ܶ���ֵ
		// curMap�� ��ǰ����ͼ
		// magMap�� �ݶȷ�ֵͼ
		//
		// �����
		// bool��   ��С�뾶���Ƿ����ҵ����ʾ��ε�ָʾ��
		// curMap�� ��ǰ����ָʾͼ
		// reg��    ��ǰ����ṹ��
		// rec��    ��ǰ�������С��Ӿ��εĽṹ��
		//
		// �������ܣ����ڼ�С����İ뾶�Ӷ����������ڵ������������ɸ����˵���С��Ӿ��Ρ�
		structRegionRadiusReducer RRR;
		RRR.boolean = true;
		RRR.curMap = curMap;
		RRR.rec = rec;
		RRR.reg = reg;
		double den = RRR.reg.num / (sqrt(pow(RRR.rec.x1 - RRR.rec.x2, 2) + pow(RRR.rec.y1 - RRR.rec.y2, 2)) * RRR.rec.wid);
		// ��������ܶ���ֵ����ֱ�ӷ���
		if (den > denThre) {
			RRR.boolean = true;
			return RRR;
		}
		// ��ԭ���������ĳ�ʼ����Ϊ���Ĳο���
		const int oriX = RRR.reg.x, oriY = RRR.reg.y;
		// ѡȡֱ����Զ���������Ĳο�������нϴ�ֵ��Ϊ�����뾶
		const double rad1 = sqrt(pow(oriX - RRR.rec.x1, 2) + pow(oriY - RRR.rec.y1, 2));
		const double rad2 = sqrt(pow(oriX - RRR.rec.x2, 2) + pow(oriY - RRR.rec.y2, 2));
		double rad = max(rad1, rad2);

		while (den < denThre) {
			// ��0.75�������ٶȼ�С�����뾶������ֱ��֧�������е�������
			rad *= 0.75;
			int i = 0;
			while (i <= RRR.reg.num) {
				if (sqrt(pow(oriX - RRR.reg.regPts_x[i], 2) + pow(oriY - RRR.reg.regPts_y[i], 2)) > rad) {
					RRR.curMap.ptr<uint8_t>(RRR.reg.regPts_y[i])[RRR.reg.regPts_x[i]] = 0;
					RRR.reg.regPts_x[i] = RRR.reg.regPts_x[RRR.reg.num - 1];
					RRR.reg.regPts_y[i] = RRR.reg.regPts_y[RRR.reg.num - 1];
					RRR.reg.regPts_x[RRR.reg.num - 1] = NULL;
					RRR.reg.regPts_y[RRR.reg.num - 1] = NULL;
					i--;
					RRR.reg.num--;
				}
				i++;
			}
			// ���ֱ��֧������������������2���������������
			if (RRR.reg.num < 2) {
				RRR.boolean = false;
				return RRR;
			}
			// ����õ�ֱ��֧������ת��Ϊ��С��Ӿ���
			RRR.rec = RectangleConverter(RRR.reg, RRR.rec.prec);
			den = RRR.reg.num / (sqrt(pow(RRR.rec.x1 - RRR.rec.x2, 2) + pow(RRR.rec.y1 - RRR.rec.y2, 2)) * RRR.rec.wid);
		}
		RRR.boolean = true;
		return RRR;
	}

	LSD::structRefiner LSD::Refiner(structReg reg, structRec rec, Mat curMap) {
		// ���룺
		// reg��   ֱ��֧������Ľṹ��
		// rec��   ֱ��֧���������С��Ӿ��εĽṹ��
		// curMap����ǰ��������ͼ
		//
		// �����
		// bool��  �Ƿ�ɹ�����ָʾ��
		// curMap����ǰ��������ָʾͼ
		// reg��   ��ǰ������������
		// rec��   ��ǰ���������������С��Ӿ���
		//
		// �������ܣ� ��������ȡ��ֱ��֧�������Լ�����Ӧ����С��Ӿ���
		structRefiner RF;
		RF.boolean = true;
		RF.curMap = curMap;
		RF.rec = rec;
		RF.reg = reg;
		double den = RF.reg.num / (sqrt(pow(RF.rec.x1 - RF.rec.x2, 2) + pow(RF.rec.y1 - RF.rec.y2, 2)) * RF.rec.wid);
		// ��������ܶ���ֵ������������
		if (den >= denThre) {
			RF.boolean = true;
			return RF;
		}
		const int oriX = RF.reg.x, oriY = RF.reg.y;
		const double cenDeg = degMap.ptr<double>(oriY)[oriX];
		double difSum = 0, squSum = 0, curDeg, degDif, ptNum = 0;
		// ����������������ʼ�����С�ھ��ο�ȵ����ؽ�����������ֵ�ع���
		for (int i = 0; i < RF.reg.num; i++) {
			if (sqrt(pow(oriX - RF.reg.regPts_x[i], 2) + pow(oriY - RF.reg.regPts_y[i], 2)) < RF.rec.wid) {
				curDeg = degMap.ptr<double>(RF.reg.regPts_y[i])[RF.reg.regPts_x[i]];
				degDif = curDeg - cenDeg;
				while (degDif <= -pi) {
					degDif += pi2;
				}
				while (degDif > pi) {
					degDif -= pi2;
				}
				difSum += degDif;
				squSum += degDif * degDif;
				ptNum++;
			}
		}
		double meanDif = difSum / ptNum;
		double degThre = 2.0 * sqrt((squSum - 2.0 * meanDif * difSum) / ptNum + pow(meanDif, 2));
		// ��������ֵ���½�����������
		structRegionGrower RG = RegionGrower(oriX, oriY, cenDeg, degThre);
		RF.curMap = RG.curMap;
		RF.reg = RG.reg;
		// �����������ֵ�������������С������ǰ����
		if (RF.reg.num < 2) {
			RF.boolean = false;
			return RF;
		}
		// ���½�����С��Ӿ���
		RF.rec = RectangleConverter(RF.reg, degThre);
		den = RF.reg.num / (sqrt(pow(RF.rec.x1 - RF.rec.x2, 2) + pow(RF.rec.y1 - RF.rec.y2, 2)) * RF.rec.wid);
		// �����δ�����ܶ���ֵ�����С����뾶
		if (den < denThre) {
			structRegionRadiusReducer RRR = RegionRadiusReducer(RF.reg, RF.rec, RF.curMap);
			//printf("RF x%d y%d\nRRR x%d y%d\n", RF.curMap.cols, RF.curMap.rows, RRR.curMap.cols, RRR.curMap.rows);
			RF.boolean = RRR.boolean;
			RF.curMap = RRR.curMap;
			RF.rec = RRR.rec;
			RF.reg = RRR.reg;
			return RF;
		}
		RF.boolean = true;
		return RF;
	}

	double LSD::RectangleNFACalculator(structRec rec) {
		// ���룺
		// rec��   ��ǰ������С��Ӿ��εĽṹ��
		// degMap��ˮ׼�߽ǻ���ͼ
		// logNT�� ���������Ķ���ֵ
		//
		// �����
		// logNFA���龯������Ȼ����ֵ
		//
		// �������ܣ�������������������������ģ�͵��龯��
		int allPixNum = 0, aliPixNum = 0;
		//����0~pi�Ļ���֮�俼�Ǿ��νǶ�
		int cnt_col, cnt_row;
		for (cnt_row = 0; cnt_row < newMapRow; cnt_row++) {
			for (cnt_col = 0; cnt_col < newMapCol; cnt_col++) {
				if (degMap.ptr<double>(cnt_row)[cnt_col] > pi)
					degMap.ptr<double>(cnt_row)[cnt_col] -= pi;
			}
		}
		//�ҵ������ĸ����������
		structRecVer recVer;
		double verX[4], verY[4];
		verX[0] = rec.x1 - rec.dy * rec.wid / 2.0;
		verX[1] = rec.x2 - rec.dy * rec.wid / 2.0;
		verX[2] = rec.x2 + rec.dy * rec.wid / 2.0;
		verX[3] = rec.x1 + rec.dy * rec.wid / 2.0;
		verY[0] = rec.y1 + rec.dx * rec.wid / 2.0;
		verY[1] = rec.y2 + rec.dx * rec.wid / 2.0;
		verY[2] = rec.y2 - rec.dx * rec.wid / 2.0;
		verY[3] = rec.y1 - rec.dx * rec.wid / 2.0;
		//��xֵ��С�ĵ���Ϊ1�ŵ㣬Ȼ����ʱ�����������
		int offset, i, j;
		if (rec.x1 < rec.x2 && rec.y1 <= rec.y2)
			offset = 0;
		else if (rec.x1 >= rec.x2 && rec.y1 < rec.y2)
			offset = 1;
		else if (rec.x1 > rec.x2 && rec.y1 >= rec.y2)
			offset = 2;
		else
			offset = 3;
		for (i = 0; i < 4; i++) {
			recVer.verX[i] = verX[(offset + i) % 4];
			recVer.verY[i] = verY[(offset + i) % 4];
		}
		//ͳ�Ƶ�ǰ������������������᷽����ͬ��С�ڽǶ����̶ȣ������ص����� aliPixNum
		//�������������ص��� allPixNum
		int xRang_len = abs(ceil(recVer.verX[0]) - floor(recVer.verX[2])) + 1;
		int* xRang = (int*)malloc(xRang_len * sizeof(int));
		for (i = 0; i < xRang_len; i++) {
			xRang[i] = i + ceil(recVer.verX[0]);
		}
		double lineK[4];
		lineK[0] = (recVer.verY[1] - recVer.verY[0]) / (recVer.verX[1] - recVer.verX[0]);
		lineK[1] = (recVer.verY[2] - recVer.verY[1]) / (recVer.verX[2] - recVer.verX[1]);
		lineK[2] = (recVer.verY[2] - recVer.verY[3]) / (recVer.verX[2] - recVer.verX[3]);
		lineK[3] = (recVer.verY[3] - recVer.verY[0]) / (recVer.verX[3] - recVer.verX[0]);
		int* yLow = (int*)malloc(xRang_len * sizeof(int));
		int* yHigh = (int*)malloc(xRang_len * sizeof(int));
		//yLow
		int cnt_yArry = 0;
		for (i = 0; i < xRang_len; i++) {
			if (xRang[i] < recVer.verX[3])
				yLow[cnt_yArry++] = (int)ceil(recVer.verY[0] + (xRang[i] - recVer.verX[0]) * lineK[3]);
		}
		for (i = 0; i < xRang_len; i++) {
			if (xRang[i] >= recVer.verX[3])
				yLow[cnt_yArry++] = (int)ceil(recVer.verY[3] + (xRang[i] - recVer.verX[3]) * lineK[2]);
		}
		//yHigh
		cnt_yArry = 0;
		for (i = 0; i < xRang_len; i++) {
			if (xRang[i] < recVer.verX[1])
				yHigh[cnt_yArry++] = (int)floor(recVer.verY[0] + (xRang[i] - recVer.verX[0]) * lineK[0]);
		}
		for (i = 0; i < xRang_len; i++) {
			if (xRang[i] >= recVer.verX[1])
				yHigh[cnt_yArry++] = (int)floor(recVer.verY[1] + (xRang[i] - recVer.verX[1]) * lineK[1]);
		}
		for (i = 0; i < xRang_len; i++) {
			for (j = yLow[i]; j <= yHigh[i]; j++) {
				if ((xRang[i] >= 0) && (xRang[i] < newMapCol) && (j >= 0) && (j < newMapRow)) {
					allPixNum++;
					double degDif = abs(rec.deg - degMap.ptr<double>(j)[xRang[i]]);
					if (degDif > pi * 3 / 2.0)
						degDif = abs(degDif - pi2);
					if (degDif < rec.prec)
						aliPixNum++;
				}
			}
		}
		//����NFA����Ȼ����ֵ
		double coefA = 0.1066 * logNT + 2.6750;
		double coefB = 0.004120 * logNT - 0.6223;
		double coefC = -0.002607 * logNT + 0.1550;
		double aliThre = allPixNum * (coefA * pow(allPixNum, coefB) + coefC);
		double logNFA2 = -1;
		if (aliPixNum > aliThre)
			logNFA2 = 1.0 * aliPixNum / allPixNum;

		return logNFA2;
	}

	LSD::structRectangleImprover LSD::RectangleImprover(structRec rec) {
		// ���룺
		// rec��    ��ǰ���νṹ��
		//
		// �����
		// logNFA�� �龯������Ȼ����ֵ
		// rec��    ������ľ��νṹ��
		//
		// �������ܣ������龯��(NFA, Number of False Alarms)����������С��Ӿ���
		structRectangleImprover RI;
		RI.logNFA = RectangleNFACalculator(rec);
		RI.rec = rec;
		return RI;
	}
}