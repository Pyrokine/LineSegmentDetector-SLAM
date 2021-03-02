#include <myLSD.h>

using namespace cv;
using namespace std;

namespace mylsd {
	// LineSegmentDetector
	LSD::structLSD LSD::myLineSegmentDetector(Mat& MapGray, const int _oriMapCol, const int _oriMapRow, const double _sca, const double _sig,
			const double _angThre, const double _denThre, const int _pseBin){
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
		//last_time = clock();
		//Mat GaussImage = GaussianSampler(MapGray, sca, sig); // ������ص���double�ͣ�ͬʱ���������趨���ű���
		Mat GaussImage;
		pyrDown(MapGray, GaussImage, Size(oriMapCol / 2.0, oriMapRow / 2.0)); // ͼ�������
		//printf("01 %lf\n", (clock() - last_time) / CLOCKS_PER_SEC);

#ifdef drawPicture
		imshow("GaussImage", GaussImage);
		//waitKey(0);
#endif

		usedMap = Mat::zeros(newMapRow, newMapCol, CV_8UC1);//��¼���ص�״̬
		degMap = Mat::zeros(newMapRow, newMapCol, CV_32FC1);//level-line������
		magMap = Mat::zeros(newMapRow, newMapCol, CV_32FC1);//��¼ÿ����ݶ�
		double degThre = angThre / 180.0 * pi; // �Ƕ���ֵ
		gradThre = 2.0 / sin(degThre); // �ݶ���ֵ

		// �����ݶȺ�level-line�����򲢴����ݶȵ�����
		//last_time = clock();
		vector<nodeBinCell> binCell;
		double maxGrad = 0, gradX, gradY, valueMagnitude, valueDegree, A, B, C, D;
		for (int y = 1; y < newMapRow; y++) {
			for (int x = 1; x < newMapCol; x++) {
				A = GaussImage.ptr<uint8_t>(y)[x];
				B = GaussImage.ptr<uint8_t>(y)[x - 1];
				C = GaussImage.ptr<uint8_t>(y - 1)[x];
				D = GaussImage.ptr<uint8_t>(y - 1)[x - 1];
				gradX = (B + D - A - C) / 2.0;
				gradY = (C + D - A - B) / 2.0;

				valueMagnitude = sqrt(pow(gradX, 2) + pow(gradY, 2));
				magMap.ptr<float>(y)[x] = valueMagnitude;

				if (valueMagnitude < gradThre)
					usedMap.ptr<uint8_t>(y)[x] = 1;

				maxGrad = max(maxGrad, valueMagnitude);
				//valueDegMap = atan2(gradX, -gradY);
				valueDegree = atan2(abs(gradX), abs(gradY));
				if (abs(valueDegree - pi) < 0.000001)
					valueDegree = 0;
				degMap.ptr<float>(y)[x] = valueDegree;

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

		//printf("02 %lf\n", (clock() - last_time) / CLOCKS_PER_SEC);

		// �ݶ�ֵ�Ӵ�С����
		//last_time = clock();
		sort(binCell.begin(), binCell.end(), compVector());
		//printf("03 %lf\n", (clock() - last_time) / CLOCKS_PER_SEC);

		// 
		logNT = 5.0 * (log10(newMapRow) + log10(newMapCol)) / 2.0;// ���������Ķ���ֵ
		regThre = -logNT / log10(angThre / 180.0); //С�������ֵ
		aliPro = angThre / 180.0;

		// ��¼��������;���
		structRec* recSaveHead = (structRec*)malloc(sizeof(structRec));
		structRec* recSaveNow = recSaveHead;
		vector<structRec> recSave;

#ifdef drawPicture
		Mat lineIm = Mat::zeros(oriMapRow, oriMapCol, CV_8UC1);// ��¼ֱ�߻Ұ�ͼ��
		Mat lineImColor = Mat::zeros(oriMapRow, oriMapCol, CV_8UC3);// ��¼ֱ�߲�ɫͼ��
#endif 
		
		//printf("1 %lf\n", (clock() - last_time2) / CLOCKS_PER_SEC);
		//last_time2 = clock();

		// ��������˳�����������������أ�������ݶȿ�ʼ����
		double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
		int regCnt = 0;
		for (int i = 0; i < binCell.size(); i++) {
			int yIdx = binCell[i].y;
			int xIdx = binCell[i].x;
			if (usedMap.ptr<uint8_t>(yIdx)[xIdx] != 0)
				continue;

			// �������� ����curMap��reg
			//last_time = clock();
			structRegionGrower RG = RegionGrower(xIdx, yIdx, degMap.ptr<float>(yIdx)[xIdx], degThre);
			//t1 += clock() - last_time;

			structReg reg = RG.reg;
			// ɾ��С����
			if (reg.num < regThre)
				continue;

			// ������� ����rec
			//last_time = clock();
			structRec rec = RectangleConverter(reg, degThre);
			//t2 += clock() - last_time;

			// �����ܶ���ֵ���������� ����boolean, curMap, rec, reg
			//last_time = clock();
			structRefiner RF = Refiner(reg, rec, RG.curMap);
			//t3 += clock() - last_time;
			reg = RF.reg;
			rec = RF.rec;
			if (!RF.boolean)
				continue;

			// ���ε��� ����logNFA, rec
			//last_time = clock();
			structRectangleImprover RI = RectangleImprover(rec);
			rec = RI.rec;
			Mat nonZeroCoordinates;
			findNonZero(RF.curMap, nonZeroCoordinates);
			if (RI.logNFA <= 0) {
				for (int j = 0; j < nonZeroCoordinates.total(); j++) {
					usedMap.ptr<uint8_t>(nonZeroCoordinates.at<Point>(j).y)[nonZeroCoordinates.at<Point>(j).x] = 2;
				}
				//t4 += clock() - last_time;
				continue;
			}
			//t4 += clock() - last_time;

			//last_time = clock();
			// �������ų߶����µ���ͼ�������ҵ���ֱ����Ϣ
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

			// �������ҵ���ֱ��֧���������Ͼ���
			recSave.push_back(rec);
			regCnt++;
			//t5 += clock() - last_time;
		}
		//printf("%lf, %lf, %lf, %lf, %lf\n", t1 / CLOCKS_PER_SEC, t2 / CLOCKS_PER_SEC, t3 / CLOCKS_PER_SEC, t4 / CLOCKS_PER_SEC, t5 / CLOCKS_PER_SEC);
		//printf("2 %lf\n", (clock() - last_time2) / CLOCKS_PER_SEC);

		// ������ȡ����ֱ�߰������ص�����ͼ�������
		Vec3b color;
		structLinesInfo* linesInfo = (structLinesInfo*)malloc(regCnt * sizeof(structLinesInfo));
		for (int i = 0; i < regCnt; i++) {
			// ���ֱ�ߵĶ˵�����
			double x1 = recSave[i].x1;
			double y1 = recSave[i].y1;
			double x2 = recSave[i].x2;
			double y2 = recSave[i].y2;
			// ��ȡֱ��б��
			double k = (y2 - y1) / (x2 - x1);
			double ang = atand(k);
			int orient = 1;
			if (ang < 0) {
				ang += 180;
				orient = -1;
			}

#ifdef drawPicture
			// ȷ��ֱ��X�������Y������Ŀ��
			int xLow, xHigh, yLow, yHigh;
			if (x1 > x2) {
				xLow = floor(x2);
				xHigh = ceil(x1);
			}
			else {
				xLow = floor(x1);
				xHigh = ceil(x2);
			}
			if (y1 > y2) {
				yLow = floor(y2);
				yHigh = ceil(y1);
			}
			else {
				yLow = floor(y1);
				yHigh = ceil(y2);
			}
			double xRang = abs(x2 - x1), yRang = abs(y2 - y1);
			// ȷ��ֱ�߿�Ƚϴ����������Ϊ�������Ტ���������ֱ������
			int xx_len = xHigh - xLow + 1, yy_len = yHigh - yLow + 1;

			color[0] = rand() % 255;
			color[1] = rand() % 255;
			color[2] = rand() % 255;
			int j, xx, yy;
			if (xx_len > yy_len) {
				for (j = 0; j < xx_len; j++) {
					xx = j + xLow;
					yy = round((xx - x1) * k + y1);
					if (xx >= 0 || xx < oriMapCol || yy >= 0 || yy < oriMapRow) {
						lineIm.ptr<uint8_t>(yy)[xx] = 255;
						lineImColor.ptr<Vec3b>(yy)[xx] = color;
					}
				}
			}
			else {
				for (j = 0; j < yy_len; j++) {
					yy = j + yLow;
					xx = round((yy - y1) / k + x1);
					if (xx >= 0 || xx < oriMapCol || yy >= 0 || yy < oriMapRow) {
						lineIm.ptr<uint8_t>(yy)[xx] = 255;
						lineImColor.ptr<Vec3b>(yy)[xx] = color;
					}
				}
			}
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
		int cell_radius2 = pow(floor(z_occ_max_dis / res), 2);
		int height = MapGray.rows, width = MapGray.cols;
		Mat mapCache = Mat(height, width, CV_32FC1, Scalar(z_occ_max_dis));
		Mat mapFlag = Mat::zeros(height, width, CV_8UC1);

		structCache* head = (structCache*)malloc(sizeof(structCache));
		structCache* now = head, * tail;

		vector<structCache> candidates;

		Mat nonZeroCoordinates;
		findNonZero(MapGray, nonZeroCoordinates);
		for (int i = 0; i < nonZeroCoordinates.total(); i++) {
			structCache temp;
			temp.srcY = nonZeroCoordinates.at<Point>(i).y;
			temp.srcX = nonZeroCoordinates.at<Point>(i).x;
			temp.curY = nonZeroCoordinates.at<Point>(i).y;
			temp.curX = nonZeroCoordinates.at<Point>(i).x;
			candidates.push_back(temp);
			mapCache.ptr<float>(nonZeroCoordinates.at<Point>(i).y)[nonZeroCoordinates.at<Point>(i).x] = 0;
			mapFlag.ptr<uint8_t>(nonZeroCoordinates.at<Point>(i).y)[nonZeroCoordinates.at<Point>(i).x] = 1;
		}

		double pnt_now = 0, pnt_end = candidates.size(), distance;
		int srcY, srcX, curY, curX;
		while (pnt_now < pnt_end) {
			srcY = candidates[pnt_now].srcY, srcX = candidates[pnt_now].srcX;
			curY = candidates[pnt_now].curY, curX = candidates[pnt_now].curX;

			if (curY >= 1 && mapFlag.ptr<uint8_t>(curY - 1)[curX] == 0) {
				distance = pow(abs(curY - srcY), 2) + pow(abs(curX - srcX), 2);

				if (distance <= cell_radius2) {
					mapCache.ptr<float>(curY - 1)[curX] = sqrt(distance) * res;
					mapFlag.ptr<uint8_t>(curY - 1)[curX] = 1;

					structCache temp;
					temp.srcY = srcY;
					temp.srcX = srcX;
					temp.curY = curY - 1;
					temp.curX = curX;
					candidates.push_back(temp);
					pnt_end += 1;
				}
			}

			if (curX >= 1 && mapFlag.ptr<uint8_t>(curY)[curX - 1] == 0) {
				distance = pow(abs(curY - srcY), 2) + pow(abs(curX - srcX), 2);

				if (distance <= cell_radius2) {
					mapCache.ptr<float>(curY)[curX - 1] = sqrt(distance) * res;
					mapFlag.ptr<uint8_t>(curY)[curX - 1] = 1;

					structCache temp;
					temp.srcY = srcY;
					temp.srcX = srcX;
					temp.curY = curY;
					temp.curX = curX - 1;
					candidates.push_back(temp);
					pnt_end += 1;
				}
			}

			if (curY < height - 1 && mapFlag.ptr<uint8_t>(curY + 1)[curX] == 0) {
				distance = pow(abs(curY - srcY), 2) + pow(abs(curX - srcX), 2);

				if (distance <= cell_radius2) {
					mapCache.ptr<float>(curY + 1)[curX] = sqrt(distance) * res;
					mapFlag.ptr<uint8_t>(curY + 1)[curX] = 1;

					structCache temp;
					temp.srcY = srcY;
					temp.srcX = srcX;
					temp.curY = curY + 1;
					temp.curX = curX;
					candidates.push_back(temp);
					pnt_end += 1;
				}
			}

			if (curX < width - 1 && mapFlag.ptr<uint8_t>(curY)[curX + 1] == 0) {
				distance = pow(abs(curY - srcY), 2) + pow(abs(curX - srcX), 2);

				if (distance <= cell_radius2) {
					mapCache.ptr<float>(curY)[curX + 1] = sqrt(distance) * res;
					mapFlag.ptr<uint8_t>(curY)[curX + 1] = 1;
					
					structCache temp;
					temp.srcY = srcY;
					temp.srcX = srcX;
					temp.curY = curY;
					temp.curX = curX + 1;
					candidates.push_back(temp);
					pnt_end += 1;
				}
			}

			pnt_now += 1;
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
		Mat auxImage = Mat::zeros(yLim, newXLim, CV_32FC1);
		Mat newImage = Mat::zeros(newYLim, newXLim, CV_32FC1);
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
				auxImage.ptr<float>(y)[x] = round(newVal);
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
					newVal += auxImage.ptr<float>(j)[x] * kerVal[i];
				}
				newImage.ptr<float>(y)[x] = newVal;
			}
		}//end for��y���������

		return newImage;
	}

	LSD::structRegionGrower LSD::RegionGrower(const int x, const int y, double regDeg, const double degThre) {
		// ����
		// x��        ��ʼ��X������
		// y��        ��ʼ��Y������
		// regDeg��   ����level - line�������
		// degThre:   �Ƕ���ֵ
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
		vector<int> regPts_x, regPts_y;
		regPts_x.push_back(x);
		regPts_y.push_back(y);

		double sinDeg = sin(regDeg), cosDeg = cos(regDeg), curDeg, degDif;
		Mat curMap = Mat::zeros(newMapRow, newMapCol, CV_8UC1);
		curMap.ptr<uint8_t>(y)[x] = 1;
		int pntEnd = 1, pntNow, m, n;

		for (pntNow = 0; pntNow < pntEnd; pntNow++) {
			// ����8���������Ƿ�����ǻ�����ֵ
			for (m = regPts_y[pntNow] - 1; m <= regPts_y[pntNow] + 1; m++) {
				for (n = regPts_x[pntNow] - 1; n <= regPts_x[pntNow] + 1; n++) {
					// �������ֵ��״̬
					if (m >= 0 && n >= 0 && m < newMapRow && n < newMapCol) {
						if (curMap.ptr<uint8_t>(m)[n] != 1 && usedMap.ptr<uint8_t>(m)[n] != 1) {
							// ����ǵ�ǰ����������ֵ ���� ��ǰ���ȼ�pi������ֵ
							curDeg = degMap.ptr<float>(m)[n];
							degDif = abs(regDeg - curDeg);
							if (degDif > pi1_5)
								degDif = abs(degDif - pi2);

							if (degDif < degThre) {
								// ����ͳ�����û��ȵ����Һ�����ֵ
								cosDeg += cos(curDeg);
								sinDeg += sin(curDeg);
								regDeg = atan2(sinDeg, cosDeg);
								// ��¼��ǰ����
								curMap.ptr<uint8_t>(m)[n] = 1;
								regPts_x.push_back(n);
								regPts_y.push_back(m);
								pntEnd++;
							}
						}
					}
				}
			}
		}

		structReg reg;
		reg.x = x;
		reg.y = y;
		reg.num = pntEnd;
		reg.deg = regDeg;
		reg.regPts_x = regPts_x;
		reg.regPts_y = regPts_y;

		structRegionGrower RG;
		RG.curMap = curMap;
		RG.reg = reg;

		return RG;
	}

	LSD::structCenterGetter LSD::CenterGetter(const int regNum, const vector<int> regX, const vector<int> regY) {
		// ���룺
		// regNum��  ����������ص���
		// regX��    ���������ص�x��������
		// regY��    ���������ص�y��������
		//
		// �����
		// cenX��    ��������x����
		// cenY��    ��������y����
		//
		// �������ܣ�����������������ص��Ȩ�أ��ҵ���������
		double cenX = 0, cenY = 0, weiSum = 0, pixWei;
		for (int k = 0; k < regNum; k++) {
			pixWei = magMap.ptr<float>(regY[k])[regX[k]];
			cenX += pixWei * regX[k];
			cenY += pixWei * regY[k];
			weiSum += pixWei;
		}
		structCenterGetter CG;
		CG.cenX = cenX / weiSum;
		CG.cenY = cenY / weiSum;

		return CG;
	}

	double LSD::OrientationGetter(const structReg reg, const double cenX, const double cenY, vector<int> regX, vector<int> regY, const double degThre) {
		// ���룺
		// reg��     ����ṹ��
		// cenX��    ��������X����
		// cenY��    ��������Y����
		// regX��    ֱ��֧�������и����X����
		// regY��    ֱ��֧�������и����Y����
		// degThre:  �Ƕ���ֵ
		//
		// �������ܣ���ȡ�����������᷽��Ľǻ���ֵ��

		double Ixx = 0, Iyy = 0, Ixy = 0, weiSum = 0, pixWei;
		// ��������������Ϊ���η���
		for (int i = 0; i < reg.num; i++) {
			pixWei = magMap.ptr<float>(reg.regPts_y[i])[reg.regPts_x[i]];
			Ixx += pixWei * pow(reg.regPts_y[i] - cenY, 2);
			Iyy += pixWei * pow(reg.regPts_x[i] - cenX, 2);
			Ixy -= pixWei * (reg.regPts_x[i] - cenX) * (reg.regPts_y[i] - cenY);
			weiSum += pixWei;
		}
		Ixx /= weiSum;
		Iyy /= weiSum;
		Ixy /= weiSum;

		const double lamb = (Ixx + Iyy - sqrt(pow(Ixx - Iyy, 2) + 4 * pow(Ixy, 2))) / 2.0;
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
		// reg��     ָʾ����Ľṹ��
		// degThre:  �Ƕ���ֵ
		//
		// ���
		// rec��     ��õľ��νṹ��
		// .x1��     ���ζ̱�ĳһ���е�X����
		// .y1��     ���ζ̱�ĳһ���е�Y����
		// .x2��     ���ζ̱���һ���е�X����
		// .y2��     ���ζ̱�ĳһ���е�Y����
		// .wid��    ���ζ̱߳���
		// .cX��     ��������X����
		// .cY��     ��������Y����
		// .deg��    ����������ǻ���
		// .dx��     ����������ǻ�������ֵ
		// .dy��     ����������ǻ�������ֵ
		// .p��      �����ڵ�level - line���ǻ��������������ǻ���������ʣ�
		// .prec��   �жϾ����ڵ�level - line���ǻ��������������ǻ��ȵ���ֵ���Ƕ����̶ȣ�
		//
		// �������ܣ������߶�֧����Ѱ����С�ڽӾ���

		// �����߶�֧���������
		structCenterGetter CG = CenterGetter(reg.num, reg.regPts_x, reg.regPts_y);
		// ȷ������������
		const double inertiaDeg = OrientationGetter(reg, CG.cenX, CG.cenY, reg.regPts_x, reg.regPts_y, degThre);

		// ȷ�����γ��Ϳ�
		const double dx = cos(inertiaDeg), dy = sin(inertiaDeg);
		double lenMin = 1e10, lenMax = 0, widMin = 1e10, widMax = 0, len, wid;
		
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
		// reg��     ��ǰ����Ľṹ��
		// rec��     ��ǰ�������С��Ӿ��εĽṹ��
		// curMap��  ��ǰ����ͼ
		//
		// �����
		// bool��    ��С�뾶���Ƿ����ҵ����ʾ��ε�ָʾ��
		// curMap��  ��ǰ����ָʾͼ
		// reg��     ��ǰ����ṹ��
		// rec��     ��ǰ�������С��Ӿ��εĽṹ��
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
		double rad = max(rad1, rad2), rad_square;

		while (den < denThre) {
			// ��0.75�������ٶȼ�С�����뾶������ֱ��֧�������е�������
			rad *= 0.75;
			rad_square = rad * rad;
			int i = 0;

			while (i <= RRR.reg.num) {
				if (sqrt(pow(oriX - RRR.reg.regPts_x[i], 2) + pow(oriY - RRR.reg.regPts_y[i], 2)) > rad) {
					RRR.curMap.ptr<uint8_t>(RRR.reg.regPts_y[i])[RRR.reg.regPts_x[i]] = 0;
					RRR.reg.regPts_x[i] = RRR.reg.regPts_x[RRR.reg.num - 1];
					RRR.reg.regPts_y[i] = RRR.reg.regPts_y[RRR.reg.num - 1];
					RRR.reg.regPts_x.pop_back();
					RRR.reg.regPts_y.pop_back();
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
		// reg��      ֱ��֧������Ľṹ��
		// rec��      ֱ��֧���������С��Ӿ��εĽṹ��
		// curMap��   ��ǰ��������ͼ
		//
		// �����
		// bool��     �Ƿ�ɹ�����ָʾ��
		// curMap��   ��ǰ��������ָʾͼ
		// reg��      ��ǰ������������
		// rec��      ��ǰ���������������С��Ӿ���
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
		const double cenDeg = degMap.ptr<float>(oriY)[oriX];
		double difSum = 0, squSum = 0, curDeg, degDif, pntNum = 0, wid_square = pow(RF.rec.wid, 2);
		// ����������������ʼ�����С�ھ��ο�ȵ����ؽ�����������ֵ�ع���
		for (int i = 0; i < RF.reg.num; i++) {
			if (pow(oriX - RF.reg.regPts_x[i], 2) + pow(oriY - RF.reg.regPts_y[i], 2) < wid_square) {
				curDeg = degMap.ptr<float>(RF.reg.regPts_y[i])[RF.reg.regPts_x[i]];
				degDif = curDeg - cenDeg;
				while (degDif <= -pi) {
					degDif += pi2;
				}
				while (degDif > pi) {
					degDif -= pi2;
				}
				difSum += degDif;
				squSum += degDif * degDif;
				pntNum++;
			}
		}
		double meanDif = difSum / pntNum;
		double degThre = 2.0 * sqrt((squSum - 2.0 * meanDif * difSum) / pntNum + pow(meanDif, 2));
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
		// rec��     ��ǰ������С��Ӿ��εĽṹ��
		// degMap��  ˮ׼�߽ǻ���ͼ
		// logNT��   ���������Ķ���ֵ
		//
		// �����
		// logNFA��  �龯������Ȼ����ֵ
		//
		// �������ܣ�������������������������ģ�͵��龯��
		int allPixNum = 0, aliPixNum = 0;
		// ����0~pi�Ļ���֮�俼�Ǿ��νǶ�
		int cnt_col, cnt_row;
		for (cnt_row = 0; cnt_row < newMapRow; cnt_row++) {
			for (cnt_col = 0; cnt_col < newMapCol; cnt_col++) {
				if (degMap.ptr<float>(cnt_row)[cnt_col] > pi)
					degMap.ptr<float>(cnt_row)[cnt_col] -= pi;
			}
		}
		// �ҵ������ĸ����������
		structRecVer recVer;
		double verX[4], verY[4], wid_half = rec.wid / 2.0;
		verX[0] = rec.x1 - rec.dy * wid_half;
		verX[1] = rec.x2 - rec.dy * wid_half;
		verX[2] = rec.x2 + rec.dy * wid_half;
		verX[3] = rec.x1 + rec.dy * wid_half;
		verY[0] = rec.y1 + rec.dx * wid_half;
		verY[1] = rec.y2 + rec.dx * wid_half;
		verY[2] = rec.y2 - rec.dx * wid_half;
		verY[3] = rec.y1 - rec.dx * wid_half;
		// ��xֵ��С�ĵ���Ϊ1�ŵ㣬Ȼ����ʱ�����������
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
		// ͳ�Ƶ�ǰ������������������᷽����ͬ��С�ڽǶ����̶ȣ������ص����� aliPixNum
		// �������������ص��� allPixNum
		int xRang_len = abs(ceil(recVer.verX[0]) - floor(recVer.verX[2])) + 1;

		vector<int> xRang;
		for (i = 0; i < xRang_len; i++) {
			xRang.push_back(i + ceil(recVer.verX[0]));
		}
		double lineK[4];
		lineK[0] = (recVer.verY[1] - recVer.verY[0]) / (recVer.verX[1] - recVer.verX[0]);
		lineK[1] = (recVer.verY[2] - recVer.verY[1]) / (recVer.verX[2] - recVer.verX[1]);
		lineK[2] = (recVer.verY[2] - recVer.verY[3]) / (recVer.verX[2] - recVer.verX[3]);
		lineK[3] = (recVer.verY[3] - recVer.verY[0]) / (recVer.verX[3] - recVer.verX[0]);

		vector<int> yLow, yHigh;
		// yLow
		int cnt_yArry = 0;
		for (i = 0; i < xRang_len; i++) {
			if (xRang[i] < recVer.verX[3])
				yLow.push_back(ceil(recVer.verY[0] + (xRang[i] - recVer.verX[0]) * lineK[3]));
			
		}
		for (i = 0; i < xRang_len; i++) {
			if (xRang[i] >= recVer.verX[3])
				yLow.push_back(ceil(recVer.verY[3] + (xRang[i] - recVer.verX[3]) * lineK[2]));
		}
		// yHigh
		cnt_yArry = 0;
		for (i = 0; i < xRang_len; i++) {
			if (xRang[i] < recVer.verX[1])
				yHigh.push_back(floor(recVer.verY[0] + (xRang[i] - recVer.verX[0]) * lineK[0]));
		}
		for (i = 0; i < xRang_len; i++) {
			if (xRang[i] >= recVer.verX[1])
				yHigh.push_back(floor(recVer.verY[1] + (xRang[i] - recVer.verX[1]) * lineK[1]));
		}

		for (i = 0; i < xRang_len; i++) {
			for (j = yLow[i]; j <= yHigh[i]; j++) {
				if ((xRang[i] >= 0) && (xRang[i] < newMapCol) && (j >= 0) && (j < newMapRow)) {
					allPixNum++;
					double degDif = abs(rec.deg - degMap.ptr<float>(j)[xRang[i]]);
					if (degDif > pi1_5)
						degDif = abs(degDif - pi2);
					if (degDif < rec.prec)
						aliPixNum++;
				}
			}
		}
		// ����NFA����Ȼ����ֵ
		const double aliThre = allPixNum * (coefA * pow(allPixNum, coefB) + coefC);
		double logNFA2 = -1;
		if (aliPixNum > aliThre)
			logNFA2 = 1.0 * aliPixNum / allPixNum;

		return logNFA2;
	}

	LSD::structRectangleImprover LSD::RectangleImprover(structRec rec) {
		// ���룺
		// rec��     ��ǰ���νṹ��
		//
		// �����
		// logNFA��  �龯������Ȼ����ֵ
		// rec��     ������ľ��νṹ��
		//
		// �������ܣ������龯��(NFA, Number of False Alarms)����������С��Ӿ���
		structRectangleImprover RI;
		RI.logNFA = RectangleNFACalculator(rec);
		RI.rec = rec;
		return RI;
	}
}