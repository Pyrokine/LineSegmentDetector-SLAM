#include <myLSD.h>

using namespace cv;
using namespace std;

namespace mylsd {
	// LineSegmentDetector
	LSD::structLSD LSD::myLineSegmentDetector(Mat& MapGray, const int _oriMapCol, const int _oriMapRow, const float _sca, const float _sig,
			const float _angThre, const float _denThre, const int _pseBin){
		oriMapCol = _oriMapCol, oriMapRow = _oriMapRow;
		sca = _sca, sig = _sig, angThre = _angThre, denThre = _denThre, pseBin = _pseBin;

		//double last_time, last_time2 = clock();
		newMapCol = floor(oriMapCol * sca);
		newMapRow = floor(oriMapRow * sca);
		// 格式化地图
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

		// 图像缩放——高斯降采样
		//last_time = clock();
		//Mat GaussImage = GaussianSampler(MapGray, sca, sig); // 这个返回的是float型，同时可以自由设定缩放比例
		Mat GaussImage;
		pyrDown(MapGray, GaussImage, Size(ceil(oriMapCol / 2.0), ceil(oriMapRow / 2.0))); // 图像金字塔，2的幂次倍缩放（应该是）
		//printf("01 %lf\n", (clock() - last_time) / CLOCKS_PER_SEC);

#ifdef drawPicture
		imshow("GaussImage", GaussImage);
		//waitKey(0);
#endif

		// usedMap中0表示待生长的栅格，1表示梯度过小不生长的栅格，2表示虚警的栅格，可生长但不能作为初始生长点
		// 因为绝大多数为1，因此稀疏矩阵储存0和2
		for (int i = 0; i < newMapRow; i++) {
			unordered_map<int, int> tempUsedMap;
			unordered_map<int, float> tempDegMap;
			usedMap2.push_back(tempUsedMap);
			degMap2.push_back(tempDegMap);
		}
		curMap2 = usedMap2; // 格式一样，直接深拷贝
		magMap2 = degMap2;

		//Mat degMap = Mat::zeros(newMapRow, newMapCol, CV_32FC1);//level-line场方向
		//Mat magMap = Mat::zeros(newMapRow, newMapCol, CV_32FC1);//记录每点的梯度
		float degThre = angThre / 180.0f * pi; // 角度阈值
		gradThre = 2.0f / sin(degThre); // 梯度阈值

		// 计算梯度和level-line场方向并储存梯度到容器
		//last_time = clock();
		vector<nodeBinCell> binCell;
		float maxGrad = 0, gradX, gradY, valueMagnitude, valueDegree, A, B, C, D;
		for (int y = 1; y < newMapRow; y++) {
			for (int x = 1; x < newMapCol; x++) {
				A = GaussImage.ptr<uint8_t>(y)[x];
				B = GaussImage.ptr<uint8_t>(y)[x - 1];
				C = GaussImage.ptr<uint8_t>(y - 1)[x];
				D = GaussImage.ptr<uint8_t>(y - 1)[x - 1];
				gradX = (B + D - A - C) / 2.0f;
				gradY = (C + D - A - B) / 2.0f;

				valueMagnitude = sqrtf(powf(gradX, 2) + powf(gradY, 2));
				//magMap.ptr<float>(y)[x] = valueMagnitude;
				if (abs(valueMagnitude) > 0.000001f) {
					magMap2[y][x] = valueMagnitude;
				}
				
				if (valueMagnitude >= gradThre) {
					usedMap2[y][x] = 0;
				}

				maxGrad = max(maxGrad, valueMagnitude);
				valueDegree = atan2(gradX, -gradY);
				if (abs(valueDegree - pi) < 0.000001f)
					valueDegree = 0;
				
				if (abs(valueDegree) > 0.000001f) {
					// 仅在0~pi的弧度之间考虑矩形角度
					if (valueDegree > pi)
						valueDegree -= pi;
					degMap2[y][x] = valueDegree;
				}
				
				//degMap.ptr<float>(y)[x] = valueDegree;

				nodeBinCell tempNode;
				tempNode.value = valueMagnitude;
				tempNode.x = x;
				tempNode.y = y;
				binCell.push_back(tempNode);
			}
		}
		GaussImage.release();
		//for (int i = 0; i < newMapRow; i++) {
		//	printf("%d\n", (int)degMap2[i].size());
		//}

#ifdef drawPicture
		//imshow("degMap", degMap);
		//imshow("magMap", magMap);
		//waitKey(0);
#endif 

		//printf("02 %lf\n", (clock() - last_time) / CLOCKS_PER_SEC);

		// 梯度值从大到小排序
		//last_time = clock();
		sort(binCell.begin(), binCell.end(), compVector());
		//printf("03 %lf\n", (clock() - last_time) / CLOCKS_PER_SEC);

		logNT = 5.0f * (log10f(newMapRow) + log10f(newMapCol)) / 2.0f;// 测试数量的对数值
		regThre = -logNT / log10f(angThre / 180.0f); // 小区域的阈值
		aliPro = angThre / 180.0f;

#ifdef drawPicture
		Mat lineIm = Mat::zeros(oriMapRow, oriMapCol, CV_8UC1);// 记录直线灰白图像
		Mat lineImColor = Mat::zeros(oriMapRow, oriMapCol, CV_8UC3);// 记录直线彩色图像
#endif 

		//printf("1 %lf\n", (clock() - last_time2) / CLOCKS_PER_SEC);
		//last_time2 = clock();

		// 记录生长区域和矩形
		vector<structRec> recSave;

		// 按照排序顺序，依次搜索种子像素，从最大梯度开始增长
		double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
		int regCnt = 0, yIdx, xIdx;
		for (int i = 0; i < binCell.size(); i++) {
			yIdx = binCell[i].y;
			xIdx = binCell[i].x;
			if (usedMap2[yIdx].find(xIdx) == usedMap2[yIdx].end() || usedMap2[yIdx][xIdx] == 2)
				continue;

			// 区域增长 返回curMap和reg
			double last_time = clock();
			structRegionGrower RG = RegionGrower(xIdx, yIdx, FetchDegMapValue(yIdx, xIdx), degThre);
			t1 += clock() - last_time;

			structReg reg = RG.reg;
			// 删除小区域
			if (reg.num < regThre)
				continue;

			// 矩阵近似 返回rec
			last_time = clock();
			structRec rec = RectangleConverter(reg, degThre);
			t2 += clock() - last_time;

			// 根据密度阈值，调整区域 返回boolean, curMap, rec, reg
			last_time = clock();
			structRefiner RF = Refiner(reg, rec, RG.curMap);
			t3 += clock() - last_time;
			reg = RF.reg;
			rec = RF.rec;
			if (!RF.boolean) {
				continue;
			}

			// 矩形调整 返回logNFA, rec
			last_time = clock();
			structRectangleImprover RI = RectangleImprover(rec);
			rec = RI.rec;

			if (RI.logNFA <= 0) {
				for (int y = 0; y < newMapRow; y++) {
					for (auto it = RF.curMap[y].begin(); it != RF.curMap[y].end(); it++) {
						usedMap2[y][it->first] = 2;
					}
				}
				
				t4 += clock() - last_time;
				continue;
			}
			t4 += clock() - last_time;

			last_time = clock();
			// 根据缩放尺度重新调整图像中所找到的直线信息
			if (sca != 1) {
				rec.x1 = (rec.x1 - 1.0f) / sca + 1.0f;
				rec.y1 = (rec.y1 - 1.0f) / sca + 1.0f;
				rec.x2 = (rec.x2 - 1.0f) / sca + 1.0f;
				rec.y2 = (rec.y2 - 1.0f) / sca + 1.0f;
				rec.wid = (rec.wid - 1.0f) / sca + 1.0f;
			}

			for (int y = 0; y < newMapRow; y++) {
				for (auto it = RF.curMap[y].begin(); it != RF.curMap[y].end(); it++) {
					usedMap2[y].erase(it->first);
				}
			}

			// 保存所找到的直线支持区域和拟合矩形
			recSave.push_back(rec);
			regCnt++;
			t5 += clock() - last_time;
		}
		printf("%lf, %lf, %lf, %lf, %lf\n", t1 / CLOCKS_PER_SEC, t2 / CLOCKS_PER_SEC, t3 / CLOCKS_PER_SEC, t4 / CLOCKS_PER_SEC, t5 / CLOCKS_PER_SEC);
		//printf("2 %lf\n", (clock() - last_time2) / CLOCKS_PER_SEC);

		// 将所提取到的直线按照像素点标记在图像矩阵中
		Vec3b color;
		vector<structLinesInfo> linesInfo(regCnt);

		float x1, y1, x2, y2, k, ang;
		int orient;
		for (int i = 0; i < regCnt; i++) {
			// 获得直线的端点坐标
			x1 = recSave[i].x1;
			y1 = recSave[i].y1;
			x2 = recSave[i].x2;
			y2 = recSave[i].y2;

			// 求取直线斜率
			k = (y2 - y1) / (x2 - x1);
			ang = atand(k);
			orient = 1;
			if (ang < 0) {
				ang += 180.0;
				orient = -1;
			}

#ifdef drawPicture
			// 确定直线X坐标轴和Y坐标轴的跨度
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

			float xRang = abs(x2 - x1), yRang = abs(y2 - y1);
			// 确定直线跨度较大的坐标轴作为采样主轴并采样，标记直线像素
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
			linesInfo[i].b = (y1 + y2) / 2.0f - k * (x1 + x2) / 2.0f;
			linesInfo[i].dx = cosd(ang);
			linesInfo[i].dy = sind(ang);
			linesInfo[i].x1 = x1;
			linesInfo[i].y1 = y1;
			linesInfo[i].x2 = x2;
			linesInfo[i].y2 = y2;
			linesInfo[i].len = sqrtf(powf(y2 - y1, 2) + powf(x2 - x1, 2));
			linesInfo[i].orient = orient;
		}
		structLSD returnLSD;
		returnLSD.linesInfo = linesInfo;
		returnLSD.len_linesInfo = regCnt;

#ifdef drawPicture
		returnLSD.lineIm = lineIm;
		imshow("lineImColor", lineImColor);
#endif

		return move(returnLSD);
	}

	float LSD::FetchDegMapValue(const int y, const int x) {
		if (degMap2[y].find(x) != degMap2[y].end())
			return degMap2[y][x];
		else
			return 0;
	}

	float LSD::FetchMagMapValue(const int y, const int x) {
		if (magMap2[y].find(x) != magMap2[y].end())
			return magMap2[y][x];
		else
			return 0;
	}

	Mat LSD::createMapCache(const Mat& MapGray, const float res) {
		//计算图中点到最近点的最小距离，在特征匹配时用作先验概率
		const int cell_radius2 = powf(floor(z_occ_max_dis / res), 2);
		const int height = MapGray.rows, width = MapGray.cols;
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

		float pnt_now = 0, pnt_end = candidates.size(), distance;
		int srcY, srcX, curY, curX;
		while (pnt_now < pnt_end) {
			srcY = candidates[pnt_now].srcY, srcX = candidates[pnt_now].srcX;
			curY = candidates[pnt_now].curY, curX = candidates[pnt_now].curX;

			if (curY >= 1 && mapFlag.ptr<uint8_t>(curY - 1)[curX] == 0) {
				distance = powf(abs(curY - srcY), 2) + powf(abs(curX - srcX), 2);

				if (distance <= cell_radius2) {
					mapCache.ptr<float>(curY - 1)[curX] = sqrtf(distance) * res;
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
				distance = powf(abs(curY - srcY), 2) + powf(abs(curX - srcX), 2);

				if (distance <= cell_radius2) {
					mapCache.ptr<float>(curY)[curX - 1] = sqrtf(distance) * res;
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
				distance = powf(abs(curY - srcY), 2) + powf(abs(curX - srcX), 2);

				if (distance <= cell_radius2) {
					mapCache.ptr<float>(curY + 1)[curX] = sqrtf(distance) * res;
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
				distance = powf(abs(curY - srcY), 2) + powf(abs(curX - srcX), 2);

				if (distance <= cell_radius2) {
					mapCache.ptr<float>(curY)[curX + 1] = sqrtf(distance) * res;
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

		return move(mapCache);
	}

	Mat LSD::GaussianSampler(const Mat& image) {
		// 输入
		// sca;   缩放尺度
		// sig:   高斯模板的标准差
		//
		// 输出
		// newIm: 经过高斯采样缩放后的图像
		const int prec = 3, xLim = image.cols, yLim = image.rows;
		const int newXLim = floor(xLim * sca), newYLim = floor(yLim * sca);
		Mat auxImage = Mat::zeros(yLim, newXLim, CV_32FC1);
		Mat newImage = Mat::zeros(newYLim, newXLim, CV_32FC1);
		
		// 如果是缩小图像则调整标准差的值
		if (sca < 1.0)
			sig = sig / sca;
		
		// 高斯模板大小
		const int h = ceil(sig * sqrt(2.0 * prec * log(10)));
		const int hSize = 1 + 2 * h, douXLim = xLim * 2, douYLim = yLim * 2;

		int x, y, i;
		// x方向采样
		for (x = 0; x < newXLim; x++) {
			const float xx = x / sca;
			const int xc = floor(xx + 0.5);
			// 确定高斯核中心位置
			const float kerMean = h + xx - xc;
			vector<float> kerVal(hSize);
			float kerSum = 0;

			// 求当前高斯核（疑似有规律可循 不需反复计算 后面再优化）
			for (i = 0; i < hSize; i++) {
				kerVal[i] = exp(-0.5 * pow((i - kerMean) / sig, 2));
				kerSum += kerVal[i];
			}
			// 高斯核归一化
			for (i = 0; i < hSize; i++) {
				kerVal[i] /= kerSum;
			}
			// 用边缘对称的方式进行X坐标高斯滤波
			for (y = 0; y < yLim; y++) {
				float newVal = 0;
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
		}

		// y方向采样
		for (y = 0; y < newYLim; y++) {
			const float yy = y / sca;
			const int yc = floor(yy + 0.5);
			// 确定高斯核中心位置
			const float kerMean = h + yy - yc;
			vector<float> kerVal(hSize);
			float kerSum = 0;
			// 求当前高斯核
			for (i = 0; i < hSize; i++) {
				kerVal[i] = exp(-0.5 * pow((i - kerMean) / sig, 2));
				kerSum += kerVal[i];
			}
			// 高斯核归一化
			for (i = 0; i < hSize; i++) {
				kerVal[i] /= kerSum;
			}
			// 用边缘对称的方式进行Y坐标高斯滤波
			for (x = 0; x < newXLim; x++)
			{
				float newVal = 0;
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
		}

		return move(newImage);
	}

	LSD::structRegionGrower LSD::RegionGrower(const int x, const int y, float regDeg, const float degThre) {
		// 输入
		// x：        起始点X轴坐标
		// y：        起始点Y轴坐标
		// regDeg：   区域level - line场的倾角
		// degThre:   角度阈值
		//
		// 输出
		// curMap：   根据所给种子点生长所得区域指示图
		// reg：      当前区域所包含的所有信息
		// .x：       初始点X坐标 
		// .y：       初始点Y坐标
		// .num：     区域所包含像素
		// .deg：     区域平均角弧度
		// .pts：     区域所包含像素所有坐标值
		//
		// 函数功能： 通过合并相同方向的level - line场实现区域增长
		vector<int> regPts_x, regPts_y;
		regPts_x.push_back(x);
		regPts_y.push_back(y);

		float sinDeg = sin(regDeg), cosDeg = cos(regDeg), curDeg, degDif;
		vector<unordered_map<int, int>> curMap = curMap2;
		curMap[y][x] = 1;
		int pntEnd = 1, pntNow, m, n;

		for (pntNow = 0; pntNow < pntEnd; pntNow++) {
			// 检验8邻域像素是否满足角弧度阈值
			for (m = regPts_y[pntNow] - 1; m <= regPts_y[pntNow] + 1; m++) {
				for (n = regPts_x[pntNow] - 1; n <= regPts_x[pntNow] + 1; n++) {
					// 检查像素值的状态
					if (m >= 0 && n >= 0 && m < newMapRow && n < newMapCol) {
						if ((curMap[m].find(n) == curMap[m].end()) && (usedMap2[m].find(n) != usedMap2[m].end())) {
							// 检查是当前弧度满足阈值 或是 当前弧度减pi满足阈值
							curDeg = FetchDegMapValue(m, n);
							degDif = abs(regDeg - curDeg);
							if (degDif > pi1_5)
								degDif = abs(degDif - pi2);

							if (degDif < degThre) {
								// 更新统计所得弧度的正弦和余弦值
								cosDeg += cos(curDeg);
								sinDeg += sin(curDeg);
								regDeg = atan2(sinDeg, cosDeg);
								// 记录当前像素
								curMap[m][n] = 1;
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
		reg.regPts_x = move(regPts_x);
		reg.regPts_y = move(regPts_y);

		structRegionGrower RG;
		RG.curMap = curMap;
		RG.reg = reg;

		return move(RG);
	}

	LSD::structCenterGetter LSD::CenterGetter(const int regNum, const vector<int>& regX, const vector<int>& regY) {
		// 输入：
		// regNum：  区域包含像素点数
		// regX：    区域内像素点x坐标向量
		// regY：    区域内像素点y坐标向量
		//
		// 输出：
		// cenX：    区域重心x坐标
		// cenY：    区域重心y坐标
		//
		// 函数功能：根据区域坐标和像素点的权重，找到区域重心
		float cenX = 0, cenY = 0, weiSum = 0, pixWei;
		for (int k = 0; k < regNum; k++) {
			pixWei = FetchMagMapValue(regY[k], regX[k]);
			cenX += pixWei * regX[k];
			cenY += pixWei * regY[k];
			weiSum += pixWei;
		}
		structCenterGetter CG;
		CG.cenX = cenX / weiSum;
		CG.cenY = cenY / weiSum;

		return move(CG);
	}

	float LSD::OrientationGetter(const structReg& reg, const float cenX, const float cenY, const float degThre) {
		// 输入：
		// reg：     区域结构体
		// cenX：    矩形重心X坐标
		// cenY：    矩形重心Y坐标
		// degThre:  角度阈值
		//
		// 函数功能：求取区域主惯性轴方向的角弧度值。

		float Ixx = 0, Iyy = 0, Ixy = 0, weiSum = 0, pixWei;
		// 计算主惯性轴作为矩形方向
		for (int i = 0; i < reg.num; i++) {
			pixWei = FetchMagMapValue(reg.regPts_y[i], reg.regPts_x[i]);
			Ixx += pixWei * powf(reg.regPts_y[i] - cenY, 2);
			Iyy += pixWei * powf(reg.regPts_x[i] - cenX, 2);
			Ixy -= pixWei * (reg.regPts_x[i] - cenX) * (reg.regPts_y[i] - cenY);
			weiSum += pixWei;
		}
		Ixx /= weiSum;
		Iyy /= weiSum;
		Ixy /= weiSum;

		const float lamb = (Ixx + Iyy - sqrtf(powf(Ixx - Iyy, 2) + 4.0f * powf(Ixy, 2))) / 2.0f;
		float inertiaDeg;
		if (abs(Ixx) > abs(Iyy))
			inertiaDeg = atan2(lamb - Ixx, Ixy);
		else
			inertiaDeg = atan2(Ixy, lamb - Iyy);

		// 调整一个pi的误差
		float regDif = inertiaDeg - reg.deg;
		while (regDif <= -pi) {
			regDif += pi2;
		}
		while (regDif > pi) {
			regDif -= pi2;
		}
		regDif = abs(regDif);
		if (regDif > degThre)
			inertiaDeg += pi;

		return move(inertiaDeg);
	}

	LSD::structRec LSD::RectangleConverter(const structReg& reg, const float degThre) {
		// 输入
		// reg：     指示区域的结构体
		// degThre:  角度阈值
		//
		// 输出
		// rec：     获得的矩形结构体
		// .x1：     矩形短边某一段中点X坐标
		// .y1：     矩形短边某一段中点Y坐标
		// .x2：     矩形短边另一段中点X坐标
		// .y2：     矩形短边某一段中点Y坐标
		// .wid：    矩形短边长度
		// .cX：     矩形重心X坐标
		// .cY：     矩形重心Y坐标
		// .deg：    矩形主方向角弧度
		// .dx：     矩形主方向角弧度余弦值
		// .dy：     矩形主方向角弧度正弦值
		// .p：      矩形内点level - line场角弧度与矩形主方向角弧度相符概率；
		// .prec：   判断矩形内点level - line场角弧度与矩形主方向角弧度的阈值（角度容忍度）
		//
		// 函数功能：根据线段支持域寻找最小内接矩阵

		// 计算线段支持域的重心
		structCenterGetter CG = CenterGetter(reg.num, reg.regPts_x, reg.regPts_y);
		// 确定矩形主方向
		const float inertiaDeg = OrientationGetter(reg, CG.cenX, CG.cenY, degThre);

		// 确定矩形长和宽
		const float dx = cos(inertiaDeg), dy = sin(inertiaDeg);
		float lenMin = 1e10, lenMax = 0, widMin = 1e10, widMax = 0, len, wid;
		
		for (int m = 0; m < reg.num; m++) {
			len = (reg.regPts_x[m] - CG.cenX) * dx + (reg.regPts_y[m] - CG.cenY) * dy;
			wid = -(reg.regPts_x[m] - CG.cenX) * dy + (reg.regPts_y[m] - CG.cenY) * dx;
			lenMin = min(len, lenMin);
			lenMax = max(len, lenMax);
			widMin = min(wid, widMin);
			widMax = max(wid, widMax);
		}
		// 保存矩形信息到结构体
		structRec rec;
		rec.x1 = CG.cenX + lenMin * dx;
		rec.y1 = CG.cenY + lenMin * dy;
		rec.x2 = CG.cenX + lenMax * dx;
		rec.y2 = CG.cenY + lenMax * dy;
		rec.wid = max(widMax - widMin, 1.0f);
		rec.cX = CG.cenX;
		rec.cY = CG.cenY;
		rec.deg = inertiaDeg;
		rec.dx = dx;
		rec.dy = dy;
		rec.p = aliPro;
		rec.prec = degThre;

		return move(rec);
	}

	LSD::structRegionRadiusReducer LSD::RegionRadiusReducer(structReg& reg, structRec& rec, vector<unordered_map<int, int>>& curMap) {
		// 输入：
		// reg：     当前区域的结构体
		// rec：     当前区域的最小外接矩形的结构体
		// curMap：  当前区域图
		//
		// 输出：
		// bool：    减小半径后是否能找到合适矩形的指示符
		// curMap：  当前区域指示图
		// reg：     当前区域结构体
		// rec：     当前区域的最小外接矩形的结构体
		//
		// 函数功能：用于减小区域的半径从而减少区域内点数以期望生成更适宜的最小外接矩形。
		structRegionRadiusReducer RRR;
		RRR.curMap = move(curMap);
		RRR.rec = move(rec);
		RRR.reg = move(reg);
		float den = RRR.reg.num / (sqrtf(powf(RRR.rec.x1 - RRR.rec.x2, 2) + powf(RRR.rec.y1 - RRR.rec.y2, 2)) * RRR.rec.wid);
		// 如果满足密度阈值，则直接返回
		if (den > denThre) {
			RRR.boolean = true;
			return RRR;
		}
		// 将原区域生长的初始点作为中心参考点
		const int oriX = RRR.reg.x, oriY = RRR.reg.y;
		// 选取直线最远两端离重心参考点距离中较大值作为搜索半径
		const float rad1 = sqrtf(powf(oriX - RRR.rec.x1, 2) + powf(oriY - RRR.rec.y1, 2));
		const float rad2 = sqrtf(powf(oriX - RRR.rec.x2, 2) + powf(oriY - RRR.rec.y2, 2));
		float rad = max(rad1, rad2), rad_square;

		while (den < denThre) {
			// 以0.75的搜索速度减小搜索半径，减少直线支持区域中的像素数
			rad *= 0.75;
			rad_square = rad * rad;
			int i = 0;

			while (i <= RRR.reg.num) {
				if (sqrtf(powf(oriX - RRR.reg.regPts_x[i], 2) + powf(oriY - RRR.reg.regPts_y[i], 2)) > rad) {
					RRR.curMap[RRR.reg.regPts_y[i]].erase(RRR.reg.regPts_x[i]);
					RRR.reg.regPts_x[i] = RRR.reg.regPts_x[floor(RRR.reg.num - 1.0)];
					RRR.reg.regPts_y[i] = RRR.reg.regPts_y[floor(RRR.reg.num - 1.0)];
					RRR.reg.regPts_x.pop_back();
					RRR.reg.regPts_y.pop_back();
					i--;
					RRR.reg.num--;
				}
				i++;
			}
			// 如果直线支持区域中像素数少于2个，则放弃该区域
			if (RRR.reg.num < 2) {
				RRR.boolean = false;
				return RRR;
			}
			// 将获得的直线支持区域转换为最小外接矩形
			RRR.rec = RectangleConverter(RRR.reg, RRR.rec.prec);
			den = RRR.reg.num / (sqrtf(powf(RRR.rec.x1 - RRR.rec.x2, 2) + powf(RRR.rec.y1 - RRR.rec.y2, 2)) * RRR.rec.wid);
		}
		RRR.boolean = true;
		return move(RRR);
	}

	LSD::structRefiner LSD::Refiner(structReg& reg, structRec& rec, vector<unordered_map<int, int>>& curMap) {
		// 输入：
		// reg：      直线支持区域的结构体
		// rec：      直线支持区域的最小外接矩形的结构体
		// curMap：   当前区域生长图
		//
		// 输出：
		// bool：     是否成功修正指示符
		// curMap：   当前区域生长指示图
		// reg：      当前所生长的区域
		// rec：      当前所生长的区域的最小外接矩形
		//
		// 函数功能： 修正所提取的直线支持区域以及所对应的最小外接矩形
		structRefiner RF;
		RF.curMap = move(curMap);
		RF.rec = move(rec);
		RF.reg = move(reg);
		float den = RF.reg.num / (sqrtf(powf(RF.rec.x1 - RF.rec.x2, 2) + powf(RF.rec.y1 - RF.rec.y2, 2)) * RF.rec.wid);
		// 如果满足密度阈值条件则不用修正
		if (den >= denThre) {
			RF.boolean = true;
			return RF;
		}
		const int oriX = RF.reg.x, oriY = RF.reg.y;
		const float cenDeg = FetchDegMapValue(oriY, oriX);
		float difSum = 0, squSum = 0, curDeg, degDif, pntNum = 0, wid_square = powf(RF.rec.wid, 2);
		// 利用离区域生长初始点距离小于矩形宽度的像素进行区域方向阈值重估计
		for (int i = 0; i < RF.reg.num; i++) {
			if (powf(oriX - RF.reg.regPts_x[i], 2) + powf(oriY - RF.reg.regPts_y[i], 2) < wid_square) {
				curDeg = FetchDegMapValue(RF.reg.regPts_y[i], RF.reg.regPts_x[i]);
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
		float meanDif = difSum / pntNum;
		float degThre = 2.0f * sqrtf((squSum - 2.0f * meanDif * difSum) / pntNum + powf(meanDif, 2));
		// 利用新阈值重新进行区域生长
		structRegionGrower RG = RegionGrower(oriX, oriY, cenDeg, degThre);
		RF.curMap = move(RG.curMap);
		RF.reg = move(RG.reg);
		// 如果由于新阈值导致生长区域过小则丢弃当前区域
		if (RF.reg.num < 2) {
			RF.boolean = false;
			return RF;
		}
		// 重新建立最小外接矩形
		RF.rec = RectangleConverter(RF.reg, degThre);
		den = RF.reg.num / (sqrtf(powf(RF.rec.x1 - RF.rec.x2, 2) + powf(RF.rec.y1 - RF.rec.y2, 2)) * RF.rec.wid);
		// 如果还未满足密度阈值，则减小区域半径
		if (den < denThre) {
			structRegionRadiusReducer RRR = RegionRadiusReducer(RF.reg, RF.rec, RF.curMap);
			RF.boolean = move(RRR.boolean);
			RF.curMap = move(RRR.curMap);
			RF.rec = move(RRR.rec);
			RF.reg = move(RRR.reg);
			return RF;
		}
		RF.boolean = true;
		return move(RF);
	}

	float LSD::RectangleNFACalculator(const structRec& rec) {
		// 输入：
		// rec：     当前区域最小外接矩形的结构体
		//
		// 输出：
		// logNFA：  虚警数的自然对数值
		//
		// 函数功能：计算矩形所在区域相对于噪声模型的虚警数
		int allPixNum = 0, aliPixNum = 0;

		// 找到矩形四个顶点的坐标
		structRecVer recVer;
		float verX[4], verY[4], wid_half = rec.wid / 2.0;
		verX[0] = rec.x1 - rec.dy * wid_half;
		verX[1] = rec.x2 - rec.dy * wid_half;
		verX[2] = rec.x2 + rec.dy * wid_half;
		verX[3] = rec.x1 + rec.dy * wid_half;
		verY[0] = rec.y1 + rec.dx * wid_half;
		verY[1] = rec.y2 + rec.dx * wid_half;
		verY[2] = rec.y2 - rec.dx * wid_half;
		verY[3] = rec.y1 - rec.dx * wid_half;
		// 将x值最小的点作为1号点，然后逆时针排序其余点
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
		// 统计当前矩形中与矩形主惯性轴方向相同（小于角度容忍度）的像素点数量 aliPixNum
		// 矩形内所有像素点数 allPixNum
		const int xRang_len = abs(ceil(recVer.verX[0]) - floor(recVer.verX[2])) + 1;

		vector<float> xRang;
		for (i = 0; i < xRang_len; i++) {
			xRang.push_back(i + ceil(recVer.verX[0]));
		}
		float lineK[4];
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

		float degDif;
		for (i = 0; i < xRang_len; i++) {
			for (j = yLow[i]; j <= yHigh[i]; j++) {
				if ((xRang[i] >= 0) && (xRang[i] < newMapCol) && (j >= 0) && (j < newMapRow)) {
					allPixNum++;
					degDif = abs(rec.deg - FetchDegMapValue(j, xRang[i]));
					if (degDif > pi1_5)
						degDif = abs(degDif - pi2);
					if (degDif < rec.prec)
						aliPixNum++;
				}
			}
		}
		// 计算NFA的自然对数值
		const float aliThre = allPixNum * (coefA * powf(allPixNum, coefB) + coefC);
		float logNFA2 = -1;
		if (aliPixNum > aliThre)
			logNFA2 = 1.0 * aliPixNum / allPixNum;

		return move(logNFA2);
	}

	LSD::structRectangleImprover LSD::RectangleImprover(structRec& rec) {
		// 输入：
		// rec：     当前矩形结构体
		//
		// 输出：
		// logNFA：  虚警数的自然对数值
		// rec：     修正后的矩形结构体
		//
		// 函数功能：利用虚警数(NFA, Number of False Alarms)修正区域最小外接矩形
		structRectangleImprover RI;
		RI.logNFA = RectangleNFACalculator(rec);
		RI.rec = move(rec);
		return move(RI);
	}
}