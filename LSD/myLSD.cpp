#include <myLSD.h>

using namespace cv;
using namespace std;

namespace mylsd {
	//LineSegmentDetector
	structRec *recSaveDisp;
	const double pi = 4.0 * atan(1.0);

	Mat createMapCache(Mat MapGray, double res, double z_occ_max_dis) {
		//计算图中点到最近点的最小距离，在特征匹配时用作先验概率
		int cell_radius = (int)floor(z_occ_max_dis / res);
		int height = MapGray.rows, width = MapGray.cols;
		Mat mapCache = Mat::zeros(height, width, CV_64FC1);
		Mat mapFlag = Mat::zeros(height, width, CV_64FC1);

		structCache *head = (structCache*)malloc(sizeof(structCache));
		structCache *now = head, *tail;
		
		int i, j;
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				if (MapGray.ptr<uint8_t>(i)[j] == 1) {
					structCache *temp = (structCache*)malloc(sizeof(structCache));
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
					structCache *temp = (structCache*)malloc(sizeof(structCache));
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
					structCache *temp = (structCache*)malloc(sizeof(structCache));
					temp->next = NULL;
					tail->src_i = src_i;
					tail->src_j = src_j;
					tail->cur_i = cur_i;
					tail->cur_j = cur_j - 1;
					tail->next = temp;
					tail = temp;
				}
			}

			if (cur_i >= 1 && mapFlag.ptr<double>(cur_i + 1)[cur_j] == 0) {
				double di = abs(cur_i - src_i);
				double dj = abs(cur_j - src_j);
				double distance = sqrt(di * di + dj * dj);

				if (distance <= cell_radius) {
					mapCache.ptr<double>(cur_i + 1)[cur_j] = distance * res;
					mapFlag.ptr<double>(cur_i + 1)[cur_j] = 1;
					structCache *temp = (structCache*)malloc(sizeof(structCache));
					temp->next = NULL;
					tail->src_i = src_i;
					tail->src_j = src_j;
					tail->cur_i = cur_i + 1;
					tail->cur_j = cur_j;
					tail->next = temp;
					tail = temp;
				}
			}

			if (cur_j >= 1 && mapFlag.ptr<double>(cur_i)[cur_j + 1] == 0) {
				double di = abs(cur_i - src_i);
				double dj = abs(cur_j - src_j);
				double distance = sqrt(di * di + dj * dj);

				if (distance <= cell_radius) {
					mapCache.ptr<double>(cur_i)[cur_j + 1] = distance * res;
					mapFlag.ptr<double>(cur_i)[cur_j + 1] = 1;
					structCache *temp = (structCache*)malloc(sizeof(structCache));
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

	structLSD myLineSegmentDetector(Mat MapGray, int oriMapCol, int oriMapRow, double sca, double sig, double angThre, double denThre, int pseBin) {
		int regCnt = 0;
		//图像缩放——高斯降采样
		int newMapCol = (int)floor(oriMapCol * sca);
		int newMapRow = (int)floor(oriMapRow * sca);
		int x, y;
		for (y = 1; y < oriMapRow; y++) {
			for (x = 1; x < oriMapCol; x++) {
				if (MapGray.ptr<uint8_t>(y)[x] == 1)
					MapGray.ptr<uint8_t>(y)[x] = 255;
				else if (MapGray.ptr<uint8_t>(y)[x] == 255)
					MapGray.ptr<uint8_t>(y)[x] = 0;
			}
		}
		Mat GaussImage = GaussianSampler(MapGray, sca, sig);

		Mat usedMap = Mat::zeros(newMapRow, newMapCol, CV_8UC1);//记录像素点状态
		Mat degMap = Mat::zeros(newMapRow, newMapCol, CV_64FC1);//level-line场方向
		Mat magMap = Mat::zeros(newMapRow, newMapCol, CV_64FC1);//记录每点的梯度
		double degThre = angThre / 180.0 * pi;
		double gradThre = 2.0 / sin(degThre);//梯度阈值

		//计算梯度和level-line场方向
		double maxGrad = 0;
		for (y = 1; y < newMapRow; y++) {
			for (x = 1; x < newMapCol; x++) {
				double gradX, gradY, valueMagMap, valueDegMap;
				double A, B, C, D;
				A = GaussImage.ptr<double>(y)[x];
				B = GaussImage.ptr<double>(y)[x - 1];
				C = GaussImage.ptr<double>(y - 1)[x];
				D = GaussImage.ptr<double>(y - 1)[x - 1];
				gradX = (B + D - A - C) / 2.0;
				gradY = (C + D - A - B) / 2.0;
				valueMagMap = sqrt(pow(gradX, 2) + pow(gradY, 2));
				magMap.ptr<double>(y)[x] = valueMagMap;
				if (valueMagMap < gradThre)
					usedMap.ptr<uint8_t>(y)[x] = 1;
				if (maxGrad < valueMagMap)
					maxGrad = valueMagMap;
				valueDegMap = atan2(gradX, -gradY);
				if (abs(valueDegMap - pi) < 0.000001)
					valueDegMap = 0;
				degMap.ptr<double>(y)[x] = valueDegMap;
			}
		}

		//储存梯度值到数组
		int len_binCell = 0;
		Mat pseIdx = Mat::zeros(newMapRow, newMapCol, CV_16UC1);
		double zoom = 1.0 * pseBin / maxGrad;
		for (y = 0; y < newMapRow; y++) {
			for (x = 0; x < newMapCol; x++) {
				int temp_value = (int)floor(magMap.ptr<double>(y)[x] * zoom);
				if (temp_value > pseBin)
					temp_value = pseBin;
				if (temp_value != 0)
					len_binCell++;
				pseIdx.ptr<uint16_t>(y)[x] = temp_value;
			}
		}
		int cnt_binCell = 0;
		nodeBinCell *binCell = (nodeBinCell*)malloc(len_binCell * sizeof(nodeBinCell));
		for (y = 0; y < newMapRow; y++) {
			for (x = 0; x < newMapCol; x++) {
				if (pseIdx.ptr<uint16_t>(y)[x] != 0) {
					binCell[cnt_binCell].value = pseIdx.ptr<uint16_t>(y)[x];
					binCell[cnt_binCell].x = x;
					binCell[cnt_binCell].y = y;
					cnt_binCell++;
				}
			}
		}
		pseIdx.release();
		//梯度值的排序
		qsort(binCell, cnt_binCell, sizeof(nodeBinCell), Comp);

		//按照伪排序的等级 依次搜索种子像素
		double logNT = 5 * (log10(newMapRow) + log10(newMapCol)) / 2.0;//测试数量的对数值
		double regThre = -logNT / log10(angThre / 180.0);//小区域的阈值
		double aliPro = angThre / 180.0;

		//记录生长区域和矩形
		structRec *recSaveHead = (structRec*)malloc(sizeof(structRec));
		structRec *recSaveNow = recSaveHead;
		Mat regIdx = Mat::zeros(newMapRow, newMapCol, CV_8UC1);
		Mat lineIm = Mat::zeros(oriMapRow, oriMapCol, CV_8UC1);//记录直线图像

		//从最大梯度开始增长
		int i = 0;
		for (i = 0; i < len_binCell; i++) {
			int yIdx = binCell[i].y;
			int xIdx = binCell[i].x;
			if (usedMap.ptr<uint8_t>(yIdx)[xIdx] != 0)
				continue;
			//区域增长 返回curMap和reg
			structRegionGrower RG = RegionGrower(xIdx, yIdx, usedMap, degMap.ptr<double>(yIdx)[xIdx], degMap, degThre);
			structReg reg = RG.reg;
			//删除小区域
			if (reg.num < regThre) {
				continue;
			}
			//矩阵近似 返回rec
			structRec rec = RectangleConverter(reg, magMap, aliPro, degThre);
			//根据密度阈值，调整区域 返回boolean, curMap, rec, reg 
			structRefiner RF = Refiner(reg, rec, denThre, degMap, usedMap, RG.curMap, magMap);
			reg = RF.reg;
			rec = RF.rec;
			if (!RF.boolean)
				continue;
			//矩形调整 返回logNFA, rec
			structRectangleImprover RI = RectangleImprover(rec, degMap, logNT);
			rec = RI.rec;
			if (RI.logNFA <= 0) {
				for (y = 0; y < newMapRow; y++) {
					for (x = 0; x < newMapCol; x++) {
						if (RF.curMap.ptr<uint8_t>(y)[x] == 1)
							usedMap.ptr<uint8_t>(y)[x] = 2;
					}
				}
				continue;
			}
			//根据缩放尺度重新调整图像中所找到的直线信息
			if (sca != 1) {
				rec.x1 = (rec.x1 - 1.0) / sca + 1;
				rec.y1 = (rec.y1 - 1.0) / sca + 1;
				rec.x2 = (rec.x2 - 1.0) / sca + 1;
				rec.y2 = (rec.y2 - 1.0) / sca + 1;
				rec.wid = (rec.wid - 1.0) / sca + 1;
			}
			for (y = 0; y < newMapRow; y++) {
				for (x = 0; x < newMapCol; x++) {
					regIdx.ptr<uint8_t>(y)[x] += RF.curMap.ptr<uint8_t>(y)[x] * (regCnt + 1);
					if (RF.curMap.ptr<uint8_t>(y)[x] == 1)
						usedMap.ptr<uint8_t>(y)[x] = 1;
				}
			}
			//保存所找到的直线支持区域和拟合矩形
			structRec *tempRec = (structRec*)malloc(sizeof(structRec));
			recSaveNow[0] = rec;
			recSaveNow[0].next = tempRec;
			recSaveNow = tempRec;
			regCnt++;
		}
		//将recSave链表变成数组
		recSaveNow = recSaveHead;
		structRec *recSave = (structRec*)malloc(regCnt * sizeof(structRec));
		for (i = 0; i < regCnt; i++) {
			recSave[i] = recSaveNow[0];
			recSaveNow = recSaveNow[0].next;
		}
		//将所提取到的直线按照像素点标记在图像矩阵中
		structLinesInfo *linesInfo = (structLinesInfo*)malloc(regCnt * sizeof(structLinesInfo));
		for (i = 0; i < regCnt; i++) {
			//获得直线的端点坐标
			double x1 = recSave[i].x1;
			double y1 = recSave[i].y1;
			double x2 = recSave[i].x2;
			double y2 = recSave[i].y2;
			//求取直线斜率
			double k = (y2 - y1) / (x2 - x1);
			double ang = atand(k);
			int orient = 1;
			if (ang < 0) {
				ang += 180;
				orient = -1;
			}
			//确定直线X坐标轴和Y坐标轴的跨度
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
			//确定直线跨度较大的坐标轴作为采样主轴并采样
			int xx_len = xHigh - xLow + 1, yy_len = yHigh - yLow + 1;
			int *xx, *yy;
			int j;
			if (xRang > yRang) {
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
			//标记直线像素
			if (xx_len > yy_len) {
				for (j = 0; j < xx_len; j++) {
					if (xx[j] != 0 && yy[j] != 0)
						lineIm.ptr<uint8_t>(yy[j])[xx[j]] = 255;
				}
			}
			else {
				for (j = 0; j < yy_len; j++) {
					if (xx[j] != 0 && yy[j] != 0)
						lineIm.ptr<uint8_t>(yy[j])[xx[j]] = 255;
				}
			}
			free(xx);
			free(yy);
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
		returnLSD.lineIm = lineIm;
		returnLSD.linesInfo = linesInfo;
		returnLSD.len_linesInfo = i;

		free(binCell);
		return returnLSD;
	}

	Mat GaussianSampler(Mat image, double sca, double sig) {
		//输入
		//sca; 缩放尺度
		//sig: 高斯模板的标准差
		//输出
		//newIm: 经过高斯采样缩放后的图像
		int prec = 3, xLim = image.cols, yLim = image.rows;
		int newXLim = (int)floor(xLim * sca);
		int newYLim = (int)floor(yLim * sca);
		Mat auxImage = Mat::zeros(yLim, newXLim, CV_64FC1);
		Mat newImage = Mat::zeros(newYLim, newXLim, CV_64FC1);
		//如果是缩小图像则调整标准差的值
		if (sca < 1)
			sig = sig / sca;
		//高斯模板大小
		int h = (int)ceil(sig * sqrt(2 * prec * log(10)));
		int hSize = 1 + 2 * h;
		int douXLim = xLim * 2;
		int douYLim = yLim * 2;

		//求高斯核
		double kerSum1 = 0, kerSum2 = 0, kerSum3 = 0;
		double *kerVal1 = (double*)malloc(hSize * sizeof(double));
		double *kerVal2 = (double*)malloc(hSize * sizeof(double));
		double *kerVal3 = (double*)malloc(hSize * sizeof(double));
		int k = 0;
		for (k = 0; k < hSize; k++) {
			kerVal1[k] = exp(-0.5 * pow((k - h) / sig, 2));
			kerVal2[k] = exp(-0.5 * pow((k - h - 1.0 / 3) / sig, 2));
			kerVal3[k] = exp(-0.5 * pow((k - h + 1.0 / 3) / sig, 2));
			kerSum1 += kerVal1[k];
			kerSum2 += kerVal2[k];
			kerSum3 += kerVal3[k];
		}
		//高斯核归一化
		for (k = 0; k < hSize; k++) {
			kerVal1[k] /= kerSum1;
			kerVal2[k] /= kerSum2;
			kerVal3[k] /= kerSum3;
		}
		//x方向采样
		int x;
		for (x = 0; x < newXLim; x++) {
			double *kerVal;
			if (x % 3 == 0)
				kerVal = kerVal1;
			else if (x % 3 == 1)
				kerVal = kerVal2;
			else
				kerVal = kerVal3;
			int xc = (int)floor(x / sca + 0.5);
			//用边缘对称的方式进行X坐标高斯滤波
			int y;
			for (y = 0; y < yLim; y++) {
				double newVal = 0;
				int i;
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
				auxImage.ptr<double>(y)[x] = newVal;
			}
		}//end for（x方向采样）

		//y方向采样
		int y;
		for (y = 0; y < newYLim; y++) {
			double *kerVal;
			if (y % 3 == 0)
				kerVal = kerVal1;
			else if (y % 3 == 1)
				kerVal = kerVal2;
			else
				kerVal = kerVal3;
			int yc = (int)floor(y / sca + 0.5);
			//用边缘对称的方式进行Y坐标高斯滤波
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
		}//end for（y方向采样）
		return newImage;
	}

	int Comp(const void *p1, const void *p2)
	{
		return(*(nodeBinCell*)p2).value > (*(nodeBinCell*)p1).value ? 1 : -1;
	}

	structRegionGrower RegionGrower(int x, int y, Mat banMap, double regDeg, Mat degMap, double degThre) {
		//输入
		//x：起始点X轴坐标
		//y：起始点Y轴坐标
		//banMap：禁止生长区域指示图 记录像素点的状态
		//regDeg：区域level - line场的倾角
		//degMap：全图level - line场倾角图
		//degThre：level - line场倾角误差阈值 角度容忍度
		//
		//输出
		//curMap为根据所给种子点生长所得区域指示图
		//reg为当前区域所包含的所有信息
		//	.x为初始点X坐标 
		//  .y为初始点Y坐标
		//	.num为区域所包含像素
		//  .deg区域平均角弧度
		//	.pts为区域所包含像素所有坐标值
		//
		//函数功能： 通过合并相同方向的level - line场实现区域增长
		structPts *regPts_now, *regPts_head, *regPts_end;
		regPts_head = regPts_end = regPts_now = (structPts*)malloc(sizeof(structPts));
		regPts_head[0].x = x;
		regPts_head[0].y = y;
		regPts_head[0].next = NULL;
		double sinDeg = sin(regDeg);
		double cosDeg = cos(regDeg);
		int yLim = banMap.rows;
		int xLim = banMap.cols;
		Mat curMap = Mat::zeros(yLim, xLim, CV_8UC1);
		curMap.ptr<uint8_t>(y)[x] = 1;
		int growNum = 1;
		int exNum = 0;
		int isFirstTime = 1;
		int temp = 0;
		while (exNum != growNum) {
			exNum = growNum;
			int i;
			regPts_now = regPts_head;
			for (i = 0; i < growNum; i++) {
				//检验8邻域像素是否满足角弧度阈值
				int m, n;
				int roi_x = regPts_now[0].x, roi_y = regPts_now[0].y;
				for (m = roi_y - 1; m <= roi_y + 1; m++) {
					for (n = roi_x - 1; n <= roi_x + 1; n++) {
						//检查像素值的状态
						if (m >= 0 && n >= 0 && m < yLim && n < xLim) {
							if (curMap.ptr<uint8_t>(m)[n] != 1 && banMap.ptr<uint8_t>(m)[n] != 1) {
								//检查是当前弧度满足阈值 或是 当前弧度减pi满足阈值
								double curDeg = degMap.ptr<double>(m)[n];
								double degDif = abs(regDeg - curDeg);
								if (degDif > pi * 3 / 2.0)
									degDif = abs(degDif - 2.0 * pi);
								if (degDif < degThre) {
									//更新统计所得弧度的正弦和余弦值
									cosDeg += cos(curDeg);
									sinDeg += sin(curDeg);
									regDeg = atan2(sinDeg, cosDeg);
									//记录当前像素
									curMap.ptr<uint8_t>(m)[n] = 1;
									growNum++;
									structPts *temp = (structPts*)malloc(sizeof(structPts));
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
			}//for (i = 0; i < growNum; i++)
		}//while (exNum != growNum)

		int *rePts_x = (int*)malloc(growNum * sizeof(int));
		int *rePts_y = (int*)malloc(growNum * sizeof(int));
		regPts_now = regPts_head;
		int i;
		for (i = 0; i < growNum; i++) {
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

	structCenterGetter CenterGetter(int regNum, int *regX, int *regY, Mat weiMap) {
		//输入：
		//regNum：区域包含像素点数
		//regX：区域内像素点x坐标向量
		//regY：区域内像素点y坐标向量
		//weiMap：区域内像素点权重图
		//
		//输出：
		//cenX：区域重心x坐标
		//cenY：区域重心y坐标
		//
		//函数功能：根据区域坐标和像素点的权重，找到区域重心
		double cenX = 0;
		double cenY = 0;
		double weiSum = 0;
		int k = 0;
		for (k = 0; k < regNum; k++) {
			double pixWei = weiMap.ptr<double>(regY[k])[regX[k]];
			cenX += pixWei * regX[k];
			cenY += pixWei * regY[k];
			weiSum += pixWei;
		}
		structCenterGetter CG;
		CG.cenX = cenX / weiSum;
		CG.cenY = cenY / weiSum;

		return CG;
	}

	double OrientationGetter(structReg reg, double cenX, double cenY, int *regX, \
		int *regY, Mat weiMap, double degThre) {
		//输入：
		//reg：区域结构体
		//cenX：矩形重心X坐标
		//cenY：矩形重心Y坐标
		//regX：直线支持区域中各点的X坐标
		//regY：直线支持区域中各点的Y坐标
		//weiMap：直线支持区域中各点的权重图
		//degThre：弧度阈值
		//
		//函数功能：求取区域主惯性轴方向的角弧度值。

		double Ixx = 0, Iyy = 0, Ixy = 0, weiSum = 0;
		//计算主惯性轴作为矩形方向
		int k;
		for (k = 0; k < reg.num; k++) {
			double pixWei = weiMap.ptr<double>(reg.regPts_y[k])[reg.regPts_x[k]];
			Ixx += pixWei * pow(reg.regPts_y[k] - cenY, 2);
			Iyy += pixWei * pow(reg.regPts_x[k] - cenX, 2);
			Ixy -= pixWei * (reg.regPts_x[k] - cenX) * (reg.regPts_y[k] - cenY);
			weiSum += pixWei;
		}
		Ixx /= weiSum;
		Iyy /= weiSum;
		Ixy /= weiSum;
		double lamb = (Ixx + Iyy - sqrt(pow((Ixx - Iyy), 2) + 4 * Ixy * Ixy)) / 2.0;
		double inertiaDeg;
		if (abs(Ixx) > abs(Iyy))
			inertiaDeg = atan2(lamb - Ixx, Ixy);
		else
			inertiaDeg = atan2(Ixy, lamb - Iyy);

		//调整一个pi的误差
		double regDif = inertiaDeg - reg.deg;
		while (regDif <= -pi) {
			regDif += 2 * pi;
		}
		while (regDif > pi) {
			regDif -= 2 * pi;
		}
		if (regDif < 0)
			regDif = -regDif;
		if (regDif > degThre)
			inertiaDeg += pi;
		return inertiaDeg;
	}

	structRec RectangleConverter(structReg reg, Mat magMap, double aliPro, double degThre) {
		//输入
		//reg：指示区域的结构体
		//magMap：梯度幅值图
		//aliPro：区域内像素梯度与矩形梯度相符的概率
		//degThre：判断角弧度相符的阈值
		//
		//输出
		//rec：获得的矩形结构体
		// .x1：  矩形短边某一段中点X坐标
		// .y1：  矩形短边某一段中点Y坐标
		// .x2：  矩形短边另一段中点X坐标
		// .y2：  矩形短边某一段中点Y坐标
		// .wid： 矩形短边长度
		// .cX：  矩形重心X坐标
		// .cY：  矩形重心Y坐标
		// .deg： 矩形主方向角弧度
		// .dx：  矩形主方向角弧度余弦值
		// .dy：  矩形主方向角弧度正弦值
		// .p：   矩形内点level - line场角弧度与矩形主方向角弧度相符概率；
		// .prec：判断矩形内点level - line场角弧度与矩形主方向角弧度的阈值（角度容忍度）
		//
		//函数功能：根据线段支持域寻找最小内接矩阵

		//计算线段支持域的重心
		structCenterGetter CG = CenterGetter(reg.num, reg.regPts_x, reg.regPts_y, magMap);
		//确定矩形主方向
		double inertiaDeg = OrientationGetter(reg, CG.cenX, CG.cenY, reg.regPts_x, reg.regPts_y, magMap, degThre);

		//确定矩形长和宽
		double dx = cos(inertiaDeg);
		double dy = sin(inertiaDeg);
		double lenMin = 0, lenMax = 0, widMin = 0, widMax = 0;
		int m;
		for (m = 0; m < reg.num; m++) {
			double len = (reg.regPts_x[m] - CG.cenX) * dx + (reg.regPts_y[m] - CG.cenY) * dy;
			double wid = -(reg.regPts_x[m] - CG.cenX) * dy + (reg.regPts_y[m] - CG.cenY) * dx;
			if (len < lenMin)
				lenMin = len;
			if (len > lenMax)
				lenMax = len;
			if (wid < widMin)
				widMin = wid;
			if (wid > widMax)
				widMax = wid;
		}
		//保存矩形信息到结构体
		structRec rec;
		rec.x1 = CG.cenX + lenMin * dx;
		rec.y1 = CG.cenY + lenMin * dy;
		rec.x2 = CG.cenX + lenMax * dx;
		rec.y2 = CG.cenY + lenMax * dy;
		rec.wid = widMax - widMin;
		rec.cX = CG.cenX;
		rec.cY = CG.cenY;
		rec.deg = inertiaDeg;
		rec.dx = dx;
		rec.dy = dy;
		rec.p = aliPro;
		rec.prec = degThre;

		if (rec.wid < 1)
			rec.wid = 1;

		return rec;
	}

	structRegionRadiusReducer RegionRadiusReducer(structReg reg, structRec rec, \
		double denThre, Mat curMap, Mat magMap) {
		//输入：
		//reg：当前区域的结构体
		//rec：当前区域的最小外接矩形的结构体
		//denThre：矩形密度阈值
		//curMap：当前区域图
		//magMap：梯度幅值图
		//
		//输出：
		//bool：减小半径后是否能找到合适矩形的指示符
		//curMap：当前区域指示图
		//reg：当前区域结构体
		//rec：当前区域的最小外接矩形的结构体
		//
		//函数功能：用于减小区域的半径从而减少区域内点数以期望生成更适宜的最小外接矩形。
		structRegionRadiusReducer RRR;
		RRR.boolean = true;
		RRR.curMap = curMap;
		RRR.rec = rec;
		RRR.reg = reg;
		double den = RRR.reg.num / (sqrt(pow(RRR.rec.x1 - RRR.rec.x2, 2) + \
			pow(RRR.rec.y1 - RRR.rec.y2, 2)) * RRR.rec.wid);
		//如果满足密度阈值，则直接返回
		if (den > denThre) {
			RRR.boolean = true;
			return RRR;
		}
		//将原区域生长的初始点作为中心参考点
		int oriX = RRR.reg.x;
		int oriY = RRR.reg.y;
		//选取直线最远两端离重心参考点距离中较大值作为搜索半径
		double rad1 = sqrt(pow(oriX - RRR.rec.x1, 2) + pow(oriY - RRR.rec.y1, 2));
		double rad2 = sqrt(pow(oriX - RRR.rec.x2, 2) + pow(oriY - RRR.rec.y2, 2));
		double rad;
		if (rad1 > rad2)
			rad = rad1;
		else
			rad = rad2;
		while (den < denThre) {
			//以0.75的搜索速度减小搜索半径，减少直线支持区域中的像素数
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
			//如果直线支持区域中像素数少于2个，则放弃该区域
			if (RRR.reg.num < 2) {
				RRR.boolean = false;
				return RRR;
			}
			//将获得的直线支持区域转换为最小外接矩形
			RRR.rec = RectangleConverter(RRR.reg, magMap, RRR.rec.p, RRR.rec.prec);
			den = RRR.reg.num / (sqrt(pow(RRR.rec.x1 - RRR.rec.x2, 2) + pow(RRR.rec.y1 - RRR.rec.y2, 2)) * RRR.rec.wid);
		}
		RRR.boolean = true;
		return RRR;
	}

	structRefiner Refiner(structReg reg, structRec rec, double denThre, \
		Mat degMap, Mat banMap, Mat curMap, Mat magMap) {
		//输入：
		//reg：直线支持区域的结构体
		//rec：直线支持区域的最小外接矩形的结构体
		//denThre：矩形具有内相同方向角弧度的像素的比例阈值
		//degMap：角弧度图
		//banMap：区域生长禁止区域图
		//curMap：当前区域生长图
		//magMap：梯度幅度图
		//
		//输出：
		//bool：是否成功修正指示符
		//curMap：当前区域生长指示图
		//reg：当前所生长的区域
		//rec：当前所生长的区域的最小外接矩形
		//
		//函数功能： 修正所提取的直线支持区域以及所对应的最小外接矩形
		structRefiner RF;
		RF.boolean = true;
		RF.curMap = curMap;
		RF.rec = rec;
		RF.reg = reg;
		double den = RF.reg.num / (sqrt(pow(RF.rec.x1 - RF.rec.x2, 2) + pow(RF.rec.y1 - RF.rec.y2, 2)) * RF.rec.wid);
		//如果满足密度阈值条件则不用修正
		if (den >= denThre) {
			RF.boolean = true;
			return RF;
		}
		int oriX = RF.reg.x;
		int oriY = RF.reg.y;
		double cenDeg = degMap.ptr<double>(oriY)[oriX];
		double difSum = 0, squSum = 0;
		int ptNum = 0, i = 0;
		//利用离区域生长初始点距离小于矩形宽度的像素进行区域方向阈值重估计
		for (i = 0; i < RF.reg.num; i++) {
			if (sqrt(pow(oriX - RF.reg.regPts_x[i], 2) + pow(oriY - RF.reg.regPts_y[i], 2)) < RF.rec.wid) {
				double curDeg = degMap.ptr<double>(RF.reg.regPts_y[i])[RF.reg.regPts_x[i]];
				double degDif = curDeg - cenDeg;
				while (degDif <= -pi) {
					degDif += 2 * pi;
				}
				while (degDif > pi) {
					degDif -= 2 * pi;
				}
				difSum += degDif;
				squSum += degDif * degDif;
				ptNum++;
			}
		}
		double meanDif = difSum / (ptNum * 1.0);
		double degThre = 2.0 * sqrt((squSum - 2 * meanDif * difSum) / (ptNum * 1.0) + meanDif * meanDif);
		//利用新阈值重新进行区域生长
		structRegionGrower RG = RegionGrower(oriX, oriY, banMap, cenDeg, degMap, degThre);
		RF.curMap = RG.curMap;
		RF.reg = RG.reg;
		//如果由于新阈值导致生长区域过小则丢弃当前区域
		if (RF.reg.num < 2) {
			RF.boolean = false;
			return RF;
		}
		//重新建立最小外接矩形
		RF.rec = RectangleConverter(RF.reg, magMap, RF.rec.p, RF.rec.prec);
		den = RF.reg.num / (sqrt(pow(RF.rec.x1 - RF.rec.x2, 2) + pow(RF.rec.y1 - RF.rec.y2, 2)) * RF.rec.wid);
		//如果还未满足密度阈值，则减小区域半径
		if (den < denThre) {
			structRegionRadiusReducer RRR = RegionRadiusReducer(RF.reg, RF.rec, denThre, RF.curMap, magMap);
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

	double LogGammaCalculator(int x) {
		// 分别利用Windschitl方法和Lanczos方法计算Gamma函数的绝对值的自然对数值。
		// 引用：
		// http://www.rskey.org/gamma.htm
		// Windschitl方法：
		// Gamma(x) = sqrt(2 * pi / x) * (x * sqrt(x * sinh(1 / x) + 1 / (810 * x^6)) / e)^x
		// 则取自然对数后为log(Gamma(x)) = 0.5 * log(2 * pi) + (x - 0.5) * log(x) - x +
		// 0.5 * x * log(x * sinh(1 / x) + 1 / (810 * x^6)))
		//
		// Lanczos方法：
		// Gamma(x) = sum{n = 0:N}(q_n * x^n) / prod{n = 0:N}(x + n) * (x + 5.5)^(x + 0.5) * e^(-z - 5.5)
		// 则取自然对数后为log(Gamma(x)) = log(sum{n = 0:N}(q_n * x^n)) + (x + 0.5) * log(x + 5.5) -
		// sum{n = 0:N}(log(x + n))
		// 其中q_0 = 75122.6331530, q_1 = 80916.6278952, q_2 = 36308.2951477,
		// q_3 = 8687.24529705, q_4 = 1168.92649479, q_5 = 83.8676043424, q_6 = 2.50662827511
		//
		// 输入：
		// x：自变量
		//
		// 输出：
		// val：计算得到的Gamma函数自然对数值
		int thre = 15;
		double val;
		//在自变量值大于15的情况下Windschitl近似效果较好
		if (x > thre) {
			//LogGammaWindschitl
			val = 0.918938533204673 + (x - 0.5) * log(x) - x + 0.5 * x *\
				log(x * sinh(1.0 / x) + 1.0 / (810 * pow(x, 6)));
		}
		else {
			//logGammaLanczos;
			double q[7] = { 75122.6331530, 80916.6278952, 36308.2951477, 8687.24529705, 1168.92649479, 83.8676043424, 2.50662827511 };
			double a = (x + 0.5) * log(x + 5.5) - (x + 5.5);
			double b = 0;
			int i;
			for (i = 0; i < 7; i++) {
				a -= log(x + i);
				b += q[i] * pow(x, i);
			}
			val = a + log(b);
		}
		return val;
	}

	double RectangleNFACalculator(structRec rec, Mat degMap, double logNT) {
		// 输入：
		// rec：当前区域最小外接矩形的结构体
		// degMap：水准线角弧度图
		// logNT：测试数量的对数值
		//
		// 输出：
		// logNFA：虚警数的自然对数值
		//
		// 函数功能：计算矩形所在区域相对于噪声模型的虚警数
		int yLim = degMap.rows, xLim = degMap.cols;
		int allPixNum = 0, aliPixNum = 0;
		//仅在0~pi的弧度之间考虑矩形角度
		int cnt_col, cnt_row;
		for (cnt_row = 0; cnt_row < yLim; cnt_row++) {
			for (cnt_col = 0; cnt_col < xLim; cnt_col++) {
				if (degMap.ptr<double>(cnt_row)[cnt_col] > pi)
					degMap.ptr<double>(cnt_row)[cnt_col] -= pi;
			}
		}
		//找到矩形四个顶点的坐标
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
		//将x值最小的点作为1号点，然后逆时针排序其余点
		int offset, i, j;
		if ((rec.x1 < rec.x2) && (rec.y1 <= rec.y2))
			offset = 0;
		else if ((rec.x1 >= rec.x2) && (rec.y1 < rec.y2))
			offset = 1;
		else if ((rec.x1 > rec.x2) && (rec.y1 >= rec.y2))
			offset = 2;
		else
			offset = 3;
		for (i = 0; i < 4; i++) {
			recVer.verX[i] = verX[(offset + i) % 4];
			recVer.verY[i] = verY[(offset + i) % 4];
		}
		//统计当前矩形中与矩形主惯性轴方向相同（小于角度容忍度）的像素点数量 aliPixNum
		//矩形内所有像素点数 allPixNum
		int xRang_len = abs((int)(ceil(recVer.verX[0]) - floor(recVer.verX[2]))) + 1;
		int *xRang = (int*)malloc(xRang_len * sizeof(int));
		for (i = 0; i < xRang_len; i++) {
			xRang[i] = (int)(i + ceil(recVer.verX[0]));
		}
		double lineK[4];
		lineK[0] = (recVer.verY[1] - recVer.verY[0]) / (recVer.verX[1] - recVer.verX[0]);
		lineK[1] = (recVer.verY[2] - recVer.verY[1]) / (recVer.verX[2] - recVer.verX[1]);
		lineK[2] = (recVer.verY[2] - recVer.verY[3]) / (recVer.verX[2] - recVer.verX[3]);
		lineK[3] = (recVer.verY[3] - recVer.verY[0]) / (recVer.verX[3] - recVer.verX[0]);
		int *yLow = (int*)malloc(xRang_len * sizeof(int));
		int *yHigh = (int*)malloc(xRang_len * sizeof(int));
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
				if ((xRang[i] >= 0) && (xRang[i] < xLim) && (j >= 0) && (j < yLim)) {
					allPixNum++;
					double degDif = abs(rec.deg - degMap.ptr<double>(j)[xRang[i]]);
					if (degDif > pi * 3 / 2.0)
						degDif = abs(degDif - 2 * pi);
					if (degDif < rec.prec)
						aliPixNum++;
				}
			}
		}
		//计算NFA的自然对数值
		double logNFA;
		if ((allPixNum == 0) || (aliPixNum == 0)) {
			logNFA = -logNT;
			return logNFA;
		}
		if (allPixNum == aliPixNum) {
			logNFA = -logNT - allPixNum * log10(rec.p);
			return logNFA;
		}
		double proTerm = rec.p / (1.0 - rec.p);
		//利用Gamma函数来近似二项式系数
		double log1Coef = LogGammaCalculator(allPixNum + 1) - LogGammaCalculator(aliPixNum + 1)\
			- LogGammaCalculator(allPixNum - aliPixNum + 1);
		//由于在二项式中，后一项term(i)与前一项term(i-1)比值为(n-i+1) / i*p / (1-p)，故以此计算减少计算量
		//term(i)表示二项式展开式中的第 i 项
		double log1Term = log1Coef + aliPixNum * log(rec.p) + (allPixNum - aliPixNum) * log(1 - rec.p);
		double term = exp(log1Term);
		//如果首项很小，则可以忽略二项式
		double eps = 2.2204e-16;
		if (abs(term) < 100 * eps) {
			if (aliPixNum > allPixNum * rec.p)
				logNFA = -log10(term) - logNT;
			else
				logNFA = -logNT;
			return logNFA;
		}
		//根据NFA=N^5 * ∑{n,i=k}(n,i)p^i * (1-p)^(n-i)式子计算二项式拖尾项
		double binTail = term, tole = 0.1;
		for (i = aliPixNum + 1; i <= allPixNum; i++) {
			double binTerm = (allPixNum - i + 1) / (i * 1.0);
			double multTerm = binTerm * proTerm;
			term *= multTerm;
			binTail += term;
			if (binTerm < 1) {
				double err = term * ((1 - pow(multTerm, allPixNum - i + 1)) / (1.0 - multTerm) - 1);
				if (err < tole * abs(-log10(binTail) - logNT) * binTail)
					break;
			}
		}
		logNFA = -log10(binTail) - logNT;
		return logNFA;
	}

	structRectangleImprover RectangleImprover(structRec rec, Mat degMap, double logNT) {
		//输入：
		//rec：当前矩形结构体
		//degMap：水准线角弧度图
		//logNT：测试数量的对数值
		//
		//输出：
		//logNFA：虚警数的自然对数值
		//rec：修正后的矩形结构体
		//
		//函数功能：利用虚警数(NFA, Number of False Alarms)修正区域最小外接矩形
		double delt = 0.5;
		double delt2 = delt / 2.0;
		structRectangleImprover RI;
		RI.logNFA = RectangleNFACalculator(rec, degMap, logNT);
		RI.rec = rec;
		//如果虚警数小于1(负对数大于0，Desolneux建议值)则满足精度
		if (RI.logNFA > 0)
			return RI;
		//尝试改善精度 (角度容忍度)
		structRec recNew = RI.rec;
		int i;
		double logNFANew;
		for (i = 0; i < 5; i++) {
			recNew.p /= 2.0;
			recNew.prec = recNew.p * pi;
			logNFANew = RectangleNFACalculator(recNew, degMap, logNT);
			if (logNFANew > RI.logNFA) {
				RI.logNFA = logNFANew;
				RI.rec = recNew;
			}
		}
		if (RI.logNFA > 0)
			return RI;
		//尝试减少宽度
		recNew = RI.rec;
		for (i = 0; i < 5; i++) {
			//printf("recNew.wid = %lf\n", recNew.wid);
			if (recNew.wid - delt >= 0.5) {
				recNew.wid -= delt;
				logNFANew = RectangleNFACalculator(recNew, degMap, logNT);
				if (logNFANew > RI.logNFA) {
					RI.logNFA = logNFANew;
					RI.rec = recNew;
				}
			}
		}
		if (RI.logNFA > 0)
			return RI;
		//尝试减少矩形的一侧
		recNew = RI.rec;
		for (i = 0; i < 5; i++) {
			if (recNew.wid - delt >= 0.5) {
				recNew.x1 -= recNew.dy * delt2;
				recNew.y1 += recNew.dx * delt2;
				recNew.x2 -= recNew.dy * delt2;
				recNew.y2 += recNew.dx * delt2;
				recNew.wid -= delt;
				logNFANew = RectangleNFACalculator(recNew, degMap, logNT);
				if (logNFANew > RI.logNFA) {
					RI.logNFA = logNFANew;
					RI.rec = recNew;
				}
			}
		}
		if (RI.logNFA > 0)
			return RI;
		//尝试减少矩形的另一侧
		recNew = RI.rec;
		for (i = 0; i < 5; i++) {
			if (recNew.wid - delt >= 0.5) {
				recNew.x1 += recNew.dy * delt2;
				recNew.y1 -= recNew.dx * delt2;
				recNew.x2 += recNew.dy * delt2;
				recNew.y2 -= recNew.dx * delt2;
				recNew.wid -= delt;
				logNFANew = RectangleNFACalculator(recNew, degMap, logNT);
				if (logNFANew > RI.logNFA) {
					RI.logNFA = logNFANew;
					RI.rec = recNew;
				}
			}
		}
		if (RI.logNFA > 0)
			return RI;
		//尝试再次改善精度
		recNew = RI.rec;
		for (i = 0; i < 5; i++) {
			recNew.p /= 2.0;
			recNew.prec = recNew.p * pi;
			logNFANew = RectangleNFACalculator(recNew, degMap, logNT);
			if (logNFANew > RI.logNFA) {
				RI.logNFA = logNFANew;
				RI.rec = recNew;
			}
		}
		return RI;
	}
}