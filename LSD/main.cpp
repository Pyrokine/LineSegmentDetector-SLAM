#include <opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fstream>
//#define _disp_
using namespace cv;
using namespace std;

const double pi = 4.0 * atan(1.0);
const int MAXLEN = 10000;
double sca = 0.3, sig = 0.6, angThre = 22.5, denThre = 0.7;
int pseBin = 1024, regCnt = 0;
#ifdef _disp_
Mat mapDispOri = Mat::zeros(128, 413, CV_64FC3);
Mat mapDispTri = Mat::zeros(384, 1239, CV_64FC3);
Mat mapDispBase = Mat::zeros(384, 1239, CV_64FC3);
Mat mapDisp = Mat::zeros(384, 1239, CV_64FC3);
#endif
typedef struct _nodeBinCell {
	int value;
	int x;
	int y;
} nodeBinCell;

typedef struct _structPts {
	int x;
	int y;
	struct _structPts *next;
}structPts;

typedef struct _structReg {
	int x;
	int y;
	int num;
	double deg;
	int *regPts_x;
	int *regPts_y;
}structReg;

typedef struct _structRegionGrower {
	Mat curMap;
	structReg reg;
}structRegionGrower;

typedef struct _structRectangleConverter {
	double x1;
	double y1;
	double x2;
	double y2;
	double wid;
	double cX;
	double cY;
	double deg;
	double dx;
	double dy;
	double p;
	double prec;
}structRec;

typedef struct _structCenterGetter {
	double cenX;
	double cenY;
}structCenterGetter;

typedef struct _structRefiner {
	bool boolean;
	Mat curMap;
	structReg reg;
	structRec rec;
}structRefiner;

typedef struct _structRegionRadiusReducer {
	bool boolean;
	Mat curMap;
	structReg reg;
	structRec rec;
} structRegionRadiusReducer;

typedef struct _structRectangleImprover {
	double logNFA;
	structRec rec;
} structRectangleImprover;

typedef struct _structRecVer {
	double verX[4] = {0};
	double verY[4] = {0};
} structRecVer;

typedef struct _structLinesInfo{
	double k;
	double b;
	double dx;
	double dy;
	double x1;
	double y1;
	double x2;
	double y2;
	double len;
	int orient;
} structLinesInfo;

typedef struct _strcutPixelCache {
	int x;
	int y;
	double B;
	double G;
	double R;
	bool FirstUsed = true;
} structPixelCache;

structRec *recSaveDisp;

int Comp(const void *p1, const void *p2);
Mat GaussianSampler(Mat image, double sca, double sig);
structRegionGrower RegionGrower(int x, int y, Mat banMap, double regDeg, Mat degMap, double degThre);
structRec RectangleConverter(structReg reg, Mat magMap, double aliPro, double degThre);
structCenterGetter CenterGetter(int regNum, int *regX, int *regY, Mat weiMap);
double OrientationGetter(structReg reg, double cenX, double cenY, int *regX, int *regY, Mat weiMap, double degThre);
structRefiner Refiner(structReg reg, structRec rec, double denThre, Mat degMap, Mat banMap, Mat curMap, Mat magMap);
structRegionRadiusReducer RegionRadiusReducer(structReg reg, structRec rec, double denThre, Mat curMap, Mat magMap);
structRectangleImprover RectangleImprover(structRec rec, Mat degMap, double logNT);
void drawPoint(Mat image, int x, int y, int B, int G, int R);
void drawSign(Mat image, int x, int y, int B, int G, int R);
void drawRecs();
void drawText(string text, int loc);
double LogGammaCalculator(int x);
double sind(double x);
double cosd(double x);
double atand(double x);


void main() {
	clock_t time_start, time_end;
	time_start = clock();
	FILE *fp = fopen("E:/NUC/LSD/mapParam.txt", "r");
	int oriMapCol, oriMapRow;
	fscanf(fp, "%d%d", &oriMapCol, &oriMapRow);
	fclose(fp);

	int cnt_row ,cnt_col;
	fp = fopen("E:/NUC/LSD/data255.txt", "r");
	Mat MapGray = Mat::zeros(oriMapRow, oriMapCol, CV_8UC1);

	int max = 0;
	for (cnt_row = 0; cnt_row < oriMapRow; cnt_row++) {
		for (cnt_col = 0; cnt_col < oriMapCol; cnt_col++) {
			int val;
			fscanf(fp, "%d", &val);
			MapGray.ptr<uint8_t>(cnt_row)[cnt_col] = val;
		}
	}
	fclose(fp);
#ifdef _disp_
	imshow("mapDisp", mapDisp);
	waitKey(0);
#endif

//test
	//Mat testMap = Mat::zeros(2, 2, CV_8UC1);
	//testMap.ptr<double>(0)[0] = 500.7;
	//testMap.ptr<double>(0)[1] = 31400.32152;
	//printf("%f %f\n", testMap.ptr<double>(0)[0], testMap.ptr<double>(0)[1]);
//end-test

	//图像缩放—— 高斯降采样
	int newMapCol = (int)floor(oriMapCol * sca);
	int newMapRow = (int)floor(oriMapRow * sca);
	Mat GaussImage = GaussianSampler(MapGray, sca, sig);
#ifdef _disp_
	for (cnt_row = 0; cnt_row < newMapRow; cnt_row++) {
		for (cnt_col = 0; cnt_col < newMapCol; cnt_col++) {
			if (GaussImage.ptr<double>(cnt_row)[cnt_col] > 0) {
				mapDispOri.ptr<double>(cnt_row)[cnt_col * 3] = 255;
				mapDispOri.ptr<double>(cnt_row)[cnt_col * 3 + 1] = 255;
				mapDispOri.ptr<double>(cnt_row)[cnt_col * 3 + 2] = 255;
			}
			else {
				mapDispOri.ptr<double>(cnt_row)[cnt_col * 3] = 155 / 255.0;
				mapDispOri.ptr<double>(cnt_row)[cnt_col * 3 + 1] = 211 / 255.0;
				mapDispOri.ptr<double>(cnt_row)[cnt_col * 3 + 2] = 254 / 255.0;
			}
		}
	}
	//ofstream Fs("E:/NUC/LSD/Disp.xls");
	//int cnt_i, cnt_j;
	//for (cnt_i = 0; cnt_i < newMapRow; cnt_i++) {
	//	for (cnt_j = 0; cnt_j < newMapCol * 3; cnt_j++) {
	//		Fs << (double)mapDispOri.ptr<double>(cnt_i)[cnt_j] << '\t';
	//	}
	//	Fs << endl;
	//}
	//Fs.close();
	resize(mapDispOri, mapDispTri, Size(), 3, 3, 0);
	mapDisp = mapDispTri.clone();
#endif
	//imshow("GaussImage", GaussImage);
	//imshow("mapDispOri", mapDispOri);
	//imshow("mapDisp", mapDisp);
	//waitKey(0);

	//计算梯度
	Mat usedMap = Mat::zeros(newMapRow, newMapCol, CV_8UC1);//记录像素点状态
	Mat degMap = Mat::zeros(newMapRow, newMapCol, CV_64FC1);//level-line场方向
	Mat magMap = Mat::zeros(newMapRow, newMapCol, CV_64FC1);//记录每点的梯度
	double degThre = angThre / 180.0 * pi;
	double gradThre = 2.0 / sin(degThre);//梯度阈值
	//printf("%f %f\n", degThre, gradThre);
	
	//感觉这个矩阵运算是一个非常低效的过程 后面再优化
	Mat A = Mat::zeros(newMapRow, newMapCol, CV_64FC1);
	Mat B = Mat::zeros(newMapRow, newMapCol, CV_64FC1);
	Mat C = Mat::zeros(newMapRow, newMapCol, CV_64FC1);
	Mat D = Mat::zeros(newMapRow, newMapCol, CV_64FC1);
	int x, y;
	//A 原矩阵
	for (y = 0; y < newMapRow; y++) {
		for (x = 0; x < newMapCol; x++) {
			A.ptr<double>(y)[x] = GaussImage.ptr<double>(y)[x];
		}
	}
	//B 右上 A左移一格
	for (y = 0; y < newMapRow; y++) {
		for (x = 1; x < newMapCol; x++) {
			B.ptr<double>(y)[x - 1] = GaussImage.ptr<double>(y)[x];
		}
	}
	for (y = 0; y < newMapRow; y++) {
			B.ptr<double>(y)[newMapCol - 1] = GaussImage.ptr<double>(y)[newMapCol - 1];
	}
	//C 左下 A上移一格
	for (y = 1; y < newMapRow; y++) {
		for (x = 0; x < newMapCol; x++) {
			C.ptr<double>(y - 1)[x] = GaussImage.ptr<double>(y)[x];
		}
	}
	for (x = 0; x < newMapCol; x++) {
		C.ptr<double>(newMapRow - 1)[x] = GaussImage.ptr<double>(newMapRow - 1)[x];
	}
	//D 右下 A左上移一格
	for (y = 1; y < newMapRow; y++) {
		for (x = 1; x < newMapCol; x++) {
			D.ptr<double>(y - 1)[x - 1] = GaussImage.ptr<double>(y)[x];
		}
	}
	for (y = 0; y < newMapRow; y++) {
		D.ptr<double>(y)[newMapCol - 1] = GaussImage.ptr<double>(y)[newMapCol - 1];
	}
	for (x = 0; x < newMapCol; x++) {
		D.ptr<double>(newMapRow - 1)[x] = GaussImage.ptr<double>(newMapRow - 1)[x];
	}

	double gradX, gradY, maxGrad = 0;
	for (y = 0; y < newMapRow; y++) {
		for (x = 0; x < newMapCol; x++) {
			gradX = (B.ptr<double>(y)[x] + D.ptr<double>(y)[x] - A.ptr<double>(y)[x] - C.ptr<double>(y)[x]) / 2.0;
			gradY = (C.ptr<double>(y)[x] + D.ptr<double>(y)[x] - A.ptr<double>(y)[x] - B.ptr<double>(y)[x]) / 2.0;
			double valueMagMap = sqrt(pow(gradX, 2) + pow(gradY, 2));
			magMap.ptr<double>(y)[x] = valueMagMap;
			if (valueMagMap < gradThre)
				usedMap.ptr<uint8_t>(y)[x] = 1;
			if (maxGrad < valueMagMap)
				maxGrad = valueMagMap;
			double valueDegMap = atan2(gradX, -gradY);
			if (abs(valueDegMap - pi) < 0.000001)
				valueDegMap = 0;
			degMap.ptr<double>(y)[x] = valueDegMap;
		}
	}//计算结果和MATLAB有出入
	
	 //ofstream Fs("E:/NUC/LSD/DEG.xls");
	 //int cnt_i, cnt_j;
	 //for (cnt_i = 0; cnt_i < newMapRow; cnt_i++) {
	 //	for (cnt_j = 0; cnt_j < newMapCol; cnt_j++) {
	 //		Fs << (double)degMap.ptr<double>(cnt_i)[cnt_j] << '\t';
	 //	}
	 //	Fs << endl;
	 //}
	 //Fs.close();

	//梯度值的伪排序
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
	qsort(binCell, cnt_binCell, sizeof(nodeBinCell), Comp);
	//int cnt;
	//printf("%d\n", cnt_binCell);
	//for (cnt = 0; cnt < len_binCell; cnt++) {
	//	printf("%d %d %d\n", binCell[cnt].x, binCell[cnt].y, binCell[cnt].value);
	//}

	//按照伪排序的等级 依次搜索种子像素
	double logNT = 5 * (log10(newMapRow) + log10(newMapCol)) / 2.0;//测试数量的对数值
	double regThre = -logNT / log10(angThre / 180.0);//小区域的阈值
	double aliPro = angThre / 180.0;
	
	//记录生长区域和矩形
	structReg *regSave = (structReg*)malloc(MAXLEN * sizeof(structReg));
	structRec *recSave = (structRec*)malloc(MAXLEN * sizeof(structRec));
	recSaveDisp = (structRec*)malloc(MAXLEN * sizeof(structRec));
	Mat regIdx = Mat::zeros(newMapRow, newMapCol, CV_8UC1);
	Mat lineIm = Mat::zeros(oriMapRow, oriMapCol, CV_8UC1);//记录直线图像
#ifdef _disp_
	mapDisp.copyTo(mapDispBase);
#endif

	int i = 0;
	cnt_binCell = 0;
	for (i = 0; i < pseBin; i++) {
		int value_pseBin = pseBin - i;//binCell为倒序排列，计算出Mat对应的值
		int cnt_sameValue = 0, len_sameValue = 0;
		//统计有多少个相同的值
		for (cnt_sameValue = cnt_binCell; cnt_sameValue < len_binCell; cnt_sameValue++) {
//debug
//			if (i == 507)
//				printf("%d %d %d\n", binCell[cnt_sameValue].value, binCell[cnt_sameValue].x, binCell[cnt_sameValue].y);
//end-debug
			if (binCell[cnt_sameValue].value == value_pseBin) {
				//数组总指针向前，同时相同数量+1
				cnt_binCell++;
				len_sameValue++;
			}
			else
				break;
		}
		
		if (len_sameValue != 0) {
			int j = 0;
			for (j = len_sameValue; j > 0; j--) {
				int yIdx = binCell[cnt_binCell - j].y;
				int xIdx = binCell[cnt_binCell - j].x;
				if (usedMap.ptr<uint8_t>(yIdx)[xIdx] != 0)
					continue;
				structReg reg;
				//区域增长 返回curMap和reg
#ifdef _disp_
				//初始化显示
				mapDispBase.copyTo(mapDisp);
#endif 
				structRegionGrower RG = RegionGrower(xIdx, yIdx, usedMap, degMap.ptr<double>(yIdx)[xIdx], degMap, degThre);
				reg = RG.reg;
//debug
				//if (i == 507) {
				//	ofstream Fsrg("E:/NUC/LSD/RG.xls");
				//	int cnt_i, cnt_j;
				//	for (cnt_i = 0; cnt_i < newMapRow; cnt_i++) {
				//		for (cnt_j = 0; cnt_j < newMapCol; cnt_j++) {
				//			Fsrg << (int)RG.curMap.ptr<uint8_t>(cnt_i)[cnt_j] << '\t';
				//		}
				//		Fsrg << endl;
				//	}
				//	Fsrg.close();
				//	printf("reg--deg:%lf num:%d x:%d y:%d\n", reg.deg, reg.num, reg.x, reg.y);
				//	for (cnt_i = 0; cnt_i < reg.num; cnt_i++) {
				//		printf("%d %d %d\n", cnt_i, reg.regPts_x[cnt_i], reg.regPts_y[cnt_i]);
				//	}
				//}
//end-debug
				//删除小区域
				if (reg.num < regThre) {
//#ifdef _disp_
//					//显示状态
//					mapDispBase.copyTo(mapDisp);
//					drawText("Too Small", 350);
//					drawRecs();
//					imshow("mapDisp", mapDisp);
//					waitKey(300);
//#endif 
					continue;
				}
				//矩阵近似 返回rec
				structRec rec = RectangleConverter(reg, magMap, aliPro, degThre);
#ifdef _disp_
				//储存rec
				mapDispBase.copyTo(mapDisp);
				recSaveDisp[regCnt] = rec;
#endif 
//debug
				//if (i == 507) {
				//	printf("rec--cX:%lf cY:%lf deg:%lf dx:%lf dy:%lf p:%lf ", rec.cX, rec.cY, rec.deg, rec.dx, rec.dy, rec.p);
				//	printf("prec:%lf wid:%lf x1:%lf y1:%lf x2:%lf y2:%lf\n", rec.prec, rec.wid, rec.x1, rec.y1, rec.x2, rec.y2);
				//}		
//end-debug

				//根据密度阈值，调整区域 返回boolean, curMap, rec, reg 
				structRefiner RF = Refiner(reg, rec, denThre, degMap, usedMap, RG.curMap, magMap);
				reg = RF.reg;
				rec = RF.rec;
#ifdef _disp_
				//更新rec
				mapDispBase.copyTo(mapDisp);
				recSaveDisp[regCnt] = rec;
				//imshow("mapDisp", mapDisp);
				//waitKey(1);
#endif 
//debug
				//if (i == 507) {
				//	ofstream Fsrf("E:/NUC/LSD/RC.xls");
				//	int cnt_i, cnt_j;
				//	for (cnt_i = 0; cnt_i < newMapRow; cnt_i++) {
				//		for (cnt_j = 0; cnt_j < newMapCol; cnt_j++) {
				//			Fsrf << (int)RF.curMap.ptr<uint8_t>(cnt_i)[cnt_j] << '\t';
				//		}
				//		Fsrf << endl;
				//	}
				//	Fsrf.close();
				//	printf("reg--deg:%lf num:%d x:%d y:%d\n", reg.deg, reg.num, reg.x, reg.y);
				//	for (cnt_i = 0; cnt_i < reg.num; cnt_i++) {
				//		printf("%d %d %d\n", cnt_i, reg.regPts_x[cnt_i], reg.regPts_y[cnt_i]);
				//	}
				//	printf("rec--cX:%lf cY:%lf deg:%lf dx:%lf dy:%lf p:%lf ", rec.cX, rec.cY, rec.deg, rec.dx, rec.dy, rec.p);
				//	printf("prec:%lf wid:%lf x1:%lf y1:%lf x2:%lf y2:%lf\n", rec.prec, rec.wid, rec.x1, rec.y1, rec.x2, rec.y2);
				//
				//	ofstream Fsdeg("E:/NUC/LSD/DEG.xls");
				//	cnt_i, cnt_j;
				//	for (cnt_i = 0; cnt_i < newMapRow; cnt_i++) {
				//		for (cnt_j = 0; cnt_j < newMapCol; cnt_j++) {
				//			Fsdeg << (double)degMap.ptr<double>(cnt_i)[cnt_j] << '\t';
				//		}
				//		Fsdeg << endl;
				//	}
				//	Fsdeg.close();
				//}
//end-debug
				if (!RF.boolean)
					continue;
				//矩形调整 返回 logNFA, rec
				structRectangleImprover RI = RectangleImprover(rec, degMap, logNT);
				rec = RI.rec;
#ifdef _disp_
				//更新rec
				mapDispBase.copyTo(mapDisp);
				recSaveDisp[regCnt] = rec;
				//imshow("mapDisp", mapDisp);
				//waitKey(1);
#endif 
				//printf("rec--cX:%lf cY:%lf deg:%lf dx:%lf dy:%lf p:%lf ", rec.cX, rec.cY, rec.deg, rec.dx, rec.dy, rec.p);
				//printf("prec:%lf wid:%lf x1:%lf y1:%lf x2:%lf y2:%lf\n", rec.prec, rec.wid, rec.x1, rec.y1, rec.x2, rec.y2);

				if (RI.logNFA <= 0){
					for (y = 0; y < newMapRow; y++) {
						for (x = 0; x < newMapCol; x++) {
							if (RF.curMap.ptr<uint8_t>(y)[x] == 1)
								usedMap.ptr<uint8_t>(y)[x] = 2;
						}
					}
#ifdef _disp_
					//显示状态
					mapDispBase.copyTo(mapDisp);
					drawText("False Alarm", 350);
					drawRecs();
					imshow("mapDisp", mapDisp);
					waitKey(800);
#endif 
					continue;
				}
				//根据缩放尺度重新调整图像中所找到的直线信息
				if (sca != 1){
					rec.x1 = (rec.x1 - 1.0) / sca + 1;
					rec.y1 = (rec.y1 - 1.0) / sca + 1;
					rec.x2 = (rec.x2 - 1.0) / sca + 1;
					rec.y2 = (rec.y2 - 1.0) / sca + 1;
					rec.wid = (rec.wid - 1.0) / sca + 1;
				}
				//printf("rec--cX:%lf cY:%lf deg:%lf dx:%lf dy:%lf p:%lf ", rec.cX, rec.cY, rec.deg, rec.dx, rec.dy, rec.p);
				//printf("prec:%lf wid:%lf x1:%lf y1:%lf x2:%lf y2:%lf\n", rec.prec, rec.wid, rec.x1, rec.y1, rec.x2, rec.y2);
				for (y = 0; y < newMapRow; y++) {
					for (x = 0; x < newMapCol; x++) {
						regIdx.ptr<uint8_t>(y)[x] += RF.curMap.ptr<uint8_t>(y)[x] * (regCnt + 1);
						if (RF.curMap.ptr<uint8_t>(y)[x] == 1)
							usedMap.ptr<uint8_t>(y)[x] = 1;
					}
				}
				//保存所找到的直线支持区域和拟合矩形
#ifdef _disp_
				//显示状态
				mapDispBase.copyTo(mapDisp);
				drawText("Save", 350);
				drawRecs();
				imshow("mapDisp", mapDisp);
				waitKey(500);
#endif 
				regSave[regCnt] = reg;
				recSave[regCnt] = rec;
				regCnt++;
			}
		}
	}
	//调整保存所用胞元(去除)

	//将所提取到的直线按照像素点标记在图像矩阵中
	structLinesInfo *linesInfo = (structLinesInfo*)malloc(regCnt * sizeof(structLinesInfo));
	for (i = 0; i < regCnt; i++){
		//获得直线的端点坐标
		double x1 = recSave[i].x1;
		double y1 = recSave[i].y1;
		double x2 = recSave[i].x2;
		double y2 = recSave[i].y2;
		//求取直线斜率
		double k = (y2 - y1) / (x2 - x1);
		double ang = atand(k);
		int orient = 1;
		if (ang < 0){
			ang += 180;
			orient = -1;
		}
		//确定直线X坐标轴和Y坐标轴的跨度
		int xLow, xHigh, yLow, yHigh;
		if (x1 > x2){
			xLow = (int)floor(x2);
			xHigh = (int)ceil(x1);
		}
		else{
			xLow = (int)floor(x1);
			xHigh = (int)ceil(x2);
		}
		if (y1 > y2){
			yLow = (int)floor(y2);
			yHigh = (int)ceil(y1);
		}
		else{
			yLow = (int)floor(y1);
			yHigh = (int)ceil(y2);
		}
		double xRang = abs(x2 - x1), yRang = abs(y2 - y1);
		//确定直线跨度较大的坐标轴作为采样主轴并采样
		int xx_len = xHigh - xLow + 1, yy_len = yHigh - yLow + 1;
		int *xx, *yy;
		int j;
		if (xRang > yRang){
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
		//imshow("lineIm", lineIm);
		//waitKey(0);
	}

#ifdef _disp_
	mapDispBase.copyTo(mapDisp);
	drawText("Finish", 350);
	imshow("mapDisp", mapDisp);
	waitKey(0);
#endif // _disp_
	//imshow("MapGray", MapGray);
	//imshow("GaussImage", GaussImage);
	//imshow("usedMap", usedMap);
	//imshow("degMap", degMap);
	time_end = clock();
	printf("time = %lf\n", (double)(time_end - time_start) / CLOCKS_PER_SEC);
	imshow("lineIm", lineIm);
	waitKey(0);
	//system("pause");
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
	//printf("newXLim=%d newYLim=%d\n", newXLim, newYLim);
	Mat auxImage = Mat::zeros(yLim, newXLim, CV_64FC1);
	Mat newImage = Mat::zeros(newYLim, newXLim, CV_64FC1);
	//如果是缩小图像则调整标准差的值
	if (sca < 1)
		sig = sig / sca;
	//printf("%f\n", sig);
	//高斯模板大小
	int h = (int)ceil(sig * sqrt(2 * prec * log(10)));
	int hSize = 1 + 2 * h;
	int douXLim = xLim * 2;
	int douYLim = yLim * 2;
	//printf("h=%d hSize=%d douXLim=%d douYLim=%d\n", h, hSize, douXLim, douYLim);
	//x方向采样
	int x;
	for (x = 0; x < newXLim; x++) {
		//if (x == 1)
		//	printf("x=%d\n", x);
		double xx = x / sca;
		int xc = (int)floor(xx + 0.5);
		//if (x == 1)
		//	printf("xx=%f xc=%d\n", xx, xc);
		//确定高斯核中心位置
		double kerMean = h + xx - xc;
		double *kerVal = (double*)malloc(hSize * sizeof(double));
		double kerSum = 0;
		int k = 0;
		//if (x == 1)
		//	printf("kerMean=%lf\n", kerMean);
		//求当前高斯核（疑似有规律可循 不需反复计算 后面再优化）
		for (k = 0; k < hSize; k++) {
			kerVal[k] = (double)exp((-0.5) * pow((k - kerMean) / sig, 2));
			kerSum += kerVal[k];
			//if (x == 1)
			//	printf("k=%d kerVal[%d]=%lf kerSum=%lf\n", k, k, kerVal[k], kerSum);
		}
		//高斯核归一化
		for (k = 0; k < hSize; k++) {
			kerVal[k] /= kerSum;
			//if (x == 0)
			//	printf("k=%d kerval[%d]=%lf\n", k, k, kerVal[k]);
		}
		//用边缘对称的方式进行X坐标高斯滤波
		int y;
		for (y = 0; y < yLim; y++){
			double newVal = 0;
			int i;
			structPixelCache *pixelCache = (structPixelCache*)malloc(hSize * sizeof(structPixelCache));
			for (i = 0; i < hSize; i++) {
			}
		
			for (i = 0; i < hSize; i++){
				int j = xc - h + i;
				//if (x == 0 && y > 80 && y < 90)
				//	printf("y=%d i=%d j1=%d ", y, i, j);
				while (j < 0) {
					j += douXLim;
				}
				while (j >= douXLim) {
					j -= douXLim;
				}
				//if (x == 0 && y > 80 && y < 90)
				//	printf("j2=%d ", j);
				if (j >= xLim)
					j = douXLim - j - 1;
				//if (x == 0 && y > 80 && y < 90) {
				//	printf("j3=%d ", j);
				//	printf("img=%d\n", image.ptr<uint8_t>(y)[j]);
				//}
				newVal += image.ptr<uint8_t>(y)[j] * kerVal[i];
			}
			auxImage.ptr<double>(y)[x] = round(newVal);
		}
		//if (x < 10)
		//	printf("\n");
	}//end for（x方向采样）

	//y方向采样
	int y;
	for (y = 0; y < newYLim; y++) {
		double yy = y / sca;
		int yc = (int)floor(yy + 0.5);
		//确定高斯核中心位置
		double kerMean = h + yy - yc;
		double *kerVal = (double*)malloc(hSize * sizeof(double));
		double kerSum = 0;
		int k = 0;
		//求当前高斯核
		for (k = 0; k < hSize; k++) {
			kerVal[k] = (double)exp((-0.5) * pow((k - kerMean) / sig, 2));
			kerSum += kerVal[k];
		}
		//高斯核归一化
		for (k = 0; k < hSize; k++) {
			kerVal[k] /= kerSum;
		}
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
	//ofstream Fs("E:/NUC/LSD/1.xls");
	//int i, j;
	//for (i = 0; i < newYLim; i++) {
	//	for (j = 0; j < newXLim; j++) {
	//		Fs << (double)newImage.ptr<double>(i)[j] << '\t';
	//	}
	//	Fs << endl;
	//}
	//Fs.close();
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
#ifdef _disp_
	drawSign(mapDisp, regPts_head[0].x, regPts_head[0].y, 122, 25, 25);
	drawText("RegionGrowing", 350);
	drawRecs();
	imshow("mapDisp", mapDisp);
	waitKey(1);
#endif
	double sinDeg = sin(regDeg);
	double cosDeg = cos(regDeg);
	int yLim = banMap.rows;
	int xLim = banMap.cols;
	Mat curMap = Mat::zeros(yLim, xLim, CV_8UC1);
	curMap.ptr<uint8_t>(y)[x] = 1;
	int growNum = 1;
	int exNum = 0;
	structPixelCache PixelCache;
	int isFirstTime = 1;
	int temp = 0;
#ifdef _disp_
	Mat mapDispBK = Mat::zeros(384, 1239, CV_64FC3);
	mapDisp.copyTo(mapDispBK);
#endif
	while (exNum != growNum) {
		exNum = growNum;
		int i;
		regPts_now = regPts_head;
		for (i = 0; i < growNum; i++) {
			//检验8邻域像素是否满足角弧度阈值
			int m, n;
			int roi_x = regPts_now[0].x, roi_y = regPts_now[0].y;
#ifdef _disp_
			mapDispBK.copyTo(mapDisp);
			rectangle(mapDisp, Point((roi_x - 1) * 3 - 1, (roi_y - 1) * 3 - 1), Point((roi_x + 1) * 3 + 1, (roi_y + 1) * 3 + 1), \
				Scalar(0 / 255.0, 252 / 255.0, 124 / 255.0), FILLED); //绿色
			drawText("RegionGrowing", 350);
			drawRecs();
			//imshow("mapDisp", mapDisp);
			//waitKey(1);
#endif 
			for (m = roi_y - 1; m <= roi_y + 1; m++) {
				for (n = roi_x - 1; n <= roi_x + 1; n++) {
					//检查像素值的状态
					if (m >= 0 && n >= 0 && m < yLim && n < xLim) {
						if (curMap.ptr<uint8_t>(m)[n] != 1 && banMap.ptr<uint8_t>(m)[n] != 1){
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
								temp[0].x = n;
								temp[0].y = m;
								temp[0].next = NULL;
								regPts_end[0].next = temp;
								regPts_end = temp;
							}
						}
					}
				}
			}
#ifdef _disp_
			//drawPoint(mapDisp, regPts_now[0].x, regPts_now[0].y, 34, 139, 34);//深绿
			drawPoint(mapDispBK, regPts_now[0].x, regPts_now[0].y, 34, 139, 34);//深绿
			drawPoint(mapDispBase, regPts_now[0].x, regPts_now[0].y, 0, 140, 255);//橙色
			//imshow("mapDisp", mapDisp);
			//waitKey(1);
#endif
			if (regPts_now != regPts_end)
				regPts_now = regPts_now[0].next;
		}//for (i = 0; i < growNum; i++)
//debug
		//if (isFirstTime == 1) {
		//	isFirstTime = 0;
		//	temp = growNum;
		//}
		//else {
		//	if (temp != growNum)
		//		printf("%d %d\n", temp, growNum);
		//}	
//end-debug
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

#ifdef _disp_
	mapDispBase.copyTo(mapDisp);
	drawRecs();
#endif

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

double OrientationGetter(structReg reg, double cenX, double cenY, int *regX,\
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
		regDif = - regDif;
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
	structPixelCache pixelCache[4];
	for (m = 0; m < reg.num; m++) {
		double len = (reg.regPts_x[m] - CG.cenX) * dx + (reg.regPts_y[m] - CG.cenY) * dy;
		double wid = -(reg.regPts_x[m] - CG.cenX) * dy + (reg.regPts_y[m] - CG.cenY) * dx;
#ifdef _disp_
		drawPoint(mapDisp, reg.regPts_x[m], reg.regPts_y[m], 0, 252, 124);//绿色
		drawText("RectangleConverter", 350);
		drawRecs();
#endif
		if (len < lenMin) {
			lenMin = len;
#ifdef _disp_
			if (pixelCache[0].FirstUsed == false) {
				drawPoint(mapDisp, pixelCache[0].x, pixelCache[0].y, 0, 252, 124);//绿色
			}
			drawPoint(mapDisp, reg.regPts_x[m], reg.regPts_y[m], 122, 25, 25);//蓝色
			pixelCache[0].x = reg.regPts_x[m];
			pixelCache[0].y = reg.regPts_y[m];
			pixelCache[0].FirstUsed = false;
#endif
		}
			if (len > lenMax) {
			lenMax = len;
#ifdef _disp_
			if (pixelCache[1].FirstUsed == false) {
				drawPoint(mapDisp, pixelCache[1].x, pixelCache[1].y, 0, 252, 124);//绿色
			}
			drawPoint(mapDisp, reg.regPts_x[m], reg.regPts_y[m], 122, 25, 25);//蓝色
			pixelCache[1].x = reg.regPts_x[m];
			pixelCache[1].y = reg.regPts_y[m];
			pixelCache[1].FirstUsed = false;
#endif
		}
			if (wid < widMin) {
			widMin = wid;
#ifdef _disp_
			if (pixelCache[2].FirstUsed == false) {
				drawPoint(mapDisp, pixelCache[2].x, pixelCache[2].y, 0, 252, 124);//绿色
			}
			drawPoint(mapDisp, reg.regPts_x[m], reg.regPts_y[m], 122, 25, 25);//蓝色
			pixelCache[2].x = reg.regPts_x[m];
			pixelCache[2].y = reg.regPts_y[m];
			pixelCache[2].FirstUsed = false;
#endif
		}
			if (wid > widMax) {
			widMax = wid;
#ifdef _disp_
			if (pixelCache[3].FirstUsed == false) {
				drawPoint(mapDisp, pixelCache[3].x, pixelCache[3].y, 0, 252, 124);//绿色
			}
			drawPoint(mapDisp, reg.regPts_x[m], reg.regPts_y[m], 122, 25, 25);//蓝色
			pixelCache[3].x = reg.regPts_x[m];
			pixelCache[3].y = reg.regPts_y[m];
			pixelCache[3].FirstUsed = false;
#endif
		}
#ifdef _disp_
		//imshow("mapDisp", mapDisp);
		//waitKey(1);
#endif
	}
#ifdef _disp_
	//waitKey(1000);
#endif
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

structRegionRadiusReducer RegionRadiusReducer(structReg reg, structRec rec,\
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
//debug
	//int cnt_i;
	//printf("reg--deg:%lf num:%d x:%d y:%d\n", RRR.reg.deg, RRR.reg.num, RRR.reg.x, RRR.reg.y);
	//for (cnt_i = 0; cnt_i < RRR.reg.num; cnt_i++) {
	//	printf("%d %d %d\n", cnt_i, RRR.reg.regPts_x[cnt_i], RRR.reg.regPts_y[cnt_i]);
	//}
//end-debug
	double den = RRR.reg.num / (sqrt(pow(RRR.rec.x1 - RRR.rec.x2, 2) +\
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
#ifdef _disp_
	drawSign(mapDisp, oriX, oriY, 122, 25, 25);
	circle(mapDisp, Point(oriX * 3.0, oriY * 3.0), rad * 3.0, Scalar(122 / 255.0, 25 / 255.0, 25 / 255.0));
#endif
	while (den < denThre) {
		//以0.75的搜索速度减小搜索半径，减少直线支持区域中的像素数
		rad *= 0.75;
#ifdef _disp_
		mapDispBase.copyTo(mapDisp);
		circle(mapDisp, Point(oriX * 3.0, oriY * 3.0), rad * 3.0, Scalar(122 / 255.0, 25 / 255.0, 25 / 255.0));
#endif
		int i = 0;
		while (i <= RRR.reg.num) {
#ifdef _disp_
			drawPoint(mapDisp, reg.regPts_x[i], reg.regPts_y[i], 34, 139, 34);//深绿
			drawText("RegionRadiusReducer", 350);
			drawRecs();
			imshow("mapDisp", mapDisp);
			waitKey(1);
#endif
			if (sqrt(pow(oriX - RRR.reg.regPts_x[i], 2) + pow(oriY - RRR.reg.regPts_y[i], 2)) > rad) {
#ifdef _disp_
				drawPoint(mapDisp, reg.regPts_x[i], reg.regPts_y[i], 0, 0, 255);//红色
				drawRecs();
				imshow("mapDisp", mapDisp);
				waitKey(1);
#endif
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
#ifdef _disp_
		mapDispBase.copyTo(mapDisp);
#endif
		RRR.rec = RectangleConverter(RRR.reg, magMap, RRR.rec.p, RRR.rec.prec);
		den = RRR.reg.num / (sqrt(pow(RRR.rec.x1 - RRR.rec.x2, 2) + pow(RRR.rec.y1 - RRR.rec.y2, 2)) * RRR.rec.wid);
	}
	RRR.boolean = true;
	return RRR;
}

structRefiner Refiner(structReg reg, structRec rec, double denThre,\
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
#ifdef _disp_
	drawSign(mapDisp, oriX, oriY, 122, 25, 25);
#endif
//debug
	//int cnt_i;
	//printf("reg--deg:%lf num:%d x:%d y:%d\n", RF.reg.deg, RF.reg.num, RF.reg.x, RF.reg.y);
	//for (cnt_i = 0; cnt_i < RF.reg.num; cnt_i++) {
	//	printf("%d %d %d\n", cnt_i, RF.reg.regPts_x[cnt_i], RF.reg.regPts_y[cnt_i]);
	//}
//end-debug
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
#ifdef _disp_
			drawPoint(mapDisp, reg.regPts_x[i], reg.regPts_y[i], 0, 0, 255);//红色
			circle(mapDisp, Point(oriX * 3.0, oriY * 3.0), RF.rec.wid * 3.0, Scalar(122 / 255.0, 25 / 255.0, 25 / 255.0));
			drawText("Refiner", 350);
			drawRecs();
			imshow("mapDisp", mapDisp);
			waitKey(1000);
#endif
		}
	}
#ifdef _disp_
	mapDispBase.copyTo(mapDisp);
#endif
	double meanDif = difSum / (ptNum * 1.0);
	double degThre = 2.0 * sqrt((squSum - 2 * meanDif * difSum) / (ptNum * 1.0) + meanDif * meanDif);
	//利用新阈值重新进行区域生长
	structRegionGrower RG = RegionGrower(oriX, oriY, banMap, cenDeg, degMap, degThre);
	RF.curMap = RG.curMap;
	RF.reg = RG.reg;
//debug
	//cnt_i;
	//printf("reg--deg:%lf num:%d x:%d y:%d\n", RF.reg.deg, RF.reg.num, RF.reg.x, RF.reg.y);
	//for (cnt_i = 0; cnt_i < RF.reg.num; cnt_i++) {
	//	printf("%d %d %d\n", cnt_i, RF.reg.regPts_x[cnt_i], RF.reg.regPts_y[cnt_i]);
	//}
//end-debug
	//如果由于新阈值导致生长区域过小则丢弃当前区域
	if (RF.reg.num < 2) {
		RF.boolean = false;
		return RF;
	}
	//重新建立最小外接矩形
#ifdef _disp_
	mapDispBase.copyTo(mapDisp);
#endif
	RF.rec = RectangleConverter(RF.reg, magMap, RF.rec.p, RF.rec.prec);
	den = RF.reg.num / (sqrt(pow(RF.rec.x1 - RF.rec.x2, 2) + pow(RF.rec.y1 - RF.rec.y2, 2)) * RF.rec.wid);
	//如果还未满足密度阈值，则减小区域半径
#ifdef _disp_
	mapDispBase.copyTo(mapDisp);
#endif
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

double LogGammaCalculator(int x){
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
	if (x > thre){
		//LogGammaWindschitl
		val = 0.918938533204673 + (x - 0.5) * log(x) - x + 0.5 * x *\
		 log(x * sinh(1.0 / x) + 1.0 / (810 * pow(x, 6)));
	}
	else{
		//logGammaLanczos;
		double q[7] = {75122.6331530, 80916.6278952, 36308.2951477, 8687.24529705, 1168.92649479, 83.8676043424, 2.50662827511};
		double a = (x + 0.5) * log(x + 5.5) - (x + 5.5);
		double b = 0;
		int i;
		for (i = 0; i < 7; i++){
			a -= log(x + i);
			b += q[i] * pow(x, i);
		}
		val = a + log(b);
	}
	return val;	
}

double RectangleNFACalculator(structRec rec, Mat degMap, double logNT){
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
	for (cnt_row = 0; cnt_row < yLim; cnt_row++){
		for (cnt_col = 0; cnt_col < xLim; cnt_col++){
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
	int offset, i ,j;
	if ((rec.x1 < rec.x2) && (rec.y1 <= rec.y2))
		offset = 0;
	else if ((rec.x1 >= rec.x2) && (rec.y1 < rec.y2))
		offset = 1;
	else if ((rec.x1 > rec.x2) && (rec.y1 >= rec.y2))
		offset = 2;
	else
		offset = 3;
	for (i = 0; i < 4; i++){
		recVer.verX[i] = verX[(offset + i) % 4];
		recVer.verY[i] = verY[(offset + i) % 4];
	}
	//统计当前矩形中与矩形主惯性轴方向相同（小于角度容忍度）的像素点数量 aliPixNum
	//矩形内所有像素点数 allPixNum
	//printf("ceil:%d floor:%d\n", (int)ceil(recVer.verX[0]), (int)floor(recVer.verX[2]));
	int xRang_len = abs((int)(ceil(recVer.verX[0]) - floor(recVer.verX[2]))) + 1;
	int *xRang = (int*)malloc(xRang_len * sizeof(int));
	for (i = 0; i < xRang_len; i++){
		xRang[i] = (int)(i + ceil(recVer.verX[0]));
		//printf("%d %d\n", i, xRang[i]);
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
	for (i = 0; i < xRang_len; i++){
		if (xRang[i] < recVer.verX[3])
			yLow[cnt_yArry++] = (int)ceil(recVer.verY[0] + (xRang[i] - recVer.verX[0]) * lineK[3]);
	}
	for (i = 0; i < xRang_len; i++){
		if (xRang[i] >= recVer.verX[3])
			yLow[cnt_yArry++] = (int)ceil(recVer.verY[3] + (xRang[i] - recVer.verX[3]) * lineK[2]);
	}
	//printf("cnt_yArry:%d ", cnt_yArry);
	//yHigh
	cnt_yArry = 0;
	for (i = 0; i < xRang_len; i++){
		if (xRang[i] < recVer.verX[1])
			yHigh[cnt_yArry++] = (int)floor(recVer.verY[0] + (xRang[i] - recVer.verX[0]) * lineK[0]);
	}
	for (i = 0; i < xRang_len; i++){
		if (xRang[i] >= recVer.verX[1])
			yHigh[cnt_yArry++] = (int)floor(recVer.verY[1] + (xRang[i] - recVer.verX[1]) * lineK[1]);
	}
	//printf("cnt_yArry:%d\n", cnt_yArry);
	//printf("%d\n", xRang_len);
	//for (i = 0; i < xRang_len; i++) {
	//	printf("Low:%d High:%d\n", yLow[i], yHigh[i]);
	//}
	for (i = 0; i < xRang_len; i++){
		for (j = yLow[i]; j <= yHigh[i]; j++){
			if ((xRang[i] >= 0) && (xRang[i] < xLim) && (j >= 0) && (j < yLim)){
				allPixNum++;
				double degDif = abs(rec.deg - degMap.ptr<double>(j)[xRang[i]]);
				if (degDif > pi * 3 / 2.0)
					degDif = abs(degDif - 2 * pi);
				if (degDif < rec.prec)
					aliPixNum++;
				//printf("%lf\n", degDif);
			}
		}
	}
	//计算NFA的自然对数值
	double logNFA;
	if ((allPixNum == 0) || (aliPixNum == 0)){
		logNFA = - logNT;
		return logNFA;
	}
	if (allPixNum == aliPixNum){
		logNFA = - logNT - allPixNum * log10(rec.p);
		return logNFA;
	}
	double proTerm = rec.p / (1.0 - rec.p);
	//利用Gamma函数来近似二项式系数
	double log1Coef = LogGammaCalculator(allPixNum + 1) - LogGammaCalculator(aliPixNum + 1)\
	 - LogGammaCalculator(allPixNum - aliPixNum + 1);
	//printf("%lf %lf %lf\n", LogGammaCalculator(allPixNum + 1), LogGammaCalculator(aliPixNum + 1), \
		LogGammaCalculator(allPixNum - aliPixNum + 1));
	//由于在二项式中，后一项term(i)与前一项term(i-1)比值为(n-i+1) / i*p / (1-p)，故以此计算减少计算量
	//term(i)表示二项式展开式中的第 i 项
	double log1Term = log1Coef + aliPixNum * log(rec.p) + (allPixNum - aliPixNum) * log(1 - rec.p);
	double term = exp(log1Term);
	//如果首项很小，则可以忽略二项式
	double eps = 2.2204e-16;
	if (abs(term) < 100 * eps){
		if (aliPixNum > allPixNum * rec.p)
			logNFA = - log10(term) - logNT;
		else
			logNFA = - logNT;
		return logNFA;
	}
	//根据NFA=N^5 * ∑{n,i=k}(n,i)p^i * (1-p)^(n-i)式子计算二项式拖尾项
	double binTail = term, tole = 0.1;
	for (i = aliPixNum + 1; i <= allPixNum; i++){
		double binTerm = (allPixNum - i + 1) / (i * 1.0);
		double multTerm = binTerm * proTerm;
		term *= multTerm;
		binTail += term;
		if (binTerm < 1){
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
	for (i = 0; i < 5; i++){
		recNew.p /= 2.0;
		recNew.prec = recNew.p * pi;
		logNFANew = RectangleNFACalculator(recNew, degMap, logNT);
		if (logNFANew > RI.logNFA){
			RI.logNFA = logNFANew;
			RI.rec = recNew;
		}
	}
	if (RI.logNFA > 0)
		return RI;
	//尝试减少宽度
	recNew = RI.rec;
	for (i =0; i < 5; i++){
		//printf("recNew.wid = %lf\n", recNew.wid);
		if (recNew.wid - delt >= 0.5){
			recNew.wid -= delt;
			logNFANew = RectangleNFACalculator(recNew, degMap, logNT);
			if (logNFANew > RI.logNFA){
				RI.logNFA = logNFANew;
				RI.rec = recNew;
			}
		}
	}
	if (RI.logNFA > 0)
		return RI;
#ifdef _disp_
	Mat mapDispBK = Mat::zeros(384, 1239, CV_64FC3);
	mapDisp.copyTo(mapDispBK);
#endif
	//尝试减少矩形的一侧
	recNew = RI.rec;
	for (i = 0; i < 5; i++){
		if (recNew.wid - delt >= 0.5){
			recNew.x1 -= recNew.dy * delt2;
			recNew.y1 += recNew.dx * delt2;
			recNew.x2 -= recNew.dy * delt2;
			recNew.y2 += recNew.dx * delt2;
#ifdef _disp_
			rectangle(mapDisp, Point(recNew.x1 * 3 + 1, recNew.y1 * 3 + 1), \
				Point(recNew.x2 * 3 + 1, recNew.y2 * 3 + 1), Scalar(122 / 255.0, 25 / 255.0, 25 / 255.0), 1);
			drawText("RectangleImprover", 350);
			drawSign(mapDisp, rec.cX, rec.cY, 122, 25, 25);
			drawRecs();
			imshow("mapDisp", mapDisp);
			waitKey(1000);
			mapDispBK.copyTo(mapDisp);
#endif
			recNew.wid -= delt;
			logNFANew = RectangleNFACalculator(recNew, degMap, logNT);
			if (logNFANew > RI.logNFA){
				RI.logNFA = logNFANew;
				RI.rec = recNew;
			}
		}
	}
	if (RI.logNFA > 0)
		return RI;
	//尝试减少矩形的另一侧
	recNew = RI.rec;
	for (i = 0; i < 5; i++){
		if (recNew.wid - delt >= 0.5){
			recNew.x1 += recNew.dy * delt2;
			recNew.y1 -= recNew.dx * delt2;
			recNew.x2 += recNew.dy * delt2;
			recNew.y2 -= recNew.dx * delt2;
#ifdef _disp_
			rectangle(mapDisp, Point(recNew.x1 * 3 + 1, recNew.y1 * 3 + 1), \
				Point(recNew.x2 * 3 + 1, recNew.y2 * 3 + 1), Scalar(122 / 255.0, 25 / 255.0, 25 / 255.0), 1);
			drawText("RectangleImprover", 350);
			drawSign(mapDisp, rec.cX, rec.cY, 122, 25, 25);
			drawRecs();
			imshow("mapDisp", mapDisp);
			waitKey(1000);
			mapDispBK.copyTo(mapDisp);
#endif
			recNew.wid -= delt;
			logNFANew = RectangleNFACalculator(recNew, degMap, logNT);
			if (logNFANew > RI.logNFA){
				RI.logNFA = logNFANew;
				RI.rec = recNew;
			}
		}
	}
	if (RI.logNFA > 0)
		return RI;
	//尝试再次改善精度
	recNew = RI.rec;
	for (i = 0; i < 5; i++){
		recNew.p /= 2.0;
		recNew.prec = recNew.p * pi;
		logNFANew = RectangleNFACalculator(recNew, degMap, logNT);
		if (logNFANew > RI.logNFA){
			RI.logNFA = logNFANew;
			RI.rec = recNew;
		}
	}
	return RI;
}

double sind(double x) {
	return sin(x / 180.0 * pi);
}

double cosd(double x) {
	return cos(x / 180.0 * pi);
}

double atand(double x) {
	return atan(x / 180.0 * pi);
}

#ifdef _disp_
void drawPoint(Mat image, int x, int y, int B, int G, int R) {
	double B2 = B / 255.0;
	double G2 = G / 255.0;
	double R2 = R / 255.0;
	x *= 9;
	y *= 3;
	//[1,1]
	image.ptr<double>(y + 0)[x + 0] = B2;
	image.ptr<double>(y + 0)[x + 1] = G2;
	image.ptr<double>(y + 0)[x + 2] = R2;
	//[1,2]
	image.ptr<double>(y + 0)[x + 3] = B2;
	image.ptr<double>(y + 0)[x + 4] = G2;
	image.ptr<double>(y + 0)[x + 5] = R2;
	//[1,3]
	image.ptr<double>(y + 0)[x + 6] = B2;
	image.ptr<double>(y + 0)[x + 7] = G2;
	image.ptr<double>(y + 0)[x + 8] = R2;
	//[2,1]
	image.ptr<double>(y + 1)[x + 0] = B2;
	image.ptr<double>(y + 1)[x + 1] = G2;
	image.ptr<double>(y + 1)[x + 2] = R2;
	//[2,2]
	image.ptr<double>(y + 1)[x + 3] = B2;
	image.ptr<double>(y + 1)[x + 4] = G2;
	image.ptr<double>(y + 1)[x + 5] = R2;
	//[2,3]
	image.ptr<double>(y + 1)[x + 6] = B2;
	image.ptr<double>(y + 1)[x + 7] = G2;
	image.ptr<double>(y + 1)[x + 8] = R2;
	//[3,1]
	image.ptr<double>(y + 2)[x + 0] = B2;
	image.ptr<double>(y + 2)[x + 1] = G2;
	image.ptr<double>(y + 2)[x + 2] = R2;
	//[3,2]
	image.ptr<double>(y + 2)[x + 3] = B2;
	image.ptr<double>(y + 2)[x + 4] = G2;
	image.ptr<double>(y + 2)[x + 5] = R2;
	//[3,3]
	image.ptr<double>(y + 2)[x + 6] = B2;
	image.ptr<double>(y + 2)[x + 7] = G2;
	image.ptr<double>(y + 2)[x + 8] = R2;
}

void drawSign (Mat image, int x, int y, int B, int G, int R) {
	double B2 = B / 255.0;
	double G2 = G / 255.0;
	double R2 = R / 255.0;
	x *= 3;
	y *= 3;
	//左上
	line(image, Point(x - 16, y + 16), Point(x - 12, y + 16), Scalar(B2, G2, R2), 2);
	line(image, Point(x - 16, y + 16), Point(x - 16, y + 12), Scalar(B2, G2, R2), 2);
	//右上
	line(image, Point(x + 16, y + 16), Point(x + 12, y + 16), Scalar(B2, G2, R2), 2);
	line(image, Point(x + 16, y + 16), Point(x + 16, y + 12), Scalar(B2, G2, R2), 2);
	//左下
	line(image, Point(x - 16, y - 16), Point(x - 12, y - 16), Scalar(B2, G2, R2), 2);
	line(image, Point(x - 16, y - 16), Point(x - 16, y - 12), Scalar(B2, G2, R2), 2);
	//右下
	line(image, Point(x + 16, y - 16), Point(x + 12, y - 16), Scalar(B2, G2, R2), 2);
	line(image, Point(x + 16, y - 16), Point(x + 16, y - 12), Scalar(B2, G2, R2), 2);
}

void drawRecs() {
	int xLeft, xRight, yDown, yUp;
	int cnt_rec;
	for (cnt_rec = 0; cnt_rec <= regCnt - 1; cnt_rec++) {
		if (recSaveDisp[cnt_rec].x1 < recSaveDisp[cnt_rec].x2) {
			xLeft = (int)round(recSaveDisp[cnt_rec].x1);
			xRight = (int)round(recSaveDisp[cnt_rec].x2);
		}
		else {
			xLeft = (int)round(recSaveDisp[cnt_rec].x2);
			xRight = (int)round(recSaveDisp[cnt_rec].x1);
		}
		if (recSaveDisp[cnt_rec].y1 < recSaveDisp[cnt_rec].y2) {
			yUp = (int)round(recSaveDisp[cnt_rec].y1);
			yDown = (int)round(recSaveDisp[cnt_rec].y2);
		}
		else {
			yUp = (int)round(recSaveDisp[cnt_rec].y2);
			yDown = (int)round(recSaveDisp[cnt_rec].y1);
		}
		rectangle(mapDisp, Point(xLeft * 3 - 1, yUp * 3 - 1), Point(xRight * 3 + 1, yDown * 3 + 1), \
			Scalar(122 / 255.0, 25 / 255.0, 25 / 255.0), 1);
	}
}

void drawText(string text, int loc) {
	int baseLine;
	Size text_size = getTextSize(text, FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, &baseLine);
	putText(mapDisp, text, Point((1239 - text_size.width) / 2.0, loc), \
		FONT_HERSHEY_SIMPLEX, 1, Scalar(122 / 255.0, 25 / 255.0, 25 / 255.0), 2, 8);
}
#endif