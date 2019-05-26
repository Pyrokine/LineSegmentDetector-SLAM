#include <myRDP.h>
#include <baseFunc.h>

using namespace cv;
using namespace std;
namespace myrdp{
	const double pi = 4.0 * atan(1.0);

	structFeatureScan FeatureScan(structMapParam mapParam, structLidarPointPolar *lidarPointPolar, int len_lp, int RegionPointLimitNumber, double threLine, double lineDistThreM) {
		////RamerDouglasPeucker
		double scanPose[3] = { 0 };
		structRegionSegmentation RS = RegionSegmentation(lidarPointPolar, len_lp, scanPose, RegionPointLimitNumber);
		SplitMerge(lidarPointPolar, len_lp, scanPose, RS, threLine);

		//处理超出阈值的值并确定图像大小
		structLidarPointRec *lidarPointRecGlobal = (structLidarPointRec*)malloc(len_lp * sizeof(structLidarPointRec));
		double minX = INFINITY, minY = INFINITY, maxX = 0, maxY = 0;
		int i;
		for (i = 0; i < len_lp; i++) {
			lidarPointRecGlobal[i].x = floor((lidarPointPolar[i].range * cos(lidarPointPolar[i].angle + scanPose[2]) + scanPose[0] - mapParam.mapOriX) / mapParam.mapResol);
			lidarPointRecGlobal[i].y = floor((lidarPointPolar[i].range * sin(lidarPointPolar[i].angle + scanPose[2]) + scanPose[1] - mapParam.mapOriY) / mapParam.mapResol);
			lidarPointRecGlobal[i].num = i;
			if (lidarPointRecGlobal[i].x < minX)
				minX = lidarPointRecGlobal[i].x;
			if (lidarPointRecGlobal[i].x > maxX)
				maxX = lidarPointRecGlobal[i].x;
			if (lidarPointRecGlobal[i].y < minY)
				minY = lidarPointRecGlobal[i].y;
			if (lidarPointRecGlobal[i].y > maxY)
				maxY = lidarPointRecGlobal[i].y;
		}
		int oriXLim = (int)ceil(maxX - minX), oriYLim = (int)ceil(maxY - minY);
		//根据地图分辨率计算真实坐标的像素坐标
		structLidarPointRec lidarPos;
		lidarPos.x = floor((scanPose[0] - mapParam.mapOriX) / mapParam.mapResol - minX);
		lidarPos.y = floor((scanPose[1] - mapParam.mapOriY) / mapParam.mapResol - minY);

		Mat lineIm = Mat::zeros(oriYLim, oriXLim, CV_8UC1);//记录直线图像
		structLinesInfo *linesInfo = (structLinesInfo*)malloc(360 * sizeof(structLinesInfo));
		int cnt_linesInfo = 0;
		double lineDistThre = lineDistThreM / mapParam.mapResol;

		//根据分割点将区块分隔开并提取直线信息
		for (i = 0; i < RS.cellNum; i++) {
			int startPoint = RS.pointCell[i].startPointNum, endPoint = RS.pointCell[i].endPointNum;
			int j, len_axis = 0, num_split = 1, *axis;
			//处理头尾衔接问题
			if (endPoint > startPoint) {
				len_axis = endPoint - startPoint + 1;
				axis = (int*)malloc(len_axis * sizeof(int));
				for (j = 0; j < len_axis; j++) {
					if (lidarPointPolar[startPoint + j].split == true)
						axis[num_split++] = startPoint + j;
				}
			}
			else {
				len_axis = len_lp + endPoint - startPoint + 1;
				axis = (int*)malloc(len_axis * sizeof(int));
				for (j = 0; j < len_axis; j++) {
					int val = startPoint + j;
					if (val >= len_lp)
						val -= len_lp;
					if (lidarPointPolar[val].split == true)
						axis[num_split++] = val;
				}
			}
			axis[0] = startPoint;
			axis[num_split++] = endPoint;

			for (j = 0; j < num_split - 1; j++) {
				structLidarPointRec pointA, pointB;
				pointA.x = lidarPointRecGlobal[axis[j]].x;
				pointA.y = lidarPointRecGlobal[axis[j]].y;
				pointB.x = lidarPointRecGlobal[axis[j + 1]].x;
				pointB.y = lidarPointRecGlobal[axis[j + 1]].y;
				double lineDist = sqrt(pow(pointA.x - pointB.x, 2) + pow(pointA.y - pointB.y, 2));
				if (lineDist >= lineDistThre) {
					//获得直线的端点坐标
					double x1 = pointA.x - minX;
					double y1 = pointA.y - minY;
					double x2 = pointB.x - minX;
					double y2 = pointB.y - minY;
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
					int m;
					if (xRang > yRang) {
						xx = (int*)malloc(xx_len * sizeof(int));
						yy = (int*)malloc(xx_len * sizeof(int));
						for (m = 0; m < xx_len; m++) {
							xx[m] = m + xLow;
							yy[m] = (int)round((xx[m] - x1) * k + y1);
							if (xx[m] < 0 || xx[m] >= oriXLim || yy[m] < 0 || yy[m] >= oriYLim) {
								xx[m] = 0;
								yy[m] = 0;
							}
						}
					}
					else {
						xx = (int*)malloc(yy_len * sizeof(int));
						yy = (int*)malloc(yy_len * sizeof(int));
						for (m = 0; m < yy_len; m++) {
							yy[m] = m + yLow;
							xx[m] = (int)round((yy[m] - y1) / k + x1);
							if (xx[m] < 0 || xx[m] >= oriXLim || yy[m] < 0 || yy[m] >= oriYLim) {
								xx[m] = 0;
								yy[m] = 0;
							}
						}
					}
					//标记直线像素
					if (xx_len > yy_len) {
						for (m = 0; m < xx_len; m++) {
							if (xx[m] != 0 && yy[m] != 0)
								lineIm.ptr<uint8_t>(yy[m])[xx[m]] = 255;
						}
					}
					else {
						for (m = 0; m < yy_len; m++) {
							if (xx[m] != 0 && yy[m] != 0)
								lineIm.ptr<uint8_t>(yy[m])[xx[m]] = 255;
						}
					}
					linesInfo[cnt_linesInfo].k = k;
					linesInfo[cnt_linesInfo].b = (y1 + y2) / 2.0 - k * (x1 + x2) / 2.0;
					linesInfo[cnt_linesInfo].dx = cosd(ang);
					linesInfo[cnt_linesInfo].dy = sind(ang);
					linesInfo[cnt_linesInfo].x1 = x1;
					linesInfo[cnt_linesInfo].y1 = y1;
					linesInfo[cnt_linesInfo].x2 = x2;
					linesInfo[cnt_linesInfo].y2 = y2;
					linesInfo[cnt_linesInfo].len = sqrt(pow(y2 - y1, 2) + pow(x2 - x1, 2));
					linesInfo[cnt_linesInfo].orient = orient;
					cnt_linesInfo++;
				}
			}
		}
		structFeatureScan FS;
		FS.lineIm = lineIm;
		FS.linesInfo = linesInfo;
		FS.lidarPos = lidarPos;
		FS.len_linesInfo = cnt_linesInfo;
		return FS;
	}

	void SplitMerge(structLidarPointPolar *lidarPointPolar, int len_lp, double *scanPose, structRegionSegmentation RS, double threLine) {
		//输入
		//ScanRanges: 列向量 激光雷达扫描点云图的扫描长度
		//ScanAngles : 列向量 激光雷达扫描点云图的扫描角度
		//ScanPose : 列向量 激光雷达在全局地图下的位姿
		//PointCell : cell(4 * cell_n)
		//PointCell	第一行表示起始点的坐标
		//	      	第二行表示终止点的坐标
		//	        第三行表示起始点的坐标对应在 ScanRanges 上的序号
		//          第四行表示终止点的坐标对应在 ScanRanges 上的序号
		//Threshold_line : 分裂成两条直线的阈值
		//输出
		//PointInflection 中的列向量 表示区域的拐点序号
		structLidarPointRec *lidarPointRec = (structLidarPointRec*)malloc(len_lp * sizeof(structLidarPointRec));
		int i;
		for (i = 0; i < len_lp; i++) {
			lidarPointRec[i].x = lidarPointPolar[i].range * cos(lidarPointPolar[i].angle + scanPose[2]) + scanPose[0];
			lidarPointRec[i].y = lidarPointPolar[i].range * sin(lidarPointPolar[i].angle + scanPose[2]) + scanPose[1];
			lidarPointRec[i].num = i;
			//printf("%d %lf %lf\n", i, lidarPointRec[i].x, lidarPointRec[i].y);
		}
		for (i = 0; i < RS.cellNum; i++) {
			//printf("%d %d %d\n", i, RS.pointCell[i].startPointNum, RS.pointCell[i].endPointNum);
			SplitMergeAssistant(lidarPointPolar, lidarPointRec, len_lp, RS.pointCell[i].startPointNum, RS.pointCell[i].endPointNum, threLine);
		}
		//for (i = 0; i < len_lp; i++) {
		//	if (lidarPointPolar[i].split == true)
		//		printf("%d\n", i + 1);
		//}

	}

	void SplitMergeAssistant(structLidarPointPolar *lidarPointPolar, structLidarPointRec *lidarPointRec, int len_lp, int startPoint, int endPoint, double threLine) {
		//Ramer-Douglas-Peucker算法
		//使用SplitMerge分解同一区域的点
		//RegionPoint  列向量为 XY 坐标
		int i, len = 0, *axis;
		if (endPoint > startPoint) {
			len = endPoint - startPoint + 1;
			axis = (int*)malloc(len * sizeof(int));
			for (i = 0; i < len; i++) {
				axis[i] = startPoint + i;
			}
		}
		else {
			len = len_lp + endPoint - startPoint + 1;
			axis = (int*)malloc(len * sizeof(int));
			for (i = 0; i < len; i++) {
				axis[i] = startPoint + i;
				if (axis[i] >= len_lp)
					axis[i] -= len_lp;
			}
		}

		if (len > 2) {
			structLidarPointRec pointA = lidarPointRec[startPoint];
			structLidarPointRec pointB = lidarPointRec[endPoint];
			//y = kx + d
			double k = (pointB.y - pointA.y) / (pointB.x - pointA.x);
			double d = pointB.y - k * pointB.x;
			double dist_max = 0;
			int i, i_max = 0;
			for (i = 1; i < len - 1; i++) {
				structLidarPointRec pointC = lidarPointRec[axis[i]];
				//计算点到直线的距离
				double dist = fabs(k * pointC.x - pointC.y + d) / sqrt(pow(k, 2) + 1);
				if (dist > dist_max) {
					dist_max = dist;
					i_max = axis[i];
				}
			}

			double threDist;
			if (lidarPointPolar[lidarPointRec[i_max].num].range > 9)
				threDist = lidarPointPolar[lidarPointRec[i_max].num].range * threLine;
			else
				threDist = threLine;

			if (dist_max > threDist) {
				SplitMergeAssistant(lidarPointPolar, lidarPointRec, len_lp, startPoint, i_max, threLine);
				SplitMergeAssistant(lidarPointPolar, lidarPointRec, len_lp, i_max, endPoint, threLine);
				lidarPointPolar[i_max].split = true;
				//printf("imax=%d\n", i_max);
			}
		}
	}

	structRegionSegmentation RegionSegmentation(structLidarPointPolar *lidarPointPolar, int len_lp, double *scanPose, int RegionPointLimitNumber) {
		//输入：
		//ScanRanges: 列向量 激光雷达扫描点云图的扫描长度
		//ScanAngles : 列向量 激光雷达扫描点云图的扫描角度
		//ScanPose : 列向量 激光雷达在全局地图下的位姿
		//RegionPointLimitNumber：分簇后，簇中的点数据个数的最小值
		//输出：
		//PointCell：簇中点的坐标，以及标号
		//函数功能：将激光点云图分簇处理，簇中的点云没有明显间距
		structLidarPointRec *lidarPointRec = (structLidarPointRec*)malloc(len_lp * sizeof(structLidarPointRec));
		int i;
		for (i = 0; i < len_lp; i++) {
			lidarPointRec[i].x = lidarPointPolar[i].range * cos(lidarPointPolar[i].angle + scanPose[2]) + scanPose[0];
			lidarPointRec[i].y = lidarPointPolar[i].range * sin(lidarPointPolar[i].angle + scanPose[2]) + scanPose[1];
			//printf("%d %lf %lf\n", i, lidarPointRec[i].x, lidarPointRec[i].y);
		}
		structLidarPointRec startPoint;
		startPoint.x = lidarPointRec[0].x;
		startPoint.y = lidarPointRec[0].y;
		int startPointNumber = 0, cellNumber = 0;

		//pointCell记录起始点和终止点的坐标以及在lidarPointRec上的序号
		structPointCell *pointCell = (structPointCell*)malloc(len_lp * sizeof(structPointCell));
		for (i = 0; i < len_lp; i++) {
			double deltaX, deltaY;
			if (i == len_lp - 1) {
				deltaX = lidarPointRec[i].x - lidarPointRec[0].x;
				deltaY = lidarPointRec[i].y - lidarPointRec[0].y;
			}
			else {
				deltaX = lidarPointRec[i].x - lidarPointRec[i + 1].x;
				deltaY = lidarPointRec[i].y - lidarPointRec[i + 1].y;
			}
			double deltaDist = sqrt(deltaX * deltaX + deltaY * deltaY);
			double threDeltaDist = getThresholdDeltaDist(lidarPointPolar[i].range);
			if (deltaDist > threDeltaDist) {
				pointCell[cellNumber].startPointLoc = startPoint;
				pointCell[cellNumber].startPointNum = startPointNumber;
				pointCell[cellNumber].endPointLoc.x = lidarPointRec[i].x;
				pointCell[cellNumber].endPointLoc.y = lidarPointRec[i].y;
				pointCell[cellNumber].endPointNum = i;
				if (abs(i - startPointNumber) >= RegionPointLimitNumber)
					cellNumber++;

				if (i < len_lp) {
					startPoint.x = lidarPointRec[i + 1].x;
					startPoint.y = lidarPointRec[i + 1].y;
					startPointNumber = i + 1;
				}
			}

			//首尾相连
			if (deltaDist <= threDeltaDist && i == len_lp - 1) {
				pointCell[0].startPointLoc = startPoint;
				pointCell[0].startPointNum = startPointNumber;
			}
		}
		structPointCell *pointCell2 = (structPointCell*)malloc(cellNumber * sizeof(structPointCell));
		for (i = 0; i < cellNumber; i++) {
			pointCell2[i].startPointLoc = pointCell[i].startPointLoc;
			pointCell2[i].startPointNum = pointCell[i].startPointNum;
			pointCell2[i].endPointLoc = pointCell[i].endPointLoc;
			pointCell2[i].endPointNum = pointCell[i].endPointNum;
		}
		//for (i = 0; i < cellNumber; i++) {
		//	printf("%d %d %d\n", i, pointCell2[i].startPointNum, pointCell2[i].endPointNum);
		//}
		structRegionSegmentation RS;
		RS.pointCell = pointCell2;
		RS.cellNum = cellNumber;
		return RS;
	}

	double getThresholdDeltaDist(double val) {
		if (val <= 0.3)
			return 0.02;
		else if (val <= 0.5)
			return 0.05;
		else if (val <= 0.8)
			return 0.11;
		else if (val <= 1)
			return 0.17;
		else if (val <= 2)
			return 0.6;
		else if (val <= 3)
			return 0.7;
		else if (val <= 4)
			return 0.85;
		else if (val <= 5)
			return 0.9;
		else if (val <= 6)
			return 1;
		else
			return 1.1;
	}
}