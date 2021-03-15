#include <myRDP.h>
#include <baseFunc.h>

using namespace cv;
using namespace std;

namespace myrdp {
	const double pi = 4.0 * atan(1.0);

	RDP::structFeatureScan RDP::FeatureScan(structMapParam& mapParam, vector<structLidarPoint>& _lidarPoint, const int _lenLidarPoint, const int _regionPointLimitNumber, const float _threLine, const double lineDistThreM) {
		// RamerDouglasPeucker
		lidarPoint = _lidarPoint, regionPointLimitNumber = _regionPointLimitNumber, threLine = _threLine, lenLidarPoint = _lenLidarPoint;

		// 生成点云的相关数据并确定绘制图像的大小
		float minX = INFINITY, minY = INFINITY, maxX = 0, maxY = 0;
		for (int i = 0; i < lenLidarPoint; i++) {
			lidarPoint[i].originX = lidarPoint[i].range * cos(lidarPoint[i].radian + scanPose[2]) + scanPose[0];
			lidarPoint[i].originY = lidarPoint[i].range * sin(lidarPoint[i].radian + scanPose[2]) + scanPose[1];
			lidarPoint[i].globalX = (lidarPoint[i].originX - mapParam.mapOriX) / mapParam.mapResol;
			lidarPoint[i].globalY = (lidarPoint[i].originY - mapParam.mapOriY) / mapParam.mapResol;
			lidarPoint[i].angle = (int)(lidarPoint[i].radian / M_PI * 180.0 + 180);
			lidarPoint[i].sn = i;
			minX = min(minX, lidarPoint[i].globalX);
			maxX = max(maxX, lidarPoint[i].globalX);
			minY = min(minY, lidarPoint[i].globalY);
			maxY = max(maxY, lidarPoint[i].globalY);
		}

		// 分割线段
		vector<structPointCell> pointCell = RegionSegmentation();
		SplitMerge(pointCell);

		int oriXLim = (int)ceil(maxX - minX), oriYLim = (int)ceil(maxY - minY);
		// 根据地图分辨率计算真实坐标的像素坐标
		structLidarPoint lidarPos;
		lidarPos.globalX = (scanPose[0] - mapParam.mapOriX) / mapParam.mapResol - minX;
		lidarPos.globalY = (scanPose[1] - mapParam.mapOriY) / mapParam.mapResol - minY;

		Mat lineIm = Mat::zeros(oriYLim, oriXLim, CV_8UC1);  // 记录直线图像
		vector<structLinesInfo> linesInfo;
		float lineDistThre = lineDistThreM / mapParam.mapResol;

		// 根据分割点将区块分隔开并提取直线信息
		vector<structPosition> scanImgPoint;
		vector<vector<structLidarPoint>> pointGroup;
		int len_axis, thisPointNumber, orient, xLow, xHigh, yLow, yHigh, xx, yy, xx_len, yy_len;
		float lineDist, x1, y1, x2, y2, k, ang;
		for (int i = 0; i < pointCell.size(); i++) {
			const int startPointNumber = pointCell[i].startPointNumber, endPointNumber = pointCell[i].endPointNumber;
			len_axis = 0;
			vector<int> listSplitNumber;
			// 处理头尾衔接问题
			if (endPointNumber > startPointNumber)
				len_axis = endPointNumber - startPointNumber + 1;
			else
				len_axis = lenLidarPoint + endPointNumber - startPointNumber + 1;

			vector<structLidarPoint> tempPointGroup;
			for (int j = 0; j < len_axis; j++) {
				thisPointNumber = startPointNumber + j;
				if (thisPointNumber >= lenLidarPoint)
					thisPointNumber -= lenLidarPoint;
				tempPointGroup.push_back(lidarPoint[thisPointNumber]);

				if (lidarPoint[thisPointNumber].split == true) {
					listSplitNumber.push_back(thisPointNumber);
					pointGroup.push_back(tempPointGroup);
					tempPointGroup.clear();
				}
			}
			if (tempPointGroup.size())
				pointGroup.push_back(tempPointGroup);
			listSplitNumber.insert(listSplitNumber.begin(), startPointNumber);
			listSplitNumber.push_back(endPointNumber);
			//std::printf("start:%d end:%d split:%d\n", startPointNumber, endPointNumber, (int)listSplitNumber.size());

			for (int j = 0; j < listSplitNumber.size() - 1; j++) {
				structLidarPoint* pointA = &lidarPoint[listSplitNumber[j]], * pointB = &lidarPoint[listSplitNumber[j + 1]];
				lineDist = sqrtf(powf(pointA->globalX - pointB->originX, 2) + powf(pointA->globalY - pointB->globalY, 2));
				if (lineDist >= lineDistThre) {
					// 获得直线的端点坐标
					x1 = pointA->globalX - minX;
					y1 = pointA->globalY - minY;
					x2 = pointB->globalX - minX;
					y2 = pointB->globalY - minY;
					// 求取直线斜率
					k = (y2 - y1) / (x2 - x1);
					ang = atand(k);
					orient = 1;
					if (ang < 0) {
						ang += 180;
						orient = -1;
					}
					// 确定直线X坐标轴和Y坐标轴的跨度
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
					// 确定直线跨度较大的坐标轴作为采样主轴并采样
					xx_len = xHigh - xLow + 1, yy_len = yHigh - yLow + 1;
					if (xx_len > yy_len) {
						for (int j = 0; j < xx_len; j++) {
							xx = j + xLow;
							yy = round((xx - x1) * k + y1);
							if (xx >= 0 and xx < oriXLim and yy >= 0 and yy < oriYLim) {
								lineIm.ptr<uint8_t>(yy)[xx] = 255;
								structPosition tempScanImgPoint = {
									xx, yy
								};
								scanImgPoint.push_back(tempScanImgPoint);
							}
						}
					}
					else {
						for (int j = 0; j < yy_len; j++) {
							yy = j + yLow;
							xx = round((yy - y1) / k + x1);
							if (xx >= 0 and xx < oriXLim and yy >= 0 and yy < oriYLim) {
								lineIm.ptr<uint8_t>(yy)[xx] = 255;
								structPosition tempScanImgPoint = {
									xx, yy
								};
								scanImgPoint.push_back(tempScanImgPoint);
							}
						}
					}

					structLinesInfo tempLinesInfo = {
						k,
						(y1 + y2) / 2.0 - k * (x1 + x2) / 2.0,
						cosd(ang),
						sind(ang),
						x1,
						y1,
						x2,
						y2,
						sqrt(pow(y2 - y1, 2) + pow(x2 - x1, 2)),
						orient
					};
					linesInfo.push_back(tempLinesInfo);
				}
			}
		}
		//cv::imshow("rdpLineIm", lineIm);
		//cv::waitKey(0);

		_lidarPoint = move(lidarPoint);
		structFeatureScan FS;
		FS.linesInfo = move(linesInfo);
		FS.lidarPos = move(lidarPos);
		FS.scanImPoint = move(scanImgPoint);
		FS.pointGroup = move(pointGroup);
		return move(FS);
	}

	void RDP::SplitMerge(vector<structPointCell>& pointCell) {
		// 按RegionSegmentation分成的分区分别进行线段分割
		for (int i = 0; i < pointCell.size(); i++) {
			SplitMergeAssistant(pointCell[i].startPointNumber, pointCell[i].endPointNumber);
		}
	}

	void RDP::SplitMergeAssistant(const int startPointNumber, const int endPointNumber) {
		// 使用SplitMerge递归分解同一区域的点
		int len = 0;
		if (endPointNumber > startPointNumber)
			len = endPointNumber - startPointNumber + 1;
		else
			len = lenLidarPoint + endPointNumber - startPointNumber + 1;

		if (len > 2) {
			structLidarPoint* pointA = &lidarPoint[startPointNumber];
			structLidarPoint* pointB = &lidarPoint[endPointNumber];
			// y = kx + d
			const float k = (pointB->originY - pointA->originY) / (pointB->originX - pointA->originX);
			const float d = pointB->originY - k * pointB->originX;
			float dist_max = 0;
			int i_max = 0, thisPointNumber;
			for (int i = 1; i < len - 1; i++) {
				// 计算点到直线的距离
				thisPointNumber = startPointNumber + i;
				if (thisPointNumber >= lenLidarPoint)
					thisPointNumber -= lenLidarPoint;
				structLidarPoint* pointC = &lidarPoint[thisPointNumber];

				const float dist = abs(k * pointC->originX - pointC->originY + d) / sqrtf(k * k + 1);
				if (dist > dist_max) {
					dist_max = dist;
					i_max = thisPointNumber;
				}
			}

			float threDist;
			if (lidarPoint[lidarPoint[i_max].sn].range > 9)
				threDist = lidarPoint[lidarPoint[i_max].sn].range * threLine;
			else
				threDist = threLine;

			//printf("%lf %lf\n", threDist, dist_max);
			if (dist_max > threDist) {
				SplitMergeAssistant(startPointNumber, i_max);
				SplitMergeAssistant(i_max, endPointNumber);
				lidarPoint[i_max].split = true;
				//printf("imax=%d\n", i_max);
			}
		}
	}

	vector<RDP::structPointCell> RDP::RegionSegmentation() {
		// 将激光点云图分簇处理，使簇中的点云没有明显间距
		int startPointNumber = 0, cellNumber = 0;
		structLidarPoint startPoint = lidarPoint[startPointNumber];

		// pointCell记录起始点和终止点的坐标以及在lidarPoint上的序号
		vector<structPointCell> pointCell;
		float deltaX, deltaY, deltaDist, threDeltaDist;
		for (int nowPointNumber = 0; nowPointNumber < lenLidarPoint; nowPointNumber++) {
			if (nowPointNumber == lenLidarPoint - 1) {
				deltaX = lidarPoint[nowPointNumber].originX - lidarPoint[0].originX;
				deltaY = lidarPoint[nowPointNumber].originY - lidarPoint[0].originY;
			}
			else {
				deltaX = lidarPoint[nowPointNumber].originX - lidarPoint[nowPointNumber + 1].originX;
				deltaY = lidarPoint[nowPointNumber].originY - lidarPoint[nowPointNumber + 1].originY;
			}

			deltaDist = sqrtf(deltaX * deltaX + deltaY * deltaY);
			threDeltaDist = getThresholdDeltaDist(lidarPoint[nowPointNumber].range);
			if (deltaDist > threDeltaDist) {
				if (abs(nowPointNumber - startPointNumber) >= regionPointLimitNumber) {
					structPointCell tempPointCell;
					tempPointCell.startPoint = startPoint;
					tempPointCell.startPointNumber = startPointNumber;
					tempPointCell.endPoint = lidarPoint[nowPointNumber];
					tempPointCell.endPointNumber = nowPointNumber;
					pointCell.push_back(tempPointCell);
				}

				if (nowPointNumber < lenLidarPoint - 1) {
					startPointNumber = nowPointNumber + 1;
					startPoint = lidarPoint[startPointNumber];
				}
			}

			// 首尾相连
			if (deltaDist <= threDeltaDist && nowPointNumber == lenLidarPoint - 1 && pointCell[0].startPointNumber == 0) {
				pointCell[0].startPoint = startPoint;
				pointCell[0].startPointNumber = startPointNumber;
			}
		}

		return move(pointCell);
	}

	float RDP::getThresholdDeltaDist(const float val) {
		if (val <= 0.3f)
			return 0.02f;
		else if (val <= 0.5f)
			return 0.05f;
		else if (val <= 0.8f)
			return 0.11f;
		else if (val <= 1.0f)
			return 0.17f;
		else if (val <= 2.0f)
			return 0.6f;
		else if (val <= 3.0f)
			return 0.7f;
		else if (val <= 4.0f)
			return 0.85f;
		else if (val <= 5.0f)
			return 0.9f;
		else if (val <= 6.0f)
			return 1.0f;
		else
			return 1.1f;
	}
}