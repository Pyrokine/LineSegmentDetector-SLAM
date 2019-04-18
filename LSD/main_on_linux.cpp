#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/MapMetaData.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <myLSD.h>
#include <myRDP.h>
#include <FeatureAssociation.h>
#include <baseFunc.h>
#include <time.h>
#include <fstream>

typedef struct _structFA {
	vector<structLinesInfo> scanLinesInfo;
	vector<structLinesInfo> mapLinesInfo;
	int lidarPos[2];
	vector<double> ScanRanges;
	vector<double> ScanAngles;
} structFA;

structFA trans2FA(myrdp::structFeatureScan FS, mylsd::structLSD LSD, structMapParam oriMapParam, myrdp::structLidarPointPolar *lidarPointPolar, int len_lp);
void laserCallback(const sensor_msgs::LaserScan::ConstPtr &msg);
void mapParamCallback(const nav_msgs::MapMetaData::ConstPtr &msg);
void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr &msg);

cv::Mat mapValue, mapCache;
myrdp::structLidarPointPolar lidarPointPolar[360];
mylsd::structLSD LSD;
structMapParam mapParam;

//如果Map没有处理完，可能会导致特征匹配报错
bool isMapReady = false;

int main(int argc, char **argv) {
	//初始化节点
	ros::init(argc, argv, "laser_listener");
	ros::NodeHandle nh;

	//订阅gmapping在gazebo仿真下的数据
	ros::Subscriber subMapParam = nh.subscribe("/map_metadata", 1, mapParamCallback);
	ros::Subscriber subMap = nh.subscribe("/map", 1, mapCallback);
	ros::Subscriber subScan = nh.subscribe("/scan", 1, laserCallback);

	//循环读取
	ros::spin();
	return 0;
}

void laserCallback(const sensor_msgs::LaserScan::ConstPtr &msg) {
	//等待地图处理结束
	if (isMapReady == false)
		return;
	//读取激光雷达数据
	std::vector<float> ranges = msg->ranges;
	int len_lp = 0;
	for (int i = 0; i < ranges.size(); i++) {
		if (ranges[i] != INFINITY) {
			lidarPointPolar[i].range = ranges[i];
			//角度是用增量式方式来储存的
			lidarPointPolar[i].angle = msg->angle_min + i * msg->angle_increment;
			lidarPointPolar[i].split = false;
			len_lp++;
		}
		//ROS_INFO("No.%d RPLIDAR:%lf %lf", i, lidarPointPolar[i].range, lidarPointPolar[i].angle);
	}

	if (len_lp != 0) {
		//匹配雷达特征到地图特征 返回像素坐标和真实坐标
		myrdp::structFeatureScan FS = FeatureScan(mapParam, lidarPointPolar, len_lp, 3, 0.04, 0.5);
		imshow("RPLidar", FS.lineIm);
		waitKey(1);

		double estimatePose_realworld[3];
		double estimatePose[3];
		Mat poseAll;
		//数据接口转换
		structFA FA = trans2FA(FS, LSD, mapParam, lidarPointPolar, len_lp);
		//特征匹配
		myfa::FeatureAssociation(FS.lineIm, FA.scanLinesInfo, FA.mapLinesInfo, mapParam, FA.lidarPos, LSD.lineIm, \
			mapCache, mapValue, FA.ScanRanges, FA.ScanAngles, estimatePose_realworld, estimatePose, poseAll);

		//将图像坐标加入地图中
		LSD.lineIm.ptr<uint8_t>((int)estimatePose[1])[(int)estimatePose[0]] = 255;
		imshow("lineIm", LSD.lineIm);
		waitKey(1);
	}
}

void mapParamCallback(const nav_msgs::MapMetaData::ConstPtr &msg) {
	mapParam.oriMapCol = msg->width;
	mapParam.oriMapRow = msg->height;
	mapParam.mapResol = msg->resolution;
	mapParam.mapOriX = msg->origin.position.x;
	mapParam.mapOriY = msg->origin.position.y;
	//ROS_INFO("%d %d %lf %lf %lf\n", oriMapRow, oriMapCol, mapOriX, mapOriY, mapResol);
}

void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr &msg) {
	if (mapParam.oriMapCol <= 0 || mapParam.oriMapRow <= 0)
		return;
	isMapReady = false;
	mapValue = cv::Mat::zeros(mapParam.oriMapRow, mapParam.oriMapCol, CV_8UC1);
	//int max = 0;
	int cnt_col, cnt_row;
	//ROS_INFO("size%d %d %lf\n", msg->info.height, msg->info.width, msg->info.resolution);

	//取消注释FILE相关的可以输出地图到文件，在Windows下运行main_on_windows使用
	//FILE *fp = fopen("mapValue.txt", "w+");
	for (cnt_row = 0; cnt_row < mapParam.oriMapRow; cnt_row++) {
		for (cnt_col = 0; cnt_col < mapParam.oriMapCol; cnt_col++) {
			uint8_t value = msg->data[cnt_row * mapParam.oriMapCol + cnt_col];
			if (value == 255) {
				//fprintf(fp, "%d ", 0);
				mapValue.ptr<uint8_t>(cnt_row)[cnt_col] = 0;
			}
			else if (value == 0) {
				//fprintf(fp, "%d ", 255);
				mapValue.ptr<uint8_t>(cnt_row)[cnt_col] = 255;
			}
			else {
				//fprintf(fp, "%d ", 1);
				mapValue.ptr<uint8_t>(cnt_row)[cnt_col] = 1;
			}
		}
	}
	//fclose(fp);
    // cv::imshow("mapValue", mapValue);
    // cv::waitKey(1);
	//计算mapCache，用于特征匹配的先验概率
	double z_occ_max_dis = 2;
	mapCache = mylsd::createMapCache(mapValue, mapParam.mapResol, z_occ_max_dis);
	//LineSegmentDetector 提取地图边界直线信息
	LSD = mylsd::myLineSegmentDetector(mapValue, mapParam.oriMapCol, mapParam.oriMapRow, 0.3, 0.6, 22.5, 0.7, 1024);
	isMapReady = true;
}

structFA trans2FA(myrdp::structFeatureScan FS, mylsd::structLSD LSD, structMapParam oriMapParam, myrdp::structLidarPointPolar *lidarPointPolar, int len_lp) {
	//将数据格式转为FeatureAssociation格式
	structFA FA;
	int i;
	//scanLinesInfo
	FA.scanLinesInfo.resize(FS.len_linesInfo);
	for (i = 0; i < FS.len_linesInfo; i++) {
		FA.scanLinesInfo[i].k = FS.linesInfo[i].k;
		FA.scanLinesInfo[i].b = FS.linesInfo[i].b;
		FA.scanLinesInfo[i].dx = FS.linesInfo[i].dx;
		FA.scanLinesInfo[i].dy = FS.linesInfo[i].dy;
		FA.scanLinesInfo[i].x1 = FS.linesInfo[i].x1;
		FA.scanLinesInfo[i].y1 = FS.linesInfo[i].y1;
		FA.scanLinesInfo[i].x2 = FS.linesInfo[i].x2;
		FA.scanLinesInfo[i].y2 = FS.linesInfo[i].y2;
		FA.scanLinesInfo[i].len = FS.linesInfo[i].len;
	}
	//mapLinesInfo
	FA.mapLinesInfo.resize(LSD.len_linesInfo);
	for (i = 0; i < LSD.len_linesInfo; i++) {
		FA.mapLinesInfo[i].k = LSD.linesInfo[i].k;
		FA.mapLinesInfo[i].b = LSD.linesInfo[i].b;
		FA.mapLinesInfo[i].dx = LSD.linesInfo[i].dx;
		FA.mapLinesInfo[i].dy = LSD.linesInfo[i].dy;
		FA.mapLinesInfo[i].x1 = LSD.linesInfo[i].x1;
		FA.mapLinesInfo[i].y1 = LSD.linesInfo[i].y1;
		FA.mapLinesInfo[i].x2 = LSD.linesInfo[i].x2;
		FA.mapLinesInfo[i].y2 = LSD.linesInfo[i].y2;
		FA.mapLinesInfo[i].len = LSD.linesInfo[i].len;
	}
	//lidarPos
	FA.lidarPos[0] = FS.lidarPos.x;
	FA.lidarPos[1] = FS.lidarPos.y;
	printf("x:%lf y:%lf\n", FS.lidarPos.x, FS.lidarPos.y);
	//ScanRanges ScanAngles
	for (i = 0; i < len_lp; i++) {
		FA.ScanRanges.push_back(lidarPointPolar[i].range);
		FA.ScanAngles.push_back(lidarPointPolar[i].angle);
	}
	return FA;
}

