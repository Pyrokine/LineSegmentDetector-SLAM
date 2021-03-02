/////////////////////////////////////////////////////////////////////////
//@Copyright(C) Pyrokine
//All rights reserved
//博客 http://www.cnblogs.com/Pyrokine/
//Github https://github.com/Pyrokine
//创建日期 20190327
//版本 1.1
//**********************************************************************
//V1.0
//建立baseFunc，简化不同库之间变量传递过程以及实现基本三角函数
//
//V1.1
//新增config功能，增加多个基础结构体
/////////////////////////////////////////////////////////////////////////

#ifndef _BASEFUNC_
#define _BASEFUNC_

#define debugMode
#define drawPicture

//必须所有自建的命名空间内都包含这个库 不管是否使用
#include <Eigen/Core>
#include <Eigen/Dense>

typedef struct _structMapParam {
	int oriMapCol;
	int oriMapRow;
	double mapResol;
	double mapOriX;
	double mapOriY;
} structMapParam;

typedef struct _structLinesInfo {
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

typedef struct _structPosition {
	double x;
	double y;
	double ang;
} structPosition;

double sind(double x);
double cosd(double x);
double atand(double x);

/////////////////////////////////////////////////////////////////////////
//                     main_on_windows.cpp
//**********************************************************************
// createMapCache 参数，为最大距离，单位:m
const double z_occ_max_dis = 1;
// 激光雷达每圈激光点数
const int pointPerLoop = 360;
// LSD输入参数
const double lsd_sca = 0.3; //缩放参数，默认0.3，修改需要手动计算新高斯核
const double lsd_sig = 0.6; //高斯核参数，默认0.6
const double lsd_angThre = 22.5; //角度阈值，默认22.5，单位：度
const double lsd_denThre = 0.7; //密度阈值，默认0.7
const int pseBin = 1024; //伪排序分组，默认1024（未使用伪排序）
// RDP输入参数
const int rdp_leastPoint = 3; //区域含有的最少点数，默认3
const double rdp_threLine = 0.08; //线段分割的长度比例阈值，默认0.08
const double rdp_leastDist = 0.5; //提取线段的最短长度，默认0.5
/////////////////////////////////////////////////////////////////////////
//                            myFA.cpp
//**********************************************************************
// ScanToMapMatch 多线程参数
const int numTHREAD = 30;
const int lenQUEUE = 50;
// scanLine 忽略长度，默认40，单位:像素
const int ignoreScanLength = 40;
// scanLine 和 mapLine 长度差比例阈值，取值范围0~1，默认0.35
const double scanToMapDiff = 0.35;
// 加权匹配候选点数量
const int lenCandidate = 30;
// 候选点距离上一帧估计位置的最大距离，默认60，单位：像素
const int maxEstiDist = 60;
/////////////////////////////////////////////////////////////////////////
#endif // ! _BASEFUNC_
