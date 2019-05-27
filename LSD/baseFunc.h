/////////////////////////////////////////////////////////////////////////
//@Copyright(C) Pyrokine
//All rights reserved
//博客 http://www.cnblogs.com/Pyrokine/
//Github https://github.com/Pyrokine
//创建日期 20190327
//版本 1.0
//**********************************************************************
//V1.0
//建立baseFunc，简化不同库之间变量传递过程以及实现基本三角函数
//
/////////////////////////////////////////////////////////////////////////

#ifndef _BASEFUNC_
#define _BASEFUNC_

#define debugMode

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

//createMapCache 参数
const double z_occ_max_dis = 1;

//ScanToMapMatch 多线程参数
const int lenTHREAD = 25;
const int lenQUEUE = 50;
#endif // ! _BASEFUNC_
