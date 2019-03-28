/////////////////////////////////////////////////////////////////////////
//@Copyright(C) Pyrokine
//All rights reserved
//博客 http://www.cnblogs.com/Pyrokine/
//Github https://github.com/Pyrokine
//创建日期 20190327
//版本 1.2
//**********************************************************************
//V1.0
//将LSD算法和RDP算法提取成单独文件并可以单独调用，增加命名空间mylsd和
//myrdp，建立baseFunc，简化不同库之间变量传递过程以及实现基本函数
//
//V1.1
//融合了FeatureAssociation算法，新增命名空间myfa，新增变量转换函数和结构体
//，将位置加入lineIm
//
//V1.2
//修改了文件结构，删除了main_with_disp.cpp，只能在发布版本1.3之前可以见到
//此文件，修改了文件读取方式，不再使用文件流读取方式
/////////////////////////////////////////////////////////////////////////

#ifndef _BASEFUNC_
#define _BASEFUNC_

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

double sind(double x);
double cosd(double x);
double atand(double x);

#endif // ! _BASEFUNC_
