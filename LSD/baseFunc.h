/////////////////////////////////////////////////////////////////////////
//@Copyright(C) Pyrokine
//All rights reserved
//���� http://www.cnblogs.com/Pyrokine/
//Github https://github.com/Pyrokine
//�������� 20190327
//�汾 1.1
//**********************************************************************
//V1.0
//����baseFunc���򻯲�ͬ��֮��������ݹ����Լ�ʵ�ֻ������Ǻ���
//
//V1.1
//����config���ܣ����Ӷ�������ṹ��
/////////////////////////////////////////////////////////////////////////

#ifndef _BASEFUNC_
#define _BASEFUNC_

#define debugMode
#define drawPicture

//���������Խ��������ռ��ڶ���������� �����Ƿ�ʹ��
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
// createMapCache ������Ϊ�����룬��λ:m
const double z_occ_max_dis = 1;
// �����״�ÿȦ�������
const int pointPerLoop = 360;
// LSD�������
const double lsd_sca = 0.3; //���Ų�����Ĭ��0.3���޸���Ҫ�ֶ������¸�˹��
const double lsd_sig = 0.6; //��˹�˲�����Ĭ��0.6
const double lsd_angThre = 22.5; //�Ƕ���ֵ��Ĭ��22.5����λ����
const double lsd_denThre = 0.7; //�ܶ���ֵ��Ĭ��0.7
const int pseBin = 1024; //α������飬Ĭ��1024��δʹ��α����
// RDP�������
const int rdp_leastPoint = 3; //�����е����ٵ�����Ĭ��3
const double rdp_threLine = 0.08; //�߶ηָ�ĳ��ȱ�����ֵ��Ĭ��0.08
const double rdp_leastDist = 0.5; //��ȡ�߶ε���̳��ȣ�Ĭ��0.5
/////////////////////////////////////////////////////////////////////////
//                            myFA.cpp
//**********************************************************************
// ScanToMapMatch ���̲߳���
const int numTHREAD = 30;
const int lenQUEUE = 50;
// scanLine ���Գ��ȣ�Ĭ��40����λ:����
const int ignoreScanLength = 40;
// scanLine �� mapLine ���Ȳ������ֵ��ȡֵ��Χ0~1��Ĭ��0.35
const double scanToMapDiff = 0.35;
// ��Ȩƥ���ѡ������
const int lenCandidate = 30;
// ��ѡ�������һ֡����λ�õ������룬Ĭ��60����λ������
const int maxEstiDist = 60;
/////////////////////////////////////////////////////////////////////////
#endif // ! _BASEFUNC_
