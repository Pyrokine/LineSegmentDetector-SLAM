# LineSegmentDetector-SLAM
基于C + CPP + OpenCV4.0 + Eigen3 + pthread实现的LineSegmentDetector定位算法，地图由Karto-Slam构建，OpenCV向下兼容3.0，编译环境为VS19，需要自行建立项目，Windows下主程序为main_on_windows.cpp

LSD算法解释https://www.cnblogs.com/Pyrokine/p/10384930.html

LSD算法动画https://www.bilibili.com/video/av43174965/

LSD在ROS下使用https://www.cnblogs.com/Pyrokine/p/10730995.html

OpenCV4.0下载链接：https://opencv.org/releases/

Eigen3下载链接：http://eigen.tuxfamily.org/index.php?title=Main_Page

pthread-win32下载链接：ftp://sourceware.org/pub/pthreads-win32/

更新日志：

V1.0 完成了基本算法的实现以及基础动画的实现

V1.1 完成了算法的优化， 将多个定长数组用动态数组实现，极大地减小了内存空间，将原算法的伪排序使用快速排序实现，优化了高斯降采样时计算高斯核的算法，改进后一张图仅需计算一次高斯核，所以将区域增长次数从多次减少到1次

V1.2 完成了Degree图的动画（在main_with_disp里面），修正了UsedMap的动画，以及增加了梯度排名和区域内像素点显示，增加了虚警数的数值显示，提取出了LSD算法为单独函数

V1.3 新增了基于Ramer-Douglas-Peucker算法的激光雷达点云分割代码

V2.0 提取出了LSD和RDP算法到单独文件并可独立调用，新增Base_Func实现共用的函数和结构体，并记录主体结构的变化，引入FeatureAssociation算法，实现RDP的数据在LSD中的匹配的算法，新增数据转换函数，增加RDP注释量，调整文件结构，删除main_with_disp函数及相关生成文件

V2.1 在myLSD中增加了mapCache的计算，用于特征匹配的先验概率，修正了LSD计算时的原始图像，清除了地图中间值为255的部分

V2.2 增加了main_on_linux入口，用于ROS使用，具体方法见前面博客，修改main_no_disp为main_on_windows，增加一组数据，以map1结尾，mapParam共用，mapValue有map1和map2

V2.3 修复了bug，简化了计算RDP像素点方式，加速了运算速度

V2.4 在特征匹配myFA中增加了基于pthread的线程池，极大地提高了计算速度

V2.5 增加隐马尔可夫链

V2.6 融合了里程计数据，增加了基于Eigen3的无迹卡尔曼滤波UKF，修复了一些已知的bug