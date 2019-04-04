# LineSegmentDetector17
基于C+CPP+OpenCV4.0实现，以C为主，OpenCV向下兼容3.0，编译环境为VS17，可以自行建立项目，mapValue是地图文件，mapParam是地图信息，其他为调试用文件

LSD算法解释https://www.cnblogs.com/Pyrokine/p/10384930.html

LSD算法动画https://www.bilibili.com/video/av43174965/

V1.0 完成了基本算法的实现以及基础动画的实现

V1.1 完成了算法的优化， 将多个定长数组用动态数组实现，极大地减小了内存空间，将原算法的伪排序使用快速排序实现，优化了高斯降采样时计算高斯核的算法，改进后一张图仅需计算一次高斯核，所以将区域增长次数从多次减少到1次

V1.2 完成了Degree图的动画（在main_with_disp里面），修正了UsedMap的动画，以及增加了梯度排名和区域内像素点显示，增加了虚警数的数值显示，提取出了LSD算法为单独函数

V1.3 新增了基于Ramer-Douglas-Peucker算法的激光雷达点云分割代码

V2.0 提取出了LSD和RDP算法到单独文件并可独立调用，新增Base_Func实现共用的函数和结构体，并记录主体结构的变化，引入FeatureAssociation算法，实现RDP的数据在LSD中的匹配的算法，新增数据转换函数，增加RDP注释量，调整文件结构，删除main_with_disp函数及相关生成文件

V2.1 在myLSD中增加了mapCache的计算，用于特征匹配的先验概率，修正了LSD计算时的原始图像，清除了地图中间值为255的部分