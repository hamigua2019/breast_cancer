# breast_cancer

一、比赛信息

用户名：hamimelong2019

名次：暂列13名

二、作业评价

这次作业就结果来说做得不怎么成功。
受CNN过拟合的影响，在归因和寻找解决方法方面花去了大半时间，导致CNN没有训练出整体的很好的分类结果，训练出来的结果基本上都是一个类别；
希望通过医学图像预处理来解决过拟合的问题，找到了几种思路和医学图像处理软件，但由于对它们不熟悉以及
。比赛中13名的成绩是上一次作业时的成绩，这次没有时间再进行修正合并改进。

时间分配方面，有点头重脚轻。

主要是过拟合、过拟合引起的CNN的参数调整、过拟合引起的图像预处理花去了较多时间。

头重脚轻，用在。表现在，用CNN训练图片，出现了过拟合的情况，力图寻找原因并解决出现了一些难题，花去了大半的时间。
，按时间顺序分别归因如下：
1）数据量够，是参数设计不对。起初出现“RuntimeError: Given groups=1, weight of size 6 1 5 5, 
expected input[4, 3, 224, 224] to have 1 channels, but got 3 channels instead”的bug，