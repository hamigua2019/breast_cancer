# breast_cancer

用户名：hamimelong2019
名次：暂列13名

这次作业就结果来说做得不怎么成功，受CNN过拟合的影响，在归因和寻找解决方法方面花去了大半的时间，导致没有训练出整体的很好的分类结果，用CNN训练出来的结果基本上都是一个类别。上一次同样的作业时

头重脚轻，用在。表现在，用CNN训练图片，出现了过拟合的情况，力图寻找原因并解决出现了一些难题，花去了大半的时间。
，按时间顺序分别归因如下：
1）数据量够，是参数设计不对。起初出现“RuntimeError: Given groups=1, weight of size 6 1 5 5, 
expected input[4, 3, 224, 224] to have 1 channels, but got 3 channels instead”的bug，
