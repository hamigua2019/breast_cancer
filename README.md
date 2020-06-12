# breast_cancer

一、比赛信息

用户名：hamimelong2019

名次：暂列13名

二、作业自评

1. 水平背景
   与NLP相比，我对计算机视觉涉猎甚少，所知甚少。NLP基本上大概听说过知道word2vec、Hanlp等等模型，复现过九歌诗歌模型，算是有一些了解。但是计算机视觉    方面，只实现过pytorch官方文档中的玩具数据集，更没有听说过过医学图像数据集，也许有两个原因，一个是计算机视觉的images对存储空间和计算空间要求较高，    处理起来多有不便，另外个人对图像计算的实际运用需求不如NLP大，由此所知甚少。而且对卷积网络参数的设置、计算也不甚了然。所以，数字图像+医学图像+CNN    参数设置，对我来说，有一定的难度。

2. 作业的教训与收获

    这次作业就结果来说，做得不是很成功，但是就过程来说，也有不少收获。不怎么成功在于没有得到一个很完美的结果，收获在于借此机会，扩大深化了对图像处理    和医学图像处理的流程的了解；提高了独立解决bug的能力。

具体来说，如下：

1. 不成功之处
1）受CNN过拟合的影响，在归因和寻找解决方法方面花去了大半时间，导致CNN没有训练出整体的很好的分类结果，训练出来的结果基本上都是一个类别；

2）比赛中13名的成绩是上一次作业时的成绩，这次没有时间再进行合并修正改进；时间分配方面，有点头重脚轻。

2. 一些收获
收获在于：
1）完成了卷积神经网络进行图像分类，了解了过程；
   对过程中出现的两个bug进行了解决。
   一个bug是，“RuntimeError: Given groups=1, weight of size 6 1 5 5, expected input[4, 3, 224, 224] to have 1 channels, but got 3 channels instead。”
   这个bug久攻不下，模模糊糊，在学堂在线作业讨论区问了助教同学，得到灰度归一化的启示，有启发，但是局于知识储备有限，搜索关键词没有找对，还是没有搜索研究思考出具体的修改方法，最后请教了百度做自动驾驶的王同学解决，解决办法是，在transform.compose()函数下，添加了一个灰度转换参数：transforms.Grayscale(num_output_channels=3)，于是解决。
   关键点：我搜索的关键词是“三通道转换为一通道”，但是正确的搜索关键词应该为“灰度转换”。
   
   第二个bug是，卷积参数计算错误。
   bug为，“shape [-1, 400] is invalid for input of size 179776”，self.fc1 = nn.Linear(16 * ？ * ？, 120)，这里的参数应该是什么，这个问题困扰很久，最后搜到一个类似的帖子解决，一个人日本人在Stack Overflow上发的帖子。后来了解到算法如下，因为179776 = 16 * 4 （batch）* 54 * 54，于是解决。
   
2）了解了医学图像处理流程，常用医学图像软件或库，这些工作虽然对结果的产生并没有实质性的帮助，但是掌握一个新的领域的知识和方法也是很重要的。
* 通过B站上海交通大学生命科学学院顾力栩教授的《医学图像处理》视频课，了解到医学图像处理流程；了解了图像二值化（threshold value）、直方图、分邻点，等等概念；了解到医学图像配准等概念和方法。配准在医学图像和区域地理方面都有深入广泛的运用。
* 下载了一些关于breast cancer的论文，了解到医学图像数据集较少，可使用迁移学习、生成学习、翻转等等生成创造新数据补充数据；
* 模型方面，医学图像适合用unet+resnet来解决；
* 医学图像识别软件有，sampleITK、Span-ITK等，可以识别出病灶的大小（见GitHub中文件1.txt）；OpenCV可以识别图像的灰度值（见GitHub中文件Img_threshold_value.py）；
但是没有找到如何批量识别这些数据，也对这些软件的参数不够熟悉，造成实际训练过程中，没有对这些值进行应用。
还找到网络软件识别了图中医生手动标注的病灶尺寸（见GitHub中文件061115591168_02_In1.txt），同样由于没有找到如何批量识别方法，所以没有进行运用。识别网站为：http://www.pdfdo.com/image-to-txt.aspx ，语言由简体中文改为英文。

3）基本了解数字图像处理原理概况，了解到一些书籍，为继续学习打下基础。比如《数字图像处理》第四版等。

三、不足与下一步需要做的工作
1. 对图像处理的高阶模型需要更进一步的学习掌握；
2. 研读《数字图像处理》书籍，研读一些医学图像处理论文；
3. 代码工程能力是一个瓶颈，需要多练多思考多总结，做到没有障碍，运用自如。

感谢老师们的教授指导，缺失不足之处，敬请不吝指正。

2020. 6. 12
