# Bender [![Build Status](https://travis-ci.org/luozan/Bender.svg?branch=master)](https://travis-ci.org/luozan/Bender)

### 目前收获的经验
                
1. tf.nn.softmax_cross_entropy_with_logits 包含激活函数softmax和交叉熵，并且处理了log导致的NaN问题。
如果传入激活函数的返回值做参数，会出现loss不下降的现象。
2. 学习率过大，会导致loss震荡，不下降。
3. 使用CNN学习的过程中，为了忽略图片像素RBG通道值中不必要的信息（亮度等），需要对图像做标准化处理。
未做处理会出现loss先短暂下降然后一直增长的现象，并且训练集的准确率会很低，不增长。
4. 当增加数据量过后，出现loss一直增长的现象，可通过添加BN层解决。

