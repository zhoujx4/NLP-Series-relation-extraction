# 项目说明:
百度2021年语言与智能技术竞赛多形态信息抽取赛道关系抽取部分Pytorch版baseline  
比赛链接:https://aistudio.baidu.com/aistudio/competition/detail/65?isFromLuge=true
> 官方的baseline版本是基于paddlepaddle框架的,我把它改写成了Pytorch框架,其中大部分代码沿用的是官方提供的代码（如评测代码、保存预测文件代码等）
>,只是对数据读取部分（感觉原代码这部分写得稍微复杂了一点，这里进行了简化）和框架部分进行了修改,习惯用Pytorch版本的可以基于此进行优化.

# 环境
- python=3.6
- torch=1.7
- transformers=4.5.0
# 训练示例
训练  
```
python run.py
--max_len=150
--model_name_or_path=下载的预训练模型路径
--per_gpu_train_batch_size=200
--per_gpu_eval_batch_size=500
--learning_rate=1e-5
--linear_learning_rate=1e-2
--num_train_epochs=100
--output_dir="./output"
--weight_decay=0.01
--early_stop=2
```
预测
```
python predict.py
--max_len=150
--model_name_or_path=下载的预训练模型路径
--per_gpu_eval_batch_size=500
--output_dir="./output"
--fine_tunning_model=微调后的模型路径
```

# 实验结果
用的baseline模型是三层的roBERTa(具体请看https://github.com/ymcui/Chinese-BERT-wwm)
在官方提供的dev集上的表现如下：

![image-20210412144557325](https://raw.githubusercontent.com/zhoujx4/PicGo/main/img/image-20210412144557325.png)



# 后续优化策略

由于数据量比较充足，可以往模型架构进行优化，做关系抽取的有几种模型架构形式，最后进行集合一下应该能显著提供效果。