# 项目说明:
本项目是关系抽取相关模型的代码复现  
包括以下四种方法
- 序列标注
- 层叠式指针网络（基于主语感知）
- Multi-head Selection
- Deep Biaffine Attention  

用的数据是百度21年语言技术经验竞赛抽取赛道的数据，四种方法的效果如下表，更详细的请看我的知乎博文
https://zhuanlan.zhihu.com/p/381894616

|                                | F1值 |
| ------------------------------ | ----- |
| 官方baseline                 | 64.69 |
| 层叠式指针网络（基于主语感知） | 61.22 |
| Multi-head Selection           | 67.90 |
| Deep Biaffine Attention        | 68.45 |

# 环境
- python=3.6
- torch=1.7
- transformers=4.5.0

# 运行示例
序列标注
```
python3 run_baseline.py
--max_len=200
--model_name_or_path=预训练模型路径
--per_gpu_train_batch_size=80
--per_gpu_eval_batch_size=100
--learning_rate=1e-4
--num_train_epochs=40
--output_dir="./output"
--weight_decay=0.01
--early_stop=2
```
层叠式指针网络（基于主语感知）
```
python3 run_mpn.py
--max_len=200
--model_name_or_path=预训练模型路径
--per_gpu_train_batch_size=100
--per_gpu_eval_batch_size=100
--learning_rate=1e-4
--num_train_epochs=40
--output_dir="./output"
--weight_decay=0.01
--early_stop=2
```
Multi-head Selection
```
python3 run_mhs.py
--max_len=200
--model_name_or_path=/data/zhoujx/prev_trained_model/rbt3
--per_gpu_train_batch_size=25
--per_gpu_eval_batch_size=30
--learning_rate=1e-4
--num_train_epochs=40
--output_dir="./output"
--weight_decay=0.01
--early_stop=2
```
Deep Biaffine Attention
```
python3 run_mhs_biaffine.py
--max_len=200
--model_name_or_path=/data/zhoujx/prev_trained_model/rbt3
--per_gpu_train_batch_size=15
--per_gpu_eval_batch_size=20
--learning_rate=1e-4
--num_train_epochs=40
--output_dir="./output"
--weight_decay=0.01
--early_stop=2
--overwrite_cache=True
```
