import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    # 数据
    parser.add_argument("--max_len", default=512, type=int, help="最大长度")
    parser.add_argument("--stride", type=int, default=100, help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    #
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="训练Batch size的大小")
    parser.add_argument("--per_gpu_eval_batch_size", default=20, type=int, help="验证Batch size的大小")

    # 训练的参数
    parser.add_argument("--do_distri_train", action="store_true", help="是否用两个卡并行训练")
    parser.add_argument("--model_name_or_path", default="/data/zhoujx/prev_trained_model/chinese_roberta_wwm_ext_pytorch", type=str, help="预训练模型的路径")
    parser.add_argument("--num_train_epochs", default=50.0, type=float, help="训练轮数")
    parser.add_argument("--early_stop", default=8, type=int, help="早停")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="transformer层学习率")
    parser.add_argument("--linear_learning_rate", default=1e-3, type=float, help="linear层学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--seed", type=int, default=66, help="random seed for initialization")
    parser.add_argument("--output_dir", default="./output", type=str, help="保存模型的路径")
    # 预测时
    parser.add_argument("--n_best_size", type=int, default=5, help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", type=int, default=150, help="Max answer length.")
    parser.add_argument("--cls_threshold", type=float, default=0.5, help="No answer threshold")

    return parser