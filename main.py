import argparse


def create_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('--vocab_path', default='model/vocab.txt', type=str)
    arg.add_argument('--triple_num', default=3691)
    arg.add_argument('--bart_config_path', default='model/bart/config.json')
    arg.add_argument('--gpt_config_path', default='model/gpt2/config.json')
    arg.add_argument('--bart_model_path', default='model/bart', type=str)
    arg.add_argument('--gpt_model_path', default='model/gpt2', type=str)
    arg.add_argument('--saved_model_path', default='model/save_model', type=str)
    arg.add_argument('--CMC_data_path', default='data/CMC/zh_CMC2.txt', help='CMC数据集')
    arg.add_argument('--literature_data_path', default='data/Chinese_literature_dataset.txt', help='无标签诗歌数据集')
    arg.add_argument('--batch_size', default=6, type=int)
    arg.add_argument('--epochs', default=20, type=int)
    arg.add_argument('--lr', default=5e-5)
    arg.add_argument('--device', default='cuda', type=str)
    arg.add_argument('--vocab_length', default=21128)
    arg.add_argument('--validation_num', default=8000)
    arg.add_argument('--max_length', default=150)
    args = arg.parse_args()
    return args
