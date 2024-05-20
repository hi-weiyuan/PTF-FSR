import argparse
import torch.cuda as cuda


def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")
    parser.add_argument('--client_device', nargs='?', default='cuda:0' if cuda.is_available() else 'cpu', help='Which device to run the model.')
    parser.add_argument('--server_device', nargs='?', default='cuda:1' if cuda.is_available() else 'cpu', help='Which device to run the model.')

    parser.add_argument('--path', nargs='?', default='./Data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='', help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of communication round.')
    parser.add_argument('--num_neg', type=int, default=1, help='Number of negative items.')
    parser.add_argument('--batch_size', type=int, default=256, help='Client Batch size.')

    parser.add_argument('--max_len', type=int, default=50, help='max length of a sequence.')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate.')
    parser.add_argument('--server_dropout', type=float, default=0.1, help='dropout rate.')

    parser.add_argument('--l2_weight', type=float, default=0.01, help='weight decay for optimizer')
    parser.add_argument('--server_emb_dim', type=int, default=256, help='Dim of server embedding.')
    parser.add_argument("--freeze_paras_before", type=int, default=0)
    parser.add_argument("--num_words_title", type=int, default=30)
    parser.add_argument('--server_dim', type=int, default=256, help='Dim of server latent vectors.')
    parser.add_argument('--server_model_type', nargs='?', default="MoRec", help='GRU4Rec, SaSRec or MoRec.')
    parser.add_argument("--bert_model_load", type=str, default='/root/bert-small', help="/root/bert-small or /root/bert-base-uncased")
    parser.add_argument('--global_lr', type=float, default=1e-4, help='global Learning rate.')
    parser.add_argument('--bert_lr', type=float, default=5e-5, help='global Learning rate for bert.')
    parser.add_argument('--server_bs', type=int, default=256, help='Server Batch size.')
    parser.add_argument('--server_n_blocks', type=int, default=2, help='Number of server_n_blocks.')
    parser.add_argument('--server_n_heads', type=int, default=2, help='Number of server_n_heads.')
    parser.add_argument('--server_epoch', type=int, default=2, help='Number of server training epochs.')

    parser.add_argument('--client_emb_dim', type=int, default=8, help='Dim of server latent vectors.')
    parser.add_argument('--client_dim', type=int, default=8, help='Dim of client latent vectors.')
    parser.add_argument('--client_n_blocks', type=int, default=2, help='Number of client_n_blocks.')
    parser.add_argument('--client_n_heads', type=int, default=1, help='Number of client_n_heads.')
    parser.add_argument('--local_epoch', type=int, default=5, help='Number of local training epochs.')
    parser.add_argument('--local_lr', type=float, default=0.01, help='local Learning rate.')
    parser.add_argument('--local_bs', type=int, default=256, help='Local Batch size.')
    parser.add_argument('--local_model_type', nargs='?', default="GRU4Rec", help='GRU4Rec or SaSRec.')

    parser.add_argument('--replace_ratio', type=float, default=0.5, help='replace ratio.')
    parser.add_argument('--epsilon', type=float, default=1., help='privacy protection, larger epsilon means less privacy protection.')
    parser.add_argument('--pc_cl', type=bool, default=True, help='whether use sequence-level contrastive learning.')
    parser.add_argument('--pc_cl_control', type=float, default=0.01, help='replace ratio.')
    parser.add_argument('--is_cl', type=bool, default=True, help='whether use target-level contrastive learning.')
    parser.add_argument('--is_cl_epoch', type=int, default=5, help='The epoch start using tag_cl')
    parser.add_argument('--is_cl_control', type=float, default=0.01, help='replace ratio.')

    return parser.parse_args()


args = parse_args()
