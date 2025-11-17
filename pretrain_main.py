import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from pretrain_trainer import Trainer
from datasets.pretraining_dataset import PretrainingDataset, CustomSampler
from torch.utils.data import ConcatDataset
from models.eegmamba import EEGMamba


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='EEG Foundation Model')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=3, help='cuda number (default: 1)')
    parser.add_argument('--parallel', type=bool, default=False, help='parallel')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight_decay')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR',
                        help='lr_scheduler: CosineAnnealingLR, ExponentialLR, StepLR, MultiStepLR, CyclicLR')

    # parser.add_argument('--project_mode', type=str, default='cnn', help='project_mode')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--in_dim', type=int, default=200, help='in_dim')
    parser.add_argument('--out_dim', type=int, default=200, help='out_dim')
    parser.add_argument('--d_model', type=int, default=200, help='d_model')
    parser.add_argument('--dim_feedforward', type=int, default=800, help='dim_feedforward')
    parser.add_argument('--seq_len', type=int, default=30, help='seq_len')
    parser.add_argument('--n_layer', type=int, default=12, help='n_layer')
    parser.add_argument('--nhead', type=int, default=8, help='nhead')
    parser.add_argument('--need_mask', type=bool, default=True, help='need_mask')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask_ratio')

    parser.add_argument('--dataset_dir', type=str, default='dataset_dir',
                        help='dataset_dir')
    parser.add_argument('--model_dir',   type=str,   default='model_dir', help='model_dir')
    params = parser.parse_args()
    print(params)
    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    dataset_dirs = [
        r'/data3/datasets/tueg/ordered_processed_200Hz_db',
        r'/data2/datasets/Physionet2018/processed',
        r'/data2/datasets/Raw EEG Data/processed_average',
        r'/data2/datasets/Siena_Scalp_EEG_Database/processed_average',
        r'/data2/datasets/szbd/processed_average',
    ]

    pretrained_datasets = ConcatDataset(
        [PretrainingDataset(dataset_dir=dataset_dir) for dataset_dir in dataset_dirs]
    )
    for cur_dataset in pretrained_datasets.datasets:
        print(len(cur_dataset))
    print("The number of samples:", len(pretrained_datasets))
    # print(len(pretrained_dataset))
    # sampler = RandomSampler(data_source=pretrained_datasets)
    data_loader = DataLoader(
        pretrained_datasets,
        sampler=CustomSampler(dataset=pretrained_datasets, batch_size=params.batch_size),
        batch_size=params.batch_size,
        num_workers=8,
    )
    model = EEGMamba(
        params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len, params.n_layer,
        params.nhead
    )
    trainer = Trainer(params, data_loader, model)
    trainer.train()


if __name__ == '__main__':
    main()
