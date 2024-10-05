import os
import argparse
import random
import numpy as np
from RankSlover import RankSlover
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import warnings

warnings.filterwarnings("ignore")


def main(config):
    folder_path = {
        'live': '/usr/Benke/data/live',
        'csiq': '/home/dataset/CSIQ',
        'tid2013': '/home/dataset/tid2013',
        'livec': '/home/user/Benke/Hulingbi/data/ChallengeDB_release/',
        'koniq-10k': '/home/dataset/koniq-10k',
        'bid': '/usr/Benke/Hulingbi/data/RBID',
    }
    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'bid': list(range(0, 586)),
    }
    sel_num= img_num[config.dataset_train]  # list(range(0, 1162))

    srcc_all = np.zeros(config.train_test_num, dtype=np.float64)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float64)
    rmse_all = np.zeros(config.train_test_num, dtype=np.float64)
    print('Training and testing on %s dataset for %d rounds...' % (config.dataset_train, config.train_test_num))
    logging.info('Training and testing on %s dataset for %d rounds...' % (config.dataset_train, config.train_test_num))

    for i in range(config.train_test_num):
        print('Round %d' % (i + 1))
        logging.info('Round %d' % (i + 1))
        random.shuffle(sel_num)
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
        # train_index = sel_num_train
        # test_index = sel_num_test

        solver = RankSlover(config, folder_path[config.dataset_train], train_index, test_index)
        srcc_all[i], plcc_all[i], rmse_all[i] = solver.train()

    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)
    rmse_med = np.median(rmse_all)
    srcc_avg = np.average(srcc_all)
    plcc_avg = np.average(plcc_all)
    rmse_avg = np.average(rmse_all)

    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f,\tmedian RMSE %4.4f' % (srcc_med, plcc_med, rmse_med))
    print('Testing average SRCC %4.4f,\taverage PLCC %4.4f,\taverage RMSE %4.4f' % (srcc_avg, plcc_avg, rmse_avg))

    logging.info('Testing median SRCC %4.4f,\tmedian PLCC %4.4f,\tmedian RMSE %4.4f' % (srcc_med, plcc_med, rmse_med))
    logging.info('Testing average SRCC %4.4f,\taverage PLCC %4.4f,\taverage RMSE %4.4f' % (srcc_avg, plcc_avg, rmse_avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train', dest='dataset_train', type=str, default='tid2013',
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--dataset_test', dest='dataset_test', type=str, default='led',
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=5,
                        help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=5,
                        help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-7, help='Weight decay')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=24, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=30, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10,
                        help='Train-test times')  # 训练轮数
    parser.add_argument('--data', dest='data', type=str, default='20')
    config = parser.parse_args()

    main(config)
