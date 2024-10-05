import logging
import torch
from scipy import stats
import numpy as np
import data_loader
import sys

sys.path.append('../')
import models2 as models


class RankSlover(object):

    def __init__(self, config, path_train, train_idx, test_idx):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num

        self.model_rank = models.RankNet(config).cuda()
        self.model_rank.train(True)

        # self.res.load_state_dict(torch.load('../model/' + config.data + '/res.pkl'))
        save_model = torch.load('../model_pth/' + config.data + '/res.pkl')
        model_dict = self.model_rank.res.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.model_rank.res.load_state_dict(model_dict)

        self.model_rank.quality_conv.load_state_dict(torch.load('../model_pth/' + config.data + '/Conv1.pkl'))
        self.model_rank.distortion_conv.load_state_dict(torch.load('../model_pth/' + config.data + '/Conv2.pkl'))
        print("导入预训练参数")
        logging.info("导入预训练参数")

        self.l1_loss = torch.nn.L1Loss().cuda()

        res_params = list(map(id, self.model_rank.res.parameters()))
        quality_conv_params = list(map(id, self.model_rank.quality_conv.parameters()))
        distortion_conv_params = list(map(id, self.model_rank.distortion_conv.parameters()))
        backbone_params = res_params + quality_conv_params + distortion_conv_params
        self.fc_params = filter(lambda p: id(p) not in backbone_params, self.model_rank.parameters())

        self.lr = config.lr
        self.weight_decay = config.weight_decay
        paras = [{'params': self.fc_params, 'lr': self.lr * 10},
                 {'params': self.model_rank.res.parameters(), 'lr': self.lr},
                 {'params': self.model_rank.quality_conv.parameters(), 'lr': self.lr},
                 {'params': self.model_rank.distortion_conv.parameters(), 'lr': self.lr},
                 ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset_train, path_train, train_idx, config.patch_size,
                                              config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset_test, path_train, test_idx, config.patch_size, config.test_patch_num,
                                             istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tRMSE')
        logging.info('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tRMSE')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for img, label in self.train_data:
                img = torch.tensor(img.cuda())  # b,3,224,224
                label = torch.tensor(label.cuda())
                self.solver.zero_grad()
                pred = self.model_rank(img)
                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss = self.l1_loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc, test_rmse = self.test(self.test_data)
            mask = ''
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                mask = '*'
                torch.save(self.model_rank.state_dict(), '../model_pth/led/' + str(t) + '.pt')
            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%s' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc, test_rmse, mask))
            logging.info('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%s' %
                         (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc, test_rmse, mask))
        print('Best test SRCC %f, PLCC %f, RMSE %f' % (best_srcc, best_plcc, test_rmse))
        logging.info('Best test SRCC %f, PLCC %f, RMSE %f' % (best_srcc, best_plcc, test_rmse))

        return best_srcc, best_plcc, test_rmse

    def test(self, data):
        """Testing"""
        self.model_rank.train(False)
        pred_scores = []
        gt_scores = []
        with torch.no_grad():
            for img, label in data:
                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda())

                pred = self.model_rank(img)
                pred_scores.append(float(pred.item()))
                gt_scores = gt_scores + label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)

        # Calculate RMSE
        test_rmse = np.sqrt(((pred_scores - gt_scores) ** 2).mean())

        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model_rank.train(True)
        return test_srcc, test_plcc, test_rmse