import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
from config import get_test_config
from data import ModelNet40
from models import MeshNet
from utils import append_feature, calculate_map


cfg = get_test_config()
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']


data_set = ModelNet40(cfg=cfg['dataset'], part='test')
data_loader = data.DataLoader(data_set, batch_size=1, num_workers=4, shuffle=True, pin_memory=False)


def test_model(model):

    correct_num = 0
    ft_all, lbl_all = None, None

    for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader):
        if torch.cuda.is_available():
            centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
            corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
            normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
            neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))
            targets = Variable(torch.cuda.LongTensor(targets.cuda()))

        outputs, feas = model(centers, corners, normals, neighbor_index)
        _, preds = torch.max(outputs, 1)

        if preds[0] == targets[0]:
            correct_num += 1

        ft_all = append_feature(ft_all, feas.detach())
        lbl_all = append_feature(lbl_all, targets.detach(), flaten=True)

    print('Accuracy: {:.4f}'.format(float(correct_num) / len(data_set)))
    print('mAP: {:.4f}'.format(calculate_map(ft_all, lbl_all)))


if __name__ == '__main__':

    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    if torch.cuda.is_available():
        model.cuda()
    model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(cfg['load_model']))
    else:
        model.load_state_dict(torch.load(cfg['load_model'],map_location='cpu'))
    model.eval()

    test_model(model)
