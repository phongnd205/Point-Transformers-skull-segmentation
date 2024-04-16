"""
Author: DucPhong
Date: Mars 2024
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import timeit
from pathlib import Path
from tqdm import tqdm
import hydra
import omegaconf

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def calculateIoU(predicts, targets, partID):
    index_partID_predict = (predicts == partID).nonzero(as_tuple=True)[0]
    index_partID_target = (targets == partID).nonzero(as_tuple=True)[0]
    combine_set = torch.unique(torch.cat((index_partID_predict,index_partID_target)))
    IoU = (len(index_partID_predict) + len(index_partID_target) - len(combine_set))/ len(combine_set)*100
    return IoU

def calculateMeanIoU(predicts, targets):
    npoint_predict = []
    npoint_target = []
    npoint_combine = []
    for partID in range(11):
        index_partID_predict = np.where(predicts == partID)[0]
        index_partID_target = np.where(targets == partID)[0]
        combine_set = np.unique(np.concatenate((index_partID_predict,index_partID_target)))
        npoint_predict.append(len(index_partID_predict))
        npoint_target.append(len(index_partID_target))
        npoint_combine.append(len(combine_set))
    IoU = (np.sum(npoint_predict) + np.sum(npoint_target) - np.sum(npoint_combine)) / np.sum(npoint_combine)*100
    return IoU


@hydra.main(config_path='config', config_name='partseg')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    #root = hydra.utils.to_absolute_path('data/shapenetcore_partanno_segmentation_benchmark_v0_normal/')

    '''MODEL LOADING'''
    args.input_dim = (6 if args.normal else 3) + 1
    args.num_class = 11
    num_category = 1
    num_part = 11
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')
    #print(f"args.model.name {args.model.name}")

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerSeg')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint = torch.load('2048_fold5/best_model.pth')
    #start_epoch = checkpoint['epoch']
    classifier.load_state_dict(checkpoint['model_state_dict'])
    logger.info('Load pretrain model')
    
    start = timeit.default_timer()
    with torch.no_grad():
        test_metrics = {}

        classifier = classifier.eval()
        mean_correct_test = []
        loss_valid_batch = []
        targets_test_all = torch.empty(0)
        predicts_test_all = torch.empty(0)

        for batch_id in range(int(targets_test.shape[0]/args.batch_size)):
            label_test_batch = np.zeros([args.batch_size,1], dtype=int)
            label_test_batch = torch.Tensor(label_test_batch)
            targets_test_batch = targets_test[0+batch_id*args.batch_size:args.batch_size+batch_id*args.batch_size,:]
            targets_test_batch = torch.Tensor(targets_test_batch)
            points_test_batch = points_test[0+batch_id*args.batch_size:args.batch_size+batch_id*args.batch_size,:,:]
            points_test_batch = torch.Tensor(points_test_batch)
            points_test_batch = points_test_batch.permute(0, 2, 1)

            points_test_batch, label_test_batch, targets_test_batch = points_test_batch.float().cuda(), label_test_batch.long().cuda(), targets_test_batch.long().cuda()

            seg_pred = classifier(torch.cat([points_test_batch, to_categorical(label_test_batch, num_category).repeat(1, points_test_batch.shape[1], 1)], -1))

            seg_pred = seg_pred.contiguous().view(-1, num_part)
            targets_test_batch = targets_test_batch.reshape(-1, 1)[:, 0]

            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(targets_test_batch.data).cpu().sum()
            mean_correct_test.append(correct.item() / (args.batch_size * args.num_point))
        
            targets_test_all = torch.cat((targets_test_all,targets_test_batch.cpu()))
            predicts_test_all = torch.cat((predicts_test_all, pred_choice.cpu()))

        print(f"Mean accuracy : {np.mean(mean_correct_test)}")
        print(f'{round(calculateMeanIoU(predicts_test_all, targets_test_all),2)}') #mean IoU: 
        
        np.save('labels_predict_fold_5.npy',predicts_test_all)
        np.save('labels_target_fold_5.npy',targets_test_all)

        for idx in range(11):
            print(round(calculateIoU(targets_test_all, predicts_test_all, idx),2))
    stop = timeit.default_timer()
    print(f"Testing time: {round(stop - start,2)}")




if __name__ == '__main__':

    point_all_parts = np.load('data/vertex_rotation_augmented_sample.npy')
    label_all_parts = np.load('data/label_rotation_augmented_sample.npy')

    points_data = point_all_parts.transpose(0,2,1)
    targets_data = label_all_parts
    
    points_test = points_data[0:528,:,0:2048]
    targets_test = targets_data[0:528,0:2048]

    # points_train = points_data[528:2368,:,0:1024]
    # targets_train = targets_data[528:2368,0:1024]
    
    # points_valid = points_data[2368:2632,:,0:1024]
    # # label_valid = label_data[2368:2632,:]
    # targets_valid = targets_data[2368:2632,0:1024]
    np.save('points_test_fold_5.npy',points_test)
    del point_all_parts, label_all_parts, points_data, targets_data
        
    main()