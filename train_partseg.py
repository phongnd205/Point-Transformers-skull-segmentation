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

@hydra.main(config_path='config', config_name='partseg')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    root = hydra.utils.to_absolute_path('data/shapenetcore_partanno_segmentation_benchmark_v0_normal/')

    '''MODEL LOADING'''
    args.input_dim = (6 if args.normal else 3) + 1
    args.num_class = 11
    num_category = 1
    num_part = 11
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerSeg')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()


    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0

    loss_train = []
    loss_valid = []
    acc_train = []
    acc_valid = []

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        logger.info('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        loss_train_batch = []
        for batch_id in range(int(targets_train.shape[0]/args.batch_size)):
            
            label_train_batch = np.zeros([args.batch_size,1], dtype=int)
            label_train_batch = torch.Tensor(label_train_batch)
            targets_train_batch = targets_train[0+batch_id*args.batch_size:args.batch_size+batch_id*args.batch_size,:]

            points_train_batch = points_train[0+batch_id*args.batch_size:args.batch_size+batch_id*args.batch_size,:,:]#.data.numpy()
            points_train_batch[:, :, 0:3] = provider.random_scale_point_cloud(points_train_batch[:, :, 0:3])
            points_train_batch[:, :, 0:3] = provider.shift_point_cloud(points_train_batch[:, :, 0:3])
            points_train_batch = torch.Tensor(points_train_batch)
            points_train_batch = points_train_batch.permute(0, 2, 1)
            optimizer.zero_grad()

            
            #print(f'points_train_batch before: {points_train_batch.shape}')
            targets_train_batch = targets_train_batch.reshape(-1, 1)[:, 0]
            targets_train_batch = torch.Tensor(targets_train_batch)
            points_train_batch, label_train_batch, targets_train_batch = points_train_batch.float().cuda(), label_train_batch.long().cuda(), targets_train_batch.long().cuda()

            #print(f"points_train_batch: {points_train_batch.shape}")
            #print(f"label_train_batch: {label_train_batch.shape}")
            #print(f"shape cateorical: {to_categorical(label_train_batch, num_category).repeat(1, points_train_batch.shape[1], 1).shape}")

            seg_pred = classifier(torch.cat([points_train_batch, to_categorical(label_train_batch, num_category).repeat(1, points_train_batch.shape[1], 1)], -1))
            #seg_pred = classifier(points_train_batch, to_categorical(label_train_batch, num_category))
            
            #print(f"seg_pred before: {seg_pred.shape}")
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            #target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(targets_train_batch.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.num_point))
            loss = criterion(seg_pred, targets_train_batch)
            loss.backward()
            optimizer.step()

            loss_train_batch.append(loss.cpu().item())

        loss_train = np.append(loss_train, np.mean(loss_train_batch))
        train_instance_acc = np.mean(mean_correct)
        acc_train = np.append(acc_train, np.mean(mean_correct))
        logger.info('Train accuracy is: %.5f' % train_instance_acc)

        np.save('acc_train.npy',acc_train)
        np.save('loss_train.npy',loss_train)

        with torch.no_grad():
            test_metrics = {}

            classifier = classifier.eval()
            mean_correct_test = []
            loss_valid_batch = []


            for batch_id in range(int(targets_valid.shape[0]/args.batch_size)):
                label_test_batch = np.zeros([args.batch_size,1], dtype=int)
                label_test_batch = torch.Tensor(label_test_batch)
                targets_test_batch = targets_valid[0+batch_id*args.batch_size:args.batch_size+batch_id*args.batch_size,:]
                targets_test_batch = torch.Tensor(targets_test_batch)
                points_test_batch = points_valid[0+batch_id*args.batch_size:args.batch_size+batch_id*args.batch_size,:,:]#.data.numpy()
                points_test_batch = torch.Tensor(points_test_batch)
                points_test_batch = points_test_batch.permute(0, 2, 1)

                points_test_batch, label_test_batch, targets_test_batch = points_test_batch.float().cuda(), label_test_batch.long().cuda(), targets_test_batch.long().cuda()
                
                seg_pred = classifier(torch.cat([points_test_batch, to_categorical(label_test_batch, num_category).repeat(1, points_test_batch.shape[1], 1)], -1))
                
                seg_pred = seg_pred.contiguous().view(-1, num_part)
                targets_test_batch = targets_test_batch.reshape(-1, 1)[:, 0]

                loss_val = criterion(seg_pred, targets_test_batch)
                loss_valid_batch.append(loss_val.cpu().item())

                pred_choice = seg_pred.data.max(1)[1]
                correct = pred_choice.eq(targets_test_batch.data).cpu().sum()
                mean_correct_test.append(correct.item() / (args.batch_size * args.num_point))

            loss_valid = np.append(loss_valid, np.mean(loss_valid_batch))
            acc_valid = np.append(acc_valid, np.mean(mean_correct_test))
            test_metrics['accuracy'] = np.mean(mean_correct_test)


            np.save('acc_valid.npy',acc_valid)
            np.save('loss_valid.npy',loss_valid)

        logger.info('Test Accuracy: %f ' % test_metrics['accuracy'])

        if (test_metrics['accuracy'] >= best_acc):
            logger.info('Save model...')
            savepath = 'best_model.pth'
            logger.info('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            logger.info('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        logger.info('Best accuracy is: %.5f' % best_acc)
        global_epoch += 1


if __name__ == '__main__':

    point_all_parts = np.load('data/vertex_rotation_augmented_sample.npy')
    label_all_parts = np.load('data/label_rotation_augmented_sample.npy')

    points_data = point_all_parts.transpose(0,2,1)
    targets_data = label_all_parts
    
    points_train = points_data[0:1840,:,0:2048]
    targets_train = targets_data[0:1840,0:2048]
    
    # test (528), val (264), train (1840)
    # Test (0:528)      (528:1056)          (1056:1584) (1584:2112) (2112:2632)
    # train (528:2368)  (0:264, 1056:2632)  (0:792, 1584:2632) (0:1320, 2112:2632)

    # points_train = np.concatenate([points_data[0:1320,:,0:2048], points_data[2112:2632,:,0:2048]], axis=0)
    # targets_train = np.concatenate([targets_data[0:1320,0:2048], targets_data[2112:2632,0:2048]], axis=0)
    
    points_valid = points_data[1840:2096,:,0:2048]
    #label_valid = label_data[2368:2632,:]
    targets_valid = targets_data[1840:2096,0:2048]

    del point_all_parts, label_all_parts, points_data, targets_data
    
    main()