#coding=utf-8
## code on 1032
from __future__ import print_function, absolute_import
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.backends import cudnn
import torch.optim as optim

import os, shutil
import argparse
import sys
import time
import numpy as np
import scipy.io as scio


import misc.custom_loss as custom_loss

import dataset.sk_dataset as sk_dataset
import dataset.sh_views_dataset as sh_views_dataset

import misc.transforms as T

import models

import models.vgg11_bn as net_def

from evaluation import map_and_auc, compute_distance, compute_map

import misc.utils as utils


from IPython.core.debugger import Pdb

from helper import *
from torch.nn import functional as F
import math
from sklearn.linear_model import SGDRegressor

os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'



lesk_path = Path.cwd() / 'Comprehensive_Lesk_Similary_Distances.csv'
lesk_distances = pd.read_csv(lesk_path)

def get_data(train_shape_views_folder, test_shape_views_folder, train_shape_flist, test_shape_flist, 
            train_sketch_folder, test_sketch_folder, train_sketch_flist, test_sketch_flist, 
             height, width, batch_size, workers, pk_flag=False):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.RectScale(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    # define sketch dataset 
    extname = '.png' if 'image' not in train_sketch_flist else '.JPEG'
    sketch_train_data = sk_dataset.Sk_Dataset(train_sketch_folder, train_sketch_flist, transform=train_transformer, ext=extname)
    sketch_test_data = sk_dataset.Sk_Dataset(test_sketch_folder, test_sketch_flist, transform=test_transformer, ext=extname)
    
    # define shape views dataset 
    shape_train_data = sh_views_dataset.Sh_Views_Dataset(train_shape_views_folder, train_shape_flist, transform=train_transformer)
    shape_test_data = sh_views_dataset.Sh_Views_Dataset(test_shape_views_folder, test_shape_flist, transform=test_transformer)
   

    if pk_flag:
        train_sketch_loader = DataLoader(
            sketch_train_data,
            batch_size=batch_size, num_workers=workers,
            pin_memory=True, drop_last=True)

        train_shape_loader = DataLoader(
            shape_train_data,
            batch_size=batch_size, num_workers=workers,
            pin_memory=True, drop_last=True)
    else:
        train_sketch_loader = DataLoader(
            sketch_train_data,
            batch_size=batch_size*2, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)

        train_shape_loader = DataLoader(
            shape_train_data,
            batch_size=batch_size, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)


    test_sketch_loader = DataLoader(
        sketch_test_data,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_shape_loader = DataLoader(
        shape_test_data,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return train_sketch_loader, train_shape_loader, test_sketch_loader, test_shape_loader 

#def semantic_bce(vgg_pred, labels, batch_size, lesk_distances, visualized_ground_truth, trained_vector):
def semantic_bce(labels, batch_size, lesk_distances, visualized_ground_truth, trained_vector):
        #print (vgg_pred)
        #print (labels)
        LAMBDA=0.5
        # Batch similarity distance
        similarity_dist = torch.zeros((batch_size, 210))
        summed_vis_truth = torch.zeros((batch_size, 1, 210))
        target_trained_vector = torch.zeros((batch_size, 1, 210))
        temp_summed_vis_truth = torch.zeros((batch_size, 1, 210))


        # For each label in the Current Batch
        for label in range(len(labels)):
            #target[label][labels[label]]=1####

            # Get label name  
            # Update the sliced similarity accordingly 
            cur_label_name = labels[label]
            similarity_dist[label] = torch.from_numpy(lesk_distances[classes[cur_label_name]].values)

            summed_vis_truth[label] = visualized_ground_truth[classes[cur_label_name]]
            temp_summed_vis_truth[label] = summed_vis_truth[label]
            summed_vis_truth[label]=summed_vis_truth[label]*similarity_dist[label]########
            target_trained_vector[label]=trained_vector[classes[cur_label_name]]
        
        sgd_reg = SGDRegressor(max_iter=100)
        X = summed_vis_truth.numpy()
        y = target_trained_vector.numpy()
        yr=y.ravel()
        sgd_reg.fit(X, yr)
        res=sgd_reg.predict(X)
        for label in range(len(labels)):
            cur_label_name = labels[label]
            lesk_distances[classes[cur_label_name]].values = summed_vis_truth[label]/temp_summed_vis_truth[label] 

        #CME = torch.dot(summed_vis_truth, similarity_dist) 
        #loss1 = (1-LAMBDA) *F.cross_entropy(vgg_pred, labels, reduction='mean') 
        loss2 = 10*LAMBDA * F.binary_cross_entropy(summed_vis_truth, target_trained_vector ,reduction='mean') 
        return loss2 #semantic loss

def main(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    cudnn.benchmark = True
    opt.checkpoint_folder += '_'+opt.backbone
    if opt.sketch_finetune:
        opt.checkpoint_folder += '_finetune'
    if not os.path.exists(opt.checkpoint_folder):
        os.makedirs(opt.checkpoint_folder)
    
    ###

    path_to_yolo_outputs = Path.cwd() / 'Xcount_result_30classes_3D_scenes_6views'

    images_per_class=420 #700 for 2D images, 910 for 13 views 3D sampled scenes, 560 for 8 views 3D sampled scenes, 420 for 6 views 3D sampled scenes，
    
    visualized_ground_truth = yolo_output_to_dict(path_to_yolo_outputs, images_per_class)
    trained_vector = load_trained_vector(path_to_yolo_outputs)######
    ####


    # Create data loaders
    if opt.height is None or opt.width is None:
        opt.height, opt.width = (224, 224)

    train_sketch_loader, train_shape_loader, test_sketch_loader, test_shape_loader  =  get_data(opt.train_shape_views_folder, 
                            opt.test_shape_views_folder, opt.train_shape_flist, opt.test_shape_flist, 
                            opt.train_sketch_folder, opt.test_sketch_folder, opt.train_sketch_flist, opt.test_sketch_flist, 
                            opt.height, opt.width, opt.batch_size, opt.workers, pk_flag=False)
    ###
    for idx, (inputs, labels) in enumerate(train_shape_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
    ###

    kwargs = {'pool_idx': opt.pool_idx} if opt.pool_idx is not None else {} 
    backbone = net_def
    net_bp = backbone.Net_Prev_Pool(**kwargs)
    net_vp = backbone.View_And_Pool()
    net_ap = backbone.Net_After_Pool(**kwargs)
    if opt.sketch_finetune:
        net_whole = backbone.Net_Whole(nclasses=30, use_finetuned=True)
    else:
        net_whole = backbone.Net_Whole(nclasses=30)
    net_cls = backbone.Net_Classifier(nclasses=30)
    crt_cls = nn.CrossEntropyLoss().cuda()
    # triplet center loss 
    crt_tlc = custom_loss.TripletCenterLoss(margin=opt.margin).cuda()
    if opt.wn:
        crt_tlc = torch.nn.utils.weight_norm(crt_tlc, name='centers')
    criterion = [crt_cls, crt_tlc, opt.w1, opt.w2]

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if opt.resume:
        checkpoint = torch.load(opt.resume)
        net_bp.load_state_dict(checkpoint['net_bp'])
        net_ap.load_state_dict(checkpoint['net_ap'])
        net_whole.load_state_dict(checkpoint['net_whole'])
        net_cls.load_state_dict(checkpoint['net_cls'])
        crt_tlc.load_state_dict(checkpoint['centers'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_prec']

    net_bp = nn.DataParallel(net_bp).cuda()
    net_vp = net_vp.cuda()
    net_ap = nn.DataParallel(net_ap).cuda()
    net_whole = nn.DataParallel(net_whole).cuda()
    net_cls = nn.DataParallel(net_cls).cuda()
    # wrap multiple models in optimizer 
    optim_shape = optim.SGD([{'params': net_ap.parameters()},
                            {'params': net_bp.parameters(), 'lr':1e-3},
                            {'params': net_cls.parameters()}],
                          lr=0.001, momentum=0.9, weight_decay=opt.weight_decay)

    base_param_ids = set(map(id, net_whole.module.features.parameters()))
    new_params = [p for p in net_whole.parameters() if id(p) not in base_param_ids]
    param_groups = [
    {'params': net_whole.module.features.parameters(), 'lr_mult':0.1},
    {'params': new_params, 'lr_mult':1.0}]

    optim_sketch = optim.SGD(param_groups, lr=0.001, momentum=0.9, weight_decay=opt.weight_decay)
    optim_centers = optim.SGD(crt_tlc.parameters(), lr=0.1)

    optimizer = (optim_sketch, optim_shape, optim_centers)
    model = (net_whole, net_bp, net_vp, net_ap, net_cls)

    # Schedule learning rate
    def adjust_lr(epoch, optimizer):
        step_size = 800 if opt.pk_flag else 80 # 40
        lr = opt.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    top1 = 0.0
    if opt.evaluate:
        # validate and compute mAP
        _, top1 = validate(test_sketch_loader, test_shape_loader, model, criterion, 0, opt)
        exit()
    best_epoch = -1
    best_metric = None
    # total_epochs = opt.max_epochs*10 if opt.pk_flag else opt.max_epochs
    for epoch in range(start_epoch, opt.max_epochs):
        train_top1 = train(train_sketch_loader, train_shape_loader, model, criterion, optimizer, epoch, opt, labels, lesk_distances, visualized_ground_truth, trained_vector)
        if epoch < opt.start_save and (epoch % opt.interval == 0):
            continue

        if train_top1 > 0.1:
            print("Test:")
            cur_metric = validate(test_sketch_loader, test_shape_loader, model, criterion, epoch, opt)
            top1 = cur_metric[-1]

        is_best = top1 > best_top1
        if is_best:
            best_epoch = epoch + 1
            best_metric = cur_metric
        best_top1 = max(top1, best_top1)


        
        checkpoint = {} 
        checkpoint['epoch'] = epoch + 1
        checkpoint['current_prec'] = top1
        checkpoint['best_prec'] = best_top1
        checkpoint['net_bp'] = net_bp.module.state_dict() 
        checkpoint['net_ap'] = net_ap.module.state_dict() 
        checkpoint['net_whole'] = net_whole.module.state_dict() 
        checkpoint['net_cls'] = net_cls.module.state_dict() 
        checkpoint['centers'] = crt_tlc.state_dict()
        
        path_checkpoint = '{0}/model_latest.pth'.format(opt.checkpoint_folder)
        
        if is_best: # save checkpoint 
            path_checkpoint = '{0}/model_best.pth'.format(opt.checkpoint_folder)
            utils.save_checkpoint(checkpoint, path_checkpoint)
            if opt.sf:
              shutil.copyfile(opt.checkpoint_folder+'/test_feat_temp.mat', opt.checkpoint_folder+'/test_feat_best.mat')

        print('\n * Finished epoch {:3d}  top1: {:5.3%}  best: {:5.3%}{} @epoch {}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else '', best_epoch))

        print('Best metric', best_metric)

def train(sketch_dataloader, shape_dataloader, model, criterion, optimizer, epoch, opt, labels, lesk_distances, visualized_ground_truth, trained_vector):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    tpl_losses = utils.AverageMeter()

    # training mode
    net_whole, net_bp, net_vp, net_ap, net_cls = model
    optim_sketch, optim_shape, optim_centers = optimizer
    crt_cls, crt_tlc, w1, w2 = criterion

    net_whole.train()
    net_bp.train()
    net_vp.train()
    net_ap.train()
    net_cls.train()

    end = time.time()
    # debug_here() 
    for i, ((sketches, k_labels), (shapes, p_labels)) in enumerate(zip(sketch_dataloader, shape_dataloader)):

        shapes = shapes.view(shapes.size(0)*shapes.size(1), shapes.size(2), shapes.size(3), shapes.size(4))

        # expanding: (bz * 12) x 3 x 224 x 224
        shapes = shapes.expand(shapes.size(0), 3, shapes.size(2), shapes.size(3))

        shapes_v = Variable(shapes.cuda())
        p_labels_v = Variable(p_labels.long().cuda())

        sketches_v = Variable(sketches.cuda())
        k_labels_v = Variable(k_labels.long().cuda())


        o_bp = net_bp(shapes_v)
        o_vp = net_vp(o_bp)
        shape_feat = net_ap(o_vp)
        sketch_feat = net_whole(sketches_v)
        feat = torch.cat([shape_feat, sketch_feat])
        target = torch.cat([p_labels_v, k_labels_v])
        score = net_cls(feat) 
        
        cls_loss = crt_cls(score, target)
        tpl_loss, _ = crt_tlc(score, target)
        # tpl_loss, _ = crt_tlc(feat, target)
        semanticLoss=semantic_bce(labels, opt.batch_size, lesk_distances, visualized_ground_truth, trained_vector)
        loss = w1 * cls_loss + w2 * tpl_loss + semanticLoss
        

        ## measure accuracy
        prec1 = utils.accuracy(score.data, target.data, topk=(1,))[0]
        losses.update(cls_loss.item(), score.size(0))  # batchsize
        tpl_losses.update(tpl_loss.item(), score.size(0))
        top1.update(prec1.item(), score.size(0))

        ## backward
        optim_sketch.zero_grad()
        optim_shape.zero_grad()
        optim_centers.zero_grad()

        loss.backward()
        nn.utils.clip_grad_value_(net_whole.module.features.parameters(), opt.gradient_clip)
        nn.utils.clip_grad_value_(optim_sketch.param_groups[1]['params'], opt.gradient_clip)
        nn.utils.clip_grad_value_(net_bp.parameters(), opt.gradient_clip)
        nn.utils.clip_grad_value_(net_ap.parameters(), opt.gradient_clip)
        nn.utils.clip_grad_value_(net_cls.parameters(), opt.gradient_clip)
        nn.utils.clip_grad_value_(crt_tlc.parameters(), opt.gradient_clip)
        '''utils.clip_gradient(optim_sketch, opt.gradient_clip)
        utils.clip_gradient(optim_shape, opt.gradient_clip)
        utils.clip_gradient(optim_centers, opt.gradient_clip)'''
        
        optim_sketch.step()
        optim_shape.step()
        optim_centers.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Trploss {triplet.val:.4f}({triplet.avg:.3f})'.format(
                epoch, i, len(sketch_dataloader), batch_time=batch_time,
                loss=losses, top1=top1, triplet=tpl_losses))
            # print('triplet loss: ', tpl_center_loss.data[0])
    print(' * Train Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg

def validate(sketch_dataloader, shape_dataloader, model, criterion, epoch, opt):

    """
    test for one epoch on the testing set
    """
    sketch_losses = utils.AverageMeter()
    sketch_top1 = utils.AverageMeter()

    shape_losses = utils.AverageMeter()
    shape_top1 = utils.AverageMeter()

    net_whole, net_bp, net_vp, net_ap, net_cls = model
    crt_cls, crt_tlc, w1, w2 = criterion

    net_whole.eval()
    net_bp.eval()
    net_vp.eval()
    net_ap.eval()
    net_cls.eval()

    sketch_features = []
    sketch_scores = []
    sketch_labels = []

    shape_features = []
    shape_scores = []
    shape_labels = []

    batch_time = utils.AverageMeter()
    end = time.time()

    for i, (sketches, k_labels) in enumerate(sketch_dataloader):
        sketches_v = Variable(sketches.cuda())
        k_labels_v = Variable(k_labels.long().cuda())
        sketch_feat = net_whole(sketches_v)
        sketch_score = net_cls(sketch_feat)

        loss = crt_cls(sketch_score, k_labels_v)

        prec1 = utils.accuracy(sketch_score.data, k_labels_v.data, topk=(1,))[0]
        sketch_losses.update(loss.item(), sketch_score.size(0)) # batchsize
        sketch_top1.update(prec1.item(), sketch_score.size(0))
        sketch_features.append(sketch_feat.data.cpu())
        sketch_labels.append(k_labels)
        sketch_scores.append(sketch_score.data.cpu())

        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(sketch_dataloader), batch_time=batch_time, loss=sketch_losses,
                      top1=sketch_top1))
    print(' *Sketch Prec@1 {top1.avg:.3f}'.format(top1=sketch_top1))

    batch_time = utils.AverageMeter()
    end = time.time()
    for i, (shapes, p_labels) in enumerate(shape_dataloader):
        shapes = shapes.view(shapes.size(0)*shapes.size(1), shapes.size(2), shapes.size(3), shapes.size(4))
        # expanding: (bz * 12) x 3 x 224 x 224
        shapes = shapes.expand(shapes.size(0), 3, shapes.size(2), shapes.size(3))

        shapes_v = Variable(shapes.cuda())
        p_labels_v = Variable(p_labels.long().cuda())

        o_bp = net_bp(shapes_v)
        o_vp = net_vp(o_bp)
        shape_feat = net_ap(o_vp)
        shape_score = net_cls(shape_feat)

        loss = crt_cls(shape_score, p_labels_v)

        prec1 = utils.accuracy(shape_score.data, p_labels_v.data, topk=(1,))[0]
        shape_losses.update(loss.item(), shape_score.size(0)) # batchsize
        shape_top1.update(prec1.item(), shape_score.size(0))
        shape_features.append(shape_feat.data.cpu())
        shape_labels.append(p_labels)
        shape_scores.append(shape_score.data.cpu())

        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(shape_dataloader), batch_time=batch_time, loss=shape_losses,
                      top1=shape_top1))
    print(' *Shape Prec@1 {top1.avg:.3f}'.format(top1=shape_top1))

    shape_features = torch.cat(shape_features, 0).numpy()
    sketch_features = torch.cat(sketch_features, 0).numpy()

    shape_scores = torch.cat(shape_scores, 0).numpy()
    sketch_scores = torch.cat(sketch_scores, 0).numpy()

    shape_labels = torch.cat(shape_labels, 0).numpy()
    sketch_labels = torch.cat(sketch_labels, 0).numpy()


    d_feat = compute_distance(sketch_features.copy(), shape_features.copy(), l2=False)
    d_feat_norm = compute_distance(sketch_features.copy(), shape_features.copy(), l2=True)
    mAP_feat = compute_map(sketch_labels.copy(), shape_labels.copy(), d_feat)
    mAP_feat_norm = compute_map(sketch_labels.copy(), shape_labels.copy(), d_feat_norm)
    print(' * Feature mAP {0:.5%}\tNorm Feature mAP {1:.5%}'.format(mAP_feat, mAP_feat_norm))


    d_score = compute_distance(sketch_scores.copy(), shape_scores.copy(), l2=False)
    mAP_score = compute_map(sketch_labels.copy(), shape_labels.copy(), d_score)
    d_score_norm = compute_distance(sketch_scores.copy(), shape_scores.copy(), l2=True)
    mAP_score_norm = compute_map(sketch_labels.copy(), shape_labels.copy(), d_score_norm)
    if opt.sf:
        shape_paths = [img[0] for img in shape_dataloader.dataset.shape_target_path_list]
        sketch_paths = [img[0] for img in sketch_dataloader.dataset.sketch_target_path_list]
        scio.savemat('{}/test_feat_temp.mat'.format(opt.checkpoint_folder), {'score_dist':d_score, 'score_dist_norm': d_score_norm, 'feat_dist': d_feat, 'feat_dist_norm': d_feat_norm,'sketch_features':sketch_features, 'sketch_labels':sketch_labels, 'sketch_scores': sketch_scores,
        'shape_features':shape_features, 'shape_labels':shape_labels, 'sketch_paths':sketch_paths, 'shape_paths':shape_paths})
    print(' * Score mAP {0:.5%}\tNorm Score mAP {1:.5%}'.format(mAP_score, mAP_score_norm))
    return [sketch_top1.avg, shape_top1.avg, mAP_feat, mAP_feat_norm, mAP_score, mAP_score_norm]


if __name__ == '__main__':
    from options import get_arguments

    opt = get_arguments()
    main(opt)
