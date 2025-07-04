import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP_CC_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
import numpy as np
import pdb

def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,
             val_cc_loader=None, num_query_cc=None,
             train_pid2clothes=None,
             ):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    i2t_loss_meter = AverageMeter()
    i2t_cloth_loss_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    if num_query_cc is not None:
        evaluator_cc = R1_mAP_CC_eval(num_query_cc, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    if cfg.MODEL.DES_TRANS:
        if cfg.DATASETS.NAMES == 'ltcc':
            train_id_non_cloth_description_file = os.path.join(cfg.DATASETS.ROOT_DIR, 'LTCC_ReID', 'train_id_non-cloth_new.txt')
        elif cfg.DATASETS.NAMES == 'prcc':
            train_id_non_cloth_description_file = os.path.join(cfg.DATASETS.ROOT_DIR, 'PRCC', 'rgb', 'train_id_non-cloth_new.txt')
        elif cfg.DATASETS.NAMES == 'deepchange':
            train_id_non_cloth_description_file = os.path.join(cfg.DATASETS.ROOT_DIR, 'DeepChangeDataset', 'train_id_non-cloth_new.txt')
            
            
        f = open(train_id_non_cloth_description_file, 'r')

        text_non_cloth_description_features = []
        for line in f.readlines():
            str_label = line.split(' ')[0]
            non_cloth_description = line[len(str_label)+1:-1][:77]
            text_non_cloth_feature = model(non_cloth_description=non_cloth_description, get_text = True)
            text_non_cloth_description_features.append(text_non_cloth_feature.cpu())

        text_non_cloth_description_features = torch.cat(text_non_cloth_description_features, 0).cuda()
        print('text_non_cloth_description_features:', text_non_cloth_description_features.shape)

    if cfg.MODEL.S2_PROMPT_TYPE == 7 or cfg.MODEL.S2_PROMPT_TYPE == 8:
        if cfg.DATASETS.NAMES == 'ltcc':
            train_id_non_cloth_description_file = os.path.join(cfg.DATASETS.ROOT_DIR, 'LTCC_ReID', 'train_id_non-cloth_new.txt')
        elif cfg.DATASETS.NAMES == 'prcc':
            train_id_non_cloth_description_file = os.path.join(cfg.DATASETS.ROOT_DIR, 'PRCC', 'rgb', 'train_id_non-cloth_new.txt')
        elif cfg.DATASETS.NAMES == 'deepchange':
            train_id_non_cloth_description_file = os.path.join(cfg.DATASETS.ROOT_DIR, 'DeepChangeDataset', 'train_id_non-cloth_new.txt')
            
            
            
        f = open(train_id_non_cloth_description_file, 'r')

        text_non_cloth_description_features = []
        for line in f.readlines():
            str_label = line.split(' ')[0]
            non_cloth_description = line[len(str_label)+1:-1]
            text_non_cloth_feature = model(non_cloth_description=non_cloth_description, get_text = True)
            text_non_cloth_description_features.append(text_non_cloth_feature.cpu())

        text_non_cloth_description_features = torch.cat(text_non_cloth_description_features, 0).cuda()
        print('text_non_cloth_description_features:', text_non_cloth_description_features.shape)


    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    
    if cfg.MODEL.DES_TRANS == False: 
        text_id_features = []
        with torch.no_grad():
            for i in range(i_ter):
                if i+1 != i_ter:
                    l_list = torch.arange(i*batch, (i+1)* batch)
                else:
                    l_list = torch.arange(i*batch, num_classes)
                with amp.autocast(enabled=True):
                    text_feature = model(label = l_list, get_text = True)
                text_id_features.append(text_feature.cpu())
            text_id_features = torch.cat(text_id_features, 0).cuda()
        
        print('text_id_features:', text_id_features.shape)


    if cfg.MODEL.DES_TRANS == False and train_pid2clothes is not None:
        assert train_pid2clothes.shape[0] == num_classes
        num_cloth_classes = train_pid2clothes.shape[1]

        if cfg.MODEL.S2_PROMPT_TYPE == 2 or cfg.MODEL.S2_PROMPT_TYPE == 5:
            text_id_cloth_features = []
            with torch.no_grad():
                for i in range(num_cloth_classes):
                    with amp.autocast(enabled=True):
                        cloth_label = torch.LongTensor([i])
                        label = np.where(train_pid2clothes[:, i] == 1)[0]
                        label_list = torch.LongTensor(label)
                        if cfg.DATASETS.NAMES == 'prcc': 
                            assert label * 2 == i or label *2 + 1 == i
                        text_id_cloth_feature = model(label = label_list, cloth_label = cloth_label, get_text = True)

                    text_id_cloth_features.append(text_id_cloth_feature.cpu())

                text_id_cloth_features = torch.stack(text_id_cloth_features).squeeze(1).cuda() 
            print('text_id_cloth_features:', text_id_cloth_features.shape)


        elif cfg.MODEL.S2_PROMPT_TYPE == 3 or cfg.MODEL.S2_PROMPT_TYPE == 6:
            i_ter = num_cloth_classes // batch
            left = num_cloth_classes-batch* (num_cloth_classes//batch)
            if left != 0 :
                i_ter = i_ter+1
            
            text_cloth_features = []
            with torch.no_grad():
                for i in range(i_ter):
                    if i+1 != i_ter:
                        l_list = torch.arange(i*batch, (i+1)* batch)
                    else:
                        l_list = torch.arange(i*batch, num_cloth_classes)
                    with amp.autocast(enabled=True):
                        text_cloth_feature = model(label = None, cloth_label = l_list, get_text = True)

                    text_cloth_features.append(text_cloth_feature.cpu())
                text_cloth_features = torch.cat(text_cloth_features, dim=0).cuda() 


            
            print('text_cloth_feature:', text_cloth_features.shape)

        elif cfg.MODEL.S2_PROMPT_TYPE == 4:
            text_id_avg_features = []
            with torch.no_grad():
                for i in range(num_classes):
                    clothes_ids = np.where(train_pid2clothes[i] == 1)[0]
                    with amp.autocast(enabled=True):
                        cloth_label = torch.LongTensor(clothes_ids)
                        label = torch.LongTensor([i]).expand(len(cloth_label))
                        text_id_cloth_feature = model(label = label, cloth_label = cloth_label, get_text = True)
                        if cfg.DATASETS.NAMES == 'prcc': 
                            assert clothes_ids[0] == i * 2 or clothes_ids[0] == i * 2 + 1

                    text_id_avg_features.append(text_id_cloth_feature.cpu().mean(0))
                text_id_avg_features = torch.stack(text_id_avg_features).cuda() 
            print('text_id_avg_features:', text_id_avg_features.shape)
    
    best_mAP, best_cmc = 0.0, 0.0 
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        i2t_loss_meter.reset()
        i2t_cloth_loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        if num_query_cc is not None:
            evaluator_cc.reset()

        scheduler.step()

        model.train()
        for n_iter, (img, vid, clothid, target_cam, target_view) in enumerate(train_loader_stage2):

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            clothid = clothid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x = img, label = target, cam_label=target_cam, view_label=target_view)
                
                if cfg.MODEL.DES_TRANS:
                    text_id_non_cloth_description_features = []
                    for i in range(i_ter):
                        if i+1 != i_ter:
                            end_label = (i+1)* batch
                        else:
                            end_label = num_classes 
                        l_list = torch.arange(i*batch, end_label)

                        non_cloth_description = [text_non_cloth_description_features[j] for j in range(i*batch, end_label)]
                        non_cloth_description = torch.stack(non_cloth_description)
                        non_cloth_description_feat = model(label=l_list, non_cloth_description=non_cloth_description, get_text = True)
                        text_id_non_cloth_description_features.extend(non_cloth_description_feat)
                    text_id_non_cloth_description_features = torch.stack(text_id_non_cloth_description_features, 0).cuda()

                    i2t_score = image_features @ text_id_non_cloth_description_features.t()

                    loss = loss_fn(score, feat, target, target_cam, i2t_score)

                else:
                    if cfg.MODEL.S2_PROMPT_TYPE == 1:
                        i2t_score = image_features @ text_id_features.t()
                        loss = loss_fn(score, feat, target, target_cam, i2tscore=i2t_score)

                    elif cfg.MODEL.S2_PROMPT_TYPE == 2:
                        i2t_cloth_score = image_features @ text_id_cloth_features.t()
                        loss = loss_fn(score, feat, target, target_cam, \
                                i2t_cloth_score=i2t_cloth_score, i2t_cloth_target=clothid)
                    
                    elif cfg.MODEL.S2_PROMPT_TYPE == 3:
                        i2t_cloth_score = image_features @ text_cloth_features.t()
                        loss = loss_fn(score, feat, target, target_cam, \
                                i2t_cloth_score=i2t_cloth_score, i2t_cloth_target=clothid)
                    
                    if cfg.MODEL.S2_PROMPT_TYPE == 4:
                        i2t_score = image_features @ text_id_avg_features.t()
                        loss = loss_fn(score, feat, target, target_cam, i2tscore=i2t_score)

                    elif cfg.MODEL.S2_PROMPT_TYPE == 5:
                        i2t_score = image_features @ text_id_features.t()
                        i2t_cloth_score = image_features @ text_id_cloth_features.t()
                        loss = loss_fn(score, feat, target, target_cam, i2tscore=i2t_score, \
                                i2t_cloth_score=i2t_cloth_score, i2t_cloth_target=clothid)
                    
                    elif cfg.MODEL.S2_PROMPT_TYPE == 6:
                        i2t_score = image_features @ text_id_features.t()
                        i2t_cloth_score = image_features @ text_cloth_features.t()
                        loss = loss_fn(score, feat, target, target_cam, i2tscore=i2t_score, \
                                i2t_cloth_score=i2t_cloth_score, i2t_cloth_target=clothid)

                    elif cfg.MODEL.S2_PROMPT_TYPE == 7:
                        i2t_score = image_features @ text_id_features.t()
                        i2t_score_description = image_features @ text_non_cloth_description_features.t()
                        loss = loss_fn(score, feat, target, target_cam, i2tscore=i2t_score, \
                                i2t_non_cloth_description_score=i2t_score_description)
                    
                    elif cfg.MODEL.S2_PROMPT_TYPE == 8:
                        i2t_score = image_features @ text_non_cloth_description_features.t()
                        loss = loss_fn(score, feat, target, target_cam,\
                                i2t_non_cloth_description_score=i2t_score)


            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            if cfg.MODEL.S2_PROMPT_TYPE == 3 or cfg.MODEL.S2_PROMPT_TYPE == 2:
                acc = (i2t_cloth_score.max(1)[1] == clothid).float().mean()
            else:
                acc = (i2t_score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
            

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0 and epoch >= cfg.SOLVER.STAGE2.EVAL_START_EPOCH:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
                    if num_query_cc is not None:
                        for n_iter, (img, vid, camid, camids, target_view, _, clothids) in enumerate(val_cc_loader):
                            with torch.no_grad():
                                img = img.to(device)
                                if cfg.MODEL.SIE_CAMERA:
                                    camids = camids.to(device)
                                else: 
                                    camids = None
                                if cfg.MODEL.SIE_VIEW:
                                    target_view = target_view.to(device)
                                else: 
                                    target_view = None
                                feat = model(img, cam_label=camids, view_label=target_view)
                                evaluator_cc.update((feat, vid, camid, clothids))
                        cmc_cc, mAP_cc, _, _, _, _, _ = evaluator_cc.compute()
                        logger.info("Cloth Change Validation Results - Epoch: {}".format(epoch))
                        logger.info("mAP: {:.1%}".format(mAP_cc))
                        for r in [1, 5, 10]:
                            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_cc[r - 1]))
                        torch.cuda.empty_cache()

            
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else: 
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else: 
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

                if num_query_cc is not None:
                    for n_iter, (img, vid, camid, camids, target_view, _, clothids) in enumerate(val_cc_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator_cc.update((feat, vid, camid, clothids))
                    cmc_cc, mAP_cc, _, _, _, _, _ = evaluator_cc.compute()
                    logger.info("Cloth Change Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP_cc))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_cc[r - 1]))
                    torch.cuda.empty_cache()
            if num_query_cc is not None:
                if mAP_cc > best_mAP:
                    best_mAP = mAP_cc
                    best_cmc = cmc_cc[0]
                    torch.save(model.state_dict(),
                                os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
            else:
                if mAP > best_mAP:
                    best_mAP = mAP
                    best_cmc = cmc[0]
                    torch.save(model.state_dict(),
                                os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
            
            logger.info('-'*40)
            if num_query_cc is not None:
                logger.info("Final results: {:.1%} {:.1%}".format(mAP_cc, cmc_cc[0], mAP, cmc[0]))


                
            else:
                logger.info("Final results: {:.1%} {:.1%}".format(mAP, cmc[0]))
            
            logger.info('-'*40)
                
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 val_cc_loader=None,
                 num_query_cc=None,
                 train_pid2clothes=None):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if num_query_cc is not None:
        evaluator_cc = R1_mAP_CC_eval(num_query_cc, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator_cc.reset()

    
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    img_path_list = []
    feats = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))



    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    
    
    if num_query_cc is not None:
        features = []
        img_path_cc_list= []
        img_labels, cloth_labels = [], []

        for n_iter, (img, pid, camid, camids, target_view, imgpath, clothids) in enumerate(val_cc_loader):
            with torch.no_grad():
                img = img.to(device)
                if cfg.MODEL.SIE_CAMERA:
                    camids = camids.to(device)
                else: 
                    camids = None
                if cfg.MODEL.SIE_VIEW:
                    target_view = target_view.to(device)
                else: 
                    target_view = None
                feat = model(img, cam_label=camids, view_label=target_view)
                evaluator_cc.update((feat, pid, camid, clothids))



        cmc_cc, mAP_cc, _, _, _, _, _ = evaluator_cc.compute()
        logger.info("Validation Results ")
        logger.info("mAP: {:.1%}".format(mAP_cc))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_cc[r - 1]))
        
            
        logger.info('-'*40)
        if num_query_cc is not None:
            logger.info("Final results: {:.1%} {:.1%} {:.1%} {:.1%}".format(mAP_cc, cmc_cc[0], mAP, cmc[0]))

        else:
            logger.info("Final results: {:.1%} {:.1%}".format(mAP, cmc[0]))
        
        logger.info('-'*40)

    return cmc[0], cmc[4]

