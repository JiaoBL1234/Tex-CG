import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
import pdb

def do_train_stage1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    it2_loss_meter = AverageMeter()
    it2_cloth_loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    labels = []
    cloth_labels = []
    with torch.no_grad():
        for n_iter, (img, vid, clothid, target_cam, target_view) in enumerate(train_loader_stage1):

            img = img.to(device)
            target = vid.to(device)
            clothid = clothid.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image = True)
                for i, cloth_id, img_feat in zip(target, clothid, image_feature):
                    labels.append(i)
                    cloth_labels.append(cloth_id)
                    image_features.append(img_feat.cpu())
        labels_list = torch.stack(labels, dim=0).cuda()
        image_features_list = torch.stack(image_features, dim=0).cuda()
        cloth_labels_list = torch.stack(cloth_labels, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_features, cloth_labels

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        it2_loss_meter.reset()
        it2_cloth_loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter+1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch:num_image]
            
            target = labels_list[b_list]
            image_features = image_features_list[b_list]
            cloth_label = cloth_labels_list[b_list]
            with amp.autocast(enabled=True):
                if cfg.MODEL.ID_PROMPT and cfg.MODEL.CLOTH_PROMPT == False:
                    text_id_features = model(label = target, get_text = True)
                
                elif cfg.MODEL.CLOTH_PROMPT:
                    text_id_cloth_features = model(label = target, cloth_label=cloth_label, get_text = True)
                    text_cloth_features = model(cloth_label=cloth_label, get_text = True)
            
            if cfg.MODEL.I2T_LOSS:
                if cfg.MODEL.CLOTH_PROMPT:
                    loss_i2t = cfg.MODEL.I2T_LOSS_WEIGHT * xent(image_features, text_id_cloth_features, cloth_label, cloth_label)
                    loss_t2i = cfg.MODEL.I2T_LOSS_WEIGHT * xent(text_id_cloth_features, image_features, cloth_label, cloth_label)
                else:
                    loss_i2t = cfg.MODEL.I2T_LOSS_WEIGHT * xent(image_features, text_id_features, target, target)
                    loss_t2i = cfg.MODEL.I2T_LOSS_WEIGHT * xent(text_id_features, image_features, target, target)
            else:
                loss_i2t, loss_t2i = 0, 0
            if cfg.MODEL.I2T_CLOTH_LOSS:
                loss_i2t_cloth = cfg.MODEL.I2T_CLOTH_LOSS_WEIGHT * xent(image_features, text_cloth_features, cloth_label, cloth_label)
                loss_t2i_cloth = cfg.MODEL.I2T_CLOTH_LOSS_WEIGHT * xent(text_cloth_features, image_features, cloth_label, cloth_label)
            else:
                loss_i2t_cloth, loss_t2i_cloth = 0, 0
            
            loss = loss_i2t + loss_t2i +  loss_i2t_cloth + loss_t2i_cloth

            
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])
            if cfg.MODEL.I2T_CLOTH_LOSS:
                it2_loss_meter.update((loss_i2t + loss_t2i).item(), img.shape[0])
                it2_cloth_loss_meter.update((loss_i2t_cloth + loss_t2i_cloth).item(), img.shape[0])
                torch.cuda.synchronize()
                if (i + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, i2t Loss: {:.3f}, i2t cloth Loss: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (i + 1), len(train_loader_stage1),
                                        loss_meter.avg, it2_loss_meter.avg, it2_cloth_loss_meter.avg,  scheduler._get_lr(epoch)[0]))
            else:
                torch.cuda.synchronize()
                if (i + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (i + 1), len(train_loader_stage1),
                                        loss_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.SIE_CAMERA:
                save_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_sie_{}.pth'.format(epoch))
            else:
                save_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch))

            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    
                    torch.save(model.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
