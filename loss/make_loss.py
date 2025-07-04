
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import pdb

def make_loss(cfg, num_classes):    
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE: 
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else: 
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on': 
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet': 
        def loss_func(score, feat, target, target_cam, i2tscore = None, \
                      i2t_non_cloth_description_score=None, i2t_cloth_score=None, i2t_cloth_target = None):

            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS) 
                    else:   
                        TRI_LOSS = triplet(feat, target)[0]
                    
                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if cfg.MODEL.STAGE2_I2T_LOSS and i2tscore != None:
                        I2TLOSS = xent(i2tscore, target)
                        loss = cfg.MODEL.STAGE2_I2T_LOSS_WEIGHT * I2TLOSS + loss
                    else:
                        I2TLOSS = 0
                    
                    if i2t_non_cloth_description_score != None:
                        I2T_NON_CLOTH_LOSS = xent(i2t_non_cloth_description_score, target)
                        loss = I2T_NON_CLOTH_LOSS + loss
                    else:
                        I2T_NON_CLOTH_LOSS = 0

                    if i2t_cloth_score != None:  
                        I2T_CLOTH_LOSS = xent(i2t_cloth_score, i2t_cloth_target)
                        loss = cfg.MODEL.STAGE2_I2T_CLOTH_LOSS_WEIGHT * I2T_CLOTH_LOSS + loss
                        
                    else:
                        I2T_CLOTH_LOSS = 0
                        


                    return loss 
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if i2tscore != None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss


                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


