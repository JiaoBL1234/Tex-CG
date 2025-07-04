import torch
import torch.nn as nn
import numpy as np
import pdb

from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from torch.nn import Dropout
from .clip.model import LayerNorm,ResidualAttentionBlock

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def init_transformer_weight(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.002, mean=0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts=None, effect_len=None): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  
        x = self.transformer(x) 
        x = x.permute(1, 0, 2) 
        x = self.ln_final(x).type(self.dtype) 

        if effect_len is not None:
            x = x[torch.arange(x.shape[0]), effect_len] @ self.text_projection 
        elif tokenized_prompts is not None:
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        else:
            x = x @ self.text_projection 

        return x

class Description_Transformer(nn.Module):
    
    def __init__(self, d_model: int = 512, n_head: int = 8, attn_mask: torch.Tensor = None, \
                 attn_weight: float = 1.0, des_fusion_type: int = 1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.ln_1.apply(init_transformer_weight)
        self.attn.apply(init_transformer_weight)

        self.attn_weight = attn_weight
        self.des_fusion_type = des_fusion_type

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def prompt_predictor(self, x: torch.Tensor):
        x = x + self.attn_weight * self.attention(self.ln_1(x))
        return x


    def forward(self, prompt_features, text_description, effect_len=None): 

        if self.des_fusion_type == 1:
            x = torch.cat([prompt_features.unsqueeze(1), text_description[torch.arange(text_description.shape[0]), :-1]], dim=1)
            x = x.permute(1, 0, 2)  
            x = self.prompt_predictor(x)
            x = x.permute(1, 0, 2) 
            hidden_features = x[:, 0]

        elif self.des_fusion_type == 2:

            x = torch.cat([prompt_features[torch.arange(prompt_features.shape[0]), :12], \
                           text_description[torch.arange(text_description.shape[0]), 1:-11]], dim=1)
            x = x.permute(1, 0, 2) 
            x = self.prompt_predictor(x)
            x = x.permute(1, 0, 2) 
            hidden_features = x[:, effect_len] 
        return hidden_features
    


class build_transformer(nn.Module):
    def __init__(self, num_classes, num_cloth_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.num_cloth_classes = num_cloth_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner(num_classes, num_cloth_classes, dataset_name, clip_model.dtype, clip_model.token_embedding, \
                                            cfg.MODEL.CLOTH_PROMPT, cfg.MODEL.ID_PROMPT)
        self.text_encoder = TextEncoder(clip_model)
        self.CLOTH_prompt = cfg.MODEL.CLOTH_PROMPT
        self.des_fusion_type = cfg.MODEL.DES_FUSION_TYPE
        self.prompt_type = cfg.MODEL.S2_PROMPT_TYPE

        if cfg.MODEL.DES_TRANS and self.des_fusion_type < 3:
            self.prompt_descroption_transformer_layer = Description_Transformer(d_model=self.in_planes_proj, \
                       attn_weight=cfg.MODEL.DES_WEIGHT, des_fusion_type=cfg.MODEL.DES_FUSION_TYPE)

    def forward(self, x = None, label=None, cloth_label=None, non_cloth_description = None, get_image = False, get_text = False, cam_label= None, view_label=None):
        if get_text == True:
            if non_cloth_description is not None and label is None: 
                if self.des_fusion_type == 3 or self.prompt_type == 7 or self.prompt_type == 8:
                    text_description, des_len = self.prompt_learner(label=None, cloth_label=None, non_cloth_description=non_cloth_description) 
                    text_features = self.text_encoder(text_description, effect_len=des_len)
                else:
                    text_description, _ = self.prompt_learner(label=None, cloth_label=None, non_cloth_description=non_cloth_description) 
                    text_features = self.text_encoder(text_description)

            else:

                if self.des_fusion_type == 1:
                    prompts, effect_len = self.prompt_learner(label, cloth_label) 
                    text_features = self.text_encoder(prompts, effect_len=effect_len)
                    if non_cloth_description is not None:  ## stage 2 
                        text_features = self.prompt_descroption_transformer_layer(text_features, non_cloth_description)
                elif self.des_fusion_type == 2:
                    prompts, effect_len = self.prompt_learner(label, cloth_label) 
                    text_features = self.text_encoder(prompts)
                    if non_cloth_description is not None:  
                        text_features = self.prompt_descroption_transformer_layer(text_features, non_cloth_description, effect_len)
                elif self.des_fusion_type == 3:
                    prompts, effect_len = self.prompt_learner(label, cloth_label) 
                    text_features = self.text_encoder(prompts, effect_len=effect_len)
                    if non_cloth_description is not None: 
                        text_features = (text_features + non_cloth_description) *0.5
                elif self.des_fusion_type == 4:
                    prompts, effect_len = self.prompt_learner(label, cloth_label) 
                    text_features = self.text_encoder(prompts, effect_len=effect_len)
                    if non_cloth_description is not None: 
                        text_features = torch.mean(torch.cat([text_features.unsqueeze(1), non_cloth_description], dim=1), dim=1)
                elif self.des_fusion_type == 5:
                    prompts, effect_len = self.prompt_learner(label, cloth_label) 
                    text_features = self.text_encoder(prompts)
                    if non_cloth_description is not None: 
                        text_features = torch.mean(torch.cat([text_features, non_cloth_description], dim=1), dim=1)
                elif self.des_fusion_type == 7:
                    prompts, effect_len = self.prompt_learner(label, cloth_label) 
                    text_features = self.text_encoder(prompts, effect_len=effect_len)

            return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':

                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:

            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, num_cloth_classes, camera_num, view_num):
    model = build_transformer(num_class, num_cloth_classes, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, num_cloth_classes, dataset_name, dtype, token_embedding, CLOTH_prompt=False, ID_prompt = True):
        super().__init__()
        
        ctx_dim = 512
        n_ctx = 4
        n_cls_ctx = 4
        self.clip_token_embedding = token_embedding
        self.clip_dtype = dtype

        self.CLOTH_prompt = CLOTH_prompt

        ctx_init = "A photo of a X X X X person with X X X X clothes."
        

        ctx_init = ctx_init.replace("_", " ")
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  
        print('self.tokenized_prompts.argmax(dim=-1):', self.tokenized_prompts.argmax(dim=-1))

    
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 

        cls_vectors = torch.empty(num_cloth_classes, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_cloth_ctx = nn.Parameter(cls_vectors) 


        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_middle", embedding[:, n_ctx + 1 + n_cls_ctx: n_ctx + 3 + n_cls_ctx, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 7 + n_cls_ctx:, :])  


        ctx_cloth_init = "A photo of X X X X clothes.".replace("_", " ")
        tokenized_prompts_cloth = clip.tokenize(ctx_cloth_init).cuda() 
        with torch.no_grad():
            embedding_cloth = token_embedding(tokenized_prompts_cloth).type(dtype) 
        self.register_buffer("token_prefix_only_cloth", embedding_cloth[:, :4, :])
        self.register_buffer("token_suffix_only_cloth", embedding_cloth[:, 8:, :])

        self.tokenized_prompts_only_cloth = tokenized_prompts_cloth  
        
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx



    def forward(self, label, cloth_label, non_cloth_description=None):
        if non_cloth_description is not None:
            non_cloth_description = clip.tokenize(non_cloth_description).cuda() 
            with torch.no_grad():
                non_cloth_tokens = self.clip_token_embedding(non_cloth_description).type(self.clip_dtype) 

            prompts = non_cloth_tokens
            effect_len = non_cloth_description.argmax(dim=-1)


        elif label is None:  
            b = cloth_label.shape[0]
            cls_cloth_ctx = self.cls_cloth_ctx[cloth_label]
            prefix = self.token_prefix_only_cloth.expand(b, -1, -1)
            suffix = self.token_suffix_only_cloth.expand(b, -1, -1)
            prompts = torch.cat(
                [
                    prefix,  
                    cls_cloth_ctx,     
                    suffix, 
                ],
                dim=1,
            )
            effect_len = 10
        elif cloth_label is None:  
            b = label.shape[0]
            cls_ctx = self.cls_ctx[label] 
            prefix = self.token_prefix.expand(b, -1, -1) 
            middle = self.token_middle[:, :1, :].expand(b, -1, -1) 
            suffix = self.token_suffix_only_cloth[:, :-2, :].expand(b, -1, -1) 
            prompts = torch.cat(
                [
                    prefix,  
                    cls_ctx,    
                    middle, 
                    suffix,  
                ],
                dim=1,
            )
            effect_len = 11
        else:
            b = label.shape[0]
            cls_ctx = self.cls_ctx[label] 
            cls_cloth_ctx = self.cls_cloth_ctx[cloth_label]
            prefix = self.token_prefix.expand(b, -1, -1) 
            middle = self.token_middle.expand(b, -1, -1) 
            suffix = self.token_suffix.expand(b, -1, -1) 
            prompts = torch.cat(
                [
                    prefix,  
                    cls_ctx,     
                    middle, 
                    cls_cloth_ctx, 
                    suffix,  
                ],
                dim=1,
            )
            
            effect_len = 17
        return prompts, effect_len
