import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from modules.transformer import Transformer
from modules.transformer import LayerNorm
from transformers import BertModel
from modules.clip.module_clip import CLIP, convert_weights
import math
from modules.attention import BilinearLayer

class RD_Base(nn.Module):
    def __init__(self, image_model_type='resnet50', language_model_type='transformer', pretrained_image=True, pretrained_language=False, 
                 freeze_image=False, freeze_language=False, expand_image=False, expand_language=False, attention_model='none', exchange=False, middle_dim=1024, drop_p=0.3, word_vocab_size=30522, max_text_len=40, exchange_early=False, language='EN', more_layer=False):
        # image_model_type: 图像模型种类，默认resnet50，可自行指定其他模型，如resnet18, resnet34, resnet101等
        # language_model_type: 语言模型种类，默认Transformer
        # pretrained_image: 图像模型是否预训练初始化，默认预训练
        # pretrained_language: 语言模型是否预训练初始化，默认随机初始化
        # freeze_image: 冻结图像模型参数，默认不冻结
        # freeze_language: 冻结语言模型参数，默认不冻结
        # middle_dim: 分类器隐藏层维度，默认1024，可以随便改
        # drop_p: dropout概率，一般小于0.5
        # word_vocab_size: 词表大小
        super(RD_Base, self).__init__()
        self.exchange_early = exchange_early and exchange
        self.image_model_type = image_model_type
        self.language_model_type = language_model_type
        self.freeze_image = freeze_image
        self.freeze_language = freeze_language
        self.expand_image = expand_image
        self.expand_language = expand_language
        self.attention_model = attention_model
        self.more_layer = more_layer
        self.more_layer_clip = more_layer and self.image_model_type=='clip'
        # self.more_layer_clip = False
        # 初始化图像模型
        if image_model_type == 'clip':
            state_dict = {}
            clip_state_dict = CLIP.get_config(pretrained_clip_name="ViT-B/32")
            for key, val in clip_state_dict.items():
                new_key = key
                if new_key not in state_dict:
                    state_dict[new_key] = val.clone()

            clip_vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            clip_vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            clip_vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            clip_grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            clip_image_resolution = clip_vision_patch_size * clip_grid_size
            clip_embed_dim = clip_state_dict["text_projection"].shape[1]
            clip_context_length = clip_state_dict["positional_embedding"].shape[0]
            clip_vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
            clip_transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            clip_transformer_heads = clip_transformer_width // 64
            clip_transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))
            clip_cut_top_layer = 0
            self.image_model = CLIP(
                clip_embed_dim,
                clip_image_resolution, clip_vision_layers-clip_cut_top_layer, clip_vision_width, clip_vision_patch_size,
                clip_context_length, clip_vocab_size, clip_transformer_width, clip_transformer_heads, clip_transformer_layers-clip_cut_top_layer,
                linear_patch='2d'
            ).float()
            for key in ["input_resolution", "context_length", "vocab_size"]:
                if key in clip_state_dict:
                    del clip_state_dict[key]
            self.image_model = self.init_preweight(self.image_model, state_dict)
        elif image_model_type == 'resnet18':
            self.image_model = resnet18(pretrained=pretrained_image)
        elif image_model_type == 'resnet50':
            self.image_model = resnet50(pretrained=pretrained_image)
        elif image_model_type == 'resnet34':
            self.image_model = resnet34(pretrained=pretrained_image)
        elif image_model_type == 'resnet101':
            self.image_model = resnet101(pretrained=pretrained_image)
        else:
            raise NotImplementedError()
        if image_model_type == 'clip':
            image_in_ch = 768 if self.more_layer_clip else 768
        else:
            image_in_ch = self.image_model.fc.in_features # 图像模型输出的embedding维度
            self.image_model.fc = nn.Identity()
        if self.expand_image:
            self.image_model.avgpool = nn.Identity()
        
        # 初始化语言模型
        if language_model_type == 'transformer':
            assert pretrained_language == False
            self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=512, padding_idx=0)
            self.position_embedding = nn.Embedding(max_text_len, 512)
            self.language_model = Transformer(width=512, layers=3, heads=8) # layers: 层数，随便改
            language_in_ch = 512
        elif language_model_type == 'bert':
            assert pretrained_language == True
            if language == 'EN':
                self.language_model = BertModel.from_pretrained('bert-base-uncased')
            else:
                self.language_model = BertModel.from_pretrained('bert-base-chinese')
            language_in_ch = 768
        elif language_model_type == 'clip':
            self.language_model = None
            language_in_ch = 512
        # elif language_model_type == 'lstm':
        #     self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=1024, padding_idx=0)
        #     self.language_model = nn.LSTM(input_size=1024, hidden_size=1024,
        #                                batch_first=True, bidirectional=False, num_layers=1) # num_layers：层数，随便改
        #     language_in_ch = 1024
        else:
            raise NotImplementedError()
        
        # 创建分类器
        if self.attention_model != 'none':
            assert self.expand_image and self.expand_language
            if self.exchange_early:
                self.fc = EarlyExchangeClassfier()
            else:
                self.fc = AttentionClassfier(exchange=exchange, more_layer=more_layer, image_dim=image_in_ch,language_dim=language_in_ch,more_layer_clip=self.more_layer_clip, attention_model=self.attention_model)
        else:
            self.fc =  nn.Sequential(
                nn.Linear(image_in_ch+language_in_ch, middle_dim),
                nn.Dropout(p=drop_p),
                nn.Linear(middle_dim, 2)
            )
        # 初始化图像模型参数
        if not pretrained_image:
            self.image_model.apply(self.init_weights) 
        # 初始化语言模型参数
        if not pretrained_language:
            self.language_model.apply(self.init_weights)
        
        self.fc.apply(self.init_weights) # 初始化分类器参数
        self.freeze_model() # 冻结模型部分参数（如果需要的话）
        
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def init_preweight(self, model, state_dict, prefix=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')
        return model
            
    def freeze_model(self):
        if self.freeze_image:
            for name, param in self.image_model.named_parameters():
                param.requires_grad = False
        if self.freeze_language and self.language_model is not None:
            for name, param in self.language_model.named_parameters():
                param.requires_grad = False
    
    def forward(self, image, text, attention_mask):
        if self.image_model_type == 'clip':
            if self.more_layer_clip:
                image_embedding = self.image_model.encode_image(image, return_hidden=True, more_layer=True)
            else:
                image_embedding, image_embedding_expand = self.image_model.encode_image(image, return_hidden=True)
                if self.expand_image:
                    image_embedding = image_embedding_expand[:,1:]
        else:
            image_embedding = self.image_model(image) # bs,3,224,224 -> bs,2048
            if self.expand_image:
                image_embedding = image_embedding.view(image_embedding.size(0),2048,7,7).view(image_embedding.size(0),2048,49).transpose(-1,-2)
        
        seq_len = text.shape[-1]
        if self.language_model_type == 'transformer':
            word_embedding = self.word_embedding(text) # bs,40 -> bs,40,512
            position_ids = torch.arange(seq_len, dtype=torch.long, device=image.device)
            position_ids = position_ids.unsqueeze(0).expand(image.size(0), -1) # bs,40
            word_embedding = word_embedding + self.position_embedding(position_ids)
            extended_mask = (1.0 - attention_mask.unsqueeze(1)) * -1000000.0
            extended_mask = extended_mask.expand(-1, attention_mask.size(1), -1)
            word_embedding = word_embedding.permute(1, 0, 2)
            text_embedding = self.language_model(word_embedding, extended_mask).permute(1, 0, 2) # bs, 40, 512
            if not self.expand_language:
                text_embedding = text_embedding[:,0,:]
        elif self.language_model_type == 'clip':
            if self.more_layer:
                text_embedding = self.image_model.encode_text(text, attention_mask, expand=True, more_layer=True)
            elif self.expand_language:
                text_embedding = self.image_model.encode_text(text, attention_mask, expand=True)
            else:
                text_embedding = self.image_model.encode_text(text, attention_mask)
        else:
            if self.more_layer:
                text_embedding = list(self.language_model(text, attention_mask, output_hidden_states=True)[2])
                for i in range(len(text_embedding)):
                    text_embedding[i] = text_embedding[i].unsqueeze(1)
            elif self.expand_language:
                text_embedding = self.language_model(text, attention_mask)[0]
            else:
                text_embedding = self.language_model(text, attention_mask)[1]
        if self.attention_model is not 'none':
            logits = self.fc(image_embedding, text_embedding, attention_mask)
        else:
            embedding = torch.cat([image_embedding,text_embedding],-1) # bs, 2560
            logits = self.fc(embedding)
        return logits

class CoAttention(nn.Module):
    def __init__(self, q_dim, kv_dim):
        super(CoAttention, self).__init__()
        self.hidden_dim = 256

        self.Wq = nn.Linear(q_dim, 256)
        self.Wk = nn.Linear(kv_dim, 256)
        self.Wv = nn.Linear(kv_dim, 256)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,y):
        q = self.Wq(x)
        k = self.Wk(y)
        v = self.Wv(y)
        score = self.softmax(torch.bmm(q, k.permute(0, 2, 1)) / math.sqrt(self.hidden_dim))  # QK/k
        weighted_v = torch.bmm(score, v)

        return weighted_v
    
class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        return [x1, x2]

class BatchNorm1dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm1dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm1d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]
        
class NewClassfier(nn.Module):
    def __init__(self):
        super(NewClassfier, self).__init__()
        self.attn1 = CoAttention(2048,768)
        self.attn2 = CoAttention(768,2048)
        self.fc =  nn.Sequential(
            nn.Linear(512, 2)
        )
        # self.insnorm_conv = InstanceNorm2dParallel(256)
        # self.exchange = Exchange()
        # self.insnorm_threshold = 0.01
        # self.insnorm_list = []
        # for module in self.insnorm_conv.modules():
        #     if isinstance(module, nn.InstanceNorm2d):
        #         self.insnorm_list.append(module)
    def forward(self, image_embedding, text_embedding):
        x1= self.attn1(image_embedding, text_embedding)
        x2 = self.attn2(text_embedding, image_embedding)
        logits = self.fc(torch.cat([x1.mean(1),x2.mean(1)],-1))
        return logits
    
class AttentionClassfier(nn.Module):
    def __init__(self, exchange = False, bn_threshold=2e-2, more_layer=False, image_dim=2048, language_dim=768, more_layer_clip=False, attention_model='simple'):
        super(AttentionClassfier, self).__init__()
        self.groups = 4 if more_layer else 1
        self.more_layer_clip = more_layer_clip
        self.language_dim = language_dim
        self.image_dim = image_dim
        self.exchange = exchange
        self.attention_layers1 = nn.ModuleList([])
        self.attention_layers2 = nn.ModuleList([])
        self.attention_model = attention_model
        if attention_model == 'simple':
            attention_dim = 256        
            for i in range(self.groups):
                if self.more_layer_clip:
                    self.attention_layers1.append(nn.ModuleList([CoAttention(image_dim,language_dim) for _ in range(self.groups)]))
                    self.attention_layers2.append(nn.ModuleList([CoAttention(language_dim,image_dim) for _ in range(self.groups)]))
                else:
                    self.attention_layers1.append(CoAttention(image_dim,language_dim))
                    self.attention_layers2.append(CoAttention(language_dim,image_dim))
        elif attention_model == 'bilinear':
            attention_dim = 768
            for i in range(self.groups):
                if self.more_layer_clip:
                    # self.attention_layers1.append(nn.ModuleList([BilinearLayer(image_dim,language_dim) for _ in range(self.groups)]))
                    # self.attention_layers2.append(nn.ModuleList([BilinearLayer(language_dim,image_dim) for _ in range(self.groups)]))
                    self.attention_layers1.append(BilinearLayer(image_dim,language_dim))
                    self.attention_layers2.append(BilinearLayer(language_dim,image_dim))
                else:
                    self.attention_layers1.append(BilinearLayer(image_dim,language_dim))
                    self.attention_layers2.append(BilinearLayer(language_dim,image_dim))
        if self.exchange:
            if self.more_layer_clip:
                self.fc1 =  nn.Sequential(
                    nn.Linear(attention_dim*self.groups, attention_dim//2)
                )
            else:
                self.fc1 =  nn.Sequential(
                    nn.Linear(attention_dim*self.groups, attention_dim//2)
                )
            self.bn_exchange = BatchNorm1dParallel(attention_dim//2, 2)
            self.bn_threshold = bn_threshold
            self.bn_list = []
            for module in self.bn_exchange.modules():
                if isinstance(module, nn.BatchNorm1d):
                    self.bn_list.append(module)
            self.exchange = Exchange()
            self.fc2 = nn.Sequential(
                nn.Linear(attention_dim, 2)
            )
        else:
            if self.more_layer_clip:
                self.fc =  nn.Sequential(
                    # nn.Linear(attention_dim*self.groups*self.groups*2, attention_dim),
                    nn.Linear(attention_dim*self.groups*2, attention_dim),
                    nn.Dropout(p=0.5),
                    nn.Linear(attention_dim, 2)
                )
            else:
                self.fc =  nn.Sequential(
                    nn.Linear(attention_dim*self.groups*2, attention_dim),
                    nn.Dropout(p=0.5),
                    nn.Linear(attention_dim, 2)
                )
                
    def forward(self, image_embedding, text_embedding, text_mask):
        if self.groups > 1:
            num_each_group = 12//self.groups
            text_embedding = torch.cat(text_embedding[1:],1).view(-1,self.groups,num_each_group,text_embedding[0].shape[-2],self.language_dim)
            text_embedding = text_embedding.sum(2)
            # text_embedding = text_embedding[:,:,-1]
            if self.more_layer_clip:
                image_embedding = torch.cat(image_embedding[1:],1).view(-1,self.groups,num_each_group,image_embedding[0].shape[-2],self.image_dim)
                image_embedding = image_embedding.sum(2)
                # image_embedding = image_embedding[:,:,-1]
        else:
            text_embedding = text_embedding.unsqueeze(1)
        
        x1_result_list = []
        x2_result_list = []
        if self.attention_model == 'bilinear':
            image_mask = torch.ones([text_mask.size(0),image_embedding.size(-2)]).to(text_mask)
            for i in range(self.groups):
                if self.more_layer_clip:
                    x1_result_list.append(self.attention_layers1[i](image_embedding[:,i],text_embedding[:,i],image_mask,text_mask))
                    x2_result_list.append(self.attention_layers2[i](text_embedding[:,i],image_embedding[:,i],text_mask,image_mask))
                    # for j in range(self.groups):
                    #     x1_result_list.append(self.attention_layers1[i][j](image_embedding[:,j],text_embedding[:,i],image_mask,text_mask))
                    #     x2_result_list.append(self.attention_layers2[i][j](text_embedding[:,i],image_embedding[:,j],text_mask,image_mask))
                else:
                    x1_result_list.append(self.attention_layers1[i](image_embedding,text_embedding[:,i],image_mask,text_mask))
                    x2_result_list.append(self.attention_layers2[i](text_embedding[:,i],image_embedding,text_mask,image_mask))
        else:
            for i in range(self.groups):
                if self.more_layer_clip:
                    for j in range(self.groups):
                        x1_result_list.append(self.attention_layers1[i][j](image_embedding[:,j],text_embedding[:,i]).mean(1))
                        x2_result_list.append(self.attention_layers2[i][j](text_embedding[:,i],image_embedding[:,j]).mean(1))
                else:
                    x1_result_list.append(self.attention_layers1[i](image_embedding,text_embedding[:,i]).mean(1))
                    x2_result_list.append(self.attention_layers2[i](text_embedding[:,i],image_embedding).mean(1))
        x1 = torch.cat(x1_result_list,-1)
        x2 = torch.cat(x2_result_list,-1)

        if self.exchange:
            x1 = self.fc1(x1)
            x2 = self.fc1(x2)
            out = self.bn_exchange([x1,x2])
            out = self.exchange(out,self.bn_list,self.bn_threshold)
            logits = self.fc2(torch.cat(out,-1))
        else:
            logits = self.fc(torch.cat([x1,x2],-1))
        return logits
    
class EarlyExchangeClassfier(nn.Module):
    def __init__(self, bn_threshold=2e-2):
        super(EarlyExchangeClassfier, self).__init__()
        self.attn1 = CoAttention(512,512)
        self.attn2 = CoAttention(512,512)
        self.fc_image = nn.Linear(2048, 768)
        self.fc_text = nn.Linear(768, 768)
        self.fc1 =  nn.Sequential(
            nn.Linear(768, 512)
        )
        self.bn_exchange = BatchNorm1dParallel(512, 2)
        self.bn_threshold = bn_threshold
        self.bn_list = []
        for module in self.bn_exchange.modules():
            if isinstance(module, nn.BatchNorm1d):
                self.bn_list.append(module)
        self.exchange = Exchange()
        self.fc2 = nn.Sequential(
            nn.Linear(512, 2)
        )
        # self.insnorm_conv = InstanceNorm2dParallel(256)
        # self.exchange = Exchange()
        # self.insnorm_threshold = 0.01
        # self.insnorm_list = []
        # for module in self.insnorm_conv.modules():
        #     if isinstance(module, nn.InstanceNorm2d):
        #         self.insnorm_list.append(module)
    def forward(self, image_embedding, text_embedding):
        image_embedding = self.fc1(self.fc_image(image_embedding)).view(-1,512)
        text_embedding = self.fc1(self.fc_text(text_embedding)).view(-1,512)
        out = self.bn_exchange([image_embedding,text_embedding])
        out = self.exchange(out,self.bn_list,self.bn_threshold)
        image_embedding = out[0].view(-1,49,512)
        text_embedding = out[1].view(-1,49,512)
        x1= self.attn1(image_embedding, text_embedding).mean(1)
        x2 = self.attn2(text_embedding, image_embedding).mean(1)
        logits = self.fc2(torch.cat([x1,x2],-1))
        return logits