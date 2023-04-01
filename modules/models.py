import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from modules.transformer import Transformer
from modules.transformer import LayerNorm
from transformers import BertModel
import math

class RD_Base(nn.Module):
    def __init__(self, image_model_type='resnet50', language_model_type='transformer', pretrained_image=True, pretrained_language=False, 
                 freeze_image=False, freeze_language=False, expand_image=False, expand_language=False, attention_model=False, middle_dim=1024, drop_p=0.3, word_vocab_size=30522, max_text_len=40):
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
        self.image_model_type = image_model_type
        self.language_model_type = language_model_type
        self.freeze_image = freeze_image
        self.freeze_language = freeze_language
        self.expand_image = expand_image
        self.expand_language = expand_language
        self.attention_model = attention_model
        # 初始化图像模型
        if image_model_type == 'resnet18':
            self.image_model = resnet18(pretrained=pretrained_image)
        elif image_model_type == 'resnet50':
            self.image_model = resnet50(pretrained=pretrained_image)
        elif image_model_type == 'resnet34':
            self.image_model = resnet34(pretrained=pretrained_image)
        elif image_model_type == 'resnet101':
            self.image_model = resnet101(pretrained=pretrained_image)
        else:
            raise NotImplementedError()
        image_in_ch = self.image_model.fc.in_features # 图像模型输出的embedding维度
        self.image_model.fc = nn.Identity()
        if self.expand_image:
            self.image_model.avgpool = nn.Identity()
        
        # 初始化语言模型
        if language_model_type == 'transformer':
            self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=512, padding_idx=0)
            self.position_embedding = nn.Embedding(max_text_len, 512)
            self.language_model = Transformer(width=512, layers=3, heads=8) # layers: 层数，随便改
            language_in_ch = 512
        elif language_model_type == 'bert':
            assert pretrained_language == True
            self.language_model = BertModel.from_pretrained('bert-base-uncased')
            language_in_ch = 768

        # elif language_model_type == 'lstm':
        #     self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=1024, padding_idx=0)
        #     self.language_model = nn.LSTM(input_size=1024, hidden_size=1024,
        #                                batch_first=True, bidirectional=False, num_layers=1) # num_layers：层数，随便改
        #     language_in_ch = 1024
        else:
            raise NotImplementedError()
        
        # 创建分类器
        if self.attention_model:
            assert self.expand_image and self.expand_language
            self.fc = NewClassfier()
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
            
    def freeze_model(self):
        if self.freeze_image:
            for name, param in self.image_model.named_parameters():
                param.requires_grad = False
        if self.freeze_language:
            for name, param in self.language_model.named_parameters():
                param.requires_grad = False
    
    def forward(self, image, text, attention_mask):
        image_embedding = self.image_model(image) # bs,3,224,224 -> bs,2048
        
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
        else:
            if self.expand_language:
                text_embedding = self.language_model(text, attention_mask)[0]
            else:
                text_embedding = self.language_model(text, attention_mask)[1]
        if self.attention_model:
            image_embedding = image_embedding.view(image_embedding.size(0),2048,7,7).view(image_embedding.size(0),2048,49).transpose(-1,-2)
            logits = self.fc(image_embedding,text_embedding)
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

    def forward(self, x, insnorm, insnorm_threshold):
        insnorm1, insnorm2 = insnorm[0].weight.abs(), insnorm[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, insnorm1 >= insnorm_threshold] = x[0][:, insnorm1 >= insnorm_threshold]
        x1[:, insnorm1 < insnorm_threshold] = x[1][:, insnorm1 < insnorm_threshold]
        x2[:, insnorm2 >= insnorm_threshold] = x[1][:, insnorm2 >= insnorm_threshold]
        x2[:, insnorm2 < insnorm_threshold] = x[0][:, insnorm2 < insnorm_threshold]
        return [x1, x2]

class InstanceNorm2dParallel(nn.Module):
    def __init__(self, num_features):
        super(InstanceNorm2dParallel, self).__init__()
        for i in range(2):
            setattr(self, 'insnorm_' + str(i), nn.InstanceNorm2d(num_features, affine=True, track_running_stats=True))

    def forward(self, x_parallel):
        return [getattr(self, 'insnorm_' + str(i))(x) for i, x in enumerate(x_parallel)]
        
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