import os
import time
import random
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from util import GradualWarmupSchedulerV2
from dataset import Pheme_Dataset
from modules.models import RD_Base
from transformers import BertTokenizer
import logging

global logger

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

# 设置随机种子
def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

# 设置device = gpu if available
def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, vocab_size):
    # Prepare model
    model = RD_Base(
        image_model_type = args.image_model_type,
        language_model_type = args.language_model_type,
        pretrained_image = args.pretrained_image,
        pretrained_language = args.pretrained_language,
        freeze_image = args.freeze_image,
        freeze_language = args.freeze_language,
        expand_image = args.expand_image,
        expand_language = args.expand_language,
        attention_model = args.attention_model,
        exchange = args.exchange,
        exchange_early = args.exchange_early,
        middle_dim = 1024,
        drop_p = 0.3,
        word_vocab_size = vocab_size,
        max_text_len = args.max_text_len
    )
    if args.init_model:
        model.load_state_dict(torch.load(args.init_model, map_location='cpu'))
    model.to(device)
    slim_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and name.endswith('weight') and 'bn_exchange' in name:
            if len(slim_params) % 2 == 0:
                slim_params.append(param[:len(param) // 2])
            else:
                slim_params.append(param[len(param) // 2:])
    return model, slim_params

# 设置优化器
def prep_optimizer(args, model, local_rank):
    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad]
    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad]

    weight_decay = args.weight_decay
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_param_tp], 'weight_decay': weight_decay, 'lr': args.lr},
        {'params': [p for n, p in no_decay_param_tp], 'weight_decay': 0.0, 'lr': args.lr},
    ]

    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1,
                                                after_scheduler=scheduler_cosine)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler_warmup, model

def get_args():
    parser = argparse.ArgumentParser()
    # 训练参数
    parser.add_argument('--output_dir', type=str, default='./output') # 模型和日志输出路径
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0') # 设置运行的GPU序号，支持多卡
    parser.add_argument('--n_epochs', type=int, default=80) # 训练迭代次数
    parser.add_argument('--do_train', action='store_true', default=False) # 是否训练
    parser.add_argument('--weight_decay', type=float, default=1e-5) # L2正则系数
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--debug', action='store_true', default=False) # debug 模式
    parser.add_argument('--seed', type=int, default=42) # 设置一个你喜欢的随机种子保证实验可复现，默认42，一个具有神秘力量的神奇数字
    parser.add_argument('--loss_weight', type=str, default='') # 损失函数每个类别的权重，用逗号分隔，形式如“2,1”，表示0类权重2，1类别权重为1，注意中间的分隔符是英文的逗号
    parser.add_argument("--world_size", default=0, type=int, help="distribted training") # 分布式训练需要的参数，不用管
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training") # 分布式训练需要的参数，不用管
    parser.add_argument("--rank", default=0, type=int, help="distribted training") # 分布式训练需要的参数，不用管
    # 模型参数
    parser.add_argument('--image_model_type', type=str, default='resnet50', choices=['resnet18','resnet34','resnet50','resnet101']) # 图像模型种类，默认resnet50，可自行指定其他模型
    parser.add_argument('--language_model_type', type=str, default='bert', choices=['transformer','bert']) # 语言模型种类，默认transformer
    parser.add_argument('--pretrained_image', action='store_true', default=True) # 加载预训练图像模型，默认加载
    parser.add_argument('--pretrained_language', action='store_true', default=False) # 加载预训练语言模型，这个参数暂时没用，不加载预训练语言模型
    parser.add_argument('--freeze_image', action='store_true', default=False) # 冻结图像模型
    parser.add_argument('--freeze_language', action='store_true', default=False) # 冻结语言模型
    parser.add_argument('--expand_image', action='store_true', default=False) # 增加图像特征维度（去掉mean pooling层）
    parser.add_argument('--expand_language', action='store_true', default=False) # 增加文本特征维度（返回单词特征）
    parser.add_argument('--init_model', type=str, default='') # 不为空则加载已保存模型
    parser.add_argument('--attention_model', action='store_true', default=False) # 使用注意力做融合
    parser.add_argument('--exchange', action='store_true', default=False) # 使用通道转换
    parser.add_argument('--exchange_early', action='store_true', default=False) # 通道转换在attention前
    parser.add_argument('--l1_lamda', type=float, default=2e-4) # 通道转换l1 loss的权重
    # Dataloader参数
    parser.add_argument('--csv_path', type=str, default='datasets/content_noid.csv') # label文件路径
    parser.add_argument('--image_folder', type=str, default='datasets/pheme_images_jpg') # 图像目录
    parser.add_argument('--train_id_file', type=str, default='datasets/train_ids.txt') # 训练集id
    parser.add_argument('--test_id_file', type=str, default='datasets/test_ids.txt') # 测试集id
    parser.add_argument('--num_workers', type=int, default=4) # 数据加载进程数
    parser.add_argument('--batch_size', type=int, default=128) # 训练阶段batchsize
    parser.add_argument('--batch_size_val', type=int, default=128) # 测试阶段batchsize
    parser.add_argument('--max_text_len', type=int, default=49) # 最大词数
    parser.add_argument('--image_size', type=int, default=224) # resize后的图像大小
    parser.add_argument('--pin_memory', action='store_true', default=False) # 某不重要的参数
    args, _ = parser.parse_known_args()
    return args


def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    return f1, pre, rec, acc

def eval_epoch(model, loader, device, debug, n_test=1):
    model.eval()
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch = tuple(t.to(device) for t in batch)
            image, text, attention_mask, target = batch
            # test 数据增强
            logits = torch.zeros((image.shape[0], 2)).to(device)
            probs = torch.zeros((image.shape[0], 2)).to(device)
            for I in range(n_test):
                l = model(get_trans(image, I), text, attention_mask)
                logits += l
                probs += l.softmax(1)
            logits /= n_test
            probs /= n_test
            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())
            print("{}/{}\r".format(step, len(loader)), end="")
            if debug and step>5:
                break
            
        LOGITS = torch.cat(LOGITS).numpy()
        PROBS = torch.cat(PROBS).numpy()
        TARGETS = torch.cat(TARGETS).numpy()
    f1, pre, rec, acc = get_metrics(TARGETS, (PROBS[:,1]>=PROBS[:,0]))
    logger.info('Classfication Metrics:')
    logger.info('f1 score: {:.4f} - precision score: {:.4f} - recall score: {:.4f} - accuracy score: {:4f}'.
            format(f1, pre, rec, acc))
    return f1

def L1_penalty(var):
    return torch.abs(var).sum()

def train_epoch(args, model, slim_params, loader, optimizer, criterion, device, local_rank, epoch, debug):
    global logger
    torch.cuda.empty_cache()
    model.train()
    start_time = time.time()
    total_loss = 0
    step_loss = 0

    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        image, text, attention_mask, target = batch
        
        logits = model(image, text, attention_mask)

        loss = criterion(logits, target.squeeze(-1))
        if len(slim_params)>0:
            L1_norm = sum([L1_penalty(m).cuda() for m in slim_params])
            loss = loss+L1_norm*args.l1_lamda
        loss.backward()
        total_loss += float(loss)
        step_loss = float(loss)
        optimizer.step()
        if local_rank == 0:
            logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Step Loss: %f, Time: %f", epoch + 1,
                        args.n_epochs, step + 1,
                        len(loader),
                        "-".join([str('%.9f' % itm) for itm in sorted(list(set([param_group['lr'] for param_group in optimizer.param_groups])))]),
                        float(loss),
                        float(step_loss),
                        (time.time() - start_time))
            start_time = time.time()
            step_loss = 0
        if step>5 and debug:
            break
    return total_loss / len(loader)

def get_dataloader(args, tokenizer):
    train_dataset = Pheme_Dataset(tokenizer, args.csv_path, args.image_folder, args.image_size, args.max_text_len, args.train_id_file, training=True)
    test_dataset = Pheme_Dataset(tokenizer, args.csv_path, args.image_folder, args.image_size, args.max_text_len, args.test_id_file, training=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = args.batch_size//args.n_gpu,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=False,
        sampler=train_sampler,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = args.batch_size_val,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=False,
        drop_last=False
    )
    return train_dataloader, test_dataloader, len(train_dataset), len(test_dataset), train_sampler

def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def main():
    global logger
    args = get_args()
    if args.debug:
        args.num_workers = 8
        args.batch_size = 16
        args.batch_size_val = 16
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    if args.debug:
        args.output_dir = os.path.join(args.output_dir,'debug')
    torch.distributed.init_process_group(backend="nccl")
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model, slim_params = init_model(args, device, tokenizer.vocab_size)

    # 加载数据集
    train_dataloader, test_dataloader, train_length, test_length, train_sampler = get_dataloader(args, tokenizer)
    loss_w = args.loss_weight
    if loss_w:
        loss_w = list(map(float,loss_w.split(',')))
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(loss_w)).float()).to(device)
    else:
        criterion = nn.CrossEntropyLoss()
    if args.do_train:
        optimizer, scheduler, model = prep_optimizer(args, model, args.local_rank)
        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("***** Running testing *****")
            logger.info("  Num examples = %d", test_length)
            logger.info("  Batch size = %d", args.batch_size_val)
        best_score = 0.00001
        best_output_model_file = "None"
        for epoch in range(args.n_epochs):
            train_sampler.set_epoch(epoch)
            train_loss = train_epoch(args, model, slim_params, train_dataloader, optimizer, criterion, device, args.local_rank, epoch, args.debug)
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.n_epochs, train_loss)
                output_model_file = save_model(epoch, args, model, type_name='')
                F1 = eval_epoch(model, test_dataloader, device, args.debug, n_test=2)
                if best_score <= F1:
                    best_score = F1
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the F1 is: {:.4f}".format(best_output_model_file, best_score))
            scheduler.step()
            if epoch == 2: scheduler.step()  # bug workaround
    else:
        if args.local_rank == 0:
            logger.info("***** Running testing *****")
            logger.info("  Num examples = %d", test_length)
            logger.info("  Batch size = %d", args.batch_size_val)
            eval_epoch(model, test_dataloader, device, args.debug, n_test=2)
    
if __name__ == '__main__':
    main()