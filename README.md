# Rumor-Detection
### 使用教程
默认环境中已经装好了cuda、pytorch、torchvision
用pip安装依赖包，有一些依赖包我可能没写完，运行时提示要装啥装上去就行了）
```
pip install -r requirements.txt
```
### 运行说明
训练模型
```bash
chmod a+x ./scripts/*
# 下面提供了三个可运行的脚本
./scripts/train.sh # transformer+resnet，端到端，resnet使用预训练模型，路径output/exp1
./scripts/train_bert.sh # bert+resnet，bert太大了数据集小训不了所以我冻结了bert参数，bert和resnet都使用预训练模型初始化，路径output/exp2
./scripts/train_bert_attention.sh # bert+resnet+attention，相比于上一个把你的attention加进去了，效果还行，目前最好的模型，输出路径在output/exp3
```
一些重要参数的说明
```
output_dir：指定输出路径
CUDA_VISIBLE_DEVICES：指定显卡，单卡直接输入0就可以了
do_train：训练模式，必选
n_epochs：迭代次数，可以改着玩
lr：学习率，可以改着玩
image_model_type：选择图像embedding模型，目前支持resnet18、resnet34、resnet50、resnet101
language_model_type：文本embedding模型，目前支持transformer和bert
pretrained_image：图像embedding模型是否加载预训练模型，建议加载
pretrained_language：文本embedding模型是否加载预训练，transformer只能False，bert只能True
csv_path：数据路径，不用改
image_folder：数据路径，不用改
train_id_file：同上
test_id_file：同上，我用的划分比例是8:2，切分的脚本是gen_train_test_split.py
batch_size：训练的batch size，如果算力不够可以改小点
batch_size_val：验证的batch size，可以和batch_size一样大或者比batch_size大点
max_text_len：文本最大长度，这个参数不太重要，比30高的估计都差不多
image_size：图像resize后的大小，不用改
num_workers：加载数据的进程数，一般越大加载数据越快，但最好不要超过10
```
文件说明
```
train.py：主进程文件，我写的是支持多卡的，所以必须用python -m torch.distributed.launch --nproc_per_node={卡数} train.py ...来启动，不过估计用不上，单卡就行了
dataset.py: 加载数据集
modules/model.py：模型的代码，你后面改模型基本只需要改这个文件
modules/transformer.py：transformer的代码，不需要改
output：输出路径，启动新实验的时候在这里建一个目录，然后对应修改output参数就行了。实验目录下的log.txt是训练日志，run.log是运行日志，pytorch_model.bin.*是训练好的模型。我跑了三个实验，日志在里面，模型太占地方我删了。
```
启动脚本后bert和resnet的预训练模型会自动下载

### 20230402 更新说明
本次更新加入了通道交换和损失函数权重，新增三个参数：
```
loss_weight：控制交叉熵损失函数各个类别的权重，默认各个类别权重相同，设置权重形式如1,2，使用英文逗号作为分隔符，第一个数字表示0类权重，第二个数字表示1类权重。
exchange：在注意力后使用通道交换，注意该选项必须和注意力机制同时使用，否则无效，默认为False
l1_lamda：通道交换的L1 loss损失权重，exchange为True时有效，默认2e-4
```
新增两个运行脚本：
```
./scripts/train_bert_attention_exchange.sh # bert+resnet+attention+exchange，在attention的基础上加入通道交换，貌似还有点用，输出路径在output/exp4
./scripts/train_bert_attention_exchange_w.sh # bert+resnet+attention+exchange，相比于上一个再加上了对类别loss的权重，权重设置的1,2，给长尾的正例设置了更大的权重，输出路径在output/exp5。
```

### 20230410 更新说明
增加了在attention前做通道交换的选项：
```
exchange_early：通道转换在attention前
```
新增一个运行脚本：
```
./scripts/train_bert_attention_earlyexchange.sh # exp6
```

### 20230425 更新说明
新增了weibo数据集：
```
dataset：可选择weibo或pheme
```
统一了数据集分割，和MFAN保持一致。加入了验证集，选择验证集上最好的模型用于测试。

### 20230430 更新说明
对attention进行了改动，使用bert的中间层做attention：
```
more_layer：增加对于中间层的attention
```
新增两个运行脚本：
```
./scripts/train_pheme_bert_attention_more.sh # exp9
./scripts/train_weibo_bert_attention_more.sh # exp10
```
两个脚本中暂时移除了exchange，exchange目前大多数情况下是负收益

### 20230601 更新说明
新增了CLIP作为embedding模型，使用bilinear attention替换原来的attention模快
#### 参数说明
```
language_model_type；指定clip则可使用clip作为图像的embedding模型
more_layer：使用clip和bert的中间层做attention，group=4
exchange：启用通道交换
attention_model：默认为none，即不设置注意力，设置为bilinear时启用双线性池化注意力
```
#### 新增实验
运行脚本：scripts_new目录下
实验结果：experiments目录下，目前最好的结果如下所示。通道交换暂时是负收益，实验结果在对应文件夹下。
pheme：
experiments/pheme/train_pheme_clip_bert_bilinear_more 
f1 score: 0.8688 - precision score: 0.8889 - recall score: 0.8496 - accuracy score: 0.924675
weibo：
experiments/weibo/train_weibo_clip_bert_bilinear_more
f1 score: 0.9000 - precision score: 0.8710 - recall score: 0.9310 - accuracy score: 0.918644
#### 模型说明
##### embedding提取
clip对图像做特征提取（vision transformer），取出中间层的结果，得到bs*12*49*768维特征，将12层中间层特征划分为四组，每组分别求和，得到bs*4*49*768维特征
bert对文本做特征提取，取出中间层的结果，得到bs*12*50*768维特征，将12层中间层特征划分为四组，每组分别求和，得到bs*4*50*768维特征
##### 交叉attention
图像的四层特征和文本的四层特征每层分别做交叉attention，即bs*50*768的文本特征和bs*49*768的图像特征互相做attention。为了降低计算量，文本作为query对图像做attention时，先求均值得到bs*768特征，然后对图像的bs*49*768维特征做attention，输出维度为bs*768，图像对文本做attention时同理，四层特征每层交叉attention后共得到8组bs*768维特征，对其进行拼接得到bs*（8*768）维特征，使用一个全连接层将其映射为bs*768，再使用一个全连接层将其映射为bs*2，然后使用交叉熵损失进行二分类。
##### 注意力模块
bs*768维度的query向量和bs*N*768维度的key向量分别计算相似度（相似度使用bilinear pooling定义），然后对value进行加权求和，最终的输出和query向量维度相同，实现细节参见x-linear。
