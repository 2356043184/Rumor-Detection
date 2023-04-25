import os
import pandas
import random
# split_point = 0.8
# df = pandas.read_csv('datasets\content_noid.csv')
# image_ids = list(df['image_id'])

# test = random.sample(image_ids, int((1-split_point)*len(image_ids)))
# train = [str(_id) for _id in image_ids if _id not in test]
# test = [str(_id) for _id in test]
# with open('datasets/train_ids.txt','w') as file:
#     file.write('\n'.join(train))
    
# with open('datasets/test_ids.txt','w') as file:
#     file.write('\n'.join(test))
fold_path = 'MFAN/dataset/pheme/pheme_files'
df = pandas.read_csv('MFAN/dataset/pheme/content.csv')
mid2imgnum = dict()
for _,line in df.iterrows():
    mid2imgnum[str(line['mid'])] = str(line['imgnum'])
for split in ['dev','train','test']:
    file_path = os.path.join(fold_path,'pheme.'+split)
    with open(file_path) as file:
        data = file.readlines()
        data = [mid2imgnum[line.strip().split('\t')[0]] for line in data]
    with open(f'datasets/pheme/{split}_ids.txt','w') as file:
        file.write('\n'.join(data))
        
fold_path = 'MFAN/dataset/weibo/weibo_files'
df = pandas.read_csv('datasets/weibo/weibo_content.csv')
mid2imgnum = dict()
for _,line in df.iterrows():
    mid2imgnum[str(line['mid'])] = str(line['imgnum'])
for split in ['dev','train','test']:
    file_path = os.path.join(fold_path,'weibo.'+split)
    with open(file_path) as file:
        data = file.readlines()
        data = [mid2imgnum[line.strip().split('\t')[0]] for line in data]
    with open(f'datasets/weibo/{split}_ids.txt','w') as file:
        file.write('\n'.join(data))