import pandas
import random
split_point = 0.8
df = pandas.read_csv('datasets\content_noid.csv')
image_ids = list(df['image_id'])

test = random.sample(image_ids, int((1-split_point)*len(image_ids)))
train = [str(_id) for _id in image_ids if _id not in test]
test = [str(_id) for _id in test]
with open('datasets/train_ids.txt','w') as file:
    file.write('\n'.join(train))
    
with open('datasets/test_ids.txt','w') as file:
    file.write('\n'.join(test))