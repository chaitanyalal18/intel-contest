import os
filee = open('data/train.txt','w')
given_dir = 'data/labels'
[filee.write(os.path.join('data/train_images',i)+'\n') for i in os.listdir(given_dir)]
