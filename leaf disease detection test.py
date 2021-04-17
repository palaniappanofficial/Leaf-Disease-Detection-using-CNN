from keras.preprocessing import image
from keras.models import load_model
import numpy as np
model=load_model('model.model')
print("Model Loaded Successfully")

def classify(img_file):
    img_name=img_file
    test_image=image.load_img(img_name,target_size=(128,128))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    arr=np.array(result[0])
    print(arr)
    maxx=np.amax(arr)
    max_prob=arr.argmax(axis=0)
    max_prob=max_prob+1
    classes=['Tomato__Tomato_mosaic_virus','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_healthy','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite']
    result=classes[max_prob-1]
    print(img_name,result)
import os
path='D:/My Projects/Leaf Disease Detection using CNN/Dataset/test'
files=[]
for r,d,f in os.walk(path):
    for file in f:
        files.append(os.path.join(r,file))
for f in files:
    classify(f)
    print('\n')
