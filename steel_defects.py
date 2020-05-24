


import os
import cv2
import numpy as np
from tqdm import tqdm

# model tools
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



# model

from keras.preprocessing import image







class steel_defects:

    def load_images(self, image_paths):
    
        fill_list = []
    
        for idx in tqdm(range(len(image_paths))):
            path = image_paths[idx]
            yield cv2.imread(path)
        
    def resize_images(self, images):
    
        img_list = []
    
        for img in images:
        
            yield np.resize(img, (64, 64, 3))
        
        
    def greyscale_images(self, images):
    
        img_list = []
    
        for img in images:
            
            width, heigh, channels = img.shape

        
            img = 0.2125*img[:,:,0] + 0.7154*img[:,:,1] + 0.0721*img[:,:,2]
            
            img = img.reshape(width, heigh, 1)
            
        
        # formula obtained from 
        # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
            yield img
        

        
    def load_imgsLabels(self, image_paths):
    
#     label = image_paths[-1]
    
        images = self.load_images(image_paths)
    
        images = self.resize_images(images)
    
        images_list = self.greyscale_images(images)

        return images_list

    def features_to_np_array(self, images):
    
        images = list(images)
    
        images = np.stack(images, axis=0)
    
        return images

    
    def make_imgs_list(self, imgs_dir, imgs_list):
    
        empty_list = []
    
        for img in imgs_list:
        
            img_dir = imgs_dir + '/' + img
        
            empty_list.append(img_dir)
        
        return empty_list
        
        


    def get_all_imgs(self, from_dir, labels_list):
    
        imgs_list = []
        labels_list = []
    
        for label in labels_list:
        
            img_dir = from_dir + '/' + str(label)
        
            img_list = os.listdir(img_dir)
        
            img_list = self.make_imgs_list(img_dir, img_list)
        
            imgs = self.load_imgsLabels(img_list)
        
            imgs =  self.features_to_np_array(imgs)
        
            labels = imgs.shape[0]*[int(label)]
        
            imgs_list.append(imgs)
        
            labels_list.append(labels)
        
#     imgs_list = features_to_np_array(imgs_list)
    
        return imgs_list, labels_list
    
    def load_defects(self, val_dir):
        
        img_list_1 = os.listdir(val_dir+'/'+'1')
        img_list_2 = os.listdir(val_dir+'/'+'2')
        img_list_3 = os.listdir(val_dir+'/'+'3')
        img_list_4 = os.listdir(val_dir+'/'+'4')



        img_list_1 = self.make_imgs_list(val_dir + '/' + '1', img_list_1)
        img_list_2 = self.make_imgs_list(val_dir + '/' + '2', img_list_2)
        img_list_3 = self.make_imgs_list(val_dir + '/' + '3', img_list_3)
        img_list_4 = self.make_imgs_list(val_dir + '/' + '4', img_list_4)


        img_list_1 = self.load_imgsLabels(img_list_1)
        img_list_2 = self.load_imgsLabels(img_list_2)
        img_list_3 = self.load_imgsLabels(img_list_3)
        img_list_4 = self.load_imgsLabels(img_list_4)


        img_list_1 = self.features_to_np_array(img_list_1)
        img_list_2 = self.features_to_np_array(img_list_2)
        img_list_3 = self.features_to_np_array(img_list_3)
        img_list_4 = self.features_to_np_array(img_list_4)

        lbl_list_1 = img_list_1.shape[0]*[1]
        lbl_list_2 = img_list_2.shape[0]*[2]
        lbl_list_3 = img_list_3.shape[0]*[3]
        lbl_list_4 = img_list_4.shape[0]*[4]


        imgs = np.concatenate((img_list_1, img_list_2, img_list_3, img_list_4))
        lbls = lbl_list_1 + lbl_list_2 + lbl_list_3 + lbl_list_4


        lbls = np.array(lbls)
        
        lbls = lbls - 1
        
        lbls = to_categorical(lbls)
        
        return imgs, lbls
    
    
  