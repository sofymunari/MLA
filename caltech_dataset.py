from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class)
        '''
        label=0
        curlab=""
        self.images=[]
        self.labels=[]
        if(self.split=='train'):
            f=open('MLA/train.txt');
            lines= f.readlines();
            for line in lines:
                if line.startswith("BACKGROUND_Google")== False:
                    parts=line.split('/')
                    if curlab=="":
                        curlab=parts[0]
                    else:
                        if curlab!=parts[0]:
                            label=label+1
                            curlab=parts[0]
                    line1=line.split('\n')
                    path1=root+"/"+line1[0]
                    self.images.append(pil_loader(path1))
                    self.labels.append(label)
        else:
            f=open('MLA/test.txt');
            lines= f.readlines();
            for line in lines:
                if line.startswith("BACKGROUND_Google")== False:
                    parts=line.split('/')
                    if curlab=="":
                        curlab=parts[0]
                    else:
                        if curlab!=parts[0]:
                            label=label+1
                            curlab=parts[0]
                    line1=line.split('\n')
                    path1=root+"/"+line1[0]
                    self.labels.append(label)
                    self.images.append(pil_loader(path1))

       
                    
    def __makesplit__ (self):
        self.trainsplit=[]
        self.validsplit=[]
        count=0
        for i in range(0,self.images.size()) :
            if count<2 :
                self.trainsplit.append(i)
                count+=1
            else :
                self.validsplit.append(i)
                count=0
                
    def __trainsplit__ (self):        
        return self.trainsplit
    
    def __validsplit__ (self):
        return self.validsplit
			

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image= self.images[index]   # Provide a way to access image and label via index
        label= self.labels[index]                                          # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.images) # Provide a way to get the length (number of elements) of the dataset
        return length
