
# coding: utf-8

import matplotlib
matplotlib.use("svg")
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np


# ## Installation
# 
# We need to install PyTorch. At this point, installation of libraries should be relatively painless however we should 
# try to install early on.
# 
# We should be able to install PyTorch using conda. 
# 
#     conda install pytorch torchvision -c pytorch
# 
# If we run into trouble, you can check out their documentation located at http://pytorch.org/.

# ## Image Dataset
# 
# We will be using the Oxford-IIIT pet dataset. 
# 
#     http://www.robots.ox.ac.uk/~vgg/data/pets/
#     
# It is a non-trivial download of ~800MB containing a collection of cats and dogs, 
# annotated with their breed, head ROI (region of interest), and a pixel level segmentation. First we will 
# need to canonicalize the images. We will do so using the pillow package, which should be already included in 
# Anaconda installations. If you'd like to know more, you can view the tutorial here: https://pillow.readthedocs.io/en/5.1.x/handbook/tutorial.html.



from PIL import Image

def plot_image(im):
    plt.imshow(np.asarray(im))

#-------------------------------------------------
im_cat = Image.open("images/Abyssinian_1.jpg")
print(im_cat.format, im_cat.size, im_cat.mode)
plot_image(im_cat)
#-------------------------------------------------
im_dog = Image.open("images/english_cocker_spaniel_198.jpg")
plot_image(im_dog)

#Every picture has at least 100 pixels in either width or height. Thus, we will crop each image down 
# to be a square picture, then scale it down to 100x100 pixels. 
# 
# ### Specification
# 1. First, crop the image to be a square by reducing one dimension to be the same size as the other. You should center
# the cropping as much as possible (e.g. crop the same number of pixels from the left as from the right).
# If you must crop an odd number of pixels, 
# then crop an extra pixel from the right or from the bottom. 
# 2. Second, resize the image to be 100 by 100 pixels. You should use the Image.ANTIALIAS filter for the resample parameter. 

def crop_and_scale_image(im):
    """ Crops and scales a given image. 
        Args: 
            im (PIL Image) : image object to be cropped and scaled
        Returns: 
            (PIL Image) : cropped and scaled image object
    """
    if im.height < im.width:
        diff = im.width - im.height
        offset = int(diff/2.0)
        off = diff - offset
        cropped = im.crop((offset, 0, im.width-off, im.height))
    elif im.height>im.width:
        diff = im.height - im.width
        offset = int(diff/2.0)
        off = diff - offset
        cropped = im.crop((0, offset, im.width, im.height-off))
    else:
        cropped = im

    return cropped.resize((100, 100), Image.ANTIALIAS)


#-----------------------------------------
plot_image(crop_and_scale_image(im_cat))
plot_image(crop_and_scale_image(im_dog))


# ## Train / Validate / Test splits for large datasets
# Next we will load the data and perform our usual data split. However, 
# the image dataset is **very large**. So large, that if you try to load every image 
# and process them at in batch, your computer may run out of memory (mine did). 
# Thus, initially we will work with the filenames until it is a manageable size. 
# This code assumes that you downloaded images exist in a folder images/, and creates an array of filepaths to each image. 

import os
dname = "images/"
im_paths = np.array([dname+fname for fname in os.listdir(dname) if fname.endswith(".jpg")])
P = np.random.permutation(len(im_paths))
split = 2000
fnames_tr, fnames_va, fnames_te = im_paths[P[:split]], im_paths[P[split:2*split]], im_paths[P[2*split:]]



plt.figure(figsize=(20,20))
for i,fname in enumerate(fnames_tr[:100]):
    plt.subplot(10,10,i+1)
    plt.axis('off')
    plot_image(crop_and_scale_image(Image.open(fname)))



# ## Filename Parsing
# 
# We will need to extract the breeds and inputs for our VGG network from these files. 
# Implement the following two helper functions to do this. 
# 
# ### Specification
# * Extract the full breed name from each filename, which is the part of the filename not including 
# the number and the extension. See the example output for an example. 
# * Using your crop_and_scale_image function from earlier, generate the image data matrix that is to be input into VGG. 
# * Note that VGG requires its input dimension to be in a slightly different order than that returned by Pillow. Use 
# np.rollaxis to rotate it until in the proper order. 
# * Some of the images are not in RGB format. You will have to convert them to RGB. 


import re
def fname_to_breed(fname):
    """ Extracts the breed from a filename
        Args: 
            fname (string) : the filename to be parsed
        Returns: 
            (string) : the breed extracted from the filename
    """
    if "/" in fname:
        fname = fname[fname.index("/")+1:]
    fname = fname[:fname.rfind("_")]
    return fname
    pass
def fname_to_vgg_input(fname):
    """ Creates the input for a VGG network from the filename 
        Args: 
            fname (string) : the filename to be parsed
        Returns: 
            (numpy ndarray) : the array to be passed into the VGG network as a single example
    """
    im = Image.open(fname)
    im = im.convert('RGB')
    im = crop_and_scale_image(im)
    data = np.asarray(im)
    roll = np.rollaxis(data,2)
    return roll
    pass

# -----------------------------------------------
print(fname_to_breed("images/english_cocker_spaniel_144.jpg"))
print(fname_to_vgg_input("images/Abyssinian_1.jpg"))



# Our implementation gets the following output: 
#      
#     english_cocker_spaniel
#     [[[59 61 61 ..., 74 72 71]
#       [58 59 61 ..., 76 72 70]
#       [59 62 65 ..., 75 71 70]
#       ..., 
#       [21 21 22 ..., 23 25 26]
#       [18 19 19 ..., 23 24 23]
#       [16 17 18 ..., 22 21 22]]
# 
#      [[70 72 72 ..., 87 85 82]
#       [69 69 72 ..., 89 84 81]
#       [69 71 74 ..., 86 83 82]
#       ..., 
#       [27 26 28 ..., 29 31 32]
#       [25 27 27 ..., 29 30 29]
#       [23 25 26 ..., 28 27 28]]
# 
#      [[53 55 55 ..., 69 67 65]
#       [52 52 55 ..., 71 66 64]
#       [53 54 57 ..., 69 65 62]
#       ..., 
#       [17 19 20 ..., 19 21 22]
#       [17 16 17 ..., 19 20 19]
#       [15 14 15 ..., 18 17 18]]]
# 

# ## VGG
# 
# One of the common ways to use deep learning is to automatically learn features. 
# For this problem, we will use a pretrained [VGG network](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) 
# to create useful features. Tuning and training your own custom neural networks takes a significant amount of 
# time and effort (much more so than running SVM with RBF features), so we will use the pretrained network VGG-16 
# in the PyTorch deep learning framework. You will find the documentation here helpful: 
# http://pytorch.org/docs/master/torchvision/models.html. 
# 
# ### Specification
# * Load the pretrained VGG network using PyTorch

import torch
import torchvision.models as models
from torch.autograd import Variable

def VGG_16():
    """ Loads a pretrained VGG network. 
        Returns: 
            (pytorch VGG model) : the VGG-16 model
    """
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model
# --------------------
truncated_vgg = VGG_16()


# ## Converting files to features and labels 
# Now we have all the parts to convert our image files to features and labels. 
# Implement the following two functions that turn filenames into labels and filenames to features. 

# ### Specification
# * We should take all the breeds and sort them in alphabetical order. Then, assign the a label of 0 to 
 # the first, and so on, so that the last breed in alphabetical order has a label of 36. 
# * The features are the flattened output of the last layer of the VGG network.


def fnames_to_labels(fnames):
    """ Given a list of filenames, generate the corresponding array of labels
        Args: 
            fnames (list) : A list of filenames
        Returns: 
            (numpy ndarray) : a 1D array of numerical labels
    """
    breed_labels = [fname_to_breed(fname) for fname in fnames]
    breed_set = set(breed_labels)
    labels = sorted(breed_set)
    
    breed_num_dict = {breed:i for i,breed in enumerate(labels)}
    
    label_nums = [breed_num_dict[breed] for breed in breed_labels]
    return np.array(label_nums)
    pass

# ---------------------------
fnames_to_labels(fnames_tr[:10])


def fnames_to_features(fnames, vgg):
    """ Given a list of filenames and a VGG16 model, generate the corresponding array of VGG features
        Args: 
            fnames (list) : A list of filenames
            vgg (pytorch model) : a pretrained VGG16 model
        Returns: 
            (pytorch Variable) : a (m x 4608)-dimensional Variable of features generated from the VGG model,
                                 where m is the number of filenames
    """
    image_array = []
    for data in fnames:
        vgg_fname = fname_to_vgg_input(data)
        image_array.append(vgg_fname)
    arr = np.array(image_array)

    tensor = Variable(torch.Tensor(arr))
    x= vgg.features(tensor).view(len(fnames), -1)
    return x
        
# ---------------------------------------
fnames_to_features(fnames_tr[:10], truncated_vgg)



# ## And now we wait... just kidding. 
# Now, you can run this on the entirety of the training, validation, and test set... 
# but wait. Our machines take about 2 seconds per example to generate VGG examples without any GPU acceleration,
# which means that processing all the images will be a several hour endeavor (about 8 hours on my machines)! 
# To save you some time, we've already run this on a random training and validation set (2k examples each), 
# in X_tr.txt and y_va.txt. You can find their labels in y_tr.txt and y_va.txt. 



#----------------------------
X_tr = np.loadtxt("X_tr.txt")
X_va = np.loadtxt("X_va.txt")


# ------------------------------
y_tr = np.loadtxt("y_tr.txt", dtype=int)
y_va = np.loadtxt("y_va.txt", dtype=int)


# ## Pet breed classification from VGG features
# 
# And now for the final task: we will now classify the pets by breed using the generated VGG features. 
# Our goal here is to get at least 50% classification accuracy on the pets dataset. We will be given the 4k random examples 
# above to build our model (the train and validation datasets), and we will evaluate our model on a holdout test set. 
# 
# ### Baseline approach
# 
# The baseline approach here is to use yet another neural network. A simple network that isn't too large can train to about 55% 
# accuracy in a minute on our machines. Try different configurations with different learners and architectures. 

import time
from sklearn import svm
def predict_from_features(X, y, X_te):
    """ Given labels and VGG features, predict the breed of the testing set. 
    Args: 
        X (numpy ndarray) : 2D array of VGG features, each row is a set of features for a single example
        y (numpy ndarray) : 1D array of labels corresponding to the features in X
        X_te (numpy ndarray) : 2D array of VGG features for unlabeled examples
    Returns: 
        (numpy ndarray) 1D array of predicted labels for the unlabeled examples in X_te
    """
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y) 
    y_pred = clf.predict(X_te)
    return y_pred
    pass

# ------------------------------
start = time.time()
y_p = predict_from_features(X_tr, y_tr, X_va)
end = time.time()
print("Validation accuracy: {} in {} seconds".format(np.mean(y_p==y_va), end-start))

