import cv2
import sys
import os
import math
import numpy as np
import scipy
from sklearn.externals import joblib
from sklearn.datasets.mldata import fetch_mldata
from skimage.feature import hog
from sklearn.svm import LinearSVC 
from skimage import img_as_uint
from array import array as pyarray
import pandas as pd
import h5py
import csv
from evaluation import eval_expr
from opencvutils import equationextractor, digitextractor
from caffelenet import LeNet

# change to respective directory
root = './'
os.chdir(root)

class ArithmaticEval(object):
    def __init__(self):
        self.dataset = fetch_mldata("mnist-original", data_home="./")
        self.features = np.array(self.dataset.data, 'int16')
        self.labels = np.array(self.dataset.target, 'str')
        
    def load_data(self):
        '''
        Append the MNIST Data with Augmented Data {'+', '-', '*', 'x'}
        '''
        for folder in os.listdir("./augmented"):
            if os.path.isdir(folder):
                for f in os.listdir(folder):
                    im = cv2.imread(f, cv2.COLOR_BGR2GRAY)
                    im = img_as_uint(im)
                    np.append(self.features, im, axis=0)
                    np.append(self.labels, np.array(folder, 'str'), axis=0)

    def save_data_as_hdf5(self, hdf5_train_data_filename, hdf5_val_data_filename):
        '''
        Saving the dataset to HDF5 format that Caffe can accept easily
        Saving image data to HDF5

        Input Parameters:
        hdf5_data_filename : numpy array to be saved in HDF5 format
        '''

        if datatype == "train" and not os.path.exists(hdf5_train_data_filename):
            with h5py.File(hdf5_train_data_filename, 'w') as f:
                f['data'] = feat_train.astype(np.float32)
                f['label'] = label_train.astype(np.float32)

        elif datatype == "test" and not os.path.exists(hdf5_data_filename):
            feat = []
            lbl = []
            for file in os.listdir("./digits"):
                if file.endswith(".txt"):
                    for line in open(file):
                        img_path = line.split(' ', 1)[0]
                        img_label = line.split(' ', 1)[1]
                        im = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
                        feat.append(im)
                        lbl.append(img_label)

            with h5py.File(hdf5_data_filename, 'w') as f:
                f['data'] = np.array(feat, 'int16')
                f['label'] = np.array(lbl, 'str')

        else:
            raise TypeError(datatype)

if __name__ == '__main__':

    olddataset = os.path.join(root, "dataset")
    equationdataset = os.path.join(root, "equations")
    digitdataset = os.path.join(root, "digits")

    # set hdf5 datapath
    hdf5_train_data_filename = os.path.join(root, 'data/hdf5/mnist_train_data.hdf5') 
    hdf5_val_data_filename = os.path.join(root, 'data/hdf5/mnist_val_data.hdf5') 
    hdf5_train_file = os.path.join(root, 'data/hdf5/train.txt')
    hdf5_val_file = os.path.join(root, 'data/hdf5/val.txt')

    # Set parameters
    solver_prototxt_filename = os.path.join(root, 'caffemodel/lenet_solver.prototxt')
    train_prototxt_filename = os.path.join(root, 'caffemodel/lenet_train.prototxt')
    val_prototxt_filename = os.path.join(root, 'caffemodel/lenet_train.prototxt')
    deploy_prototxt_filename  = os.path.join(root, 'caffemodel/lenet_deploy.prototxt')
    
    '''
	------------------------------------------------------
    Declare object to Arithmatic class and CaffeNet class
    ------------------------------------------------------
    '''

	arth = ArithmaticEval()
    
    '''
	------------------------------------------------------
    Load the Training & Validation Dataset
    ------------------------------------------------------
    '''

    # Load MNIST Dataset and Training Dataset
    arth.load_data()

    # Validation extract equation
    equationextractor(olddataset, equationdataset)

    # Validation extract digits
    digitextractor(equationdataset, digitdataset)
    
    # Save Both the dataset in HDF5 file
    arth.save_data_as_hdf5(hdf5_train_data_filename, hdf5_val_data_filename)

    '''
	------------------------------------------------------
    Initialise the Network Parameters and start Training
    ------------------------------------------------------
    '''
    net = LeNet(solver_prototxt_filename, train_prototxt_filename, val_prototxt_filename)
    # create the solver file
    net.set_solver()

    net.writelenet(train_prototxt_filename, hdf5_train_file, hdf5_val_file,
    			   batch_size_train=64, batch_size_test=10)

    max_iters = 10000
    net.train(max_iters, display=20, snapshot_interval=1)

    '''
	------------------------------------------------------
    Prediction for the Testing, here I am doing 
    for Validation set
    Compute the accuracy of the system
    ------------------------------------------------------
    '''

    # dict to store Path to image file and Predicted Labels to overlay on Image 
    data = {}
    
    # Get predicted outputs and Pass it on to Final Model
    for folder in os.listdir("./digits"):
        s = ''.join(map(str, [td.get_predicted_output(img) for img in os.listdir(folder)]))
        print s
        result = eval_expr(s)
        print result
        data[folder+'.jpg'] = str(result)
        with open('./Prediction.csv', 'w+', newline='') as mycsvfile:
            a = csv.writer(mycsvfile, delimiter=',', quoting = csv.QUOTE_NONE)
            a.writerows([str(folder + '.jpg'), s, str(result)])

    # check the accuracy by reading true values of the Image file 
    # from data.csv file and comparing with predicted label
    for file in os.listdir("./dataset"):
    	if file.endswith(".csv"):
    		with open(file, 'rb') as mycsvfile:
			    df = pd.read_csv(csv_file)
	# drop those values that are not predicted
	# for k,_ in data.iteritems():
	# 	df = df[df.file != k]

	# add the predicted value to the dataframe
	for k, v in data.iteritems():
	    df.loc[df[df.file == k].index, 'pred'] = v

	# compute the accuracy of the system
	net.get_accuracy(df)

    # Paste the Result overlay on the image
    for file in os.listdir("./dataset"):
        if file.endswith(".jpg", ".png", ".jpeg"):
            img = cv2.imread(file, cv2.IMREAD_COLOR)
            cv2.putText(img, data[file], (img.shape[0]/2,img.shape[1]/2l), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

    '''
	------------------------------------------------------
    Testing of the Full Model
    ------------------------------------------------------
    '''        
    '''
	------------------------------------------------------
    Display and Save Network Architecture
    ------------------------------------------------------
    '''

    net.print_network(deploy_prototxt_filename)
    net.print_network(train_prototxt_filename)
    net.print_network_weights(train_prototxt_filename)