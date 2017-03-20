import sys
import os
import math
import caffe
from caffe.proto import caffe_pb2
from caffe import layers as L, params as P
import caffe.draw
import google.protobuf 
import numpy as np
from numpy import append, array, int8, uint8, zeros
import scipy
import h5py

sys.path.append("pycaffe/layers") # the datalayers we will use are in this directory.
sys.path.append("pycaffe") # the tools file is in this folder

class LeNet(object):
    """docstring for LeNet"""
    def __init__(self, arg):
        self.solver_prototxt_filename = solver_prototxt_filename
        self.train_prototxt_filename = train_prototxt_filename
        self.val_prototxt_filename = val_prototxt_filename

    def lenet(self, hdf5, batch_size):
        '''
        Initialise the Basic LeNet layers

        Input Parameters:
        hdf5 : train hdf5 file
        batch_size : batch size for training the system

        Output Parameters:
        n.to_proto() : dictionary of network layers

        '''
        n = caffe.NetSpec()
        n.data, n.label = L.HDF5Data(name="data", data_param={'source': hdf5,'batch_size':batch_size}, ntop=2,
            include={'phase':caffe.TRAIN})        
        n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
        n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
        n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
        n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
        n.ip1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
        n.relu1 = L.ReLU(n.ip1, in_place=True)
        n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
        n.loss =  L.SoftmaxWithLoss(n.ip2, n.label)
        n.accuracy = L.Accuracy(n.ip2, n.label, include={'phase':caffe.TEST})
        print (str(n.to_proto()))
        return n.to_proto()

    def lenet_test(self, hdf5, batch_size):
        '''
        Declaring the validation phase for the network layer

        Input Parameters:
        hdf5 : val hdf5 file
        batch_size : batch size for validating the system
        '''
        n = caffe.NetSpec()
        n.data, n.label = L.HDF5Data(name="data", data_param={'source':hdf5, 'batch_size':batch_size}, ntop=2,
            include={'phase':caffe.TEST})
        return n.to_proto()

    def writelenet(self, prototxt_filename, hdf5_train_file, hdf5_test_file, batch_size_train, batch_size_test):
        '''
        Store the training and testing parameters in same file

        Input Parameters:
        prototxt_filename : Path to the prototxt file to store Layer Parameters
        hdf5_train_file   : Path of hdf5 training data
        hdf5_test_file    : Path of hdf5 validation data
        batch_size_train  : training data batch size
        batch_size_test   : validation data batch size 
        '''
        if not os.path.exists(prototxt_filename):
            with open((prototxt_filename), 'w') as f:
                f.write(str(self.lenet_test(hdf5_test_file, batch_size_test)))
                f.write(str(self.lenet(hdf5_train_file, batch_size_train)))

    def train(self, snapshot_interval=100):
        '''
        Train the ANN
        '''
        caffe.set_mode_gpu()
        solver = caffe.get_solver(self.solver_prototxt_filename)
        solver.solve()

        while solver.iter < solver.max_iters:
            # Make one SGD update
            try:
                solver.step(1)
            except Exception as e:
                print str(e)
                import traceback
                traceback.print_exc()
                raise
            if solver.iter % snapshot_interval == 0:
                self.snapshot(solver)

    def snapshot(self, solver):
        '''
        Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        '''
        net = solver.net
        self.caffemodel_filename = 'snapshot_iter_{:d}'.format(solver.iter) + '.caffemodel'
        
        net.save(str(self.caffemodel_filename))
        solver.snapshot()
        print 'Wrote snapshot to: {:s}'.format(self.caffemodel_filename)

    def set_solver(self):

        '''
        Define the solver required by model

        Input Parameters : None
        Output Parameters :
        s : solver object
        '''
        
        s = caffe_pb2.SolverParameter()

        # Set a seed for reproducible experiments:
        # this controls for randomization in training.
        s.random_seed = 0xCAFFE

        # Specify locations of the train and (maybe) test networks.
        s.train_net = self.train_prototxt_filename
        s.test_net.append(self.val_prototxt_filename)
        s.test_interval = 500  # Test after every 500 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.

        s.max_iter = 10000     # no. of times to update the net (training iterations)
        
        # Set the initial learning rate for SGD.
        s.base_lr = 0.01  # EDIT HERE to try different learning rates
        # Set momentum to accelerate learning by
        # taking weighted average of current and previous updates.
        s.momentum = 0.9
        # Set weight decay to regularize and prevent overfitting
        s.weight_decay = 5e-4

        # Set `lr_policy` to define how the learning rate changes during training.
        # This is the same policy as our default LeNet.
        s.lr_policy = 'inv'
        s.gamma = 0.0001
        s.power = 0.75
        # EDIT HERE to try the fixed rate (and compare with adaptive solvers)
        # `fixed` is the simplest policy that keeps the learning rate constant.
        # s.lr_policy = 'fixed'

        # Display the current training loss and accuracy every 1000 iterations.
        s.display = 100

        # Snapshots are files used to store networks we've trained.
        # We'll snapshot every 5K iterations -- twice during training.
        s.snapshot = 1000
        s.snapshot_prefix = './caffemodel/lenet'

        # Train on the GPU
        s.solver_mode = caffe_pb2.SolverParameter.GPU

        # Write the solver to a temporary file and return its filename.
        with open(self.solver_prototxt_filename, 'w') as f:
            f.write(str(s))

        return s

    def print_network_parameters(self,net):
        '''
        Print the parameters of the network
        '''
        print(net)
        print('net.inputs: {0}'.format(net.inputs))
        print('net.outputs: {0}'.format(net.outputs))
        print('net.blobs: {0}'.format(net.blobs))
        print('net.params: {0}'.format(net.params)) 

    def get_predicted_output(self, img, net = None):
        '''
        Get the predicted output, i.e. perform a forward pass
        '''
        if net is None:
            net = caffe.Net(self.deploy_prototxt_filename,self.caffemodel_filename, caffe.TEST)  

        # transform it and copy it into the net
        image = caffe.io.load_image(img)
        net.blobs['data'].data[...] = transformer.preprocess('data', image)

        # perform classification
        net.forward()

        # obtain the output probabilities
        output_prob = net.blobs['prob'].data[0]

        # sort top five predictions from softmax output
        top_inds = output_prob.argsort()[::-1][:5]

        plt.imshow(image)

        print 'probabilities and labels:'
        zip(output_prob[top_inds], labels[top_inds])

        return output_prob.argmax()  

    def print_network(self, prototxt_filename):
        '''
        Draw the ANN architecture

        Input Parameter:
        prototxt_filename : Path to the Prototxt file to draw
        '''
        _net = caffe.proto.caffe_pb2.NetParameter()
        f = open(prototxt_filename)
        google.protobuf.text_format.Merge(f.read(), _net)
        caffe.draw.draw_net_to_file(_net, prototxt_filename + '.png' )
        print('Draw ANN done!')

    def get_accuracy(self,true_outputs, prediction_outputs, df):
        '''
        Get the accuracy of the system

        Input Parameters : 
        true_outputs : actual labels
        prediction_output : prediction by the system
        '''

        pred_output = []
        pred_output.append(np.where(df.value == df.pred, 1, 0))
        true_outputs = []
        true_outputs.append(np.where(df.value, 1, 0))

        print('accuracy: {0}'.format(sklearn.metrics.accuracy_score(true_outputs, pred_output)))