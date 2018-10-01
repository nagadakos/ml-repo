import numpy as np
import mxnet as mx
import mxnet.ndarray as F
from mxnet.gluon import nn
from mxnet import nd, gluon, init, autograd
from mxnet.gluon.data.vision import datasets, transforms
import matplotlib.pyplot as plt
from time import time
from multiprocessing import cpu_count



# Get the number of cores to use for multithread
# data loading!
CPU_COUNT = cpu_count()


# Index definitions for history metrics
trainAcc = 0
trainLoss= 1
testAcc  = 2
testLoss = 3


# Define the transformation to be applied to data
# Turn the images to Tensor representation and normalize them.
tensor_normalize_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31)])


trainSource = mx.gluon.data.vision.datasets.MNIST(train=True)
validSource = mx.gluon.data.vision.datasets.MNIST(train=False)

train_dataset = trainSource.transform_first(tensor_normalize_tf)
valid_dataset = validSource.transform_first(tensor_normalize_tf)

batch_size = 32
trainDataLoader = mx.gluon.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=CPU_COUNT)
validDataLoader = mx.gluon.data.DataLoader(valid_dataset, batch_size, num_workers=CPU_COUNT)


# Gluon provides 2 methods of creating a network architecture
# 1) Select a bulidng pattern such as sequential, Recurrent etc
#    and use the guideline to build the network, i.e in sequential
#    like Keras, each layer has as input the output of the prev one.
# 2) Like PyTorch, define a subclass of a class, here block, and
#    define 2 functions __init__ and forward. In init you declare
#    the available building blocks and their structure, and in forward
#    how they are laid out to form the actual network.
class mnist_Net(gluon.Block): 

    history = [[],[],[],[]]
    def __init__(self, **kwargs):
        # Run `nn.Block`'s init method
        super(mnist_Net, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = nn.Conv2D(32, kernel_size=3)
            self.pool1 = nn.MaxPool2D(pool_size=(2,2))
            self.conv2 = nn.Conv2D(64, kernel_size=3)
            self.pool2 = nn.MaxPool2D(pool_size=(2,2))
            self.drop2D= nn.Dropout(0.25)
            self.drop1D= nn.Dropout(0.5)
            self.fc1 = nn.Dense(128)
            self.fc2 = nn.Dense(10)
            # self.fc3 = nn.Dense(10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2D(x)
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        # Essentially flattens to 1D.
        x = x.reshape((0, -1))
        x = F.relu(self.fc1(x))
        x = self.drop1D(x)
        x = F.relu(self.fc2(x))
        x = F.softmax(x)
        return x

    def train(self, args, trainData):
        epoch       = args[0]
        lossFunction = args[1]
        trainer     = args[2]
        device      = args[3]

        acc    = 0.0
        tLoss   = 0.0
        for batch_idx, (data, label) in enumerate(trainDataLoader):
            # if batch_idx % 100 == 0:
                # print("Batch: {}/{} size: {}".format(batch_idx, len(trainDataLoader), data.shape))
            data  = data.as_in_context(device)
            label = label.as_in_context(device)

            with autograd.record():
                output = self(data)
                loss = lossFunction(output, label)
                loss.backward()
            trainer.step(batch_size) 

            # Get loss and accuracy
            # Get th e
            predictions = nd.argmax(output, axis=1)
            acc += (predictions == label.astype('float32') ).mean().asscalar() 
            # Have to devide by the number or categories to convert to
            # Traditional categorical loss.
            tLoss += loss.mean().asscalar() / 10 
        self.history[trainAcc].append(acc)
        self.history[trainLoss].append(tLoss)
        print("Epoch: {} -> Train accuracy: {:.4f} Train Loss: {:.4f}" .format(epoch, acc /
                                                                               len(trainDataLoader),
                                                                              tLoss
                                                                               /len(trainDataLoader)
                                                                              ))
    def test(self, args, testData):
        epoch       = args[0]
        lossFunction = args[1]
        device      = args[3]
        acc    = 0.0
        tLoss   = 0.0
        for batch_idx, (data, label) in enumerate(testData):
            # if batch_idx % 100 == 0:
                # print("Batch: {}/{} size: {}".format(batch_idx, len(trainDataLoader), data.shape))
            data  = data.as_in_context(device)
            label = label.as_in_context(device)

            output = self(data)
            predictions = nd.argmax(output, axis=1)
            loss = lossFunction(output, label)

            # Get loss and accuracy
            acc += (predictions == label.astype('float32') ).mean().asscalar() 
            tLoss += loss.mean().asscalar() / 10
            self.history[testAcc].append(acc)
            self.history[testLoss].append(tLoss)
        print("Epoch: {} -> Test accuracy: {:.4f} Test Loss: {:.4f}" .format(epoch, acc /
                                                                               len(testData),
                                                                              tLoss
                                                                               /len(testData)
                                                                              ))

def main():

    lossFunction = gluon.loss.SoftmaxCrossEntropyLoss()
    
    epochs = 12
    # holds arguments for training.
    # 1. current epoch 2.lossFunction 3. trainer type
    args = []
     
    net = mnist_Net()
    device = mx.gpu(0)
    # initialize weight with Xavier specification, on GPU
    net.initialize(mx.init.Xavier(), ctx=device)
    trainer      = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

    args.append(epochs)
    args.append(lossFunction)
    args.append(trainer)
    args.append(device)
    for e in range(epochs):
        args[0] = e
        net.train(args, trainDataLoader)
        net.test(args, validDataLoader)

    # save the model
    net.save_parameters('./Models/Gluon_mnist_drop_model.params')











if __name__ == "__main__":
    main()






