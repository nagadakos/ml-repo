import matplotlib.pyplot as plt
import matplotlib.axes  as  ax
import numpy as np



file1 = "./PyTorch/PyTorch_drop_rep.txt"
file2 = './Keras/mnistKeras_report.txt'
file3 = './PyTorch/PyTorch_no_drop_report.txt'
file4 = './Keras/mnistKeras_no_drop_report.txt'
file5 = './PyTorch/PyTorch_drop_eval.txt'
file6 = './PyTorch/PyTorch_no_drop_eval.txt'

# This list has the filelabels to examine
files = [file1, file2, file3, file4]
eval_files = [file5, file6]
labels = ["PyTorch-Drop", "Keras-Drop", "PyTorch-NO_Drop", "Keras-NO_Drop"]
# Lists acting as data holders.
reps  = []
eval_reps  = []
# index variables
trainAcc  = 0
trainLoss = 1
testAcc   = 2 
testLoss  = 3


epochs = 0
# open files and read data
for i,f in enumerate(files):
    reps.append([[],[],[],[]])
    with open(f, 'r') as p:
        for j,l in enumerate(p):
            # Ignore last character from line parser as it is just the '/n' char.
            report = l[:-1].split(' ')
            reps[i][trainAcc].append(report[trainAcc])
            reps[i][trainLoss].append(report[trainLoss])
            reps[i][testAcc].append(report[testAcc])
            reps[i][testLoss].append(report[testLoss])

           # kerasRep[testLoss].append(report[testLoss])

        epochs = len(reps[0][0])
        reps[i] = np.asarray(reps[i], dtype = np.float32)

for i,f in enumerate(eval_files):
    eval_reps.append([[],[]])
    with open(f, 'r') as p:
        for j,l in enumerate(p):
            # Ignore last character from line parser as it is just the '/n' char.
            report = l[:-1].split(' ')
            eval_reps[i][trainAcc].append(report[trainAcc])
            eval_reps[i][trainLoss].append(report[trainLoss])


        evalEpochs = len(eval_reps[0][0])
        eval_reps[i] = np.asarray(eval_reps[i], dtype = np.float32)
# Sanity Print
# print(reps)

#************************************
#Function: Plot Accuracy
#Description:   This function will plot the accuracy curve of 2 statistics
 #              Reports, given they are in the format train acc, train loss
#               test acc, test loss.
#Arguments:     rep1:   list of stas report 1 
#               rep2:   list of stas report 2 
#               epochs: int. Number of epochs
#               title:  string, used to anotate the plot figure.
#***********************************
def plot_acc( rep1, rep2, epochs, title):
   
    a = np.asarray(rep1, dtype = np.float32)
    b = np.asarray(rep2, dtype = np.float32)
    ymin =np.asscalar(np.fmin(a[0][0], b[0][0]))
    epchs = np.arange(1, epochs+1)
    fig = plt.figure()
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    # plt.plot(epchs, a[0], 'r', label = 'Keras_train', epchs, b[0], 'b', 'PyTorch_train')
    plt.plot(epchs, a[0],  'r', epchs, b[0], 'b')
    plt.plot(epchs, a[2], 'm--', epchs, b[2], 'c--')
    labels = ['Keras_train', 'PyTorch_train', 'Keras_test', 'PyTorch_test']
    plt.legend( labels, loc='lower right')
    # plt.draw()
    # plt.pause(10)

    # Detailed plot of lists. This is very cumbersoeme to handle
    # in pyplot.
    # fig = plt.figure()
    # plt.title(' Keras-PyTorch model Detailed Accuracy vs Epoch')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # f1  = fig.add_subplot(111, label="1")
    # f2 = fig.add_subplot(111, label="2", frame_on=False)
    # f1.plot(epchs, rep1[0], 'r--') 
    # f2.plot(epchs, rep2[0], 'g--') 
    # f2.xaxis.tick_top()
    # f2.yaxis.tick_right()
    # f2.set_xlabel('Epochs ', color="C1") 
    # f2.set_ylabel('Accuracy', color="C1")       
    # f2.xaxis.set_label_position('top') 
    # f2.yaxis.set_label_position('right') 
    # f2.tick_params(axis='x', colors="C1")
    # f2.tick_params(axis='y', colors="C1")
    # plt.yticks(rep1[0][0::5])
    # plt.yticks(rep1[0][0::5])
    # plt.ylim(ymin, 1)
    # plt.plot(epchs, rep1[0], 'r--', epchs, rep2[0], 'b-.')
    # plt.axis([epchs[0], epchs[-1], ymin, 1])
    # plt.yticks(rep1[0][0::5])
    # plt.plot(epchs, rep2[0], 'b--')
    # tell pyplot to write a y-axis tick every 5 units

    # plt.draw()
    # plt.pause(10)
# TODO: This Function will plot all reports on the same figure!
def plot_all_in_one(reps, epochs, title):

    print("Hellos")

#############################################################
#
# Description:  This function computer the moments from training
#               reports. They are 4 Columns: train acc, train loss
#               test acc, test loss.
#############################################################
def comp_train_moments(reps, epochs, labels):
    print("Size of report: {} {} {}".format(len(reps), len(reps[0]), len(reps[0][1])))
    moments = [[], []]
    moments[0] = np.mean(reps, axis = 2)
    moments[1] = np.std(reps, axis = 2)
    for i, l in enumerate(labels):
        print("{:<20}: {}" .format(labels[i], moments[0][i]))
    return moments

#############################################################
#
# Description:  This function computes the moments of evaluation
#               reports in the format accuracy, loss
#############################################################
def comp_eval_moments(reps, epochs, labels):
    print("Size of report: {} {} {}".format(len(reps), len(reps[0]), len(reps[0][1])))
    moments = [[], []]
    print("{} {} {}".format(len(reps), len(reps[0]), len(reps[0][0])) )
    print("{} {} {}".format(len(reps), len(reps[1]), len(reps[1][0])) )
    moments[0] = np.mean(reps, axis = 2)
    moments[1] = np.std(reps, axis = 2)
    for i, l in enumerate(labels):
        print("{:<20} MEAN: {}" .format(labels[i], moments[0][i]))
        print("{:<20} STD : {}" .format(labels[i], moments[1][i]))
    return moments
def main():

    # Compute moments of reported data
    moments  = comp_train_moments(reps, epochs, labels)
    moments2 = comp_eval_moments(eval_reps, evalEpochs, ["PyTorch Drop Eval", "PyTorch NO-Drop Eval"])
    title = 'Keras-PyTorch DROPOUT Model: Accuracy vs Epoch'
    plot_acc(reps[1],reps[0], epochs, title)
    plt.savefig("./plots/keras_vs_pytorch_dropout_train-epoch.png")
    plt.close()

    title = 'Keras-PyTorch NO-DROP Model: Accuracy vs Epoch'
    plot_acc(reps[3],reps[2], epochs, title)
    plt.savefig("./plots/keras_vs_pytorch_NO_dropout_train-epoch.png")
    plt.close()
if __name__ == '__main__':
    main()
