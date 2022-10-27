
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import string

"""def get_onehot_seq(seq): 
    chr_str = seq.upper()
    d = np.array(['A','C','G','T'])
    y = np.frombuffer(chr_str, dtype='|S1')[:, np.newaxis] == d
    return y
"""



def cal_entropy(probabilities):
    ent = entropy(probabilities, axis=-1, base = 2)
    return ent




def saveGeneratedSeq(oneHotSeq):

    B = oneHotSeq.shape[0]

    fileName = './GeneratedSeq/SeqDecoderVAE_Nucleotides.txt'

    with open(fileName, 'w') as filehandle:
        for i in range(B):
            indexSeq = oneHotSeq[i,:,:].argmax(axis=1)

            results = np.where(indexSeq==0,['A'],indexSeq)
            results = np.where(results=='1',['C'],results)
            results = np.where(results=='2',['G'],results)
            results = np.where(results=='3',['T'],results)


            filehandle.write('\n>>>>>>>>>>>\n')
            for listitem in results:
                filehandle.write('%s' % listitem)




def saveGeneratedSeq_Pro(oneHotSeq):

    B = oneHotSeq.shape[0]
    charr = list(string.ascii_uppercase)
    time = now.strftime("%H:%M:%S")
    fileName = './GeneratedSeq/'+ time + '_SeqDecoderVAE_Pro.txt'

    with open(fileName, 'w') as filehandle:
        for i in range(B):
            indexSeq = oneHotSeq[i,:,:].argmax(axis=1)

            results = np.where(indexSeq==0,charr[0],indexSeq)

            for j in range(25):
                results = np.where(indexSeq==j+1,charr[j+1],results)


            filehandle.write('\n>>>>>>>>>>>\n')
            for listitem in results:
                filehandle.write('%s' % listitem)



def get_onehot_seq_pro(seq):
    chr_str = seq.upper()
    d = np.array(list(string.ascii_uppercase))
    chr_str = np.array(chr_str, dtype = '<U')
    y = np.frombuffer(chr_str, dtype='|U1')[:, np.newaxis] == d
    return y



def read_seq_new_pro(seq_file,label):
    print("Loading Class {} data".format(label))
    seq_list = []
    seq = ''
    MaxNumberofSeqInClass = 20000
    q = 0
    with open(seq_file, 'r') as fp:
        numline = 1
        for line in fp:
            if q < MaxNumberofSeqInClass:
                if label == 6 and numline == 40:
                    if len(seq):
                        seq_list.append(seq)
                        numline= 1
                    q = q + 1

                if line[0] == '>':
                    if len(seq):
                        seq_list.append(seq)
    #                     print(len(seq))
                    seq = ''
                    q = q + 1
                else:
                    seq = seq + line[:]
                    numline+=1

        if len(seq):
            seq_list.append(seq)
    seq_data=[]
    #seq_feat=[]
    all_data = []

    MaxSeqLen = 3000

    
    for s in seq_list:
        # print('  ',len(s))
        if len(s) > MaxSeqLen:
            tmp=s[0:MaxSeqLen]
            d=get_onehot_seq_pro(tmp)
            # seq_data.append(d)
            all_data.append([d,label])
    # return seq_data, all_data

    print("Number of Sequences in Class : {} ".format(len(all_data)))
    return all_data

           


def get_onehot_seq(seq):
    chr_str = seq.upper()
    d = np.array(['A','C','G','T'])
    chr_str = np.array(chr_str, dtype = '<U')
    y = np.frombuffer(chr_str, dtype='|U1')[:, np.newaxis] == d
    return y


def read_seq_new(seq_file,label):
    print("Loading Class {} data".format(label))
    seq_list = []
    seq = ''
    MaxNumberofSeqInClass = 1500
    q = 0
    with open(seq_file, 'r') as fp:
        numline = 1
        for line in fp:
            if q < MaxNumberofSeqInClass:
                if label == 6 and numline == 40:
                    if len(seq):
                        seq_list.append(seq)
                        numline= 1
                    q = q + 1

                if line[0] == '>':
                    if len(seq):
                        seq_list.append(seq)
    #                     print(len(seq))
                    seq = ''
                    q = q + 1
                else:
                    seq = seq + line[:]
                    numline+=1

        if len(seq):
            seq_list.append(seq)
    seq_data=[]
    #seq_feat=[]
    all_data = []

    MaxSeqLen = 3000

    print("Number of Sequences in Class : {} ".format(len(seq_list)))
    for s in seq_list:
        # print('  ',len(s))
        if len(s) > MaxSeqLen:
            tmp=s[0:MaxSeqLen]
            d=get_onehot_seq(tmp)
            # seq_data.append(d)
            all_data.append([d,label])
    # return seq_data, all_data
    return all_data


def plot_ROC(y_test,y_scores,  classNames, fileNamePLT):
    # Compute ROC curve and ROC area for each class
    n_classes = len(classNames)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='Average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                    ''.format(classNames[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.title('Receiver operating characteristic of COVID-19 Variants Classification')
    plt.legend(loc="lower right")
    fileNamePLT = fileNamePLT+ "_ROC.png"
    plt.savefig(fileNamePLT)


def plot_Acc_Loss(history,fileNamePLT):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Train', 'Val'], loc='upper left')
    fileNamePLT = fileNamePLT+ "_Accuracy.png"
    plt.savefig(fileNamePLT)
    plt.clf()


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Train', 'Val'], loc='upper left')
    fileNamePLT = fileNamePLT+ "_Loss.png"
    plt.savefig(fileNamePLT)

def plot_confusion_matrix(y_test,y_scores, classNames, fileNamePLT):
    y_test=np.argmax(y_test, axis=1)
    y_scores=np.argmax(y_scores, axis=1)
    classes = len(classNames)
    cm = confusion_matrix(y_test, y_scores)
    print("**** Confusion Matrix ****")
    print(cm)
    print("**** Classification Report ****")
    print(classification_report(y_test, y_scores, target_names=classNames))
    con = np.zeros((classes,classes))
    for x in range(classes):
        for y in range(classes):
            con[x,y] = cm[x,y]/np.sum(cm[x,:])

    plt.figure(figsize=(40,40))
    sns.set(font_scale=3.0) # for label size
    df = sns.heatmap(con, annot=True,fmt='.2', cmap='Blues',xticklabels= classNames , yticklabels= classNames)
    fileNamePLT = fileNamePLT+ "_ConfusionMatrix.png"
    df.figure.savefig(fileNamePLT)



def plot_Gen_Loss(history,fileNamePLT):
    # plt.plot(history.history['loss'])
    plt.plot(history.history['construction_loss'][1:]/np.max(history.history['construction_loss'][1:]))
    plt.plot(history.history['kl_loss'][1:]/np.max(history.history['kl_loss'][1:])) 
    plt.title('Variant Generator Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Variant Construction Loss', 'KL-D Loss'], loc='upper right')
    fileNamePLT = fileNamePLT+ "_Loss.png"
    plt.savefig(fileNamePLT)

    

