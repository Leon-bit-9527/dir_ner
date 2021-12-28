from matplotlib import pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def plot_confusion_matrix(y_true, y_pred):

    classes = list(set(y_true))  
    # 排序
    classes.sort()
    confusion = confusion_matrix(y_true,y_pred)
    indices = range(len(confusion))
    x_labels = ['uncertain','certain']
    y_labels = ['inaccurate','accurate']

    plt.imshow(confusion, cmap = plt.cm.Blues)
    plt.xticks(indices,x_labels)
    plt.yticks(indices,y_labels)
    plt.colorbar()

    #显示数据
    for first_index in range(len(confusion)):
        #print(len(confusion[first_index]))
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index,second_index,confusion[first_index][second_index])
    plt.show()

def get_auc(y_true, y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr,tpr)
    print('roc_auc值为：',roc_auc)
    print(threshold)
    plt.figure(figsize = [10,10])
    plt.plot(fpr, tpr, color='darkorange',lw=2,label='ROC curve(area=%0.2f)'%roc_auc)
    plt.plot([0,1],[0,1],color='navy', lw=2, linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()

def get_label_prob(filepath1,filepath2):
    raw_list = []
    a = []
    prob_list = []
    pred_list = []
    prob_soft = []

    with open(filepath1,'r+',encoding='utf-8') as f:
        for line in f.readlines(): 
            if len(line[:-1].split(' '))==4:
                a= line[:-1].split(' ')
                raw_list.append(a[3])


    with open(filepath2,'r+',encoding='utf-8') as f:
        for line in f.readlines(): 
             if len(line[:-1].split(' '))==4:
                a= line[:-1].split(' ')
                pred_list.append(a[3])
                prob_list.append(a[1])
                prob_soft.append(a[2])

    return raw_list,pred_list,prob_list,prob_soft

def get_count(filepath1,filepath2,threshold):
    raw_list,pred_list,prob_list,prob_soft = get_label_prob(filepath1,filepath2)
    num_list=len(raw_list)
    y_true = []
    y_score = []
    y_pred = []
    for i in range(num_list):
        if raw_list[i] != 'O':
            y_score.append(float(prob_soft[i]))
            if raw_list[i] == pred_list[i]:
                y_true.append(1)
                if float(prob_list[i]) <= threshold:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            else:
                y_true.append(0)
                if float(prob_list[i]) <= threshold:
                    y_pred.append(0)
                else:
                    y_pred.append(1)
    return y_true, y_pred, y_score

if __name__ == '__main__':
    y_true, y_pred, y_score = get_count('./data/ontonote/test.txt','./outs/onto/results.txt',threshold=0.1) 
    # plot_confusion_matrix(y_true, y_pred)
    get_auc(y_true, y_score)
