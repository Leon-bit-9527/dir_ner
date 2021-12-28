from matplotlib import pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_ece(y):
    bins=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
    plt.bar(bins,y,width=0.097,color='#1F77B4',alpha=1,edgecolor="k",linewidth=0.7)#alpha设置透明度，0为完全透明
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    ident = [0.0, 1.0]
    plt.plot(ident,ident,color='0.5',ls="--",linewidth=1)
    plt.xlim(0,1)#设置x轴分布范围
    plt.ylim(0,1)#设置x轴分布范围
    plt.grid()  # 生成网格
    #     plt.savefig('out/stage1_results.jpg', dpi=50)

    plt.show()

def get_label_prob(filepath1,filepath2):
    raw_list = []
    a = []
    prob_list = []
    pred_list = []

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
                prob_list.append(a[2])

    return raw_list,pred_list,prob_list

def num_prob(filepath1,filepath2):    
    raw_list,pred_list,prob_list = get_label_prob(filepath1,filepath2)

    gold_p = []
    entity = 0
    count = [0]*10
    p_count=[0]*10
    gold_count = [0]*10
    num_list=len(raw_list)
    for i in range(num_list):
        if raw_list[i] != 'O':
            entity += 1
            prob_list[i] = float(prob_list[i])
            if prob_list[i]>0 and prob_list[i]<=0.1:
                gold_count[0] += 1
            elif prob_list[i]>0.1 and prob_list[i]<=0.2:
                gold_count[1] += 1
            elif prob_list[i]>0.2 and prob_list[i]<=0.3:
                gold_count[2] += 1        
            elif prob_list[i]>0.3 and prob_list[i]<=0.4:
                gold_count[3] += 1        
            elif prob_list[i]>0.4 and prob_list[i]<=0.5:
                gold_count[4] += 1
            elif prob_list[i]>0.5 and prob_list[i]<=0.6:
                gold_count[5] += 1
            elif prob_list[i]>0.6 and prob_list[i]<=0.7:
                gold_count[6] += 1        
            elif prob_list[i]>0.7 and prob_list[i]<=0.8:
                gold_count[7] += 1        
            elif prob_list[i]>0.8 and prob_list[i]<=0.9:
                gold_count[8] += 1
            elif prob_list[i]>0.9 and prob_list[i]<=1.0:
                gold_count[9] += 1

            if raw_list[i] == pred_list[i]:
                gold_p.append(prob_list[i])
                if prob_list[i]>0 and prob_list[i]<=0.1:
                    count[0] += 1
                elif prob_list[i]>0.1 and prob_list[i]<=0.2:
                    count[1] += 1
                elif prob_list[i]>0.2 and prob_list[i]<=0.3:
                    count[2] += 1        
                elif prob_list[i]>0.3 and prob_list[i]<=0.4:
                    count[3] += 1        
                elif prob_list[i]>0.4 and prob_list[i]<=0.5:
                    count[4] += 1
                elif prob_list[i]>0.5 and prob_list[i]<=0.6:
                    count[5] += 1
                elif prob_list[i]>0.6 and prob_list[i]<=0.7:
                    count[6] += 1        
                elif prob_list[i]>0.7 and prob_list[i]<=0.8:
                    count[7] += 1        
                elif prob_list[i]>0.8 and prob_list[i]<=0.9:
                    count[8] += 1
                elif prob_list[i]>0.9 and prob_list[i]<=1.0:
                    count[9] += 1
    for i in range(len(count)):
        p_count[i]=round((count[i]/(gold_count[i]+0.1)),2)
    print('num_entity:', entity/num_list)
    return gold_count,count,p_count,num_list

def ECE_comp(gold_count,count,p_count,num_list):
    ece=0
    for i in range(len(count)):
        ece += abs(p_count[i]-0.1*(i+1))*count[i]/num_list
    return ece

if __name__ == '__main__':
    gold_count,count,p_count,num_list = num_prob('./data/ontonote/test.txt','./outs/onto/results.txt') 
    plot_ece(p_count)
    print('gold_count:',gold_count,' \ncorrect_count:',count, '\np_count:',p_count)
    ECE = ECE_comp(gold_count,count,p_count,num_list)
    print('ECE:',ECE)