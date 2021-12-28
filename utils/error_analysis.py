def get_class_span_dict(filepath,label_type):
    class_span = {}
    instance_span = {}
    fined_span = {}
    current_label = None
    n = 0
    i = 0
    error = 0
    with open(filepath,'r+',encoding='utf-8') as f:
        label_span = []
        entity_span= []
        each =None
        m = -1
        for line in f.readlines(): 
            m += 1
            if each != None:
                if each in ['\n','\r\n']:
                    s.append('')
                else:
                    a = each[:-1].split(' ')
                    s = a[3].split('-')
                    if len(s) ==1 :
                        s.append('O')
                    label_span.append(s[0])
                    entity_span.append(s[1])
            each = line
            a = each[:-1].split(' ')
            
#         label_span.append(a[2])
#         entity_span.append(a[2])
    
    fined = entity_span
    while i < len(fined):
        if fined[i] != 'O' or fined[i] != '':
            begin = i+1
            current_fined = fined[i]
            i += 1
            while i < len(fined) and fined[i] == current_fined and fined[i] != 'O':
                i += 1
            if current_fined in fined_span:
                fined_span[current_fined].append(tuple([begin, i]))
            else:
                fined_span[current_fined] = [tuple([begin, i])]
        else:
            i += 1

    entity = entity_span
    label = label_span
    if label_type=="BIES":
        while n < len(label):
            if label[n] == 'O' or label[n] == '':
                n += 1
            else:
                if label[n] == 'S':
                    current_label = 'S'
                    current_instance = entity[n]
                    if current_instance in instance_span:
                        instance_span[current_instance].append(tuple([n,n]))
                    else:
                        instance_span[current_instance] = [tuple([n,n])]
                    if current_label in class_span:
                        class_span[current_label].append(tuple([n,n]))
                    else:
                        class_span[current_label] = [tuple([n,n])]
                    n += 1
                elif label[n] == 'B':
                    start = n+1
                    current_label = 'BIE'
                    current_instance = entity[n]
                    n += 1
                    while n < len(label) and label[n] == 'E' or label[n] == 'I':
                        n += 1
                    if current_instance in instance_span:
                        instance_span[current_instance].append(tuple([start,n]))
                    else:
                        instance_span[current_instance] = [tuple([start,n])]
                        
                    if current_label in class_span:
                        class_span[current_label].append(tuple([start, n]))
                    else:
                        class_span[current_label] = [tuple([start, n])]
                else:
                    error += 1
                    n += 1
    if label_type=="BIO":
        while n < len(label):
            if label[n] == 'O' or label[n] == '':
                n += 1
            else:
                if label[n] == 'B':
                    start = n+1
                    current_label = 'BI'
                    n += 1
                    while n < len(label) and label[n] == 'I':
                        n += 1
                    if current_label in class_span:
                        class_span[current_label].append(tuple([start, n]))
                    else:
                        class_span[current_label] = [tuple([start, n])]
                else:
                    n += 1
    return error,fined_span, instance_span, class_span

def metrics_by_entity(error,pred_span, class_span,instance_pre,instance_class,fined_pre,fined_class):
    pred_span_list = []
    label_span_list = []
    pre_instance_list = []
#     label_instance_list = []
    error_spanE = []
    error_spanE_list = []
    correct_spanE = []
    correct_spanE_list = []
    temp_pre = []
    temp_glod = []
    error_fined = []
    error_fined_list = []
    correct_fined = []
    correct_fined_list = []
    
    for pred in pred_span:
        pred_span_list += pred_span[pred]
    for label in class_span:
        label_span_list += class_span[label]
        
    for instance in instance_pre:
        temp_pre = instance_pre[instance]
        temp_glod = instance_class[instance]
        error_spanE.append(list(set(temp_pre).difference(set(temp_glod))))
        correct_spanE.append(list(set(temp_pre).intersection(set(temp_glod))))
        pre_instance_list += instance_class[instance] 
#         label_instance_list += instance_class[instance]
    for key in range (len(error_spanE)):
        error_spanE_list += error_spanE[key]
    for key in range (len(correct_spanE)):
        correct_spanE_list += correct_spanE[key]

    for fined in fined_pre:
        tempf_pre = fined_pre[fined]
        tempf_glod = fined_class[fined]
        error_fined.append(list(set(tempf_pre).difference(set(tempf_glod))))
        correct_fined.append(list(set(tempf_pre).intersection(set(tempf_glod))))
    for key2 in range (len(error_fined)):
        error_fined_list += error_fined[key2]
    for key2 in range (len(correct_fined)):
        correct_fined_list += correct_fined[key2]
        
    error_spanB = list(set(pred_span_list).difference(set(label_span_list))) ## boundaries × ，instance -
    correct_spanB = list(set(pred_span_list).intersection(set(label_span_list)))
#     both_error = list(set(error_spanB).union(set(error_spanE_list))) ## both boundaries × and instance ×, all
    classify_error = list(set(correct_spanB).intersection(set(error_spanE_list))) ##  实体错的里面边界错误的个数
    classify_correct = list(set(correct_spanB).intersection(set(correct_spanE_list)))
#     both_error = list(set(error_spanB).intersection(set(error_spanE_list)))
    span_error = list(set(error_spanB).intersection(set(error_spanE_list)))
    both_error = list((set(error_spanB).intersection(set(error_fined_list))))

#     print(both_error2)
    only_span_error = list(set(span_error).difference(set(both_error)))
    
#     print('correct_spanB:',(correct_spanB),'boundaries √ ,instance -')
#     print('error_spanB:',len(error_spanB),'boundaries × ,instance -')
#     print(len(correct_spanB))
#     print('error_spanE:',len(error_spanE_list),'boundaries -, instance ×')
#     print('both_error:',len(both_error),' boundaries ×, instance ×')
#     print(len(instance_correct))
    print('classify_correct:',len(classify_correct),'proportion:',len(classify_correct)/len(pre_instance_list),'实体预测正确的个数及比例')
    print('correct_spanB:',len(correct_spanB),'proportion:',len(correct_spanB)/len(label_span_list),'实体边界预测正确的个数及比例')
    print('error_spanE_list:',len(error_spanE_list),'proportion:',len(error_spanE_list)/len(pre_instance_list),'实体预测错误的个数及其比例')
    print('classify_error:',len(classify_error),'proportion:',len(classify_error)/len(error_spanE_list),'分类错误导致结果错误')
    print('only_span_error:',len(only_span_error),len(only_span_error)/len(error_spanE_list),'边界错误导致结果错误')
    print('both_error:',len(both_error)+error,(len(both_error)+error)/len(error_spanE_list),'边界错误且分类错误的个数')

    return both_error
if __name__ == '__main__':
    error,fined_pre, instance_pre, pred_span = get_class_span_dict(filepath='./outs/onto/results.txt',label_type="BIES")
    _,fined_class, instance_class, class_span = get_class_span_dict(filepath='./data/ontonote/test.txt',label_type="BIES")
    both_error = metrics_by_entity(error,pred_span,class_span,instance_pre,instance_class,fined_pre,fined_class)