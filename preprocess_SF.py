# 1: food
# 2: infra
# 3: med
# 4: search
# 5: shelter
# 6: utils
# 7: water
# 8:crimeviolence
# 9: out-of-domain
import codecs
import numpy as np
from random import *
from collections import defaultdict

# type2hypothesis = {
# 'food': ['people there need food', 'people are in the shortage of food'],
# 'infra':['the infrastructures are destroyed, we need build new'],
# 'med': ['people need medical assistance'],
# 'search': ['some people are missing or buried, we need to provide search and rescue'],
# 'shelter': ['Many houses collapsed and people are in desperate need of new places to live.'],
# 'utils': ['The water supply and power supply system is broken, and the basic living supply is urgently needed.'],
# 'water': ['people are in the shortage of water', 'Clean drinking water is urgently needed'],
# 'crimeviolence': ['There was violent criminal activity in that place.'],
# 'terrorism': ['There was a terrorist activity in that place, such as an explosion, shooting'],
# 'evac': ['This place is very dangerous and it is urgent to evacuate people to safety.'],
# 'regimechange': ['Regime change happened in this country']}

type2hypothesis = {
'food': ['people there need food', 'people there need any substance that can be metabolized by an animal to give energy and build tissue'],
'infra':['people there need infrastructures', 'people there need the basic structure or features of a system or organization'],
'med': ['people need medical assistance', 'people need an allied health professional who supports the work of physicians and other health professionals'],
'search': ['people there need search', 'people there need the activity of looking thoroughly in order to find something or someone'],
'shelter': ['people there need shelter', 'people there need a structure that provides privacy and protection from danger'],
'utils': ['people there need utilities', 'people there need the service (electric power or water or transportation) provided by a public utility'],
'water': ['people there need water', 'Clean drinking water is urgently needed'],
'crimeviolence': ['crime violence happened there', 'an act punishable by law; usually considered an evil act happened there'],
'terrorism': ['There was a terrorist activity in that place, such as an explosion, shooting'],
'evac': ['This place is very dangerous and it is urgent to evacuate people to safety.'],
'regimechange': ['Regime change happened in this country']}

def statistics_BBN_SF_data():
    '''
    label2co: [('regimechange', 16), ('terrorism', 49), ('search', 85), ('evac', 101), ('infra', 240), ('crimeviolence', 293), ('utils', 308), ('water', 348), ('shelter', 398), ('med', 421), ('food', 448), ('out-of-domain', 1976)]
    '''
    label2co = defaultdict(int)
    BBNread = codecs.open('/save/wenpeng/datasets/LORELEI/SF-BBN-Mark-split/full_BBN_multi.txt', 'r', 'utf-8')

    labelset = set(['search','evac','infra','utils','water','shelter','med','food','out-of-domain'])
    valid_size = 0
    for line in BBNread:
        valid=False
        parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
        if len(parts)==3:
            label_strset = set(parts[1].strip().split())
            for label in label_strset:
                label2co[label]+=1
                if label in labelset:
                    valid=True
        if valid:
            valid_size+=1


    BBNread.close()
    print('label2co:', sorted(label2co.items(), key = lambda kv:(kv[1], kv[0])))
    print('valid_size:',valid_size)


def reformat_BBN_SF_data():
    '''this function just split the BBN data into train and test parts, not related with unseen or seen'''
    type_list = ['food', 'infra', 'med', 'search', 'shelter', 'utils', 'water', 'crimeviolence', 'terrorism','evac','regimechange','out-of-domain']
    type_set = set(type_list)
    BBNread = codecs.open('/save/wenpeng/datasets/LORELEI/SF-BBN-Mark-split/full_BBN_multi.txt', 'r', 'utf-8')
    # writefile = codecs.open('/home/wyin3/Datasets/MNLI-SNLI-SciTail-RTE-SICK/BBN.SF.txt', 'w', 'utf-8')
    trainfile = codecs.open('/home/wyin3/Datasets/MNLI-SNLI-SciTail-RTE-SICK/BBN.SF.train.txt', 'w', 'utf-8')
    testfile = codecs.open('/home/wyin3/Datasets/MNLI-SNLI-SciTail-RTE-SICK/BBN.SF.test.txt', 'w', 'utf-8')
    size = 0
    size_split = [0,0]
    OOD_size = 0
    out_of_domain_size = 0
    for line in BBNread:
        parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
        if len(parts)==3:
            probility = random()
            if probility < (2/3):
                writefile = trainfile
                writesize_index = 0
            else:
                writefile = testfile
                writesize_index = 1

            sent1=parts[2].strip()
            label_strset = set(parts[1].strip().split())
            if len(label_strset - type_set) >0:
                print(label_strset)
                exit(0)

            if len(label_strset) == 1 and list(label_strset)[0] == type_list[-1]:
                OOD_size+=1
                if out_of_domain_size < 5000000:
                    for idd, label in enumerate(type_list[:-1]):
                        hypo_list = type2hypothesis.get(label)
                        for hypo in hypo_list:
                            writefile.write('0\t'+sent1+'\t'+hypo+'\t'+str(size_split[writesize_index])+':'+str(idd)+'\n')
                    out_of_domain_size+=1
                    size_split[writesize_index]+=1

            else:
                for idd, label in enumerate(type_list[:-1]):
                    hypo_list = type2hypothesis.get(label)
                    for hypo in hypo_list:
                        if label in label_strset:
                            writefile.write('1\t'+sent1+'\t'+hypo+'\t'+str(size_split[writesize_index])+':'+str(idd)+'\n')
                        else:
                            writefile.write('0\t'+sent1+'\t'+hypo+'\t'+str(size_split[writesize_index])+':'+str(idd)+'\n')
                size_split[writesize_index]+=1
            size+=1

    BBNread.close()
    # writefile.close()
    trainfile.close()
    testfile.close()
    print('BBN SF data reformat over, size:', size, '...OOD_size:', OOD_size, '..OOD written size:', out_of_domain_size)


def reformat_full_BBN_SF_2_entai_data():
    '''this function convert the whole BBN data into (premise, hypothesis) with 1/0'''
    type_list = ['food', 'infra', 'med', 'search', 'shelter', 'utils', 'water', 'crimeviolence', 'terrorism','evac','regimechange','out-of-domain']
    type_set = set(type_list)
    BBNread = codecs.open('/save/wenpeng/datasets/LORELEI/SF-BBN-Mark-split/full_BBN_multi.txt', 'r', 'utf-8')
    # writefile = codecs.open('/home/wyin3/Datasets/MNLI-SNLI-SciTail-RTE-SICK/BBN.SF.txt', 'w', 'utf-8')
    writefile = codecs.open('/home/wyin3/Datasets/MNLI-SNLI-SciTail-RTE-SICK/BBN.SF.whole2entail.txt', 'w', 'utf-8')
    # testfile = codecs.open('/home/wyin3/Datasets/MNLI-SNLI-SciTail-RTE-SICK/BBN.SF.test.txt', 'w', 'utf-8')
    size = 0
    OOD_size = 0
    out_of_domain_size = 0
    for line in BBNread:
        parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
        if len(parts)==3:
            sent1=parts[2].strip()
            label_strset = set(parts[1].strip().split())
            if len(label_strset - type_set) >0:
                print(label_strset)
                exit(0)

            if len(label_strset) == 1 and list(label_strset)[0] == type_list[-1]:
                '''out-of-domain'''
                OOD_size+=1
                if out_of_domain_size < 5000000:
                    for idd, label in enumerate(type_list[:-1]):
                        hypo_list = type2hypothesis.get(label)
                        for hypo in hypo_list:
                            writefile.write('0\t'+sent1+'\t'+hypo+'\t'+str(size)+':'+str(idd)+'\n')
                    out_of_domain_size+=1

            else:
                for idd, label in enumerate(type_list[:-1]):
                    hypo_list = type2hypothesis.get(label)
                    for hypo in hypo_list:
                        if label in label_strset:
                            writefile.write('1\t'+sent1+'\t'+hypo+'\t'+str(size)+':'+str(idd)+'\n')
                        else:
                            writefile.write('0\t'+sent1+'\t'+hypo+'\t'+str(size)+':'+str(idd)+'\n')
            size+=1

    BBNread.close()
    # writefile.close()
    writefile.close()
    print('BBN SF data reformat over, size:', size, '...OOD_size:', OOD_size, '..OOD written size:', out_of_domain_size)


def reformat_BBN_SF_2_ZeroShot_data():
    '''
    label2co: [('regimechange', 16), ('terrorism', 49), ('search', 85), ('evac', 101), ('infra', 240),
    ('crimeviolence', 293), ('utils', 308), ('water', 348), ('shelter', 398), ('med', 421),
    ('food', 448), ('out-of-domain', 1976)]
    '''
    # type_list = ['food', 'med', 'shelter', 'water', 'utils', 'crimeviolence', 'infra','evac','search','terrorism','regimechange','out-of-domain']
    # train_type_set = set(type_list[:5])
    type_list = ['water', 'utils', 'crimeviolence', 'infra','evac','food', 'med', 'shelter', 'search','terrorism','regimechange','out-of-domain']
    train_type_set = set(type_list[:5])
    type_set = set(type_list)
    # type2hypothesis = {
    # 'food': ['people there need food', 'people are in the shortage of food'],
    # 'infra':['the infrastructures are destroyed, we need build new'],
    # 'med': ['people need medical assistance'],
    # 'search': ['some people are missing or buried, we need to provide search and rescue'],
    # 'shelter': ['Many houses collapsed and people are in desperate need of new places to live.'],
    # 'utils': ['The water supply and power supply system is broken, and the basic living supply is urgently needed.'],
    # 'water': ['people are in the shortage of water', 'Clean drinking water is urgently needed'],
    # 'crimeviolence': ['There was violent criminal activity in that place.'],
    # 'terrorism': ['There was a terrorist activity in that place, such as an explosion, shooting'],
    # 'evac': ['This place is very dangerous and it is urgent to evacuate people to safety.'],
    # 'regimechange': ['Regime change happened in this country']}
    BBNread = codecs.open('/save/wenpeng/datasets/LORELEI/SF-BBN-Mark-split/full_BBN_multi.txt', 'r', 'utf-8')
    # writefile = codecs.open('/home/wyin3/Datasets/MNLI-SNLI-SciTail-RTE-SICK/BBN.SF.txt', 'w', 'utf-8')
    trainfile = codecs.open('/home/wyin3/Datasets/MNLI-SNLI-SciTail-RTE-SICK/BBN.SF.train.zeroshot.v2.txt', 'w', 'utf-8')
    testfile = codecs.open('/home/wyin3/Datasets/MNLI-SNLI-SciTail-RTE-SICK/BBN.SF.test.zeroshot.v2.txt', 'w', 'utf-8')
    def train_or_test_set(train_file, test_file):
        probility = random()
        if probility < (2/3):
            return trainfile, 0
        else:
            return testfile, 1
    size = 0
    size_split = [0,0]
    OOD_size = 0
    out_of_domain_size = 0
    for line in BBNread:
        parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
        if len(parts)==3:
            sent1=parts[2].strip()
            label_strset = set(parts[1].strip().split())
            if len(label_strset - type_set) >0:
                print(label_strset)
                exit(0)

            if len(label_strset) == 1 and list(label_strset)[0] == type_list[-1]:
                '''
                if is "out-of-domain"
                '''
                OOD_size+=1
                if out_of_domain_size < 5000000:
                    train_and_test = [False, False]
                    for idd, label in enumerate(type_list[:-1]):
                        hypo_list = type2hypothesis.get(label)
                        if label in train_type_set:
                            writefile, writesize_index = train_or_test_set(trainfile, testfile)
                        else:
                            writefile = testfile
                            writesize_index = 1
                        train_and_test[writesize_index] = True
                        for hypo in hypo_list:
                            writefile.write('0\t'+sent1+'\t'+hypo+'\t'+str(size_split[writesize_index])+':'+str(idd)+'\n')
                    out_of_domain_size+=1
                    if train_and_test[0] == True:
                        size_split[0]+=1
                    if train_and_test[1] == True:
                        size_split[1]+=1

            else:
                train_and_test = [False, False] # indicate if this row will be written into train and test
                for idd, label in enumerate(type_list[:-1]):
                    hypo_list = type2hypothesis.get(label)
                    if label in train_type_set:
                        writefile, writesize_index = train_or_test_set(trainfile, testfile)
                    else:
                        writefile = testfile
                        writesize_index = 1
                    train_and_test[writesize_index] = True
                    for hypo in hypo_list:
                        if label in label_strset:
                            writefile.write('1\t'+sent1+'\t'+hypo+'\t'+str(size_split[writesize_index])+':'+str(idd)+'\n')
                        else:
                            writefile.write('0\t'+sent1+'\t'+hypo+'\t'+str(size_split[writesize_index])+':'+str(idd)+'\n')
                if train_and_test[0] == True:
                    size_split[0]+=1
                if train_and_test[1] == True:
                    size_split[1]+=1
            size+=1

    BBNread.close()
    # writefile.close()
    trainfile.close()
    testfile.close()
    print('BBN SF data reformat over, size:', size, '...OOD_size:', OOD_size, '..OOD written size:', out_of_domain_size)



def f1_two_col_array(vec1, vec2):
    overlap = sum(vec1*vec2)
    pos1 = sum(vec1)
    pos2 = sum(vec2)
    recall = overlap*1.0/(1e-8+pos1)
    precision = overlap * 1.0/ (1e-8+pos2)
    return 2*recall*precision/(1e-8+recall+precision)

def average_f1_two_array_by_col(arr1, arr2):
    '''
    arr1: pred
    arr2: gold
    '''
    col_size = arr1.shape[1]
    f1_list = []
    class_size_list = []
    for i in range(col_size):
        f1_i = f1_two_col_array(arr1[:,i], arr2[:,i])
        class_size = sum(arr2[:,i])
        f1_list.append(f1_i)
        class_size_list.append(class_size)
    print('f1_list:',f1_list)
    print('class_size:', class_size_list)
    mean_f1 = sum(f1_list)/len(f1_list)
    weighted_f1 = sum([x*y for x,y in zip(f1_list,class_size_list)])/sum(class_size_list)
    # print 'mean_f1, weighted_f1:', mean_f1, weighted_f1
    # exit(0)
    return mean_f1, weighted_f1


def average_f1_two_array_by_col_zeroshot(arr1, arr2, k):
    '''
    arr1: pred
    arr2: gold
    '''
    col_size = arr1.shape[1]
    f1_list = []
    class_size_list = []
    for i in range(col_size):
        f1_i = f1_two_col_array(arr1[:,i], arr2[:,i])
        class_size = sum(arr2[:,i])
        f1_list.append(f1_i)
        class_size_list.append(class_size)
    print('f1_list:',f1_list)
    print('class_size:', class_size_list)
    mean_f1 = sum(f1_list)/len(f1_list)
    weighted_f1_first_k = sum([x*y for x,y in zip(f1_list[:k],class_size_list[:k])])/sum(class_size_list[:k])
    weighted_f1_remain = sum([x*y for x,y in zip(f1_list[k:-1],class_size_list[k:-1])])/sum(class_size_list[k:-1])
    # print 'mean_f1, weighted_f1:', mean_f1, weighted_f1
    # exit(0)
    return weighted_f1_first_k, weighted_f1_remain

def evaluate_SF(preds, gold_label_list, coord_list):
    pred_list = list(preds)
    assert len(pred_list) == len(gold_label_list)


    def create_binary_matrix(binary_list, position_list, gold_flag):
        example_size = 1+int(position_list[-1].split(':')[0])
        # print('example_size:',example_size)
        # print('binary_list:', binary_list[:200])
        matrix = np.zeros((example_size, 12), dtype=int)
        for idd, pair in enumerate(position_list):
            parts = pair.split(':')
            row = int(parts[0])
            col = int(parts[1])
            '''
            note that the 0^th label in BERT RTE is entailment
            but in our data, "1" means entailment
            '''
            if gold_flag:
                if binary_list[idd] == 1:
                    matrix[row, col]=1
            else:
                if binary_list[idd] == 0:
                    matrix[row, col]=1
        all_multiplication = np.sum(matrix, axis=1) == 0
        matrix[:,-1] = all_multiplication
        return matrix

    gold_matrix = create_binary_matrix(gold_label_list, coord_list, True)
    print('gold_matrix:', gold_matrix[:20])
    pred_matrix = create_binary_matrix(pred_list, coord_list, False)
    print('pred_matrix:', pred_matrix[:20])
    # exit(0)
    return average_f1_two_array_by_col(pred_matrix, gold_matrix)


def evaluate_SF_zeroshot(preds, gold_label_list, coord_list, k):
    pred_list = list(preds)
    assert len(pred_list) == len(gold_label_list)


    def create_binary_matrix(binary_list, position_list, gold_flag):
        example_size = 1+int(position_list[-1].split(':')[0])
        # print('example_size:',example_size)
        # print('binary_list:', binary_list[:200])
        matrix = np.zeros((example_size, 12), dtype=int)
        for idd, pair in enumerate(position_list):
            parts = pair.split(':')
            row = int(parts[0])
            col = int(parts[1])
            '''
            note that the 0^th label in BERT RTE is entailment
            but in our data, "1" means entailment
            '''
            if gold_flag:
                if binary_list[idd] == 1:
                    matrix[row, col]=1
            else:
                if binary_list[idd] == 0:
                    matrix[row, col]=1
        all_multiplication = np.sum(matrix, axis=1) == 0
        matrix[:,-1] = all_multiplication
        return matrix

    gold_matrix = create_binary_matrix(gold_label_list, coord_list, True)
    print('gold_matrix:', gold_matrix[:20])
    pred_matrix = create_binary_matrix(pred_list, coord_list, False)
    print('pred_matrix:', pred_matrix[:20])
    # exit(0)
    return average_f1_two_array_by_col_zeroshot(pred_matrix, gold_matrix, k)


if __name__ == '__main__':
    # reformat_BBN_SF_data()
    # reformat_BBN_SF_2_ZeroShot_data()
    statistics_BBN_SF_data()

    # reformat_full_BBN_SF_2_entai_data()
