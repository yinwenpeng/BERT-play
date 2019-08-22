import codecs
from collections import defaultdict
import numpy as np
import statistics
yahoo_path = '/home/wyin3/Datasets/YahooClassification/yahoo_answers_csv/'

'''
Society & Culture
Science & Mathematics
Health
Education & Reference
Computers & Internet
Sports
Business & Finance
Entertainment & Music
Family & Relationships
Politics & Government
'''
type2hypothesis = {
0: ['it is related with society or culture', 'this text  describes something about an extended social group having a distinctive cultural and economic organization or a particular society at a particular time and place'],
1:['it is related with science or mathematics', 'this text  describes something about a particular branch of scientific knowledge or a science (or group of related sciences) dealing with the logic of quantity and shape and arrangement'],
2: ['it is related with health', 'this text  describes something about a healthy state of wellbeing free from disease'],
3: ['it is related with education or reference', 'this text  describes something about the activities of educating or instructing or activities that impart knowledge or skill or an indicator that orients you generally'],
4: ['it is related with computers or Internet', 'this text  describes something about a machine for performing calculations automatically or a computer network consisting of a worldwide network of computer networks that use the TCP/IP network protocols to facilitate data transmission and exchange'],
5: ['it is related with sports', 'this text  describes something about an active diversion requiring physical exertion and competition'],
6: ['it is related with business or finance', 'this text  describes something about a commercial or industrial enterprise and the people who constitute it or the commercial activity of providing funds and capital'],
7: ['it is related with entertainment or music', 'this text  describes something about an activity that is diverting and that holds the attention or an artistic form of auditory communication incorporating instrumental or vocal tones in a structured and continuous manner'],
8: ['it is related with family or relationships', 'this text  describes something about a social unit living together, primary social group; parents and children or a relation between people'],
9: ['it is related with politics or government', 'this text  describes something about social relations involving intrigue to gain authority or power or the organization that is the governing authority of a political unit']}
# society, culture, science, mathematics, health, education, reference, computers, Internet, sports, business, finance, entertainment, music, family, relationships, politics, government

# type2hypothesis = {
# 0: ['this text describes something about society or culture, not about science, mathematics, health, education, reference, computers, Internet, sports, business, finance, entertainment, music, family, relationships, politics, government'],
# 1:['this text describes something about science or mathematics, not about society or culture, not about health, education, reference, computers, Internet, sports, business, finance, entertainment, music, family, relationships, politics, government'],
# 2: ['this text describes something about health, not about society, culture, science, mathematics, education, reference, computers, Internet, sports, business, finance, entertainment, music, family, relationships, politics, government'],
# 3: ['this text describes something about education or reference, not about society, culture, science, mathematics, education, reference, computers, Internet, sports, business, finance, entertainment, music, family, relationships, politics, government'],
# 4: ['this text describes something about computers or Internet, not about society, culture, science, mathematics, health, education, reference, sports, business, finance, entertainment, music, family, relationships, politics, government'],
# 5: ['this text describes something about sports, not about society, culture, science, mathematics, health, education, reference, computers, Internet, business, finance, entertainment, music, family, relationships, politics, government'],
# 6: ['this text describes something about business or finance, not about society, culture, science, mathematics, health, education, reference, computers, Internet, sports, entertainment, music, family, relationships, politics, government'],
# 7: ['this text describes something about entertainment or music, not about society, culture, science, mathematics, health, education, reference, computers, Internet, sports, business, finance, family, relationships, politics, government'],
# 8: ['this text describes something about family or relationships, not about society, culture, science, mathematics, health, education, reference, computers, Internet, sports, business, finance, entertainment, music, politics, government'],
# 9: ['this text describes something about politics or government, not about society, culture, science, mathematics, health, education, reference, computers, Internet, sports, business, finance, entertainment, music, family, relationships']}

def load_labels(word2id, maxlen):

    texts=[]
    text_masks=[]

    readfile=codecs.open(yahoo_path+'classes.txt', 'r', 'utf-8')
    for line in readfile:
        wordlist = line.strip().replace('&', ' ').lower().split()

        text_idlist, text_masklist=transfer_wordlist_2_idlist_with_maxlen(wordlist, word2id, maxlen)
        texts.append(text_idlist)
        text_masks.append(text_masklist)

    print('\t\t\t totally :', len(texts), 'label names')

    return texts, text_masks, word2id


def convert_yahoo_train_zeroshot():
    train_type_set  = set([0,2,4,6,8])
    # train_type_set  = set([1,3,5,7,9])
    id2size = defaultdict(int)

    readfile=codecs.open(yahoo_path+'train_tokenized.txt', 'r', 'utf-8')
    writefile = codecs.open(yahoo_path+'zero-shot-split/train.two.phases.txt', 'w', 'utf-8')
    line_co=0
    for line in readfile:
        parts = line.strip().split('\t')
        if len(parts)==2:
            label_id = int(parts[0])
            if label_id in train_type_set:
                id2size[label_id]+=1
                sent1 = parts[1].strip()
                # start write hypo
                idd=0
                while idd < 10:
                    '''only consider pos and neg in seen labels'''
                    if idd in train_type_set:
                        hypo_list = type2hypothesis.get(idd)
                        for hypo in hypo_list:
                            if idd == label_id:
                                writefile.write('1\t'+sent1+'\t'+hypo+'\t'+str(line_co)+':'+str(idd)+'\n')
                            else:
                                writefile.write('0\t'+sent1+'\t'+hypo+'\t'+str(line_co)+':'+str(idd)+'\n')
                    idd +=1
                line_co+=1
                if line_co%10000==0:
                    print('line_co:', line_co)
    print('dataset loaded over, id2size:', id2size, 'total read lines:',line_co )

def convert_yahoo_test_zeroshot():
    id2size = defaultdict(int)

    readfile=codecs.open(yahoo_path+'test_tokenized.txt', 'r', 'utf-8')
    writefile = codecs.open(yahoo_path+'zero-shot-split/test.two.phases.txt', 'w', 'utf-8')
    line_co=0
    for line in readfile:
        parts = line.strip().split('\t')
        if len(parts)==2:
            label_id = int(parts[0])
            id2size[label_id]+=1
            sent1 = parts[1].strip()
            # start write hypo
            idd=0
            while idd < 10:
                hypo_list = type2hypothesis.get(idd)
                for hypo in hypo_list:
                    if idd == label_id:
                        writefile.write('1\t'+sent1+'\t'+hypo+'\t'+str(line_co)+':'+str(idd)+'\n')
                    else:
                        writefile.write('0\t'+sent1+'\t'+hypo+'\t'+str(line_co)+':'+str(idd)+'\n')
                idd +=1
            line_co+=1
            if line_co%10000==0:
                print('line_co:', line_co)
    print('dataset loaded over, id2size:', id2size, 'total read lines:',line_co )

def evaluate_Yahoo_zeroshot(preds, gold_label_list, coord_list, seen_col_set):
    '''
    preds: probability vector
    '''
    pred_list = list(preds)
    assert len(pred_list) == len(gold_label_list)
    seen_hit=0
    unseen_hit = 0
    seen_size = 0
    unseen_size = 0


    start = 0
    end = 0
    total_sizes = [0.0]*10
    hit_sizes = [0.0]*10
    while end< len(coord_list):
        # print('end:', end)
        # print('start:', start)
        # print('len(coord_list):', len(coord_list))
        while end< len(coord_list) and int(coord_list[end].split(':')[0]) == int(coord_list[start].split(':')[0]):
            end+=1
        preds_row = pred_list[start:end]
        # print('preds_row:',preds_row)
        gold_label_row  = gold_label_list[start:end]
        # print('gold_label_row:',gold_label_row)
        # print(start,end)
        # assert sum(gold_label_row) >= 1
        coord_list_row = [int(x.split(':')[1]) for x in coord_list[start:end]]
        # print('coord_list_row:',coord_list_row)
        # print(start,end)
        # assert coord_list_row == [0,0,1,2,3,4,5,6,7,8,9]
        '''max_pred_id = np.argmax(np.asarray(preds_row)) is wrong, since argmax can be >=10'''
        max_pred_id = np.argmax(np.asarray(preds_row))
        pred_label_id = coord_list_row[max_pred_id]
        gold_label = -1
        for idd, gold in enumerate(gold_label_row):
            if gold == 1:
                gold_label = coord_list_row[idd]
                break
        # assert gold_label!=-1
        if gold_label == -1:
            if end == len(coord_list):
                break
            else:
                print('gold_label_row:',gold_label_row)
                exit(0)

        total_sizes[gold_label]+=1
        if gold_label == pred_label_id:
            hit_sizes[gold_label]+=1

        start = end

    # seen_acc = statistics.mean([hit_sizes[i]/total_sizes[i] for i in range(10) if i in seen_col_set])
    seen_hit = sum([hit_sizes[i] for i in range(10) if i in seen_col_set])
    seen_total = sum([total_sizes[i] for i in range(10) if i in seen_col_set])
    unseen_hit = sum([hit_sizes[i] for i in range(10) if i not in seen_col_set])
    unseen_total = sum([total_sizes[i] for i in range(10) if i not in seen_col_set])
    # unseen_acc = statistics.mean([hit_sizes[i]/total_sizes[i] for i in range(10) if i not in seen_col_set])
    print('acc for each label:', [hit_sizes[i]/total_sizes[i] for i in range(10)])
    print('total_sizes:',total_sizes)
    print('hit_sizes:',hit_sizes)

    return seen_hit/(1e-6+seen_total), unseen_hit/(1e-6+unseen_total)


def evaluate_Yahoo_zeroshot_2phases(pred_probs, pred_labels, gold_label_list, coord_list, seen_col_set):
    '''
    pred_probs: probability vector
    pred_labels: a list of 0/1
    '''
    pred_list = list(pred_probs)
    pred_labels = list(pred_labels)
    assert len(pred_list) == len(gold_label_list)
    seen_hit=0
    unseen_hit = 0
    seen_size = 0
    unseen_size = 0


    start = 0
    end = 0
    total_sizes = [0.0]*10
    hit_sizes = [0.0]*10
    while end< len(coord_list):
        # print('end:', end)
        # print('start:', start)
        # print('len(coord_list):', len(coord_list))
        while end< len(coord_list) and int(coord_list[end].split(':')[0]) == int(coord_list[start].split(':')[0]):
            end+=1
        pred_probs_row = pred_list[start:end]
        pred_label_row = pred_labels[start:end]

        gold_label_row  = gold_label_list[start:end]
        # print('gold_label_row:',gold_label_row)
        # print(start,end)
        # assert sum(gold_label_row) >= 1
        coord_list_row = [int(x.split(':')[1]) for x in coord_list[start:end]]
        # print('coord_list_row:',coord_list_row)
        # print(start,end)
        # assert coord_list_row == [0,0,1,2,3,4,5,6,7,8,9]
        '''max_pred_id = np.argmax(np.asarray(pred_probs_row)) is wrong, since argmax can be >=10'''
        '''pred label -- finalize'''
        pred_label = -1
        unseen_col_with_max_prob = -1
        max_prob = -10.0
        for idd, col in enumerate(coord_list_row):
            if col in seen_col_set and pred_label_row[idd] == 0: # 0 is entailment
                pred_label = col
            elif col not in seen_col_set: # unseen class
                if pred_probs_row[idd] > max_prob:
                    max_prob = pred_probs_row[idd]
                    unseen_col_with_max_prob = col
        pred_label = unseen_col_with_max_prob if pred_label==-1 else pred_label


        # max_pred_id = np.argmax(np.asarray(pred_probs_row))
        # pred_label_id = coord_list_row[max_pred_id]
        '''gold label'''
        gold_label = -1
        for idd, gold in enumerate(gold_label_row):
            if gold == 1:
                gold_label = coord_list_row[idd]
                break
        # assert gold_label!=-1
        if gold_label == -1:
            if end == len(coord_list):
                break
            else:
                print('gold_label_row:',gold_label_row)
                exit(0)
        print('pred_probs_row:',pred_probs_row)
        print('pred_label_row:',pred_label_row)
        print('gold_label_row:',gold_label_row)
        print('coord_list_row:',coord_list_row)
        print('gold_label:',gold_label)
        print('pred_label:',pred_label)

        total_sizes[gold_label]+=1
        if gold_label == pred_label:
            hit_sizes[gold_label]+=1

        start = end

    # seen_acc = statistics.mean([hit_sizes[i]/total_sizes[i] for i in range(10) if i in seen_col_set])
    seen_hit = sum([hit_sizes[i] for i in range(10) if i in seen_col_set])
    seen_total = sum([total_sizes[i] for i in range(10) if i in seen_col_set])
    unseen_hit = sum([hit_sizes[i] for i in range(10) if i not in seen_col_set])
    unseen_total = sum([total_sizes[i] for i in range(10) if i not in seen_col_set])
    # unseen_acc = statistics.mean([hit_sizes[i]/total_sizes[i] for i in range(10) if i not in seen_col_set])
    print('acc for each label:', [hit_sizes[i]/total_sizes[i] for i in range(10)])
    print('total_sizes:',total_sizes)
    print('hit_sizes:',hit_sizes)

    return seen_hit/(1e-6+seen_total), unseen_hit/(1e-6+unseen_total)

if __name__ == '__main__':
    convert_yahoo_train_zeroshot()
    # convert_yahoo_test_zeroshot()
