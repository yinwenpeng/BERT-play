import codecs
from collections import defaultdict
import numpy as np
import statistics
yahoo_path = '/home/wyin3/Datasets/agnews/ag_news_csv/'

'''
World
Sports
Business
Sci/Tech


'''
type2hypothesis = {
0:['it is related with world', 'this text  describes something about everything that exists anywhere'],
1: ['it is related with sports', 'this text  describes something about an active diversion requiring physical exertion and competition'],
2: ['it is related with business or finance', 'this text  describes something about a commercial or industrial enterprise and the people who constitute it or the commercial activity of providing funds and capital'],
3: ['it is related with science or technology', "this text  describes something about a particular branch of scientific knowledge or the application of the knowledge and usage of tools (such as machines or utensils) and techniques to control one's environment"]}


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


def convert_AGnews_train_zeroshot_keep_full_labels():
    train_type_set  = set([0,1,2,3])
    # train_type_set  = set([1,3,5,7,9])
    id2size = defaultdict(int)

    readfile=codecs.open(yahoo_path+'train.csv', 'r', 'utf-8')
    writefile = codecs.open(yahoo_path+'zero-shot-split/train.full.txt', 'w', 'utf-8')
    line_co=0
    for line in readfile:
        parts = line.strip().split('","')
        if len(parts)==3:
            label_id = int(parts[0][-1])-1
            if label_id in train_type_set:
                id2size[label_id]+=1
                sent1 = parts[1].strip()+' '+parts[2].strip()
                # start write hypo
                idd=0
                while idd < 4:
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
        else:
            print(line)
            print(parts)
            exit(0)
    print('dataset loaded over, id2size:', id2size, 'total read lines:',line_co )


def convert_AGnews_train_zeroshot():
    '''so make the 0th label as unseen here'''
    train_type_set  = set([0,1,2])
    # train_type_set  = set([1,3,5,7,9])
    id2size = defaultdict(int)

    readfile=codecs.open(yahoo_path+'train.csv', 'r', 'utf-8')
    writefile = codecs.open(yahoo_path+'zero-shot-split/train.wo.3.fun.txt', 'w', 'utf-8')
    line_co=0
    for line in readfile:
        parts = line.strip().split('","')
        if len(parts)==3:
            label_id = int(parts[0][-1])-1
            if label_id in train_type_set:
                id2size[label_id]+=1
                sent1 = parts[1].strip()+' '+parts[2].strip()
                # start write hypo
                idd=0
                while idd < 4:
                    '''only create positive and negative in seen labels'''
                    if idd in train_type_set:
                        hypo_list = type2hypothesis.get(idd)
                        for hypo in hypo_list:
                            if idd == label_id:
                                writefile.write('1\t'+sent1+'\t'+hypo+'\t'+str(line_co)+':'+str(idd)+'\n')
                            else:
                                writefile.write('0\t'+sent1+'\t'+hypo+'\t'+str(line_co)+':'+str(idd)+'\n')

                    else:
                        '''create positive and negative in unseen labels'''
                        hypo_list = type2hypothesis.get(idd)
                        for hypo in hypo_list:
                            '''for unseen label, make positive and negative pairs'''
                            if np.random.uniform() <0.25:
                                writefile.write('1\t'+sent1+'\t'+hypo+'\t'+str(line_co)+':'+str(idd)+'\n')
                            if np.random.uniform() >0.25:
                                writefile.write('0\t'+sent1+'\t'+hypo+'\t'+str(line_co)+':'+str(idd)+'\n')
                    idd +=1
                line_co+=1
                if line_co%10000==0:
                    print('line_co:', line_co)
        else:
            print(line)
            print(parts)
            exit(0)
    print('dataset loaded over, id2size:', id2size, 'total read lines:',line_co )

def convert_AGnews_test_zeroshot():
    id2size = defaultdict(int)

    readfile=codecs.open(yahoo_path+'test.csv', 'r', 'utf-8')
    writefile = codecs.open(yahoo_path+'zero-shot-split/test.v1.txt', 'w', 'utf-8')
    line_co=0
    for line in readfile:
        parts = line.strip().split('","')
        if len(parts)==3:
            '''label_id is 1,2,3,4'''
            label_id = int(parts[0][-1]) -1
            id2size[label_id]+=1
            sent1 = parts[1].strip()+' '+parts[2].strip()
            # start write hypo
            idd=0
            while idd < 4:
                hypo_list = type2hypothesis.get(idd)
                for hypo in hypo_list:
                    if idd == label_id:
                        writefile.write('1\t'+sent1+'\t'+hypo+'\t'+str(line_co)+':'+str(idd)+'\n')
                    else:
                        writefile.write('0\t'+sent1+'\t'+hypo+'\t'+str(line_co)+':'+str(idd)+'\n')
                idd +=1
            line_co+=1
            if line_co%1000==0:
                print('line_co:', line_co)
        else:
            print(line+'\n')
            print(parts)
            exit(0)
    print('dataset loaded over, id2size:', id2size, 'total read lines:',line_co )

def evaluate_AGnews_zeroshot(preds, gold_label_list, coord_list, seen_col_set):
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
    total_sizes = [0.0]*4
    hit_sizes = [0.0]*4
    while end< len(coord_list):
        # print('end:', end)
        # print('start:', start)
        # print('len(coord_list):', len(coord_list))
        while end< len(coord_list) and int(coord_list[end].split(':')[0]) == int(coord_list[start].split(':')[0]):
            end+=1
        preds_row = pred_list[start:end]
        # print('preds_row:',preds_row)
        gold_label_row  = gold_label_list[start:end]
        '''we need -1 because the col label are 1,2,3,4'''
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
    seen_hit = sum([hit_sizes[i] for i in range(4) if i in seen_col_set])
    seen_total = sum([total_sizes[i] for i in range(4) if i in seen_col_set])
    unseen_hit = sum([hit_sizes[i] for i in range(4) if i not in seen_col_set])
    unseen_total = sum([total_sizes[i] for i in range(4) if i not in seen_col_set])
    # unseen_acc = statistics.mean([hit_sizes[i]/total_sizes[i] for i in range(10) if i not in seen_col_set])
    print('acc for each label:', [hit_sizes[i]/total_sizes[i] for i in range(4)])
    print('total_sizes:',total_sizes)
    print('hit_sizes:',hit_sizes)

    return seen_hit/(1e-6+seen_total), unseen_hit/unseen_total

if __name__ == '__main__':
    # convert_AGnews_train_zeroshot_keep_full_labels()
    convert_AGnews_train_zeroshot()
    # convert_AGnews_test_zeroshot()
