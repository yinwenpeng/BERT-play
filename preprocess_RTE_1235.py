
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
import codecs
from collections import defaultdict

RTE1_label2id = {'TRUE':1, 'FALSE':0}
RTE2_label2id = {'YES':1, 'NO':0}
RTE3_label2id = {'YES':1, 'NO':0, 'UNKNOWN':0}
RTE5_label2id = {'ENTAILMENT':1, 'CONTRADICTION':0, 'UNKNOWN':0}
rootpath = '/home/wyin3/Datasets/RTE/'

def preprocess_RTE1(filename, writefile):
    # rootpath = '/save/wenpeng/datasets/RTE/'
    tree = ET.parse(rootpath+filename)
    root = tree.getroot()
    # writefile = codecs.open(rootpath+filename+'.txt', 'w', 'utf-8')
    size=0
    for pair in root.findall('pair'):
        sent1 = pair.find('t').text.strip()#' '.join(word_tokenize(pair.find('t').text.strip()))
        sent2 = pair.find('h').text.strip()#' '.join(word_tokenize(pair.find('h').text.strip()))
        label_str= pair.get('value')
        label = RTE1_label2id.get(label_str)
        writefile.write(str(label)+'\t'+sent1+'\t'+sent2+'\n')
        size+=1
    print('parsing ', filename, 'over...size:', size)
    # writefile.close()

def preprocess_RTE2(filename, writefile):
    # rootpath = '/save/wenpeng/datasets/RTE/'
    tree = ET.parse(rootpath+filename)
    root = tree.getroot()
    # writefile = codecs.open(rootpath+filename+'.txt', 'w', 'utf-8')
    size=0
    for pair in root.findall('pair'):
        sent1 = pair.find('t').text.strip()#' '.join(word_tokenize(pair.find('t').text.strip()))
        sent2 = pair.find('h').text.strip()#' '.join(word_tokenize(pair.find('h').text.strip()))
        label_str= pair.get('entailment')
        label = RTE2_label2id.get(label_str)
        writefile.write(str(label)+'\t'+sent1+'\t'+sent2+'\n')
        size+=1
    print('parsing ', filename, 'over...size:', size)
    # writefile.close()

def preprocess_RTE3(filename, writefile):
    # rootpath = '/save/wenpeng/datasets/RTE/'
    tree = ET.parse(rootpath+filename)
    root = tree.getroot()
    # writefile = codecs.open(rootpath+filename+'.txt', 'w', 'utf-8')
    size=0
    for pair in root.findall('pair'):
        sent1 = pair.find('t').text.strip()#' '.join(word_tokenize(pair.find('t').text.strip()))
        sent2 = pair.find('h').text.strip()#' '.join(word_tokenize(pair.find('h').text.strip()))
        label_str= pair.get('entailment')
        label = RTE3_label2id.get(label_str)
        writefile.write(str(label)+'\t'+sent1+'\t'+sent2+'\n')
        size+=1
    print('parsing ', filename, 'over...size:', size)
    # writefile.close()


def preprocess_RTE5(filename, writefile):
    # rootpath = '/save/wenpeng/datasets/RTE/'
    # labelsize = defaultdict(int)
    tree = ET.parse(rootpath+filename)
    root = tree.getroot()
    # writefile = codecs.open(rootpath+filename+'.txt', 'w', 'utf-8')
    size=0
    for pair in root.findall('pair'):
        sent1 = pair.find('t').text.strip()#' '.join(word_tokenize(pair.find('t').text.strip()))
        sent2 = pair.find('h').text.strip()#' '.join(word_tokenize(pair.find('h').text.strip()))
        label_str= pair.get('entailment')
        # labelsize[label_str]+=1
        label = RTE5_label2id.get(label_str)
        writefile.write(str(label)+'\t'+sent1+'\t'+sent2+'\n')
        size+=1
    print('parsing ', filename, 'over...size:', size)
    # writefile.close()

def preprocess_GLUE_RTE(filename, writefile):
    readfile = codecs.open(filename, 'r', 'utf-8')
    size=0
    for line in readfile:
        if size>0:
            parts = line.strip().split('\t')
            sent1 = parts[1]#' '.join(word_tokenize(pair.find('t').text.strip()))
            sent2 = parts[2]#' '.join(word_tokenize(pair.find('h').text.strip()))
            label = 0 if parts[3] == 'not_entailment' else 1
            writefile.write(str(label)+'\t'+sent1+'\t'+sent2+'\n')
        size+=1
    print('parsing ', filename, 'over...size:', size)
    # writefile.close()

if __name__ == '__main__':
    # writefile = codecs.open(rootpath+'test_RTE_1235.txt', 'w', 'utf-8')
    # preprocess_RTE1('annotated_test.xml', writefile)
    # preprocess_RTE2('RTE2_test.annotated.xml', writefile)
    # preprocess_RTE3('RTE3_test_3ways.xml', writefile)
    # preprocess_RTE5('RTE5_MainTask_TestSet_Gold.xml', writefile)
    # writefile.close()

    writefile = codecs.open(rootpath+'dev_RTE_1235.txt', 'w', 'utf-8')
    preprocess_RTE1('dev.xml', writefile)
    preprocess_RTE1('dev2.xml', writefile)
    preprocess_RTE2('RTE2_dev.xml', writefile)
    preprocess_RTE3('RTE3_dev_3ways.xml', writefile)
    preprocess_RTE5('RTE5_MainTask_DevSet.xml', writefile)
    writefile.close()

    # writefile = codecs.open(rootpath+'train_RTE_1235.txt', 'w', 'utf-8')
    # preprocess_GLUE_RTE('/home/wyin3/Datasets/glue_data/RTE/train.tsv', writefile)
    # writefile.close()
