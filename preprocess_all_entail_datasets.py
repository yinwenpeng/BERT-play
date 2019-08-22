
import codecs
def integrate_all_entail_training_data():
    '''
    MNLI, SNLI, SciTail, RTE-glue, SICK; into 2-way
    '''
    root = '/home/wyin3/Datasets/MNLI-SNLI-SciTail-RTE-SICK/'
    writefile = codecs.open(root+'all.6.train.txt', 'w', 'utf-8')

    readfile = codecs.open('/home/wyin3/Datasets/glue_data/MNLI/train.tsv', 'r', 'utf-8')
    line_co = 0
    valid=0
    for line in readfile:
        if line_co>0:
            parts=line.strip().split('\t')
            labelstr = '1' if parts[-1] == 'entailment' else '0'
            sent1 = parts[8].strip()#' '.join(word_tokenize(parts[5]))
            sent2 = parts[9].strip()
            writefile.write(labelstr+'\t'+sent1+'\t'+sent2+'\t'+'MNLI\n')
            valid+=1
        line_co+=1
    readfile.close()
    print('load MNLI over, size:', valid)

    readfile = codecs.open('/home/wyin3/Datasets/glue_data/SNLI/train.tsv', 'r', 'utf-8')
    #index	captionID	pairID	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2	label1	gold_label
    line_co = 0
    valid=0
    for line in readfile:
        if line_co>0:
            parts=line.strip().split('\t')
            labelstr = '1' if parts[-1] == 'entailment' else '0'
            sent1 = parts[7].strip()#' '.join(word_tokenize(parts[5]))
            sent2 = parts[8].strip()
            writefile.write(labelstr+'\t'+sent1+'\t'+sent2+'\t'+'SNLI\n')
            valid+=1
        line_co+=1
    readfile.close()
    print('load SNLI over, size:', valid)

    readfile = codecs.open('/home/wyin3/Datasets/glue_data/RTE/train.tsv', 'r', 'utf-8')
    valid=0
    for line in readfile:
        parts = line.strip().split('\t')
        sent1 = parts[1].strip()#' '.join(word_tokenize(pair.find('t').text.strip()))
        sent2 = parts[2].strip()#' '.join(word_tokenize(pair.find('h').text.strip()))
        label = '0' if parts[3] == 'not_entailment' else '1'
        writefile.write(label+'\t'+sent1+'\t'+sent2+'\t'+'GLUE-RTE\n')
        valid+=1
    readfile.close()
    print('load GLUE RTE over, size:', valid)


    readfile=codecs.open('/save/wenpeng/datasets/SciTailV1/tsv_format/scitail_1.0_train.tsv', 'r', 'utf-8')
    valid=0
    for line in readfile:
        parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
        if len(parts)==3:
            label= '1' if parts[2] == 'entails' else '0' # keep label be 0 or 1
            sent1 = parts[0].strip()
            sent2 = parts[1].strip()
            writefile.write(label+'\t'+sent1+'\t'+sent2+'\t'+'SciTail\n')
            valid+=1
    readfile.close()
    print('load SciTail over, size:', valid)

    readfile=codecs.open('/save/wenpeng/datasets/Dataset/SICK/train.txt', 'r', 'utf-8')
    valid=0
    for line in readfile:
        parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
        if len(parts)==4:
            label= '1' if parts[2] == '1' else '0' # keep label be 0 or 1
            sent1 = parts[0].strip()
            sent2 = parts[1].strip()
            writefile.write(label+'\t'+sent1+'\t'+sent2+'\t'+'SICK\n')
            valid+=1
    readfile.close()
    print('load SICK over, size:', valid)

    readfile=codecs.open('/home/wyin3/Datasets/FEVER_2_Entailment/train.txt', 'r', 'utf-8')
    valid=0
    for line in readfile:
        parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
        if len(parts)==3:
            label= '1' if parts[2] == 'entailment' else '0' # keep label be 0 or 1
            sent1 = parts[0].strip()
            sent2 = parts[1].strip()
            writefile.write(label+'\t'+sent1+'\t'+sent2+'\t'+'FEVER\n')
            valid+=1
    readfile.close()
    print('load FEVER over, size:', valid)

    writefile.close()


if __name__ == '__main__':
    '''
    load MNLI over, size: 392702
    load SNLI over, size: 549367
    load GLUE RTE over, size: 2491
    load SciTail over, size: 23596
    load SICK over, size: 4439
    load MNLI over, size: 109082
    '''
    integrate_all_entail_training_data()
