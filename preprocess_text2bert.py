import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from bert_common_functions import sent_pair_to_embedding
import codecs


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# model.eval()
# model.to('cuda')
# emb = sent_pair_to_embedding('Who was Jim Henson?', 'Jim Henson was a puppeteer.', tokenizer, model, False)
# print(' '.join(emb.cpu().numpy().astype('str')))
# print(len(emb))


#
def SciTail_to_Bert_emb():
    root="/save/wenpeng/datasets/SciTailV1/tsv_format/"
    newroot = '/home/wyin3/Datasets/SciTailV1/BERT/'
    files=['scitail_1.0_train.tsv', 'scitail_1.0_dev.tsv', 'scitail_1.0_test.tsv']
    writefiles=['scitail_1.0_train_to_Bert.txt', 'scitail_1.0_dev_to_Bert.txt', 'scitail_1.0_test_to_Bert.txt']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    model.to('cuda')

    for i in range(len(files)):
        print('loading file:', root+files[i], '...')
        readfile=codecs.open(root+files[i], 'r', 'utf-8')
        writefile=codecs.open(newroot+writefiles[i], 'w', 'utf-8')
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)==3:

                label=parts[2]  # keep label be 0 or 1
                sent1 = parts[0]
                sent2 = parts[1]
                emb = sent_pair_to_embedding(sent1, sent2, tokenizer, model, False)
                emb2str = ' '.join(emb.cpu().numpy().astype('str'))
                writefile.write(label+'\t'+emb2str+'\n')
        readfile.close()
        writefile.close()

def RTE_to_Bert_emb():
    root="/home/wyin3/Datasets/RTE/"
    # newroot = '/home/wyin3/Datasets/SciTailV1/BERT/'
    files=['dev_RTE_1235.txt', 'test_RTE_1235.txt']
    writefiles=['dev_RTE_1235_to_Bert.txt', 'test_RTE_1235_to_Bert.txt']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    model.to('cuda')

    for i in range(len(files)):
        print('loading file:', root+files[i], '...')
        readfile=codecs.open(root+files[i], 'r', 'utf-8')
        writefile=codecs.open(root+writefiles[i], 'w', 'utf-8')
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)==3:

                label=parts[0] # keep label be 0 or 1
                sent1 = parts[1]
                sent2 = parts[2]
                emb = sent_pair_to_embedding(sent1, sent2, tokenizer, model, False)
                emb2str = ' '.join(emb.cpu().numpy().astype('str'))
                writefile.write(label+'\t'+emb2str+'\n')
        readfile.close()
        writefile.close()

def RTE_GLUE_to_Bert_emb():
    root="/home/wyin3/Datasets/glue_data/RTE/"
    # newroot = '/home/wyin3/Datasets/SciTailV1/BERT/'
    files=['train.tsv']
    writefiles=['train_to_Bert.txt']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    model.to('cuda')

    for i in range(len(files)):
        print('loading file:', root+files[i], '...')
        readfile=codecs.open(root+files[i], 'r', 'utf-8')
        writefile=codecs.open(root+writefiles[i], 'w', 'utf-8')
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)==4:

                label='0' if parts[3] == 'not_entailment' else '1' # keep label be 0 or 1
                sent1 = parts[1]
                sent2 = parts[2]
                emb = sent_pair_to_embedding(sent1, sent2, tokenizer, model, False)
                emb2str = ' '.join(emb.cpu().numpy().astype('str'))
                writefile.write(label+'\t'+emb2str+'\n')
        readfile.close()
        writefile.close()

if __name__ == '__main__':
    # SciTail_to_Bert_emb()
    # RTE_to_Bert_emb()
    RTE_GLUE_to_Bert_emb()
