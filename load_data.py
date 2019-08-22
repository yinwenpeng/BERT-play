
import codecs
def load_SciTail_bert_version():
    root = '/home/wyin3/Datasets/SciTailV1/BERT/'
    files=['scitail_1.0_train_to_Bert.txt', 'scitail_1.0_dev_to_Bert.txt', 'scitail_1.0_test_to_Bert.txt']
    features_dataset = []
    labels_dataset = []
    for i in range(len(files)):
        print('loading file:', root+files[i], '...')
        readfile=codecs.open(root+files[i], 'r', 'utf-8')
        feature_split = []
        label_split = []
        for line in readfile:
            parts = line.strip().split('\t')
            label_str = parts[0]
            features = [float(x) for x in parts[1].split()]
            feature_split.append(features)
            label_split.append(0 if label_str=='neutral' else 1)
        readfile.close()
        features_dataset.append(feature_split)
        labels_dataset.append(label_split)
    print('load scitail succeed!')
    return features_dataset, labels_dataset

def load_RTE_bert_version():
    root = '/home/wyin3/Datasets/RTE/'
    glueroot = '/home/wyin3/Datasets/glue_data/RTE/'
    files=[glueroot+'train_to_Bert.txt', root+'test_RTE_1235_to_Bert.txt']
    features_dataset = []
    labels_dataset = []
    for i in range(len(files)):
        pos_co=0
        print('loading file:', files[i], '...')
        readfile=codecs.open(files[i], 'r', 'utf-8')
        feature_split = []
        label_split = []
        for line in readfile:
            parts = line.strip().split('\t')
            label_int = int(parts[0])
            features = [float(x) for x in parts[1].split()]
            feature_split.append(features)
            label_split.append(label_int)
            if label_int == 1:
                pos_co+=1
        readfile.close()
        features_dataset.append(feature_split)
        labels_dataset.append(label_split)
        print('pos vs. all:', pos_co/len(label_split))
    print('load RTE succeed!')
    return features_dataset, labels_dataset

#
# def get_test_examples_wenpeng(filename):
#     readfile = codecs.open(filename, 'r', 'utf-8')
#     line_co=0
#     delete_co=0
#     for row in readfile:
#         print(row)
#         line=row.strip().split('\t')
#         if len(line)==3:
#             text_a = line[1]
#             text_b = line[2]
#             label = 'entailment' if line[0] == '1' else 'not_entailment'
#             line_co+=1
#         else:
#             delete_co+=1
#
#
#     readfile.close()
#     print('loaded test size:', line_co, delete_co)

if __name__ == "__main__":
    # get_test_examples_wenpeng('/home/wyin3/Datasets/RTE/test_RTE_1235.txt')
