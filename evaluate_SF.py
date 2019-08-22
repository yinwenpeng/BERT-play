import codecs
import numpy as np
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

def get_examples_SF_wenpeng(filename):
    readfile = codecs.open(filename, 'r', 'utf-8')
    label_list = []
    coord_list = []
    for row in readfile:
        line=row.strip().split('\t')
        if len(line)==4:
            label_list.append(int(line[0]))
            coord_list.append(line[3])

    readfile.close()
    return label_list, coord_list

def evaluate_SF(preds, filename):
    '''
    preds: a vector (1 or 0) of label prediction for all test examples
    filename: the file path of the SF test file

    output: two values, one is "Mean F1", the other is "Weighted F1"
    '''
    gold_label_list, coord_list = get_examples_SF_wenpeng(filename)
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
    # print('gold_matrix:', gold_matrix[:20])
    pred_matrix = create_binary_matrix(pred_list, coord_list, False)
    # print('pred_matrix:', pred_matrix[:20])
    # exit(0)
    return average_f1_two_array_by_col(pred_matrix, gold_matrix)
