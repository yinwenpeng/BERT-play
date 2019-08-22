import torch
import torch.optim as optim
import torch.nn as nn
from load_data import load_RTE_bert_version
from bert_common_functions import LogisticRegression
import numpy as np

features_dataset, labels_dataset = load_RTE_bert_version()
train_feature_lists = np.asarray(features_dataset[0], dtype='float32')
train_label_list = np.asarray(labels_dataset[0], dtype='int32')

# dev_feature_lists = np.asarray(features_dataset[1], dtype='float32')
# dev_label_list = np.asarray(labels_dataset[1], dtype='int32')

test_feature_lists = np.asarray(features_dataset[1], dtype='float32')
test_label_list = np.asarray(labels_dataset[1], dtype='int32')

train_size = len(train_label_list)
test_size = len(test_label_list)
print('train size:', train_size)
print('test size:', test_size)

model = LogisticRegression(768, 2)
model.to('cuda')
loss_function = nn.NLLLoss() #nn.CrossEntropyLoss()#
optimizer = optim.Adagrad(model.parameters(), lr=0.001)


batch_size=50
n_train_batches=train_size//batch_size
train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]

n_test_batches=test_size//batch_size
test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]
train_indices = list(range(train_size))
iter_co = 0
loss_co = 0.0
max_test_acc = 0.0
for epoch in range(100):
    for batch_id in train_batch_start: #for each batch
        train_id_batch = train_indices[batch_id:batch_id+batch_size]
        train_batch_input = train_feature_lists[train_id_batch]
        train_batch_input_tensor = torch.from_numpy(train_batch_input).float().to('cuda')
        train_batch_gold = train_label_list[train_id_batch]
        train_batch_gold_tensor = torch.from_numpy(train_batch_gold).long().to('cuda')

        model.zero_grad()
        log_probs = model(train_batch_input_tensor)

        loss = loss_function(log_probs, train_batch_gold_tensor)
        loss.backward()
        optimizer.step()
        iter_co+=1
        loss_co+=loss.item()
        if iter_co%5==0:
            print('epoch:', epoch,' loss:', loss_co/iter_co)
    if epoch % 1 ==0:
        model.eval()
        with torch.no_grad():
            n_test_correct=0
            for idd, test_batch_id in enumerate(test_batch_start): # for each test batch
                test_batch_input = test_feature_lists[test_batch_id:test_batch_id+batch_size]
                test_batch_input_tensor = torch.from_numpy(test_batch_input).float().to('cuda')
                test_batch_gold = test_label_list[test_batch_id:test_batch_id+batch_size]
                test_batch_gold_tensor = torch.from_numpy(test_batch_gold).long().to('cuda')
                test_log_probs = model(test_batch_input_tensor)
                n_test_correct += (torch.argmax(test_log_probs, 1).view(test_batch_gold_tensor.size()) == test_batch_gold_tensor).sum().item()
            test_acc = n_test_correct/(batch_size*len(test_batch_start))
            if test_acc > max_test_acc:
                max_test_acc = test_acc
        print('\t\t\t\t test acc:', test_acc, '\tmax_test_acc:', max_test_acc)
