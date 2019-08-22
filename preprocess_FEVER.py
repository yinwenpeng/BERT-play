import jsonlines
import codecs

root = '/save/wenpeng/datasets/FEVER/'
newroot = '/home/wyin3/Datasets/FEVER_2_Entailment/'
def convert_fever_into_entailment():
    '''
    label, statement, #50sentcand, #50binaryvec
    '''
    #load wiki
    title2sentlist={}
    readwiki = codecs.open(root+'wiki_title2sentlist.txt' ,'r', 'utf-8')
    wiki_co = 0
    for line in readwiki:
        parts = line.strip().split('\t')
        title2sentlist[parts[0]] = parts[1:]
        wiki_co+=1
        if wiki_co % 1000 ==0:
            print('wiki_co....', wiki_co)
        # if wiki_co == 1000:
        #     break
    readwiki.close()
    print('wiki pages loaded over, totally ', len(title2sentlist), ' pages')

    readfiles = ['paper_dev.jsonl','paper_test.jsonl']
    writefiles = ['dev.txt','test.txt']
    label2id = {'SUPPORTS':1, 'REFUTES':0, 'NOT ENOUGH INFO':2}

    for i in range(2):
        readfile = jsonlines.open(root+readfiles[i] ,'r')
        writefile = codecs.open(newroot+writefiles[i] ,'w', 'utf-8')
        co = 0
        for line2dict in readfile:
            origin_label = line2dict.get('label')
            if origin_label != 'NOT ENOUGH INFO':
                label = 'entailment' if origin_label == 'SUPPORTS' else 'not_entailment'
                claim =  line2dict.get('claim').strip()
                all_evi_list = line2dict.get('evidence')
                if all_evi_list is None:
                    continue
                title2evi_idlist = {}
                for evi in all_evi_list:
                    title = evi[0][2]
                    sent_id = int(evi[0][3])
                    evi_idlist = title2evi_idlist.get(title)
                    if evi_idlist is None:
                        evi_idlist = [sent_id]
                    else:
                        evi_idlist.append(sent_id)
                    title2evi_idlist[title] = evi_idlist
                premise_str = ''
                for title, idlist in title2evi_idlist.items():
                    id_set = set(idlist)
                    title_sents = title2sentlist.get(title)
                    if title_sents is None:
                        continue
                    else:
                        for idd in idlist:
                            if idd < len(title_sents):
                                premise_str+=' '+title_sents[idd].strip()
                writefile.write(premise_str+'\t'+claim+'\t'+label+'\n')
                co+=1
                if co % 1000==0:
                    print('loading...', i, '.....',co)
        writefile.close()
        readfile.close()
    print('reformat FEVER over')


if __name__ == '__main__':
    convert_fever_into_entailment()
