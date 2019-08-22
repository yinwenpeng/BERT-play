
import jsonlines
from collections import defaultdict
path = '/home/wyin3/Datasets/Stuttgart_Emotion/unify-emotion-datasets-master/'

def statistics():
    readfile = jsonlines.open(path+'unified-dataset.jsonl' ,'r')
    domain2size = defaultdict(int)
    source2size = defaultdict(int)
    emotion2size = defaultdict(int)
    single2size = defaultdict(int)
    emo_dom_size = defaultdict(int)
    line_co = 0
    valid_line_co = 0
    for line2dict in readfile:
        valid_line = False
        text = line2dict.get('text')
        domain = line2dict.get('domain')


        source_dataset = line2dict.get('source')

        single = line2dict.get('labeled')

        emotions =line2dict.get('emotions')
        if domain == 'headlines' or domain == 'facebook-messages':
            print(emotions)
        for emotion, label in emotions.items():
            if label == 1:
                emotion2size[emotion]+=1
                emo_dom_size[(emotion, domain)]+=1
                valid_line = True
        if valid_line:
            valid_line_co+=1
            domain2size[domain]+=1
            source2size[source_dataset]+=1
            single2size[single]+=1
        line_co+=1
        if line_co%100==0:
            print(line_co)
    readfile.close()
    print('domain2size:',domain2size)
    print('source2size:',source2size)
    print('emotion2size:',emotion2size)
    print('single2size:',single2size)
    print('emo_dom_size:',emo_dom_size)
    print('line_co:', line_co)
    print('valid_line_co:', valid_line_co)

'''
domain2size: defaultdict(<class 'int'>, {'tweets': 54203, 'emotional_events': 7666, 'fairytale_sentences': 14771, 'artificial_sentences': 2268})
source2size: defaultdict(<class 'int'>, {'grounded_emotions': 2585, 'ssec': 4776, 'isear': 7666, 'crowdflower': 39740, 'tales-emotion': 14771, 'emotion-cause': 2268, 'emoint': 7102})
emotion2size: defaultdict(<class 'int'>, {'sadness': 12947, 'joy': 17833, 'anger': 8335, 'disgust': 3931, 'trust': 2700, 'fear': 14752, 'surprise': 4304, 'shame': 1096, 'guilt': 1093, 'noemo': 18765, 'love': 3820})
single2size: defaultdict(<class 'int'>, {'single': 74132, 'multi': 4776})


emo_dom_size: defaultdict(<class 'int'>, {('sadness', 'tweets'): 10355, ('joy', 'tweets'): 14433, ('anger', 'tweets'): 6024, ('disgust', 'tweets'): 2362, ('trust', 'tweets'): 2700, ('fear', 'tweets'): 12522, ('surprise', 'tweets'): 3285, ('joy', 'emotional_events'): 1094, ('fear', 'emotional_events'): 1095, ('anger', 'emotional_events'): 1096, ('sadness', 'emotional_events'): 1096, ('disgust', 'emotional_events'): 1096, ('shame', 'emotional_events'): 1096, ('guilt', 'emotional_events'): 1093, ('noemo', 'tweets'): 9370, ('love', 'tweets'): 3820, ('noemo', 'fairytale_sentences'): 9395, ('disgust', 'fairytale_sentences'): 378, ('joy', 'fairytale_sentences'): 1827, ('surprise', 'fairytale_sentences'): 806, ('fear', 'fairytale_sentences'): 712, ('anger', 'fairytale_sentences'): 732, ('sadness', 'fairytale_sentences'): 921, ('joy', 'artificial_sentences'): 479, ('sadness', 'artificial_sentences'): 575, ('surprise', 'artificial_sentences'): 213, ('disgust', 'artificial_sentences'): 95, ('anger', 'artificial_sentences'): 483, ('fear', 'artificial_sentences'): 423})
'''
if __name__ == '__main__':
    statistics()
