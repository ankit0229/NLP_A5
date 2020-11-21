import json
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models.doc2vec import Doc2Vec

model= Doc2Vec.load("Task2_doc2vec.model")

t = []
with open(r'C:\Users\ANKIT\PycharmProjects\NLP_A5\test.jsonl') as fle:
    for i in fle:
        line = json.loads(i)
        t.append(line)

len_questions = len(t)

def do_preprocessing(text):
    text = text.lower()
    text = text.translate(text.maketrans('', '', string.punctuation))
    return text

predicted_options = []
actual_answers = []
score = 0
counting = 0

for q in t:
    counting = counting + 1
    print(counting)
    opt_max_values = []
    predicted_options = []
    actual_answers.append(q['answerKey'])
    curr = q['question']
    qi = curr['stem']
    qi = do_preprocessing(qi)
    qi = qi + ' '
    optns = curr['choices']
    list_ono = []
    for o in optns:
        oi = o['text']
        ono = o['label']
        oi = do_preprocessing(oi)
        list_ono.append(ono)
        tot_query = qi + oi
        tot_query = word_tokenize(tot_query)
        query_vector = model.infer_vector(tot_query)
        max_sim = model.docvecs.most_similar(positive=[query_vector], topn=1)
        req = max_sim[0]
        max_cos = req[1]
        opt_max_values.append(max_cos)

    pred_val = max(opt_max_values)
    len_list_ono = len(list_ono)
    for e in range(len_list_ono):
        if opt_max_values[e] == pred_val:
            predicted_options.append(list_ono[e])
    if q['answerKey'] in predicted_options:
        len_pred_options = len(predicted_options)
        score = score + (1 / len_pred_options)

print(f"Score = {score}")
accuracy = score / len_questions
accuracy = accuracy * 100
print(f"Accuracy is = {accuracy}")



