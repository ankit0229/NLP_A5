import json
import pickle
import string
import math
import numpy
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter


Picklefile1 = open('sentences_file', 'rb')
sentences = pickle.load(Picklefile1)
len_sent = len(sentences)

Picklefile2 = open('words_file', 'rb')
unique_words = pickle.load(Picklefile2)
len_words = len(unique_words)

Picklefile3 = open('nt_file', 'rb')
dict_nt = pickle.load(Picklefile3)


Picklefile4 = open('term_doc_file', 'rb')
term_doc = pickle.load(Picklefile4)

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

def make_vector(qry):
    q_vector = numpy.zeros(len_words, dtype=float)
    qry = word_tokenize(qry)
    coll_qry = Counter(qry)
    N = len_sent
    for i in range(len_words):
        curr_wd = unique_words[i]
        curr_nt = dict_nt[curr_wd]
        ftd = coll_qry[curr_wd]
        term1 = 1 + math.log10(1 + ftd)
        tmp = N / (1 + curr_nt)
        term2 = math.log10(tmp)
        prod = term1 * term2
        q_vector[i] = prod
    return q_vector

def find_similarity(q_vector):
    norm_q_vector = numpy.linalg.norm(q_vector)
    for i in range(len_sent):
        doc_vector = term_doc[i]
        doc_np_vector = numpy.array(doc_vector)
        final_val = numpy.dot(q_vector, doc_np_vector) / (norm_q_vector * dict_doc_norm[i])
        if i == 0:
            max_val = final_val
        elif final_val > max_val:
            max_val = final_val
        else:
            max_val = max_val
    return max_val

predicted_options = []
actual_answers = []
score = 0
counting = 0
dict_doc_norm = {}

for i in range(len_sent):
    doc_vector = term_doc[i]
    doc_np_vector = numpy.array(doc_vector)
    norm_doc_vector = numpy.linalg.norm(doc_np_vector)
    dict_doc_norm[i] = norm_doc_vector

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
        query_vector = make_vector(tot_query)
        tmp = find_similarity(query_vector)
        opt_max_values.append(tmp)

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
