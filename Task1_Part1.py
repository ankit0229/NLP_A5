import string
import math
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

fp = open(r"C:\Users\ANKIT\PycharmProjects\NLP_A5\data.txt","r",encoding="utf8")
text = fp.read()
fp.close()
text = text.lower()
text = text.translate(text.maketrans('','',string.punctuation))
sentences = text.split("\n")
unique_words = []
words = []
for sent in sentences:
    temp = word_tokenize(sent)
    words.extend(temp)

coll_words = Counter(words)
for w in coll_words:
    unique_words.append(w)

print(f"vocab size = {len(unique_words)}")
len_sent = len(sentences)
len_words = len(unique_words)
word_count = [[0 for i in range(len_words)] for j in range(len_sent)]
term_doc = [[0 for i in range(len_words)] for j in range(len_sent)]

dict_master = {}
for s in sentences:
    wd_list = word_tokenize(s)
    coll_wd = Counter(wd_list)
    dict_master[s] = coll_wd

for j in range(len_words):
    curr_word = unique_words[j]
    line = ''
    for i in range(len_sent):
        curr_doc = sentences[i]
        coll_doc = dict_master[curr_doc]
        curr_cnt = coll_doc[curr_word]
        word_count[i][j] = curr_cnt
        temp_cnt = str(curr_cnt)
        line = line + temp_cnt
        line = line + ', '
    fp2 = open(r"C:\Users\ANKIT\PycharmProjects\NLP_A5\word_count.txt", "a")
    fp2.write(line)
    fp2.write("\n")
    fp2.close()

dict_nt = {}
for j in range(len_words):
    curr_wd = unique_words[j]
    nt = 0
    for i in range(len_sent):
        if word_count[i][j] != 0:
            nt = nt + 1
    dict_nt[curr_wd] = nt

N = len_sent

all_len = []
for z in range(len_sent):
    cu = sentences[z]
    cu = word_tokenize(cu)
    cu_len = len(cu)
    all_len.append(cu_len)

mx_se = max(all_len)
mx_se_idx = all_len.index(mx_se)

for j in range(len_words):
    curr_wd = unique_words[j]
    curr_nt = dict_nt[curr_wd]
    line = ''
    for i in range(len_sent):
        ftd = word_count[i][j]
        term1 = 1 + math.log10(1 + ftd)
        tmp = N / (1 + curr_nt)
        term2 = math.log10(tmp)
        prod = term1 * term2
        term_doc[i][j] = prod
        temp_cnt = str(prod)
        line = line + temp_cnt
        line = line + ', '
    fp3 = open(r"C:\Users\ANKIT\PycharmProjects\NLP_A5\term_doc.txt", "a")
    fp3.write(line)
    fp3.write("\n")
    fp3.close()

Picklefile1 = open('sentences_file', 'wb')
pickle.dump(sentences, Picklefile1)
Picklefile1.close()

Picklefile2 = open('words_file', 'wb')
pickle.dump(unique_words, Picklefile2)
Picklefile2.close()

Picklefile3 = open('nt_file', 'wb')
pickle.dump(dict_nt, Picklefile3)
Picklefile3.close()

Picklefile4 = open('term_doc_file', 'wb')
pickle.dump(term_doc, Picklefile4)
Picklefile4.close()

print("Pickles created successfully")


