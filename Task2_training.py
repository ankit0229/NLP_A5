import self as self
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import string

fp = open(r"C:\Users\ANKIT\PycharmProjects\NLP_A5\data.txt","r",encoding="utf8")
text = fp.read()
fp.close()
text = text.lower()
text = text.translate(text.maketrans('','',string.punctuation))
sentences = text.split("\n")
print(sentences)

docs_tagged = []

for i in range(len(sentences)):
        x = sentences[i]
        x = word_tokenize(x)
        y = TaggedDocument(x, tags=[str(i)])
        docs_tagged.append(y)

print(docs_tagged)
episodes_count = 100
vector_size = 100
alpha = 0.025

model = Doc2Vec(size=vector_size,
                alpha=alpha,
                min_alpha=0.025,
                min_count=1,
                dm=1)

model.build_vocab(docs_tagged)

for epoch in range(episodes_count):
        print(f"iteration  = {epoch}")
        model.train(docs_tagged,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        model.min_alpha = model.alpha

model.save("Task2_doc2vec.model")
print("Model Saved successfully")
