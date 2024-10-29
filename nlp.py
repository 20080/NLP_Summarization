import gensim
import numpy as np
import pandas as pd
import nltk
import re
from transformers import BertTokenizer, BertModel
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import networkx as nx
import gensim.downloader as api


def funglove(inp,num_lines):
    df = pd.read_csv("tennis_articles_v4.csv", encoding='unicode-escape')

    sentences = [sent_tokenize(inp)]
    sentences = sentences[0]

    word_embeddings = {}
    # f = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  #
    # specific to google model
    f = open('glove.6B.100d.txt', encoding='utf-8')
    # separating words and coefficients
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    nltk.download('stopwords')

    from nltk.corpus import stopwords

    stop_words = stopwords.words('english')

    # function to remove stopwords
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new

    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = \
                    cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[
                        0, 0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # Extract top 10 sentences as the summary
    temp = ""
    for i in range(num_lines):
        temp += ranked_sentences[i][1]
    return temp


def fungoogle(inp,num_lines):
    df = pd.read_csv("tennis_articles_v4.csv", encoding='unicode-escape')

    sentences = sent_tokenize(inp)

    word_embeddings = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    clean_sentences = [s.lower() for s in sentences]

    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')

    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new

    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = np.zeros((300,))
            count = 0
            for w in i.split():
                try:
                    v += word_embeddings.get_vector(w)
                    count += 1
                except KeyError:
                    continue
            if count != 0:
                v /= count
        else:
            v = np.zeros((300,))
        sentence_vectors.append(v)

    sim_mat = np.zeros([len(sentences), len(sentences)])

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 300),
                                                  sentence_vectors[j].reshape(1, 300))[0, 0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    temp = ""
    for i in range(num_lines):
        temp += ranked_sentences[i][1]
    return temp


def funbert(inp,num_lines):
    df = pd.read_csv("tennis_articles_v4.csv", encoding='unicode-escape')

    sentences = sent_tokenize(inp)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    clean_sentences = [s.lower() for s in sentences]

    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')

    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new

    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            inputs = tokenizer.encode_plus(i, add_special_tokens=True, return_tensors='pt')
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
            v = embeddings / (len(i.split()) + 0.001)
        else:
            v = np.zeros((768,))
        sentence_vectors.append(v)

    sim_mat = np.zeros([len(sentences), len(sentences)])

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 768),
                                                  sentence_vectors[j].reshape(1, 768))[0, 0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    temp = ""
    for i in range(num_lines):
        temp += ranked_sentences[i][1]
    return temp
