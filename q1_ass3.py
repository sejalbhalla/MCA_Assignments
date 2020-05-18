# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:40:14 2020

@author: Sejal Bhalla
"""

import nltk
nltk.download('abc')
nltk.download('punkt')
from nltk.corpus import abc, stopwords
from collections import Counter
import numpy as np
import pickle
import string
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

from keras.models import Model
from keras.layers import Input, Dense, Reshape, dot
from keras.layers.embeddings import Embedding


import matplotlib
matplotlib.use('AGG')

def pre_process():
    """Remove stop words and punctuation marks from corpus
    """
    if 'cleaned_corpus.pkl' not in os.listdir(os.curdir) or 'cleaned_sentences.pkl' not in os.listdir(os.curdir):
        print('Pre-processing...')
        words = abc.words()
        words = [w for w in words]
        
        sentences = abc.sents()
        sentences = [s for s in sentences]
        
        stop_words = stopwords.words('english')
        punctuation = list(string.punctuation)
        for i in range(len(sentences)):
            print(i)
            for j in sentences[i]:
                prev = len(sentences[i])
                #print(i*j)
                if set(j) - set(punctuation) == set() or j.lower() in stop_words:
                    print(j)
                    print('removed')
                    if j in words:
                        words.remove(j)
                    sentences[i].remove(j)
                    assert prev == len(sentences[i])+1
                    
        for s in sentences:
            if len(s) <= 1:
                print(s)
                sentences.remove(s)
        
        pickle.dump(words, open('cleaned_corpus.pkl', 'wb'))
        pickle.dump(sentences, open('cleaned_sentences.pkl', 'wb'))
        
    else:
        print('Pre processed data already present..')
        words = pickle.load(open('cleaned_corpus.pkl', 'rb'))
        sentences = pickle.load(open('cleaned_sentences.pkl', 'rb'))
        
    return words, sentences


def process_data(words, sentences):
    """Process raw inputs into a dataset."""
    count = []
    count.extend(Counter(words).most_common(31510))
    dictionary = {}
    index = 0
    for tup in count:
        word = tup[0]
        dictionary[word] = index
        index += 1
        
    data = []
    for word in words:
        try:
            data.append(dictionary[word])
        except Exception:
            print('word', word)
        
    sentences_encoded = []
    for sent in sentences:
        s = []
        for word in sent:
            s.append(dictionary[word])
        sentences_encoded.append(s)
    
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary, sentences_encoded

def generate_training_data(window_size, sentences, vocab_size):
    if 'X.pkl' in os.listdir(os.curdir) and 'Y.pkl' in os.listdir(os.curdir):
        print('Training data already present..')
        couples = pickle.load(open('X.pkl', 'rb'))
        labels= pickle.load(open('Y.pkl', 'rb'))
        return couples, labels
    
    print('generating training data..')
    couples = []
    labels = []
    for s in sentences:
        for i in range(0, len(s), 2):
            window = [i-window_size, i+window_size+1]
            if window[0] < 0:
                window[0] = 0
            if window[1] > len(s):
                window[1] = len(s)
            valid_pos_window = np.setdiff1d(np.arange(window[0], window[1]), i)
            valid_neg_window = np.delete(np.arange(vocab_size), s)
            pos_word = s[np.random.choice(valid_pos_window)]
            neg_word = np.random.choice(valid_neg_window)
            
            couples.append([s[i], pos_word])
            labels.append(1)
            couples.append([s[i], neg_word])
            labels.append(0)
                
    return np.array(couples), np.array(labels)


# print(couples[:10], labels[:10])

########################## Model ##################################

def intermediate_model(embedding, vector_dim):
    input_target = Input((1,))   #input is just the word
    target = embedding(input_target)
    target = Reshape((1, vector_dim))(target)
    return input_target, target

def build_model(vocab_size, vector_dim):
    print('Building the model..')
    embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
    input_target, target = intermediate_model(embedding, vector_dim)
    input_context, context = intermediate_model(embedding, vector_dim)
    similarity = dot([target, context], axes = -1, normalize = True)
    dot_product = dot([target, context], axes = -1)
    dot_product = Reshape((1,))(dot_product)
    # add the sigmoid output layer
    output = Dense(1, activation='sigmoid')(dot_product)
    model = Model(input=[input_target, input_context], output=output)
    validation_model = Model(input=[input_target, input_context], output=similarity)
    embedding_model = Model(input=model.input[0], output=target)
    
    return model, validation_model, embedding_model

    
def get_embedding(word, model, dictionary):
    word_key = np.zeros((1,))
    word_key[0,] = dictionary[word]
    embedding = model.predict_on_batch(word_key)
    return embedding[0][0]


def tsne_plot_2d(label, embeddings, name, words, a=1):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, 1))
    x = embeddings[:,0]
    y = embeddings[:,1]
    plt.scatter(x, y, c=colors, alpha=a, label=label)
    for i, word in enumerate(words):
        plt.annotate(word, alpha=0.3, xy=(x[i], y[i]), xytext=(5, 2), 
                     textcoords='offset points', ha='right', va='bottom', size=10)
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig(name+'.png', format='png', dpi=150, bbox_inches='tight')
    #plt.show()


def tsne_plot_3d(title, label, embeddings, name, a=1):
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = cm.rainbow(np.linspace(0, 1, 1))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=colors, alpha=a, label=label)
    plt.legend(loc=4)
    plt.title(title)
    plt.savefig(name+'.png', format='png', dpi=150, bbox_inches='tight')
    #plt.show()

   
############################ Training ############################

# def word2vec():
#     pass
#     #Processing
vocab_size = 31510
words, sentences = pre_process()  #corpus
data, count, dictionary, reverse_dictionary, sentences_encoded = process_data(words, sentences)

#del words  #if short on memory

#important variables
window_size = 2
vector_dim = 300

couples, labels = generate_training_data(window_size, sentences_encoded, vocab_size)
 
#Build Model
model, validation_model, embedding_model = build_model(vocab_size, vector_dim)
    
#Make Training Data
target = np.array([np.array(c[0]).reshape(1,) for c in couples])
context = np.array([np.array(c[1]).reshape(1,) for c in couples])

#Compile and Train Model
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
num_epochs = 30

for i in range(1, num_epochs):
    print('Epoch', str(i+1))
    model.fit([target, context], labels, epochs = 1, validation_split = 0.1, shuffle=True)
    
    if i%5 == 0:  #After every 5 iterations
        #Get embeddings
        print('Obtaining the embeddings..')
        words_abc = []
        embeddings = []
        for word in dictionary:
            embeddings.append(get_embedding(word, embedding_model, dictionary))
            words_abc.append(word)
        
        #Apply TSNE
        #1. 2 components
        print('Applying TSNE with 2 components..')
        tsne_2d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500, random_state=32)
        embeddings_2d = tsne_2d.fit_transform(embeddings[:10000])
        print('Saving 2d plot')
        tsne_plot_2d('NLTK ABC Corpus', embeddings_2d, 'plots/epoch_'+str(i)+'_2d', words_abc[:10000], a=0.1)
    
        #2. 3 components
        print('Applying TSNE with 3 components..')
        tsne_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=500, random_state=12)
        embeddings_wp_3d = tsne_3d.fit_transform(embeddings[:10000])
        print('Saving 3d plot')
        tsne_plot_3d('Visualizing Embeddings using t-SNE', 'NLTK ABC Corpus', embeddings_wp_3d, 'plots/epoch_'+str(i)+'_3d', a=0.1)
            
    
pickle.dump(model, open('model.pkl', 'wb'))
 

################# T-SNE #################

def most_similar(model, word, n, vocab_size, reverse_dictionary):
    print('Finding', str(n), 'most similar words for', word)
    sims = np.ones((vocab_size,))
    word_idx = dictionary[word]
    word_key = np.zeros((1,))
    var_key = np.zeros((1,))
    word_key[0,] = word_idx
    for i in range(vocab_size):
        var_key[0,] = i
        out = model.predict_on_batch([word_key, var_key])
        sims[i] = out
    sims = np.array(sims)
    nearest = (-sims).argsort()[:n]
    nearest_words = [reverse_dictionary[nearest[w]] for w in range(len(nearest))]   
    return nearest_words    

keys = Counter(words).most_common(50)[20:35]
keys = [k[0] for k in keys]

embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words_abc = []
    for similar_word in most_similar(validation_model, word, 20, vocab_size, reverse_dictionary):
        words_abc.append(similar_word)
        embeddings.append(get_embedding(similar_word, embedding_model, dictionary))
    embedding_clusters.append(embeddings)
    word_clusters.append(words_abc)

embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape

tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=1000, random_state=32)
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


tsne_plot_similar_words('Similar words from Google News', keys, embeddings_en_2d, word_clusters, 0.7,
                        'similar_words.png')
   
        