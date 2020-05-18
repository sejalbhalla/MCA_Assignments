import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    
    beta = 0.8
    gamma = 0.6
    
    vec_queries = vec_queries.toarray()
    vec_docs = vec_docs.toarray()
    
    # alpha = 0.25
    #gamma = np.round(1 - beta, 2)
    for iteration in range(3):            
        for i in range(sim.shape[1]):
            relevant = np.argsort(-sim[:, i])[:n]
            non_relevant = np.argsort(-sim[:, i])[-n:]
            
            add = (beta/n)*np.sum(vec_docs[relevant], axis = 0) - (gamma/n)*np.sum(vec_docs[non_relevant], axis = 0)
            vec_queries[i] += add          
            
        sim = cosine_similarity(vec_docs, vec_queries)
        
    rf_sim = sim # change
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10, k=5):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant
        k: integer
            number of words to be added to the query

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    
    beta = 0.6
    gamma = 0.4
    
    vec_queries = vec_queries.toarray()
    vec_docs = vec_docs.toarray()
    
    all_indices = np.arange(0, len(vec_queries[0])) #helper
    indices = []  #non-zero indices for each query
    for i in range(sim.shape[1]):
        indices.append(np.where(vec_queries[i] > 0)[0])
        
    for _ in range(3):    
        for i in range(sim.shape[1]):
            relevant = np.argsort(-sim[:, i])[:n]      #relevant docs index
            non_relevant = np.argsort(-sim[:, i])[-n:] #non relevant docs index
            
            #Query extension
            all_rel = np.sum(vec_docs[relevant], axis = 0) #sum of relevant docs
            rel_words = np.argsort(-all_rel)               #words in order of relevance from relevant docs
            no_words_added = 0 #no of words added 
            words_added = []   #list of words added
            for w in rel_words:
                if w not in indices[i]:     #word not present
                    words_added.append(w)
                    np.append(indices[i], w)
                    no_words_added += 1
                if no_words_added == k:    #stop if required number of words are added
                    break
                
            index_table = tfidf_model.get_feature_names()                      #words list
            words_added2 = [index_table[w] for w in words_added]               #words corresponding to added indices
            org_words = list(tfidf_model.inverse_transform(vec_queries[i])[0]) #words corresponding to original query
            
            temp = ' '.join(org_words)
            temp2 = ' '.join(words_added2)
            temp = temp + ' '+temp2       #convert the list of words into a string
            
            vec_queries[i] = tfidf_model.transform([temp]).toarray()[0]  #Vectorize using tfidf
            
            zero_indices = np.setdiff1d(all_indices, indices[i])  #Make all other entries 0
            vec_queries[i][zero_indices] = 0
            
            #Vector Adjustment
            add = (beta/n)*np.sum(vec_docs[relevant], axis = 0) - (gamma/n)*np.sum(vec_docs[non_relevant], axis = 0)
            vec_queries[i] += add
            
            
        sim = cosine_similarity(vec_docs, vec_queries)       
            
            
    rf_sim = cosine_similarity(vec_docs, vec_queries)
    #rf_sim = sim  # change
    return rf_sim