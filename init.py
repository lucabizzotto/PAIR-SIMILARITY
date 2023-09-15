import read_data
from sklearn.feature_extraction.text import TfidfVectorizer


# object containing tfidf matrix and doc_id

class Init:

    def __init__(self, file_name):
        # get data
        corpus = read_data.get_data(file_name)
        # Convert a collection of raw documents to a matrix of TF-IDF features
        # save the corpus
        corpus_text = [corpus[key]['text'] for key in corpus.keys()]
        vectorizer = TfidfVectorizer()
        # scipy.sparse._csr.csr_matrix
        self.tf_idf = vectorizer.fit_transform(corpus_text)
        self.doc_key = corpus.keys()

