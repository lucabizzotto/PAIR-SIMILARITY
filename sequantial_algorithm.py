import init
import numpy as np
from tqdm import tqdm


# subclass of init.Init
class SequentialAlgorithm(init.Init):

    def __init__(self, file_name, threshold):
        self.threshold = threshold
        super().__init__(file_name)

    def summary_sort_doc(self):
        """
        :return: list[(doc_key, tf_idf, len_tf_idf)] sorted by "length of the doc"
        """
        # key = doc_id, value = tf_idf in CSR format
        summary = []
        for doc_key, tf_idf_csr in zip(self.doc_key, self.tf_idf):
            # count entry different from 0 to know "length" of each doc
            summary.append((doc_key, (tf_idf_csr, len(tf_idf_csr.nonzero()[0]))))
        # sort in descending order by "length" of document corpus
        # summary list -> (doc_key, (csr_tfidf, "length" of doc))
        summary_sorted = sorted(summary, key=lambda x: x[1][1], reverse=True)
        return summary_sorted

    @staticmethod
    def compute_k(sort_documents, index, threshold):
        """
        compute the least length the doc must have to overtake the threshold requirement
        :param sort_documents:
        :param index:
        :param threshold:
        :return: the least length the doc must have to overtake the threshold requirement
        """
        #                   (doc_id,(csr, length))
        tf_idf = sort_documents[index][1][0]
        # get value different from 0 of tf_idf
        tf_idf = tf_idf[tf_idf.nonzero()].copy()
        # sort
        tf_idf.sort(axis=1)
        # we want in decreasing order
        tf_idf = tf_idf.flat[::-1]
        # our counter
        k = 0
        for i in tf_idf.flat:
            # inner product is maximized when the vector are equal
            threshold -= np.power(i, 2)
            k += 1
            if threshold >= 0:
                continue
            return k

    def count_similar_doc(self):
        """
        count the similarity doc we find
        :return: the number of similar  doc
        """
        cosine_list = []
        similar_doc = 0
        # list containing all the documents sorted by length
        # (doc_key, (csr_tfidf, "length" of doc))
        sort_documents = self.summary_sort_doc()
        # doc sorted by length

        for i in tqdm(range(len(sort_documents))):
            # calculate k
            # k is the minimum number of doc length we need to overtake the threshold
            teta = self.threshold
            k = self.compute_k(sort_documents, i, teta)
            # d(x,y) = d(y,x) is symmetric so we explore only the upper bound of the similarity matrix
            for j in range(i + 1, len(sort_documents)):
                # if the 'length' of the document is not enough long skipp all the rest od documents that are shorter
                # since they cannot overtake the threshold we fixed
                if sort_documents[j][1][1] >= k:
                    cos_similarity = sort_documents[i][1][0].dot(np.transpose(sort_documents[j][1][0]))
                    # print(f'd1: {sort_documents[i][0]}, d2: {sort_documents[j][0]} cos_score {cos_similarity}')
                    if cos_similarity >= self.threshold:
                        similar_doc += 1
                else:
                    break
        return similar_doc
