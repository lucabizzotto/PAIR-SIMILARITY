import spark_first_method
import numpy as np


class SparkSecondMethod(spark_first_method.SparkFirstMethod):

    def __init__(self, file_name, threshold):
        super().__init__(file_name, threshold)
        # calculate the mean for each column
        mean = self.tf_idf.mean(0)
        # argsort return index sorted
        index_term = np.argsort(mean)
        # we want decreasing order
        # index we have to follow
        index_term = index_term.flat[::-1]
        # easier to use
        my_new_index = [i for i in index_term.flat]
        # rearrange the order
        # it return a copy of tf_idf
        tf_idf_sorted = self.tf_idf[:, my_new_index]
        # get d*
        d_star = tf_idf_sorted.max(axis=0)
        # CALCULATE b_d
        # l [(doc_id, (tf_idf, b(d)))]
        # my new triple we will work in second method
        my_new_list = []
        # for each doc
        for i, doc_key in enumerate(self.doc_key):
            # threshold
            still = self.threshold
            b_d = 0
            # accessing the doc term by term by the decreasing tfidf
            for term in range(tf_idf_sorted[i].shape[1]):
                # find b(d)
                still = still - (tf_idf_sorted[i, term] * d_star.tocsr()[0, term])
                if still >= 0.:
                    b_d += 1
                else:
                    #                   doc_id                tf_idf          b_d
                    my_new_list.append((doc_key, (tf_idf_sorted[i], b_d)))
                    break
        self.triple_summary = my_new_list

    def summary_doc(self):
        """
        create the data structure we will use in spark context
        :return: [(doc_key, tfidf_doc),.....]
        """
        return self.triple_summary


    @staticmethod
    def foo(x):
        l = []
        # i have to check if there is some term after the b_d
        # otherwise we will not compute similarity
        # we want only the term after the b_d
        #              b_d         till the end
        for i in range(x[1][1], x[1][0].shape[1]):
            if x[1][0][0, i] != 0.:
                # (term_id, (id_doc, doc, b_d))
                l.append((i, (x[0], x[1][0], x[1][1])))
        return l


    @staticmethod
    def compute_cosine(x):
        """
        compute cosine similarity
        :param x:
        :return:
        """
        l = []
        # term -> list of documents containing that term
        # x[1] -> list[ (term, [ (doc_id, tf_idf) ]) ]
        # for each i_doc in the list
        for i in range(len(x[1])):
            # for each j_doc > i_doc (we are calculating only the upper bound of the similarity matrix since
            # sim(doc_i, doc_j) == sim(doc_j, doc_i))
            for j in range(i + 1, len(x[1])):
                cosine_score = x[1][i][1].dot(x[1][j][1].transpose())
                l.append((x[1][i][0], x[1][j][0], cosine_score[0, 0]))
        return l

    def compute_similarity(self):
        """
        it calculate the number of similarity documents in respect to a given threshold
        using spark
        :return:
        """
        summary_doc = self.summary_doc()
        # x -> [ (doc_id, tf_idf) ]
        # create rdd object
        myrdd = self.sc.parallelize(summary_doc)
        # x[0] key, x[1] value
        # transformations it create a new rdd
        # we need flatMap to get the result we want
        t = myrdd.flatMap(SparkSecondMethod.foo)
        # we need groupByKey() because we want for each term the list of document containing it
        # t -> [(doc_id, tf_idf)]
        #                       cast at list
        # [ (term, [(doc_id, tf_idf) ........,(doc_id, tf_idf)]) ]
        group_by = t.groupByKey().mapValues(list)
        # compute cosine similarity
        # we need DISTINCT because many pair of documents are the same
        # since we have for each term the list of documents containing it
        # and documents share terms and the when we compute similarity
        # we obtain many equal pair
        distinct = group_by.flatMap(SparkSecondMethod.compute_cosine).distinct()
        # accumulator we use to keep track of the similar doc
        similarity_doc = self.sc.accumulator(0)
        teta = self.threshold
        res = distinct.filter(lambda x: x[2] >= teta)
        result = res.collect()
        return len(result)








