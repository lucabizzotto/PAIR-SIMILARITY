import spark_second_method


class SparkThirdMethod(spark_second_method.SparkSecondMethod):

    def __init__(self, file_name, threshold):
        super().__init__(file_name, threshold)

    @staticmethod
    def compute_cosine(x):
        l = []
        # term -> list of documents containing that term
        # x[1] -> list[ (term, [ (doc_id, tf_idf, b_d) ]) ]
        # for each i_doc in the list
        for i in range(len(x[1])):
            # for each j_doc > i_doc (we are calculating only the upper bound of the similarity matrix since
            # sim(doc_i, doc_j) == sim(doc_j, doc_i))
            for j in range(i + 1, len(x[1])):
                # doc_i and doc_j for sure they have a term in common
                # only the reducer that have the "smaller term" will do the cosine similarity
                # find the  term in common from max
                # tf_idf more frequent order
                #            max b_d from two doc term they have in common
                for z in range(max(x[1][i][2], x[1][j][2]), x[1][i][1].shape[1]):
                    # min in common
                    if x[1][i][1][0, z] != 0. and x[1][j][1][0, z] != 0.:
                        # equal to the term min(d_i intersection d_j)
                        # for sure is after max(b(d_i), b(d_j))
                        if z == x[0]:
                            cosine_score = x[1][i][1].dot(x[1][j][1].transpose())
                            l.append((x[1][i][0], x[1][j][0], cosine_score[0, 0]))
                        # we want the minimun
                        break
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
        t = myrdd.flatMap(SparkThirdMethod.foo)
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
        distinct = group_by.flatMap(SparkThirdMethod.compute_cosine).distinct()
        # accumulator we use to keep track of the similar doc
        similarity_doc = self.sc.accumulator(0)
        teta = self.threshold
        res = distinct.filter(lambda x: x[2] >= teta)
        result = res.collect()
        return len(result)

