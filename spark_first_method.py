import init
import findspark
from pyspark.sql import SparkSession


class SparkFirstMethod(init.Init):
    def __init__(self, file_name, threshold):
        # threshold to valuate similarity
        self.threshold = threshold
        # open a spark session
        spark = SparkSession.builder.master("local").config("spark.ui.port", "4041").appName("hope").getOrCreate()
        # our spark context object
        self.sc = spark.sparkContext
        # threshold for similarity
        # tfidf, doc_key
        super().__init__(file_name)

    def close(self):
        self.sc.stop()

    def summary_doc(self):
        """
        create the data structure we will use in spark context
        :return: [(doc_key, tfidf_doc),.....]
        """
        summary = []
        for doc_key, tf_idf_csr in zip(self.doc_key, self.tf_idf):
            summary.append((doc_key, tf_idf_csr))
        return summary


    @staticmethod
    def foo(x):
        """
        :param x:
        :return: emit [term, (doc_id, tf_idf)]
        """
        l = []
        # x-> (doc_id, tfidf)
        # for each term in the doc
        for i in range(x[1].shape[1]):
            # to take only the values different from 0
            if x[1][0, i] != 0.:
                # (term_id, (id_doc, doc))
                l.append((i, (x[0], x[1])))
        # l -> [term, (doc_id, tf_idf)]
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
        t = myrdd.flatMap(SparkFirstMethod.foo)
        # we need groupByKey() because we want for each term the list of document containing it
        # t -> [(doc_id, tf_idf)]
        #                       cast at list
        # [ (term, [(doc_id, tf_idf) ........,(doc_id, tf_idf)]) ]
        # mapValues it is applied only on values leaving key unchanged 
        group_by = t.groupByKey().mapValues(list)
        # compute cosine similarity
        # we need DISTINCT because many pair of documents are the same
        # since we have for each term the list of documents containing it
        # and documents share terms and the when we compute similarity
        # we obtain many equal pair
        distinct = group_by.flatMap(SparkFirstMethod.compute_cosine).distinct()
        teta = self.threshold
        res = distinct.filter(lambda x: x[2] >= teta)
        result = res.collect()
        return len(result)

