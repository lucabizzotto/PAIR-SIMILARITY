import create_data_set
import csv_maker
import read_data
import sequantial_algorithm
import init
import spark_first_method
import spark_second_method
import spark_third_method
from tqdm import tqdm
import numpy as np
import time
import sys

import pyspark
from pyspark.sql import SparkSession

if __name__ == '__main__':

    num_worker = 1# sys.argv[1]
    # threshold we use
    threshold = [0.15, 0.55, 0.95]
    # datasets
    data_set = ["corpus_100.npy", "corpus_200.npy", "corpus_500.npy"]

    for d in tqdm(data_set):
        for teta in threshold:
            # sequantial_algorithm
            seq = sequantial_algorithm.SequentialAlgorithm(d,teta)
            start = time.time()
            seq_res = seq.count_similar_doc()
            print(f'{time.time() - start}  {seq_res}')

            # first method
            first = spark_first_method.SparkFirstMethod(d, teta)
            start = time.time()
            first_res = first.compute_similarity()
            # time used
            t = time.time() - start
            row = [d, teta, "first_method", t, num_worker]
            csv_maker.add_data(row)
            print(f'{time.time() - start}  {first_res}')
            first.close()

            # second method
            second = spark_second_method.SparkSecondMethod(d, teta)
            start = time.time()
            res_second = second.compute_similarity()
            # time used
            t = time.time() - start
            print(f'{time.time() - start}  {res_second}')
            row = [d, teta, "second_method", t, num_worker]
            csv_maker.add_data(row)
            second.close()

            # third method
            third = spark_third_method.SparkThirdMethod(d, teta)
            start = time.time()
            res_third = third.compute_similarity()
            # time used
            t = time.time() - start
            print(f'{time.time() - start} {res_third}')
            row = [d, teta, "third_method", t, num_worker]
            csv_maker.add_data(row)
            third.close()
            break
        break


