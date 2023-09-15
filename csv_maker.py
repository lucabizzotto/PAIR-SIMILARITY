import csv


def create_csv():
    # open the file in the write mode
    f = open('/mnt/c/Users/bizzo/Desktop/AssigmentIII_LMD/analysis.csv', 'w')
    # create the csv writer
    writer = csv.writer(f)
    row = ["data_set", "threshold", "algorithm", "time", "worker"]
    # write a row to the csv file
    writer.writerow(row)
    # close the file
    f.close()


def add_data(row):
    """
    :param row: [data_set, threshold, algorithm, time]
    :return:
    """
    # update csv file
    f = open('/mnt/c/Users/bizzo/Desktop/AssigmentIII_LMD/analysis.csv', 'a')
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    writer.writerow(row)
    # close the file
    f.close()
