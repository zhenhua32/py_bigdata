#! /usr/bin/env python3

import re
from pyspark import SparkContext


if __name__ == "__main__":
    sc = SparkContext(appName="Word Count")
    PAT = re.compile(r"[-./:\s\xa0]+")
    fp = "./night.txt"
    text_file = sc.textFile(fp)
    xs = text_file.flatMap(lambda x: PAT.split(x)).filter(lambda x: len(x) > 6).countByValue()

    for k, v in xs.items():
        print("{:<30}{}".format(k, v))
