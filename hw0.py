from pyspark import SparkConf, SparkContext
import re
import sys
from pathlib import Path
import string
conf = SparkConf()
sc = SparkContext(conf=conf)

s = set(w.lower() for w in open(r"C:\Users\akash\Desktop\pg1661.txt").read().split())
print(sorted(s))

counter = {}
for w in open(r"C:\Users\akash\Desktop\pg1661.txt").read().split():
    if w in counter:
        counter[w] += 1
    else:
        counter[w] = 1
for word, times in counter.items():
    print("%s was found %d times" % (word, times))

sc.stop()

