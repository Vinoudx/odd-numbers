import random
import sys
import time
import csv

random.seed(time.time())


with open("./valid.csv", 'w') as f:
    writer = csv.writer(f, delimiter=' ')

    for _ in range(2000):
        ran = random.randint(0, sys.maxsize)
        label = 1 if ran % 2 == 0 else 0
        writer.writerow([ran, label])

