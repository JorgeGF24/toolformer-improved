# Python script that starts 3 processes concurrently each of them waits 10 seconds before exiting.

import os
import time
from multiprocessing import Process, cpu_count
import random

def run_process(process, callable):
    print(f"Process {process} started")
    # wait 10 seconds before exiting
    time.sleep(random.randint(1, 10)/10)
    print(callable())
    time.sleep(4)

if __name__ == "__main__":
    processes = ["process1.py", "process2.py", "process3.py"]
    print(cpu_count())
    print(len(os.sched_getaffinity(0)))
    # Define an iterable that goes throuugh the first 100 numbers
    iterable = iter(range(10,220))

    def next_data():
        return next(iterable)
    
    for i in range(200):
        p = Process(target=run_process, args=("process " + str(i), next_data))
        p.start()
