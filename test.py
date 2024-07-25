#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example of thread pool
https://docs.python.org/3/library/concurrent.futures.html
https://docs.python.org/3/library/multiprocessing.html
"""
import concurrent.futures as confu
import multiprocessing.pool as mpp
import time


def slow_square_even(x):
    # print('calculating: {}'.format(x))
    if x % 2:
        raise ValueError("Error: {} is odd number!".format(x))
    time.sleep(x / 4.0)
    return x ** 2


def confu_tpe(tasks, max_workers):
    print('concurrent.futures.ThreadPoolExecutor')
    with confu.ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(slow_square_even, x) for x in tasks]
        for future in confu.as_completed(futures):
            try:
                print(future.result())
            except ValueError as e:
                print(e)


def mpp_tp(tasks, max_workers):
    print('multiprocessing.pool.ThreadPool')
    with mpp.ThreadPool(max_workers) as pool:
        results = [pool.apply_async(slow_square_even, [x]) for x in tasks]
        for async_result in results:
            try:
                print(async_result.get())
            except ValueError as e:
                print(e)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', type=int, default=2)
    args = parser.parse_args()

    tasks = list(reversed(range(6)))
    confu_tpe(tasks, args.jobs)
    mpp_tp(tasks, args.jobs)