import numpy as np
import time
import operator

sizes = [2**i for i in range(11, 17)]

def vec_vec_performance(size: int, func):
    start_time = time.time()

    v1 = np.random.uniform(0, 1, size)
    v2 = np.random.uniform(0, 1, size)


    func(v1, v2)

    end_time = time.time()

    duration = end_time - start_time

    print(f"size= {size:>6}k | operation= {func.__name__:<5}| time= {duration * 1000:5.2f} ms")

v_v_operators = [
    operator.add,
    operator.sub,
    operator.iadd
]

for op in v_v_operators:
    print(f"\n--------------------- {op.__name__.upper():<4} ---------------------")
    for size in sizes:
        vec_vec_performance(size, op)


def vec_scalar_performance(size: int, func):
    start_time = time.time()
    v1 = np.random.uniform(0, 1, size)

    func(67, v1)
    end_time = time.time()

    duration = end_time - start_time
    print(f"size= {size:>6}k | operation= {func.__name__:<5}| time= {duration * 1000:5.2f} ms")

v_s_operators = [
    operator.mul,
    operator.imul,
]

for op in v_s_operators:
    print(f"\n--------------------- {op.__name__.upper():<4} ---------------------")
    for size in sizes:
        vec_scalar_performance(size, op)