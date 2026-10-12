from multiprocessing import Pool
import time

import snake


def sweep():
    process_arr = []
    with Pool(processes=10) as pool:
        for i in range(5):
            process_arr.append(pool.apply_async(snake.main))
        for i in process_arr:
            i.get()


if __name__ == "__main__":
    start_time = time.time()
    sweep()
    print("--- %s seconds ---" % (time.time() - start_time))
