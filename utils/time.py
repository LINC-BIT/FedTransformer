__all__ = [
    "current_datetime_string"
]

import datetime
import time


def current_datetime_string(time_str_format="%Y-%m-%d-%H-%M-%S") -> str:
    _datetime = datetime.datetime.now()
    _str = datetime.datetime.strftime(_datetime, time_str_format)
    return _str


def record_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        cost = round(time.perf_counter() - t, 3)
        # print(f"function {func.__name__} uses time: {cost} s")
        return result, cost
    return fun


@record_time
def func_1():
    time.sleep(3)


if __name__ == '__main__':
    res = func_1()
    print(res[1:])
