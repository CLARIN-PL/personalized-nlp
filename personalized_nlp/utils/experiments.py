from itertools import product

def product_kwargs(kwargs):
    kwargs_listed = {}
    for k in kwargs:
        if isinstance(kwargs[k], list):
            kwargs_listed[k] = list(kwargs[k])
        else:
            kwargs_listed[k] = [kwargs[k]]

    return list(dict(zip(kwargs_listed.keys(), values)) for values in product(*kwargs_listed.values()))