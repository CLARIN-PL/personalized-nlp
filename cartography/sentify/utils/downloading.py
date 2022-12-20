from sys import stderr
from time import sleep
from urllib.error import HTTPError


def safe_from_pretrained(model_class, name, *args, **kwargs):
    while True:
        try:
            model = model_class.from_pretrained(name, *args, **kwargs)
        except (HTTPError, ValueError):
            print("Connection error. Zzz...", file=stderr)
            sleep(10)
        else:
            return model
