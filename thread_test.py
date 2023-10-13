from cnvrg_endpoint_binary import EndpointThread
from time import sleep
import random
from datetime import datetime


def constant_loop(**kwargs):
    text = kwargs.get("text", "default_text")
    endpoint = kwargs["endpoint"]
    while (True):
        random.seed(datetime.now().timestamp())
        print(text)
        try:
            endpoint.log_metric(text, random.random())
        except AttributeError:
            pass
        sleep(5)


def predict(arg):
    return "OK"


args = {}
args["text"] = "data_point"
et = EndpointThread(function_name=constant_loop, function_kwargs=args)
et.run_thread()
