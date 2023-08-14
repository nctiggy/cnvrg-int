from subprocess import Popen, PIPE, STDOUT
from threading import Thread
from cnvrgv2 import Endpoint


def is_tag(stdout):
    spl_str = stdout.split("_")
    if "".join(spl_str[:2]) == "cnvrgtag":
        return spl_str[2:]


def deal_with_stdout(process):
    for line in process.stdout:
        tag = is_tag(line.decode("utf-8"))
        if tag:
            kv = "_".join(tag).split(":")
            key = kv[0].strip()
            value = float(kv[1].strip())
            print(f'key: {key}\nvalue: {value}')
            if ep:
                ep.log_metric(key, value)


def predict(*args):
    p = Popen(["./main"], stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    t = Thread(target=deal_with_stdout, args=(p,), daemon=True)
    t.start()
    t.join()
    return "Success"


if __name__ == "__main__":
    ep = None
    predict()
else:
    ep = Endpoint()
