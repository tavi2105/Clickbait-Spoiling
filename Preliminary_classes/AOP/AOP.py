import logging
import time

import aspectlib, socket, sys
import aspectlib.debug


def time_log():
    @aspectlib.Aspect
    def time_log_aspect(*args):
        st = time.time()
        # logging.info("Execution started...")
        yield aspectlib.Proceed
        et = time.time()
        elapsed_time = et - st
        logging.info("Execution finished. Time: "+str(elapsed_time) )

    return time_log_aspect


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
with aspectlib.weave(
        socket.socket,
        [time_log(),aspectlib.debug.log(
            module=False,
            use_logging="INFO",
         print_to=sys.stdout,
         stacktrace=None,
    )],
        lazy=True,
):
    s = socket.socket()
    s.connect(('example.com', 80))
    s.send(b'GET / HTTP/1.1\r\nHost: example.com\r\n\r\n')
    s.recv(8)
    s.close()
