# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license
# TODO: Dashboard needs to be tested.

from IPython import display
import re
import requests
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from . import udash

from gevent.pywsgi import WSGIServer
from flask import Flask

import socket
import random

import logging

log = logging.getLogger(__name__)

app = Flask(__name__)
app.logger.disabled = True


class AppRunner:
    def __init__(self, addr=None):
        self.app = DispatcherApp()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        if addr is None:
            # Allocate port
            self.ip = "127.0.0.1"
            self.port = -1
            max_attempts = 10
            for _ in range(max_attempts):
                port = random.randint(7000, 7999)
                if self._local_port_available(port, rais=False):
                    self.port = port
                    log.debug("Found open port: {0}".format(port))
                    break
                else:
                    log.debug("Port already in use: {0}".format(port))

            else:
                msg = "Could not find open port"
                log.error(msg)
                raise RuntimeError(msg)
        else:
            self.ip = addr[0]
            self.port = addr[1]

    def _local_port_available(self, port, rais=True):
        """
        Borrowed from:
        https://stackoverflow.com/questions/19196105/how-to-check-if-a-network-port-is-open-on-linux
        """
        try:
            backlog = 5
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", port))
            sock.listen(backlog)
            sock.close()
            sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            sock.bind(("::1", port))
            sock.listen(backlog)
            sock.close()
        except socket.error:
            if rais:
                raise RuntimeError(
                    "The server is already running on port {0}".format(port)
                )
            else:
                return False
        return True

    def stop(self):
        # Shutdown
        log.debug("Triggering shutdown")
        try:
            url = "http://{0}:{1}/shutdown".format(self.ip, self.port)
            r = requests.post(url)
            log.debug(r)
        except requests.exceptions.RequestException as e:
            log.debug("Dashboard stop failed: {0}".format(e))
            return False

        try:
            if self.future is not None:
                self.future.result(timeout=5)
        except concurrent.futures.TimeoutError as e:
            log.error(e)
            return False

        return True

    def _run(self):
        try:

            class devnull:
                write = lambda _: None  # noqa: E731

            server = WSGIServer((self.ip, self.port), self.app, log=devnull)
            self.app.config["server"] = server
            server.serve_forever()
        except Exception as e:
            log.error(e, exc_info=True)

    def _obj_id(self, obj):
        return str(id(obj))

    def start(self):
        log.debug("Running app runner on: {0}:{1}".format(self.ip, self.port))
        self.future = self.executor.submit(lambda: self._run())

    def register(self, ctx, **kwargs):
        # The path to this instance should be id based.
        self.app.register(ctx, **kwargs)

    def display(self, ctx, width="100%", height=800, open_link=False):
        path = "/" + self._obj_id(ctx) + "/"
        url = "http://{0}:{1}{2}".format(self.ip, self.port, path)
        log.debug("Display URL: {0}".format(url))

        html_str = ""
        if open_link:
            html_str += r'<a href="{url}" target="_new">Open in new window</a>'.format(
                url=url
            )

        html_str += """
            <iframe src="{url}" width={width} height={height} frameBorder="0"></iframe>
        """.format(
            url=url, width=width, height=height
        )

        display.display_html(html_str, raw=True)
        return None


class DispatcherApp:
    def __init__(self):
        self.default_app = Flask(__name__)
        self.pool = {}
        self.config = {}
        self.app_pattern = re.compile(r"/?(.+?)(/|$)")

    def obj_id(self, obj):
        return str(id(obj))

    def register(self, ctx, share_tables=None):
        ctx_id = self.obj_id(ctx)
        if ctx_id not in self.pool:
            log.debug("App Entry not found: {0}".format(ctx_id))
            app = udash.generate_app(
                ctx, {"share_tables": share_tables}, "/" + ctx_id + "/"
            )
            app.css.config.serve_locally = True
            app.scripts.config.serve_locally = True

            self.pool[ctx_id] = app.server
        else:
            log.debug("App Entry found: {0}".format(ctx_id))

    def _split(self, strng, sep, pos):
        strng = strng.split(sep)
        return sep.join(strng[:pos]), sep.join(strng[pos:])

    def __call__(self, environ, start_response):
        old_path_info = environ.get("PATH_INFO", "")

        if old_path_info == "/shutdown":
            log.debug("Shutting down.")
            server = self.config["server"]
            server.stop()
            start_response("200 OK", [("content-type", "text/html")])
            return ["Shutdown".encode("utf-8")]

        match = re.search(self.app_pattern, old_path_info)
        if match is None:
            msg = "URL not supported: {0}".format(old_path_info)
            log.error(msg)
            start_response("400 Bad Request Error", [("content-type", "text/html")])
            return [msg.encode("utf-8")]

        ctx_id = match.group(1)
        app = self.pool[ctx_id]
        return app(environ, start_response)
