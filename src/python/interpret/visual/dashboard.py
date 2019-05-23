# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license
# TODO: Dashboard needs to be tested.

from IPython import display
import re
import requests
import threading
from . import udash

from gevent.pywsgi import WSGIServer
from flask import Flask

import socket
import random
from string import Template

import logging

log = logging.getLogger(__name__)

app = Flask(__name__)
app.logger.disabled = True


class AppRunner:
    def __init__(self, addr=None):
        self.app = DispatcherApp()

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

        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            log.error("Thread still alive despite shutdown called.")
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

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def register(self, ctx, **kwargs):
        # The path to this instance should be id based.
        self.app.register(ctx, **kwargs)

    def display(self, ctx, width="100%", height=800, open_link=False):
        path = "/" + self._obj_id(ctx) + "/"
        url = "http://{0}:{1}{2}".format(self.ip, self.port, path)
        log.debug("Display URL: {0}".format(url))

        html_str = "<!-- {0} -->\n".format(url)
        if open_link:
            html_str += r'<a href="{url}" target="_new">Open in new window</a>'.format(
                url=url
            )

        html_str += """<iframe src="{url}" width={width} height={height} frameBorder="0"></iframe>""".format(
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

        if old_path_info == "/":
            log.debug("Root path requested")

            start_response("200 OK", [("content-type", "text/html")])
            content = self._root_content()
            return [content.encode("utf-8")]

        if old_path_info == "/shutdown":
            log.debug("Shutting down.")
            server = self.config["server"]
            server.stop()
            start_response("200 OK", [("content-type", "text/html")])
            return ["Shutdown".encode("utf-8")]

        match = re.search(self.app_pattern, old_path_info)
        if match is None or self.pool.get(match.group(1), None) is None:
            msg = "URL not supported: {0}".format(old_path_info)
            log.error(msg)
            start_response("400 Bad Request Error", [("content-type", "text/html")])
            return [msg.encode("utf-8")]

        ctx_id = match.group(1)
        app = self.pool[ctx_id]
        return app(environ, start_response)

    def _root_content(self):
        body = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Visualization Backend</title>
</head>
<style>
body {
    background-color: white;
    margin: 0;
    padding: 0;
    min-height: 100vh;
}
.banner {
    height: 65px;
    margin: 0;
    padding: 0;
    background-color: rgb(20, 100, 130);
    box-shadow: rgba(0, 0, 0, 0.1) 1px 2px 3px 0px;
}
.banner h2{
    color: white;
    margin-top: 0px;
    padding: 15px 0;
    text-align: center;
    font-family: Georgia, Times New Roman, Times, serif;
}
.app {
    background-color: rgb(245, 245, 250);
    min-height: 100vh;
    overflow: hidden;
}
.card-header{
    padding-top: 12px;
    padding-bottom: 12px;
    padding-left: 20px;
    padding-right: 20px;
    position: relative;
    line-height: 1;
    border-bottom: 1px solid #eaeff2;
    background-color: rgba(20, 100, 130, 0.78);
}
.card-body{
    padding-top: 30px;
    padding-bottom: 30px;
    position: relative;
    padding-left: 20px;
    padding-right: 20px;
}
.card-title{
    display: inline-block;
    margin: 0;
    color: #ffffff;
}
.card {
    border-radius: 3px;
    background-color: white;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    border: 1px solid #d1d6e6;
    margin: 30px 20px;
}
.link-container {
    text-align: center;
}
.link-container ul {
    display: inline-block;
    margin: 0px;
    padding: 0px;
}
.link-container li {
    display: block;
    padding: 15px;
}
.center {
    position: absolute;
    left: 50%;
    top: 50%;
    -webkit-transform: translate(-50%, -50%);
    transform: translate(-50%, -50%);
}
</style>
<body>
<div class="app">
    <div class="banner"><h2>Backend Server</h2></div>
    <div class="card">
        <div class="card-header">
            <div class="card-title"><div class="center">Active Links</div></div>
        </div>
        <div class="card-body">
            <div class="link-container">
                <ul>
                    $list
                </ul>
            </div>
        </div>
    </div>
</div>
</body>
</html>
"""
        if not self.pool:
            items = "<li>No active links.</li>"
        else:
            items = "\n".join(
                [
                    r'<li><a href="/{0}/">{0}</a></li>'.format(key)
                    for key in self.pool.keys()
                ]
            )
        content = Template(body).substitute(list=items)
        return content
