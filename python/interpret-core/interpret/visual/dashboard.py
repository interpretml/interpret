# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from IPython import display
import re
import requests
import threading
import os
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


def _build_path(path, base_url=None):
    if base_url:
        return "{0}/{1}".format(base_url, path)
    else:
        return path


class AppRunner:
    def __init__(self, addr=None, base_url=None, use_relative_links=False):
        self.app = DispatcherApp(
            base_url=base_url, use_relative_links=use_relative_links
        )
        self.base_url = base_url
        self.use_relative_links = use_relative_links
        self._thread = None

        if addr is None:
            # Allocate port
            self.ip = "127.0.0.1"
            self.port = -1
            max_attempts = 10
            for _ in range(max_attempts):
                port = random.randint(7000, 7999)
                if self._local_port_available(port, rais=False):
                    self.port = port
                    log.info("Found open port: {0}".format(port))
                    break
                else:  # pragma: no cover
                    log.info("Port already in use: {0}".format(port))

            else:  # pragma: no cover
                msg = """Could not find open port.
                Consider calling `interpret.set_show_addr(("127.0.0.1", 7001))` first.
                """
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
        except socket.error:  # pragma: no cover
            if rais:
                raise RuntimeError(
                    "The server is already running on port {0}".format(port)
                )
            else:
                return False
        return True

    def stop(self):

        # Shutdown
        if self._thread is None:
            return True

        log.info("Triggering shutdown")
        try:
            path = _build_path("shutdown")
            url = "http://{0}:{1}/{2}".format(self.ip, self.port, path)
            r = requests.post(url)
            log.debug(r)
        except requests.exceptions.RequestException as e:  # pragma: no cover
            log.info("Dashboard stop failed: {0}".format(e))
            return False

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                log.error("Thread still alive despite shutdown called.")
                return False

            self._thread = None

        return True

    def _run(self):
        try:

            class devnull:
                write = lambda _: None  # noqa: E731

            server = WSGIServer((self.ip, self.port), self.app, log=devnull)
            self.app.config["server"] = server
            server.serve_forever()
        except Exception as e:  # pragma: no cover
            log.error(e, exc_info=True)

    def _obj_id(self, obj):
        return str(id(obj))

    def start(self):
        log.info("Running app runner on: {0}:{1}".format(self.ip, self.port))

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def ping(self):
        """ Returns true if web service reachable, otherwise False."""

        try:
            path = _build_path("")
            url = "http://{0}:{1}/{2}".format(self.ip, self.port, path)
            requests.get(url)
            log.info("Dashboard ping succeeded")
            return True
        except requests.exceptions.RequestException as e:  # pragma: no cover
            log.info("Dashboard ping failed: {0}".format(e))
            return False

    def status(self):
        status_dict = {}
        status_dict["addr"] = self.ip, self.port
        status_dict["base_url"] = self.base_url
        status_dict["use_relative_links"] = self.use_relative_links
        status_dict["thread_alive"] = self._thread.is_alive() if self._thread else False

        http_reachable = self.ping()
        status_dict["http_reachable"] = http_reachable

        return status_dict

    def register(self, ctx, **kwargs):
        # The path to this instance should be id based.
        self.app.register(ctx, **kwargs)

    def display_link(self, ctx):
        obj_path = self._obj_id(ctx) + "/"
        path = (
            obj_path
            if self.base_url is None
            else "{0}/{1}".format(self.base_url, obj_path)
        )
        start_url = (
            "/"
            if self.use_relative_links
            else "http://{0}:{1}/".format(self.ip, self.port)
        )

        url = "{0}{1}".format(start_url, path)
        log.info("Display URL: {0}".format(url))

        return url

    def display(self, ctx, width="100%", height=800, open_link=False):
        url = self.display_link(ctx)

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
    def __init__(self, base_url=None, use_relative_links=False):
        self.base_url = base_url
        self.use_relative_links = use_relative_links

        self.root_path = "/"
        self.shutdown_path = "/shutdown"
        self.favicon_path = "/favicon.ico"
        self.favicon_res = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "favicon.ico"
        )

        self.default_app = Flask(__name__)
        self.pool = {}
        self.config = {}
        if self.base_url is None:
            self.app_pattern = re.compile(r"/?(.+?)(/|$)")
        else:
            self.app_pattern = re.compile(
                r"/?(?:{0}/)?(.+?)(/|$)".format(self.base_url)
            )

    def obj_id(self, obj):
        return str(id(obj))

    def register(self, ctx, share_tables=None):
        ctx_id = self.obj_id(ctx)
        if ctx_id not in self.pool:
            log.info("Creating App Entry: {0}".format(ctx_id))
            ctx_path = (
                "/{0}/".format(ctx_id)
                if self.base_url is None
                else "/{0}/{1}/".format(self.base_url, ctx_id)
            )
            app = udash.generate_app(
                ctx,
                {"share_tables": share_tables},
                # url_base_pathname=ctx_path,
                requests_pathname_prefix=ctx_path,
                routes_pathname_prefix=ctx_path,
            )
            app.css.config.serve_locally = True
            app.scripts.config.serve_locally = True

            self.pool[ctx_id] = app.server
        else:
            log.debug("App Entry found: {0}".format(ctx_id))

    def __call__(self, environ, start_response):
        path_info = environ.get("PATH_INFO", "")
        script_name = environ.get("SCRIPT_NAME", "")
        log.debug("PATH INFO  : {0}".format(path_info))
        log.debug("SCRIPT NAME: {0}".format(script_name))

        try:

            if path_info == self.root_path:
                log.info("Root path requested.")
                start_response("200 OK", [("content-type", "text/html")])
                content = self._root_content()
                return [content.encode("utf-8")]

            if path_info == self.shutdown_path:
                log.info("Shutting down.")
                server = self.config["server"]
                server.stop()
                start_response("200 OK", [("content-type", "text/html")])
                return ["Shutdown".encode("utf-8")]

            if path_info == self.favicon_path:
                log.info("Favicon requested.")

                start_response("200 OK", [("content-type", "image/x-icon")])
                with open(self.favicon_res, "rb") as handler:
                    return [handler.read()]

            match = re.search(self.app_pattern, path_info)
            if match is None or self.pool.get(match.group(1), None) is None:
                msg = "URL not supported: {0}".format(path_info)
                log.error(msg)
                start_response("400 BAD REQUEST ERROR", [("content-type", "text/html")])
                return [msg.encode("utf-8")]

            ctx_id = match.group(1)
            log.info("Routing request: {0}".format(ctx_id))
            app = self.pool[ctx_id]
            if self.base_url and not environ["PATH_INFO"].startswith(
                "/{0}".format(self.base_url)
            ):
                log.info("No base url in path. Rewrite to include in path.")
                environ["PATH_INFO"] = "/{0}{1}".format(
                    self.base_url, environ["PATH_INFO"]
                )

            return app(environ, start_response)

        except Exception as e:  # pragma: no cover
            log.error(e, exc_info=True)
            try:
                start_response(
                    "500 INTERNAL SERVER ERROR", [("Content-Type", "text/plain")]
                )
            except Exception:
                pass
            return [
                "Internal Server Error caught by Dispatcher. See logs if available.".encode(
                    "utf-8"
                )
            ]

    def _root_content(self):
        body = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Backend Server</title>
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
                    r'<li><a href="{0}">{1}</a></li>'.format(
                        "/{0}/".format(key)
                        if self.base_url is None
                        else "/{0}/{1}/".format(self.base_url, key),
                        key,
                    )
                    for key in self.pool.keys()
                ]
            )

        content = Template(body).substitute(list=items)
        return content
