# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from abc import ABC, abstractmethod
import logging

from ..utils.environment import EnvironmentDetector, is_cloud_env, ENV_DETECTED
from warnings import warn

from ..version import __version__

JS_URL = "https://unpkg.com/@interpretml/interpret-inline@{}/dist/interpret-inline.js".format(
    __version__
)

log = logging.getLogger(__name__)


class VisualizeProvider(ABC):
    @abstractmethod
    def render(self, explanation, key=-1, **kwargs):
        pass  # pragma: no cover


class AutoVisualizeProvider(VisualizeProvider):
    def __init__(self, app_runner=None, **kwargs):
        self.has_initialized = False
        self.environment_detector = None
        self.in_cloud_env = ENV_DETECTED
        self.provider = None
        self.app_runner = app_runner
        self.kwargs = kwargs

    def _lazy_initialize(self):
        self.environment_detector = EnvironmentDetector()
        detected_envs = self.environment_detector.detect()
        self.in_cloud_env = is_cloud_env(detected_envs)

        # NOTE: This is tested manually per release. Ignoring for coverage.
        if self.in_cloud_env == ENV_DETECTED.CLOUD:  # pragma: no cover
            log.info("Detected cloud environment.")
            self.provider = InlineProvider(detected_envs=detected_envs, js_url=JS_URL)
        elif "docker-dev-mode" in detected_envs:
            log.info("Operating in docker development mode.")
            self.provider = InlineProvider(detected_envs=detected_envs)
        elif self.in_cloud_env == ENV_DETECTED.BOTH_CLOUD_AND_NON_CLOUD:
            log.info("Detected both cloud and non cloud environment.")
            # val = input("Type 'C' if you want to choose Cloud environment or 'NC' for Non Cloud Environment :")
            val = 'C'
            if val == 'C':
                self.provider = InlineProvider(detected_envs=detected_envs, js_url=JS_URL)
            else:
                if self.app_runner:
                    self.provider = DashProvider(self.app_runner)
                else:
                    self.provider = DashProvider.from_address()
        else: # ENV_DETECTED.NON_CLOUD
            log.info("Detected non-cloud environment.")
            if self.app_runner:
                self.provider = DashProvider(self.app_runner)
            else:
                self.provider = DashProvider.from_address()

    def render(self, explanation, key=-1, **kwargs):
        if not self.has_initialized:
            self._lazy_initialize()
            self.has_initialized = True

        self.provider.render(explanation, key=key, **kwargs)


class PreserveProvider(VisualizeProvider):
    def render(self, explanation, key=-1, **kwargs):
        file_name = kwargs.pop("file_name", None)

        # NOTE: Preserve didn't support returning everything. If key is -1 default to key is None.
        # This is for backward-compatibility. All of this will be deprecated shortly anyway.
        if key == -1:
            key = None

        # Get visual object
        visual = explanation.visualize(key=key)

        # Output to front-end/file
        self._preserve_output(
            explanation.name, visual, selector_key=key, file_name=file_name, **kwargs
        )
        return None

    def _preserve_output(
        self, explanation_name, visual, selector_key=None, file_name=None, **kwargs
    ):
        from plotly.offline import iplot, plot, init_notebook_mode
        from IPython.display import display, display_html
        from base64 import b64encode

        from plotly import graph_objs as go
        from pandas.core.generic import NDFrame
        import dash.development.base_component as dash_base

        init_notebook_mode(connected=True)

        def render_html(html_string):
            base64_html = b64encode(html_string.encode("utf-8")).decode("ascii")
            final_html = """<iframe src="data:text/html;base64,{data}" width="100%" height=400 frameBorder="0"></iframe>""".format(
                data=base64_html
            )
            display_html(final_html, raw=True)

        if visual is None:  # pragma: no cover
            msg = "No visualization for explanation [{0}] with selector_key [{1}]".format(
                explanation_name, selector_key
            )
            log.error(msg)
            if file_name is None:
                render_html(msg)
            else:
                pass
            return False

        if isinstance(visual, go.Figure):
            if file_name is None:
                iplot(visual, **kwargs)
            else:
                plot(visual, filename=file_name, **kwargs)
        elif isinstance(visual, NDFrame):
            if file_name is None:
                display(visual, **kwargs)
            else:
                visual.to_html(file_name, **kwargs)
        elif isinstance(visual, str):
            if file_name is None:
                render_html(visual)
            else:
                with open(file_name, "w") as f:
                    f.write(visual)
        elif isinstance(visual, dash_base.Component):  # pragma: no cover
            msg = "Preserving dash components is currently not supported."
            if file_name is None:
                render_html(msg)
            log.error(msg)
            return False
        else:  # pragma: no cover
            msg = "Visualization cannot be preserved for type: {0}.".format(
                type(visual)
            )
            if file_name is None:
                render_html(msg)
            log.error(msg)
            return False

        return True


class DashProvider(VisualizeProvider):
    """ Provides rendering via Plotly's Dash.

    This works in the event of an environment that can expose HTTP(s) ports.
    """
    def __init__(self, app_runner):
        """ Initializes class.

        This requires an instantiated `AppRunner`, call `.from_address` instead
        to initialize both.

        Args:
            app_runner: An AppRunner instance.
        """
        self.app_runner = app_runner

    @classmethod
    def from_address(cls, addr=None, base_url=None, use_relative_links=False):
        """ Initialize a new `AppRunner` along with the provider.

        Args:
            addr: A tuple that is (ip_addr, port).
            base_url: Base URL, this useful when behind a proxy.
            use_relative_links: Relative links for rendered pages instead of full URI.
        """
        from ..visual.dashboard import AppRunner

        app_runner = AppRunner(
            addr=addr, base_url=base_url, use_relative_links=use_relative_links
        )
        return cls(app_runner)

    def idempotent_start(self):
        status = self.app_runner.status()
        if not status["thread_alive"]:
            self.app_runner.start()

    def link(self, explanation, **kwargs):
        self.idempotent_start()

        # Register
        share_tables = kwargs.pop("share_tables", None)
        self.app_runner.register(explanation, share_tables=share_tables)

        url = self.app_runner.display_link(explanation)
        return url

    def render(self, explanation, **kwargs):
        self.idempotent_start()

        # Register
        share_tables = kwargs.pop("share_tables", None)
        self.app_runner.register(explanation, share_tables=share_tables)

        # Display
        open_link = isinstance(explanation, list)
        self.app_runner.display(explanation, open_link=open_link)


class InlineProvider(VisualizeProvider):
    """ Provides rendering via JavaScript that are invoked within Jupyter cells."""

    def __init__(self, detected_envs=None, js_url=None):
        """ Initializes class.

        Args:
            detected_envs: Environments targetted as defined in `interpret.utils.environment`.
            js_url: If defined, will load the JavaScript bundle for interpret-inline from the given URL.
        """
        self.detected_envs = [] if detected_envs is None else detected_envs
        self.js_url = js_url

    def render(self, explanation, key=-1, **kwargs):
        from ..visual.inline import render

        render(
            explanation,
            default_key=key,
            detected_envs=self.detected_envs,
            js_url=self.js_url,
        )
