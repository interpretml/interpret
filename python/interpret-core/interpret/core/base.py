# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from abc import abstractmethod, ABCMeta


class LocalExplainer(metaclass=ABCMeta):
    """Abstract base for explainers that provide local explanations."""

    @abstractmethod
    def explain_local(self, X, y=None, name=None):
        pass  # pragma: no cover


class GlobalExplainer(metaclass=ABCMeta):
    """Abstract base for explainers that provide global explanations."""

    @abstractmethod
    def explain_global(self, name=None):
        pass  # pragma: no cover


class DataExplainer(metaclass=ABCMeta):
    """Abstract base for explainers that provide data explanations."""

    @abstractmethod
    def explain_data(self, X, y, name=None):
        pass  # pragma: no cover


class PerfExplainer(metaclass=ABCMeta):
    """Abstract base for explainers that provide performance explanations."""

    @abstractmethod
    def explain_perf(self, X, y, name=None):
        pass  # pragma: no cover


class BaseExplanation(metaclass=ABCMeta):
    """The result of calling explain_* from an Explainer. Responsible for providing data and/or visualization.
        This is a contract required for InterpretML.

    Attributes:
        explanation_type: A string indicating the kind of explanation.
            Should be one of "perf", "data", "local", "global".
        name: A string that denotes the name of the explanation
            for display purposes.
        selector: An optional dict with "columns" (list of str) and
            "data" (list of dicts) that describes the data.
            Each entry in "data" corresponds with a respective data item.
    """

    @property
    @abstractmethod
    def explanation_type(self):
        pass  # pragma: no cover

    @abstractmethod
    def data(self, key=None):
        """Provides specific explanation data.

        Args:
            key: A number/string that references a specific data item.
        Returns:
            A serializable dictionary.
        """
        # pragma: no cover

    @abstractmethod
    def visualize(self, key=None):
        """Provides interactive visualizations.

        Args:
            key: Either a scalar or list
                that indexes the internal object for sub-plotting.
                If an overall visualization is requested, pass None.

        Returns:
            A Plotly figure, html as string, or a Dash component.
        """
        # pragma: no cover

    name = "An Explanation"
    selector = None
