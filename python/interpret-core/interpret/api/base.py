# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from abc import ABC, abstractmethod


# TODO v.3 PK Possibly rename explainer types to (blackbox, glassbox, greybox)
class ExplainerMixin(ABC):
    """ An object that computes explanations.
        This is a contract required for InterpretML.

    Attributes:
        available_explanations: A list of strings subsetting the following
            - "perf", "data", "local", "global".
        explainer_type: A string that is one of the following
            - "blackbox", "model", "specific", "data", "perf".
    """

    @property
    @abstractmethod
    def available_explanations(self):
        pass  # pragma: no cover

    @property
    @abstractmethod
    def explainer_type(self):
        pass  # pragma: no cover


class ExplanationMixin(ABC):
    """ The result of calling explain_* from an Explainer. Responsible for providing data and/or visualization.
        This is a contract required for InterpretML.

    Attributes:
        explanation_type: A string that is one of the
            explainer's available explanations.
            Should be one of "perf", "data", "local", "global".
        name: A string that denotes the name of the explanation
            for display purposes.
        selector: An optional dataframe that describes the data.
            Each row of the dataframe corresponds with a respective data item.
    """

    @property
    @abstractmethod
    def explanation_type(self):
        pass  # pragma: no cover

    @abstractmethod
    def data(self, key=None):
        """ Provides specific explanation data.

        Args:
            key: A number/string that references a specific data item.
        Returns:
            A serializable dictionary.
        """
        pass  # pragma: no cover

    @abstractmethod
    def visualize(self, key=None):
        """ Provides interactive visualizations.

        Args:
            key: Either a scalar or list
                that indexes the internal object for sub-plotting.
                If an overall visualization is requested, pass None.

        Returns:
            A Plotly figure, html as string, or a Dash component.
        """
        pass  # pragma: no cover

    name = "An Explanation"
    selector = None
