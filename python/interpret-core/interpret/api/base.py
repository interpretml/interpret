# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import abc

# TODO: instead of having interpret.api.base, it would probably be easier to just have a single base.py file
#      in the root directory parallel to develop.py called base.py and also put the templates.py file contents in
#      there


# TODO v.3 PK Possibly rename explainer types to (blackbox, glassbox, greybox)
class ExplainerMixin(abc.ABC):
    """An object that computes explanations.
        This is a contract required for InterpretML.

    Attributes:
        available_explanations: A list of strings subsetting the following
            - "perf", "data", "local", "global".
        explainer_type: A string that is one of the following
            - "blackbox", "model", "specific", "data", "perf".
    """

    @property
    @abc.abstractmethod
    def available_explanations(self):
        pass  # pragma: no cover

    @property
    @abc.abstractmethod
    def explainer_type(self):
        pass  # pragma: no cover


class ExplanationMixin(abc.ABC):
    """The result of calling explain_* from an Explainer. Responsible for providing data and/or visualization.
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
    @abc.abstractmethod
    def explanation_type(self):
        pass  # pragma: no cover

    @abc.abstractmethod
    def data(self, key=None):
        """Provides specific explanation data.

        Args:
            key: A number/string that references a specific data item.
        Returns:
            A serializable dictionary.
        """
        # pragma: no cover

    @abc.abstractmethod
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
