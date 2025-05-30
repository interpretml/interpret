from typing import List

from slicer import Alias, Obj


class Component:
    @classmethod
    def from_fields(cls, fields: List):
        instance = cls.__new__(cls)
        instance.fields = fields
        return instance


class Attribution(Component):
    def __init__(self, values, base_values=None, units=None):
        self.fields = locals()
        del self.fields["self"]


class BinnedData(Component):
    def __init__(
        self,
        data,
        data_counts=None,
        feature_names=None,
        feature_types=None,
        feature_indexes=None,
    ):
        if not isinstance(feature_names, (Obj, Alias, type(None))):
            feature_names = Alias(feature_names, 0)

        if not isinstance(feature_types, (Obj, Alias, type(None))):
            feature_types = Alias(feature_types, 0)

        if not isinstance(feature_indexes, (Obj, Alias, type(None))):
            feature_indexes = Alias(feature_indexes, 0)

        self.fields = locals()
        del self.fields["self"]


class TabularData(Component):
    def __init__(
        self, data, feature_names=None, feature_types=None, feature_indexes=None
    ):
        if not isinstance(feature_names, (Obj, Alias, type(None))):
            feature_names = Alias(feature_names, 1)

        if not isinstance(feature_types, (Obj, Alias, type(None))):
            feature_types = Alias(feature_types, 1)

        if not isinstance(feature_indexes, (Obj, Alias, type(None))):
            feature_indexes = Alias(feature_indexes, 1)

        self.fields = locals()
        del self.fields["self"]


class Bound(Component):
    def __init__(self, lower_bounds, upper_bounds):
        self.fields = locals()
        del self.fields["self"]


# TODO: Consider separation of concerns for each field.
class Meta(Component):
    def __init__(self, source, pivots, dimension_names=None):
        self.fields = locals()
        del self.fields["self"]


class Extra(Component):
    def __init__(
        self,
        display_data=None,
        output_names=None,
        output_indexes=None,  # Alias?
        main_effects=None,
        hierarchical_values=None,
        clustering=None,
    ):
        self.fields = locals()
        del self.fields["self"]
