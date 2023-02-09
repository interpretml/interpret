import json

from slicer import Slicer as S
from interpret.newapi.component import Component

# TODO: .component and .append to be made protected for minimal API surface.


class Explanation(S):
    @classmethod
    def _init_explanation(cls, instance, *args):
        super(Explanation, instance).__init__()

        instance.components = {}
        instance._field_components_map = {}

        for value in args:
            if value is not None:
                instance.append(value)

    def __init__(self, **kwargs):
        self.__class__._init_explanation(self, *list(kwargs.values()))

    # TODO: Needs further discussion at design-level.
    def append(self, component):
        if not isinstance(component, Component):
            raise Exception(f"Can't append object of type {type(component)} to this object.")

        self.components[type(component)] = component
        for field_name, field_value in component.fields.items():
            self.__setattr__(field_name, field_value)
            self._field_components_map[field_name] = type(component)

        return self

    def __contains__(self, item):
        return item in self.components

    def __repr__(self):
        record = self.components.copy()
        fields = []
        shape_str = f"shape: {self.shape}"
        fields.append(shape_str)
        fields.append("-" * len(shape_str))
        for record_key, record_val in record.items():
            for field_name, field_val in record_val.fields.items():
                field_value = str(self.__getattr__(field_name))

                if field_name in self._dims:
                    field_value_str = f"Dim\t{field_name} = {field_value}"
                else:
                    if field_name in self._objects:
                        field_type = 'O'
                        field_dim = ','.join(str(x) for x in self._objects[field_name].dim)
                    else:
                        field_type = 'A'
                        field_dim = ','.join(str(x) for x in self._aliases[field_name].dim)
                    field_value_str = f"{field_type}{{{field_dim}}}\t{field_name} = {field_value}"

                if len(field_value_str) > 60:
                    field_value_str = field_value_str[:57] + "..."
                fields.append(field_value_str)
        fields = "\n".join(fields)

        return fields

    @classmethod
    def from_json(cls, json_str):
        from interpret.newapi.serialization import ExplanationJSONDecoder

        d = json.loads(json_str, cls=ExplanationJSONDecoder)
        instance = d["content"]
        return instance

    @classmethod
    def from_components(cls, components):
        instance = cls.__new__(cls)
        cls._init_explanation(instance, *components)

        return instance

    def to_json(self, **kwargs):
        from interpret.newapi.serialization import ExplanationJSONEncoder
        version = "0.0.1"
        di = {
            "version": version,
            "content": self,
        }

        return json.dumps(di, cls=ExplanationJSONEncoder, **kwargs)


class AttribExplanation(Explanation):
    def __init__(self, attrib, data=None, perf=None, bound=None, meta=None, **kwargs):
        super().__init__(
            attrib=attrib,
            data=data,
            perf=perf,
            bound=bound,
            meta=meta,
            **kwargs,
        )

