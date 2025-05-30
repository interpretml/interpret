from importlib import import_module
from json import JSONDecoder, JSONEncoder

import numpy as np
from slicer import Alias, Dim, Obj


class ExplanationJSONEncoder(JSONEncoder):
    def default(self, o):
        from interpret.newapi.component import Component
        from interpret.newapi.explanation import Explanation

        if isinstance(o, np.ndarray):
            return {
                "_type": "array",
                "value": o.tolist(),
            }
        if isinstance(o, Explanation):
            return {
                "_type": "explanation",
                "module": o.__class__.__module__,
                "class": o.__class__.__name__,
                "components": list(o.components.values()),
            }
        if isinstance(o, Component):
            return {
                "_type": "component",
                "module": o.__class__.__module__,
                "class": o.__class__.__name__,
                "fields": o.fields,
            }
        if isinstance(o, Obj):
            return {
                "_type": "obj",
                "value": o.o,
                "dim": o.dim,
            }
        if isinstance(o, Alias):
            return {
                "_type": "alias",
                "value": o.o,
                "dim": o.dim,
            }
        if isinstance(o, Dim):
            return {
                "_type": "dim",
                "value": o.o,
            }
        return JSONEncoder.default(self, o)


class ExplanationJSONDecoder(JSONDecoder):
    def __init__(self, *args, **kwargs):
        JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "_type" not in obj:
            return obj
        _type = obj["_type"]
        if _type == "array":
            return np.array(obj["value"])
        if _type == "explanation":
            cls = getattr(import_module(obj["module"]), obj["class"])
            return cls.from_components(obj["components"])
        if _type == "component":
            cls = getattr(import_module(obj["module"]), obj["class"])
            return cls.from_fields(obj["fields"])
        if _type == "obj":
            return Obj(obj["value"], obj["dim"])
        if _type == "alias":
            return Alias(obj["value"], obj["dim"])
        if _type == "dim":
            return Dim(obj["dim"])
        return obj
