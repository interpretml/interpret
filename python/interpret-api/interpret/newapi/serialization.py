import numpy as np
from json import JSONEncoder, JSONDecoder
from slicer import Alias
from slicer import Obj
from importlib import import_module


class ExplanationJSONEncoder(JSONEncoder):
    def default(self, o):
        from interpret.newapi.explanation import Explanation
        from interpret.newapi.component import Component

        if isinstance(o, np.ndarray):
            return {
                "_type": "array",
                "value": o.tolist(),
            }
        elif isinstance(o, Explanation):
            return {
                "_type": "explanation",
                "module": o.__class__.__module__,
                "class": o.__class__.__name__,
                "components": list(o.components.values()),
            }
        elif isinstance(o, Component):
            return {
                "_type": "component",
                "module": o.__class__.__module__,
                "class": o.__class__.__name__,
                "fields": o.fields,
            }
        elif isinstance(o, Obj):
            return {
                "_type": "obj",
                "value": o.o,
                "dim": o.dim,
            }
        elif isinstance(o, Alias):
            return {
                "_type": "alias",
                "value": o.o,
                "dim": o.dim,
            }
        else:
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
        elif _type == "explanation":
            cls = getattr(import_module(obj["module"]), obj["class"])
            return cls.from_components(obj["components"])
        elif _type == "component":
            cls = getattr(import_module(obj["module"]), obj["class"])
            return cls.from_fields(obj["fields"])
        elif _type == "obj":
            return Obj(obj["value"], obj["dim"])
        elif _type == "alias":
            return Alias(obj["value"], obj["dim"])
        return obj
