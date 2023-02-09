from .explanation import *
from slicer import Obj as O
from slicer import Alias as A

from interpret.newapi.explanation import AttribExplanation
from interpret.newapi.component import BinnedData, Attribution


def test_explanation_serialize():
    data = [[0, 1], [1, 2]]
    data_counts = [1, 2]
    feature_names = ['f1', 'f2']
    binned = BinnedData(
        O(data),
        O(data_counts),
        A(feature_names, 0),
    )

    values = [[1, 2], [3, 4]]
    base_values = [0, 1]
    units = "logits"
    attrib = Attribution(
        values,
        base_values,
        O(units, None),
    )

    expl = AttribExplanation(attrib, binned)
    serialized = expl.to_json()
    deserialized = Explanation.from_json(serialized)

    assert deserialized.data == [[0, 1], [1, 2]]
    assert deserialized.data_counts == [1, 2]
    assert deserialized.feature_names == ['f1', 'f2']
    assert deserialized.values == [[1, 2], [3, 4]]
    assert deserialized.base_values == [0, 1]
    assert deserialized.units == "logits"


def test_explanation():
    data = [[0, 1], [1, 2]]
    data_counts = [1, 2]
    feature_names = ['f1', 'f2']
    binned = BinnedData(
        O(data),
        O(data_counts),
        A(feature_names, 0),
    )

    values = [[1, 2], [3, 4]]
    base_values = [0, 1]
    units = "logits"
    attrib = Attribution(
        values,
        base_values,
        O(units, None),
    )

    expl = AttribExplanation(attrib, binned)
    expl = AttribExplanation.from_json(expl.to_json())
    actual = expl[0]
    assert actual.data == [0, 1]
    assert actual.feature_names == 'f1'
    assert actual.units == 'logits'
    assert actual.base_values == 0

    actual = expl['f1']
    assert actual.data == [0, 1]
    assert actual.feature_names == 'f1'
    assert actual.units == 'logits'
    assert actual.base_values == 0

    expl = AttribExplanation(attrib, extra=binned)
    actual = expl[0]
    assert actual.data == [0, 1]
    assert actual.feature_names == 'f1'
    assert actual.units == 'logits'
    assert actual.base_values == 0