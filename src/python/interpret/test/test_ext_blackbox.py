import interpret.ext.blackbox


def test_import_demo_explainer():
    print(dir(interpret.ext.blackbox))
    from interpret.ext.blackbox import BlackboxExplainerExample
    print("Loaded {}".format(BlackboxExplainerExample.__name__))
