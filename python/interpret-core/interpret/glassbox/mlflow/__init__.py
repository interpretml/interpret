import json
import os
import yaml

from tempfile import TemporaryDirectory

import interpret


def load_model(*args, **kwargs):
    import mlflow.pyfunc
    return mlflow.pyfunc.load_model(*args, **kwargs)


def _load_pyfunc(path):
    import cloudpickle as pickle
    with open(os.path.join(path, "model.pkl"), "rb") as f:
        return pickle.load(f)

def _save_model(model, output_path):
    import cloudpickle as pickle
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    with open(os.path.join(output_path, "model.pkl"), "wb") as stream:
        pickle.dump(model, stream)
    try:
        with open(os.path.join(output_path, "global_explanation.json"), "w") as stream:
            json.dump(model.explain_global().data(-1)["mli"], stream)
    except ValueError as e:
        raise Exception("Unsupported glassbox model type {}. Failed with error {}.".format(type(model), e))

def log_model(path, model):
    try:
        import mlflow.pyfunc
    except ImportError as e:
        raise Exception("Could not log_model to mlflow. Missing mlflow dependency, pip install mlflow to resolve the error: {}.".format(e))

    with TemporaryDirectory() as tempdir:
        _save_model(model, tempdir)

        conda_env = {"name": "mlflow-env",
                     "channels": ["defaults"],
                     "dependencies": ["interpret=".format(interpret.version.__version__),
                                      "cloudpickle==0.5.8"
                                     ]
                    }
        conda_path = os.path.join(tempdir, "conda.yaml")  # TODO Open issue and bug fix for dict support
        with open(conda_path, "w") as stream:
            yaml.dump(conda_env, stream)
        mlflow.pyfunc.log_model(path, loader_module="interpret.glassbox.mlflow", data_path=tempdir, conda_env=conda_path)
