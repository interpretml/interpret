# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from contextlib import AbstractContextManager


class SharedDataset(AbstractContextManager):
    def __init__(self):
        self.shared_memory = None
        self.dataset = None
        self.name = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reset()

    def reset(self):
        shared_memory = self.shared_memory
        self.name = None
        self.dataset = None
        self.shared_memory = None
        if shared_memory is not None:
            shared_memory.close()
            shared_memory.unlink()
