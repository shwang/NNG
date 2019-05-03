from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

_FLAGS = {}


def add_to_collection(key, value):
    _FLAGS[key] = value

def get_collection(key):
    return _FLAGS[key]

def clear_collection(key):
    _FLAGS.clear()

def get_layer_input_activations(n_layers):
    for i in range(n_layers):
        yield get_collection("a{}".format(i))

def get_layer_outputs(n_layers):
    for i in range(n_layers):
        yield get_collection("s{}".format(i))
