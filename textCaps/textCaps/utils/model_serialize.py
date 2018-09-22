# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

if sys.version_info.major < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

import tensorflow as tf


class ModelSerialize(object):
    def __init__(self, model_dir, model_tag, signature_tag):
        self._model_dir = model_dir
        self._model_tag = model_tag
        self._signatrue_tag = signature_tag
        self._signature_method_name = model_tag
        self._inputs = {}
        self._outputs = {}

    def add_input(self, name, tensor):
        self._inputs[name] = tf.saved_model.utils.build_tensor_info(tensor)

    def add_output(self, name, tensor):
        self._outputs[name] = tf.saved_model.utils.build_tensor_info(tensor)

    def set_signature_method_name(self, signature_method_name):
        self._signature_method_name = signature_method_name

    def save(self, sess):
        builder = tf.saved_model.builder.SavedModelBuilder(self._model_dir)
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=self._inputs,
            outputs=self._outputs,
            method_name=self._signature_method_name)
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[self._model_tag],
            signature_def_map={self._signatrue_tag: signature},
            clear_devices=True)
        builder.save()
        return self._model_dir


class ModelUnSerizlize(object):
    def __init__(self, sess, model_dir, model_tag, signature_tag):
        self._sess = sess
        meta_graph_def = tf.saved_model.loader.load(sess, [model_tag], model_dir)
        signature = meta_graph_def.signature_def

        inputs = signature[signature_tag].inputs
        self._inputs = {}
        for k, v in inputs.items():
            self._inputs[k] = v.name

        outputs = signature[signature_tag].outputs
        self._outputs = {}
        for k, v in outputs.items():
            self._outputs[k] = v.name

    def get_input_tensor(self, name):
        return self._sess.graph.get_tensor_by_name(self._inputs[name])

    def get_output_tensor(self, name):
        return self._sess.graph.get_tensor_by_name(self._outputs[name])
