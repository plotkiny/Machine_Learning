
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from pydoc import locate
from main.model.tensorflow.configurable import Configurable
from main.resources.helper_function import get_join_dir
from main.resources.helper_function import Loading

class ModelBase(Configurable):

    def __init__(self, params, mode, output_dir, name="base_model"):
        self.name = name
        Configurable.__init__(self, params, mode, output_dir)
        self.encoder_class = locate(self.params["encoder.class"])
        self.decoder_class = locate(self.params["decoder.class"])
        self.checkpoint = self.params["checkpoint_dir"]

        self.word_to_ind = None
        if "word_to_ind" in self.params and self.params["word_to_ind"]:
            file1 = get_join_dir(self.output_dir, self.params["word_to_ind"])
            self.word_to_ind = Loading.load_pickle(file1)

        self.ind_to_word = None
        if "ind_to_word" in self.params and self.params["ind_to_word"]:
            file2 = get_join_dir(self.output_dir, self.params["ind_to_word"])
            self.ind_to_word = Loading.load_pickle(file2)

        self.embed_matrix = None
        file3 = get_join_dir(self.output_dir, self.params["embed_matrix"])
        if "embed_matrix" in self.params and self.params["embed_matrix"]:
            self.embed_matrix = Loading.load_pickle(file3)

        if any([x == None for x in [file1, file2, file3]]):
            OSError("One of the following required files is missing: %s" %("word_to_ind, ind_to_word, embed_matrix"))

    def _clip_gradients(self, grads_and_vars):
        gradients, variables = zip(*grads_and_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.params["clip_gradients"])
        return list(zip(clipped_gradients ,variables))

    def _create_optimizer(self ,name, global_step):
        """Creates the optimizer"""
        opt_types = ["adam", "gradient_descent"]
        learning_rate = self.params["learning_rate"]
        if name == "adam":
            optimizer = tf.train.AdamOptimizer
            rate = learning_rate
        elif name == "gradient_descent":
            optimizer = tf.train.GradientDescentOptimizer
            rate = tf.train.exponential_decay(learning_rate,
                                              global_step * self.params["batch_size"],
                                              100000,
                                              self.params["learning_rate_decay"], staircase=True)
        else:
            ValueError("The following training optimizers are currently supported: %s" %opt_types)

        return optimizer(rate, **self.params["optimizer.params"])  # name='{}.Optimizer'.format(name.upper()),

    #TODO: add  _encoder_output, _features, _labels as parameters
    def _create_decoder(self, summ_length, max_length):
        """Creates a decoder instance based on the passed parameters."""
        return self.decoder_class(params=self.params,
                                  mode=self.mode,
                                  output_dir=self.output_dir,
                                  vocab_size=len(self.word_to_ind)+1,
                                  summ_length=summ_length,
                                  max_length=max_length)

    def _build_train_op(self):
        """Creates the training operation"""

        # TODO: Optimizer: Optionally wrap with SyncReplicasOptimizer
        # TODO: look into clipping embedding gradients

        name = self.params["optimizer.name"].lower()
        global_step = tf.Variable(0, trainable=False)

        optimizer = self._create_optimizer(name, global_step)

        grads_and_vars = optimizer.compute_gradients(self.batch_loss)
        clipped_gradients = self._clip_gradients(grads_and_vars)

        if name == "gradient_descent":
            train_op = optimizer.apply_gradients(clipped_gradients, global_step=global_step)
        elif name == "adam":
            train_op = optimizer.apply_gradients(clipped_gradients)

        return train_op

    def _batch_size(self):
        """Returns the batch size for a batch of examples"""
        raise NotImplementedError()

    def __call__(self, features, labels, params):
        """Creates the model graph. See the model_fn documentation in
        tf.contrib.learn.Estimator class for a more detailed explanation.
        """
        with tf.variable_scope("model"):
            with tf.variable_scope(self.name):
                return self._build(features, labels, params)

    def _build(self, features, labels, params):
        """Subclasses should implement this method. See the `model_fn` documentation
        in tf.contrib.learn.Estimator class for a more detailed explanation.
        """
        raise NotImplementedError