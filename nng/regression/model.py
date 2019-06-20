from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from typing import Iterable, List, Optional, Tuple, Sequence, TYPE_CHECKING

import tensorflow as tf
from tensorflow.contrib.framework import with_shape

from nng.core.base_model import BaseModel
from nng.misc.registry import get_model
from nng.regression.controller.bayesian_learning import BayesianLearning
from nng.regression.misc.collections import (
    get_collection, get_layer_input_activations, get_layer_outputs)
from nng.regression.misc.layers import append_homog, EMVGLayer, MVGLayer
from nng.regression.controller.sample import NormalOutSample
import nng.regression.network.ffn as ffn

if TYPE_CHECKING:
    from irdplus.bnn.problem.problem import Problem
    from irdplus.bnn.model.model import Model


class Model(BaseModel):

    n_particles = ...  # type: tf.Tensor
    inputs_shape = ... # type: tf.Tensor
    inputs = ... # type: tf.Tensor
    targets = ... # type: tf.Tensor
    alpha = ... # type: tf.Tensor
    beta = ... # type: tf.Tensor
    omega = ... # type: tf.Tensor

    def __init__(self, config, input_dim: Iterable[int], n_data: int, *,
                 logger,
                 n_batch_size: Optional[int] = None,
                 n_particles_ph: Optional[tf.Tensor] = None,
                 inputs_ph: Optional[tf.Tensor] = None,
                 problem: "Optional[Problem]" = None,
                 scale_ll: Optional[float] = None,
                 ):
        """ Initialize a class Model.
        :param config: Configuration Bundle.
        :param input_dim: int
        :param n_data: int
        """
        super().__init__(config, logger)
        # Set the approximation type specifically.
        if config.optimizer == "ekfac":
            self.logger.info("[!] Optimizer: eKFAC")
            self.layer_type = "emvg"
        elif config.optimizer == "kfac":
            self.logger.info("[!] Optimizer: KFAC")
            self.layer_type = "mvg"
        else:
            self.logger.info("[!] Optimizer: {}".format(config.optimizer))
            self.layer_type = None
        self.input_dim = input_dim  # type: List[int]
        self.n_data = n_data  # type: int
        self.scale_ll = scale_ll
        self.logger = logger
        self.logger.info("model.n_data = {}".format(n_data))
        self.problem = problem
        self.n_batch_size = n_batch_size  # Forced inverse scaling factor for w_grads.

        # Initialize attributes.
        if n_particles_ph is None:
            self.n_particles = tf.placeholder(tf.int32, shape=[],
                    name="n_particles")
        else:
            self.n_particles = n_particles_ph

        if inputs_ph is None:
            inputs_ph = tf.placeholder(tf.float32,
                        shape=[None] + list(self.input_dim))
        self.inputs = inputs_ph

        self.targets = tf.placeholder(tf.float32, [None])
        self.alpha = tf.placeholder(tf.float32, shape=[], name='alpha')
        self.beta = tf.placeholder(tf.float32, shape=[], name='beta')
        self.omega = tf.placeholder(tf.float32, shape=[], name='omega')

        if not self.problem:
            self.ird_tag = "regression"
        else:
            self.ird_tag = self.problem.nng_tag

        # Build the model.
        self._build_model()
        self.init_saver()

    def _build_model(self):
        outsample_cls = NormalOutSample if self.ird_tag == "regression" else None
        if self.layer_type == "emvg":
            layer_cls = EMVGLayer
        elif self.layer_type == "mvg":
            layer_cls = MVGLayer

        if self.config.model_name == "ffn50":
            default_hidden_size = 50
        elif self.config.model_name == "ffn100":
            default_hidden_size = 100
        else:
            raise ValueError(self.config.model_name)

        hidden_sizes = self.config.get("hidden_sizes", None) or \
                [default_hidden_size]
        layer_sizes = [int(self.inputs.shape[-1])] + list(hidden_sizes)
        end_with_sum = self.config.get("end_with_sum", False)
        if not end_with_sum:
            layer_sizes.append(1)
        self.n_layers = len(layer_sizes) - 1
        layer_types = [layer_cls] * self.n_layers
        layer_params = [{}] * self.n_layers

        self.logger.info("layer_sizes: {}".format(layer_sizes))
        layers, init_ops = ffn.ffn(layer_type=self.layer_type,
                                           input_size=int(self.inputs.shape[-1]),
                                           num_data=self.n_data,
                                           kl_factor=self.config.kl,
                                           ita=self.config.eta,
                                           alpha=self.alpha,
                                           beta=self.beta,
                                           damp=self.config.damping,
                                           omega=self.omega,
                                           layer_sizes=layer_sizes)

        self.learn = BayesianLearning(
                layer_sizes=layer_sizes,
                layer_types=layer_types,
                layer_params=layer_params,
                out_params={},
                activation_fn=tf.nn.relu,
                outsample_cls=outsample_cls,
                x=self.inputs,
                y=self.targets,
                n_particles=self.n_particles,
                std_y_train=self.config.std_train,
                ird_tag=self.ird_tag,
                end_with_sum=end_with_sum)

        self.h_pred = tf.squeeze(self.learn.h_pred, 2)
        if self.ird_tag == "ird":
            main_raw, _, self._bnn = \
                    self.problem.gather_standard_rewards(self.h_pred)
            self._main = tf.reshape(main_raw, [self.n_particles, -1])
        else:
            self._main = self._aux = self._bnn = None

        self._log_py_xw = self._build_log_py_xw()
        self.kl = tf.check_numerics(self.learn.build_kl(), "kl")
        self.loss_prec = self._build_loss_prec()

        n_outputs = tf.cast(tf.shape(self.inputs)[0] * self.n_particles, tf.float32)
        self.mean_log_py_xw = self._log_py_xw / n_outputs
        self.lower_bound = self._build_lower_bound()
        self.rmse = self.learn.rmse
        self.ll = self.learn.log_likelihood

        self.init_ops = tf.group(init_ops) if init_ops != [] else None
        weight_update_op, self.basis_update_op, self.scale_update_op = \
                self._build_layer_update_ops(layers)
        prec_op = self._build_prec_op()
        self.train_op = tf.group([weight_update_op, prec_op], name="train_op")

    def get_train_output(self) -> tf.Tensor:
        """
        Returns the output that should be used for calculating training loss.
        """
        if self.ird_tag == "regression":
            return self.learn.y_pred
        elif self.ird_tag == "ird":
            return self.h_pred  # Idea: Try using noisy outputs for IRD?
        else:
            raise ValueError(self.ird_tag)

    def get_test_output(self) -> tf.Tensor:
        """
        Returns the output that should be used at test time.
        """
        if self.ird_tag == "regression":
            return self.h_pred
        elif self.ird_tag == "ird":
            return self.h_pred
        else:
            raise ValueError(self.ird_tag)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def _build_log_py_xw(self) -> tf.Tensor:
        if self.ird_tag == "regression":
            # return self.learn.log_py_xw
            return tf.reduce_sum(self.learn.log_py_xw)
        elif self.ird_tag == "ird":
            assert self.scale_ll is not None
            output = self.get_train_output()
            loss = self.problem.build_data_loss(
                output, l1_loss=0.0, l2_loss=0.0) * self.scale_ll
            return -loss
        else:
            raise ValueError(self.problem.stupid_ird_tag)

    def _build_loss_prec(self) -> tf.Tensor:
        if self.ird_tag != "regression":
            return tf.constant(0.0, name="loss_prec")

        # log_alpha, log_beta = tf.trainable_variables()[-2:]  # Ewww
        outsample = self.learn._net._outsample
        alpha_ = outsample.q_alpha
        beta_ = outsample.q_beta
        # alpha_ = tf.exp(log_alpha)
        # beta_ = tf.exp(log_beta)
        y_obs = tf.tile(tf.expand_dims(self.targets, 0), [self.n_particles, 1])
        loss_prec = 0.5 * (tf.stop_gradient(tf.reduce_mean((y_obs - self.h_pred) ** 2)) *
                           alpha_ / beta_ - (tf.digamma(alpha_) - tf.log(beta_ + 1e-10)))
        return loss_prec

    def _build_lower_bound(self) -> tf.Tensor:
        """
        Note that this Tensor is only used for logging, not visualization.
        A better name for it would be mean_lower_bound.
        """
        lower_bound = with_shape([],
                self.mean_log_py_xw - self.config.kl * self.kl / self.n_data)
        lower_bound = lower_bound - tf.reduce_mean(self.loss_prec)

        return tf.check_numerics(lower_bound, "lb")

    def _build_prec_op(self) -> tf.Operation:
        """Build an operation used to update outsampling precision."""
        if self.ird_tag == "regression":
            outsample = self.learn._net._outsample
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            prec_vars = [outsample.prec_logalpha, outsample.prec_logbeta]
            prec_grads = optimizer.compute_gradients(-self.lower_bound,
                    prec_vars)
            prec_op = optimizer.apply_gradients(prec_grads, name="prec_op")
        else:
            prec_op = tf.no_op(name="prec_op")
        return prec_op

    def _build_layer_update_ops(self, layers: Sequence) -> Tuple[
            tf.Operation, tf.Operation, tf.Operation]:
        """
        Builds the following layer Operations:
        `weight_update_op, scale_update_op, basis_update_op`.
        """
        qws = [self.learn.qws['w' + str(i)].tensor
                for i in range(len(self.learn.qws))]
        # w_grads = tf.gradients(tf.reduce_sum(
        #     tf.reduce_mean(self._log_py_xw, 1), 0), qws)

        # NOTE: _log_py_xw is a scalar.
        n_batches = self.n_batch_size or tf.cast(
                tf.shape(self.inputs)[0], tf.float32)
        w_grads = tf.gradients(self._log_py_xw / n_batches, qws)

        n_layers = len(layers)
        activations = [append_homog(a)
                       for a in get_layer_input_activations(n_layers)]
        s = list(get_layer_outputs(n_layers))
        assert len(layers) == len(activations) == len(s)

        if self.ird_tag == "regression":
            if self.config.true_fisher:
                # True fisher: sample model and y from the var. distribution.
                sampled_log_prob = self.learn.sampled_log_prob
                s_grads = tf.gradients(tf.reduce_sum(sampled_log_prob), s)
            else:
                # Empirical fisher: sample model from var distribution, setting
                # y = target.
                s_grads = tf.gradients(self._log_py_xw, s)
        elif self.ird_tag == "ird":
            assert self.config.true_fisher, "Only true fisher supported"
            # Sample model and y from the var. distribution.
            # (Yes, _log_py_xw holds true fisher even though in regression case
            # this is the empiral fisher).
            s_grads = [tf.check_numerics(x, "ird s_grads")
                for x in tf.gradients(self._log_py_xw, s)]
        else:
            raise ValueError("Invalid ird tag: {}".format(self.ird_tag))

        weight_updates = []
        scale_updates = []
        basis_updates = []
        for l, w, w_grad, a, s_grad in zip(layers, qws, w_grads, activations, s_grads):
            # Adds the regular KFAC update.
            weight_updates.extend(l.update(w, w_grad, a, s_grad))
            if self.layer_type == "emvg":
                scale_updates.extend(l.update_scale(w, w_grad, a, s_grad))
                basis_updates.extend(l.update_basis(w, w_grad, a, s_grad))

        return (tf.group(*weight_updates, name="weight_updates"),
                tf.group(*scale_updates, name="scale_updates"),
                tf.group(*basis_updates, name="basis_updates"))
