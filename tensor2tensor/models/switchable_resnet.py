# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resnets."""
# Copied from cloud_tpu/models/resnet/resnet_model.py and modified

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
import tensorflow as tf
from ..layers.swichable_norm import switch_norm as sn

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5



def batch_norm(inputs,
                    is_training,
                    init_zero=False,
                    data_format="channels_first"):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == "channels_first":
    axis = 1
  else:
    axis = 3

  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      training=is_training,
      fused=True,
      gamma_initializer=gamma_initializer)

  return inputs

def switch_norm(inputs,
                hparams,
                    is_training,
                    init_zero=False,
                    data_format="channels_first"):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == "channels_first":
    axis = 1
  else:
    axis = 3

  inputs = sn(hparams=hparams,
              inputs=inputs,
              axis=axis,
              momentum=BATCH_NORM_DECAY,
              epsilon=BATCH_NORM_EPSILON,
              center=True,
              scale=True,
              is_training=is_training,
              fused=False,
              gamma_initializer=gamma_initializer
             )
  return inputs


def normalization(inputs,
                  hparams,
                  data_format,
                  relu=True,
                  init_zero=False,
                  scope='switch_norm'):
    is_training = (hparams.mode == tf.estimator.ModeKeys.TRAIN)
    with tf.variable_scope(scope):
        if hparams.relu_first & relu:
           inputs = tf.nn.relu(inputs)
        print('data format', data_format)

        if hparams.is_switchable:
           print("NOT BATCH NORMALIZATION")
           inputs = switch_norm(inputs, hparams, is_training, data_format=data_format, init_zero=init_zero)
        else:
           print("BATCH NORMALIZATION")
           inputs = batch_norm(inputs, is_training,  data_format=data_format, init_zero=init_zero)
        if (not hparams.relu_first & relu):
            inputs = tf.nn.relu(inputs)
    return inputs


def fixed_padding(inputs, kernel_size, data_format="channels_first"):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
        operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == "channels_first":
    padded_inputs = tf.pad(
        inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         data_format="channels_first",
                         use_td=False,
                         targeting_rate=None,
                         keep_prob=None,
                         is_training=None):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    use_td: `str` one of "weight" or "unit". Set to False or "" to disable
      targeted dropout.
    targeting_rate: `float` proportion of weights to target with targeted
      dropout.
    keep_prob: `float` keep probability for targeted dropout.
    is_training: `bool` for whether the model is in training.

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.

  Raises:
    Exception: if use_td is not valid.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  if use_td:
    inputs_shape = common_layers.shape_list(inputs)
    if use_td == "weight":
      if data_format == "channels_last":
        size = kernel_size * kernel_size * inputs_shape[-1]
      else:
        size = kernel_size * kernel_size * inputs_shape[1]
      targeting_count = targeting_rate * tf.to_float(size)
      targeting_fn = common_layers.weight_targeting
    elif use_td == "unit":
      targeting_count = targeting_rate * filters
      targeting_fn = common_layers.unit_targeting
    else:
      raise Exception("Unrecognized targeted dropout type: %s" % use_td)

    y = common_layers.td_conv(
        inputs,
        filters,
        kernel_size,
        targeting_count,
        targeting_fn,
        keep_prob,
        is_training,
        do_prune=True,
        strides=strides,
        padding=("SAME" if strides == 1 else "VALID"),
        data_format=data_format,
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer())
  else:
    y = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=("SAME" if strides == 1 else "VALID"),
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)

  return y


def residual_block(inputs,
                   filters,
                   is_training,
                   projection_shortcut,
                   strides,
                   final_block,
                   hparams,
                   data_format="channels_first",
                   use_td=False,
                   targeting_rate=None,
                   keep_prob=None):
  """Standard building block for residual networks with BN before convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    projection_shortcut: `function` to use for projection shortcuts (typically
        a 1x1 convolution to match the filter dimensions). If None, no
        projection is used and the input is passed as unchanged through the
        shortcut connection.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    final_block: unused parameter to keep the same function signature as
        `bottleneck_block`.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    use_td: `str` one of "weight" or "unit". Set to False or "" to disable
      targeted dropout.
    targeting_rate: `float` proportion of weights to target with targeted
      dropout.
    keep_prob: `float` keep probability for targeted dropout.

  Returns:
    The output `Tensor` of the block.
  """
  del final_block
  shortcut = inputs
  inputs = normalization(inputs,
                  hparams, data_format=data_format, scope='1')

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format,
      use_td=use_td,
      targeting_rate=targeting_rate,
      keep_prob=keep_prob,
      is_training=is_training)

  inputs = normalization(inputs,
                  hparams, data_format, scope='2')
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format,
      use_td=use_td,
      targeting_rate=targeting_rate,
      keep_prob=keep_prob,
      is_training=is_training)

  return inputs + shortcut


def bottleneck_block(inputs,
                     filters,
                     is_training,
                     projection_shortcut,
                     strides,
                     final_block,
                     hparams,
                     data_format="channels_first",
                     use_td=False,
                     targeting_rate=None,
                     keep_prob=None):
  """Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    projection_shortcut: `function` to use for projection shortcuts (typically
        a 1x1 convolution to match the filter dimensions). If None, no
        projection is used and the input is passed as unchanged through the
        shortcut connection.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    final_block: `bool` set to True if it is this the final block in the group.
        This is changes the behavior of batch normalization initialization for
        the final batch norm in a block.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    use_td: `str` one of "weight" or "unit". Set to False or "" to disable
      targeted dropout.
    targeting_rate: `float` proportion of weights to target with targeted
      dropout.
    keep_prob: `float` keep probability for targeted dropout.

  Returns:
    The output `Tensor` of the block.
  """
  # TODO(chrisying): this block is technically the post-activation resnet-v1
  # bottleneck unit. Test with v2 (pre-activation) and replace if there is no
  # difference for consistency.
  shortcut = inputs
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format,
      use_td=use_td,
      targeting_rate=targeting_rate,
      keep_prob=keep_prob,
      is_training=is_training)

  inputs = normalization(inputs, hparams, data_format=data_format, scope='3')
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format,
      use_td=use_td,
      targeting_rate=targeting_rate,
      keep_prob=keep_prob,
      is_training=is_training)

  inputs = normalization(inputs, hparams, data_format=data_format, scope='4')
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format,
      use_td=use_td,
      targeting_rate=targeting_rate,
      keep_prob=keep_prob,
      is_training=is_training)
  inputs = normalization(inputs,
                         hparams,
                         relu=False,
                         init_zero=final_block,
                         data_format=data_format,scope='5')
  return tf.nn.relu(inputs + shortcut)


def block_layer(inputs,
                filters,
                block_fn,
                blocks,
                strides,
                is_training,
                name,
                hparams,
                data_format="channels_first",
                use_td=False,
                targeting_rate=None,
                keep_prob=None

):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
    is_training: `bool` for whether the model is training.
    name: `str`name for the Tensor output of the block layer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    use_td: `str` one of "weight" or "unit". Set to False or "" to disable
      targeted dropout.
    targeting_rate: `float` proportion of weights to target with targeted
      dropout.
    keep_prob: `float` keep probability for targeted dropout.

  Returns:
    The output `Tensor` of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    """Project identity branch."""
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format,
        use_td=use_td,
        targeting_rate=targeting_rate,
        keep_prob=keep_prob,
        is_training=is_training)

    return normalization(
        inputs, hparams, relu=False, data_format=data_format, scope='6')

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(
      inputs,
      filters,
      is_training,
      projection_shortcut,
      strides,
      False,
      hparams,
      data_format,
      use_td=use_td,
      targeting_rate=targeting_rate,
      keep_prob=keep_prob)

  for i in range(1, blocks):
    with tf.variable_scope('block_{}'.format(i)):
        inputs = block_fn(
            inputs,
            filters,
            is_training,
            None,
            1, (i + 1 == blocks),
            hparams,
            data_format,
            use_td=use_td,
            targeting_rate=targeting_rate,
            keep_prob=keep_prob)

  return tf.identity(inputs, name)


def resnet_v2(inputs,
              block_fn,
              layers,
              filters,
              hparams,
              data_format="channels_first",
              is_training=False,
              is_cifar=False,
              use_td=False,
              targeting_rate=None,
              keep_prob=None):
  """Resnet model.

  Args:
    inputs: `Tensor` images.
    block_fn: `function` for the block to use within the model. Either
        `residual_block` or `bottleneck_block`.
    layers: list of 3 or 4 `int`s denoting the number of blocks to include in
      each of the 3 or 4 block groups. Each group consists of blocks that take
      inputs of the same resolution.
    filters: list of 4 or 5 `int`s denoting the number of filter to include in
      block.
    data_format: `str`, "channels_first" `[batch, channels, height,
        width]` or "channels_last" `[batch, height, width, channels]`.
    is_training: bool, build in training mode or not.
    is_cifar: bool, whether the data is CIFAR or not.
    use_td: `str` one of "weight" or "unit". Set to False or "" to disable
      targeted dropout.
    targeting_rate: `float` proportion of weights to target with targeted
      dropout.
    keep_prob: `float` keep probability for targeted dropout.

  Returns:
    Pre-logit activations.
  """
  print("hparams eval esteps", hparams.eval_steps)
  with tf.variable_scope('block_group_1'):
      inputs = block_layer(
          inputs=inputs,
          filters=filters[1],
          block_fn=block_fn,
          blocks=layers[0],
          strides=1,
          is_training=is_training,
          name="block_layer1",
          hparams=hparams,
          data_format=data_format,
          use_td=use_td,
          targeting_rate=targeting_rate,
          keep_prob=keep_prob)
  with tf.variable_scope('block_group_2'):
      inputs = block_layer(
          inputs=inputs,
          filters=filters[2],
          block_fn=block_fn,
          blocks=layers[1],
          strides=2,
          is_training=is_training,
          name="block_layer2",
          hparams=hparams,
          data_format=data_format,
          use_td=use_td,
          targeting_rate=targeting_rate,
          keep_prob=keep_prob)
  with tf.variable_scope('block_group_3'):
      inputs = block_layer(
          inputs=inputs,
          filters=filters[3],
          block_fn=block_fn,
          blocks=layers[2],
          strides=2,
          is_training=is_training,
          name="block_layer3",
          hparams=hparams,
          data_format=data_format,
          use_td=use_td,
          targeting_rate=targeting_rate,
          keep_prob=keep_prob)
  if not is_cifar:
    with tf.variable_scope('block_group_4'):
        inputs = block_layer(
            inputs=inputs,
            filters=filters[4],
            block_fn=block_fn,
            blocks=layers[3],
            strides=2,
            is_training=is_training,
            name="block_layer4",
            hparams=hparams,
            data_format=data_format,
            use_td=use_td,
            targeting_rate=targeting_rate,
            keep_prob=keep_prob)

  return inputs

@registry.register_model
class SwitchableResnet(t2t_model.T2TModel):
  """Residual Network."""


  def body(self, features):
    print("#Relu First",  self.hparams.relu_first)
    print("#Using Is Switchable", self.hparams.is_switchable)
    print("#IS original shake shake", self.hparams.original_shake_shake)
    hp = self.hparams
    block_fns = {
        "residual": residual_block,
        "bottleneck": bottleneck_block,
    }
    assert hp.block_fn in block_fns
    is_training = hp.mode == tf.estimator.ModeKeys.TRAIN
    if is_training:
      targets = features["targets_raw"]

    inputs = features["inputs"]

    data_format = "channels_last"
    if hp.use_nchw:
      # Convert from channels_last (NHWC) to channels_first (NCHW). This
      # provides a large performance boost on GPU.
      inputs = tf.transpose(inputs, [0, 3, 1, 2])
      data_format = "channels_first"

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=hp.filter_sizes[0],
        kernel_size=7,
        strides=1 if hp.is_cifar else 2,
        data_format=data_format)
    inputs = tf.identity(inputs, "initial_conv")
    with tf.variable_scope('after_conv_ident'):
        inputs = normalization(inputs, hp, data_format=data_format, scope='7')

    if not hp.is_cifar:
      inputs = tf.layers.max_pooling2d(
          inputs=inputs,
          pool_size=3,
          strides=2,
          padding="SAME",
          data_format=data_format)
      inputs = tf.identity(inputs, "initial_max_pool")

    with tf.variable_scope("rest_of_net"):
        out = resnet_v2(
            inputs,
            block_fns[hp.block_fn],
            hp.layer_sizes,
            hp.filter_sizes,
            hp,
            data_format,
            is_training=is_training,
            is_cifar=hp.is_cifar,
            use_td=hp.use_td,
            targeting_rate=hp.targeting_rate,
            keep_prob=hp.keep_prob)

    if hp.use_nchw:
      out = tf.transpose(out, [0, 2, 3, 1])

    if not hp.is_cifar:
      return out

    out = tf.reduce_mean(out, [1, 2])
    num_classes = self._problem_hparams.modality["targets"].top_dimensionality
    logits = tf.layers.dense(out, num_classes, name="logits")

    losses = {"training": 0.0}
    if is_training:
      loss = tf.losses.sparse_softmax_cross_entropy(
          labels=tf.squeeze(targets), logits=logits)
      loss = tf.reduce_mean(loss)

      losses = {"training": loss}

    logits = tf.reshape(logits, [-1, 1, 1, 1, logits.shape[1]])

    return logits, losses

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    """Predict."""
    del decode_length, beam_size, top_beams, alpha, use_tpu
    assert features is not None
    logits, _ = self(features)  # pylint: disable=not-callable
    assert len(logits.get_shape()) == 5
    logits = tf.squeeze(logits, [1, 2, 3])
    log_probs = common_layers.log_prob_from_logits(logits)
    predictions, scores = common_layers.argmax_with_score(log_probs)
    return {
        "outputs": predictions,
        "scores": scores,
    }






