from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from operator import mul
from tensorflow.python.keras import initializers
import tensorflow as tf

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

def switch_norm(x, hparams, dataformat, is_training, scope='switch_norm'):
    with tf.variable_scope(scope):
        moving_mean_initializer = initializers.get('zeros')
        batch_size = hparams.batch_size
        num_branches = 3

        rand_forward = [tf.random_uniform([batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32)
                        for _ in range(num_branches)]
        rand_backward = [tf.random_uniform([batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32)
                         for _ in range(num_branches)]
        rand_eval = [1/num_branches]*num_branches
        if not hparams.original_shake_shake:
            means = [tf.get_variable('normalize_means_{}'.format(i), shape=[1, 1, 1, 1])
            for i in range(num_branches)]
            means = [tf.math.abs(x) for x in means]
            means_sum = tf.add_n(means)
            means = [x / means_sum for x in means]
            step = tf.to_float(tf.train.get_or_create_global_step())
            if hparams.weight_lower_bound:
                means_lower_treshhold = lower_bound_scheduler(step, num_branches, hparams.train_steps)
                tf.summary.scalar('lower_bound', means_lower_treshhold)
                means = [(1 - means_lower_treshhold * num_branches) * means[i] + means_lower_treshhold for i in
                range(num_branches)]
            rand_forward = [2 * means[i] * rand_forward[i] for i in range(num_branches)]
            rand_backward = [2 * means[i] * rand_backward[i] for i in range(num_branches)]
            rand_eval = means

            tf.summary.scalar('mean_0_', tf.squeeze(means[0]))
            tf.summary.scalar('mean_1_', tf.squeeze(means[1]))
        total_forward = tf.add_n(rand_forward)
        total_backward = tf.add_n(rand_backward)
        rand_forward_normal = [samp / total_forward for samp in rand_forward]
        rand_backward_normal = [samp / total_backward for samp in rand_backward]

        if dataformat == "channels_first":
            ch = x.shape[1]
            intance_index = [2, 3]
            batch_index = [0, 2, 3]
            running_shape = [1, ch, 1, 1]
        elif dataformat == "channels_last":
            ch = x.shape[-1]
            intance_index = [1, 2]
            batch_index = [0, 1, 2]
            running_shape = [1, 1, 1, ch]
        else:
            raise Exception("data format not defined")
        layer_index = [1, 2, 3]

        running_mean = tf.get_variable(
            'running_mean',
            shape=running_shape,
            initializer=moving_mean_initializer,
            trainable=False,
        )

        running_var = tf.get_variable(
            'running_var',
            shape=running_shape,
            initializer=moving_mean_initializer,
            trainable=False,
        )
        ins_mean, ins_var = tf.nn.moments(x, intance_index, keep_dims=True)
        layer_mean, layer_var = tf.nn.moments(x, layer_index, keep_dims=True)
        if is_training:
            batch_mean, batch_var = tf.nn.moments(x, batch_index, keep_dims=True)
            running_mean = BATCH_NORM_DECAY*running_mean + (1-BATCH_NORM_DECAY)*batch_mean
            # tf.assign(running_mean, new_running_mean)
            running_var = BATCH_NORM_DECAY*running_var + (1-BATCH_NORM_DECAY)*batch_var
            # tf.assign(running_var, new_running_var)

        else:
            batch_mean = running_mean
            batch_var = running_var
        tf.summary.scalar('batch_mean_0_', batch_mean[0][0][0][0])
        tf.summary.scalar('running_mean_0_', running_mean[0][0][0][0])
        tf.summary.scalar('instance_mean_0_', batch_mean[0][0][0][0])
        tf.summary.scalar('layer_mean_0_', batch_mean[0][0][0][0])


        gamma = tf.get_variable("gamma", running_shape, initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", running_shape, initializer=tf.constant_initializer(0.0))

        #mean_weight = tf.nn.softmax(tf.get_variable("mean_weight", [3], initializer=tf.constant_initializer(1.0)))
        #var_wegiht = tf.nn.softmax(tf.get_variable("var_weight", [3], initializer=tf.constant_initializer(1.0)))
        layer_mean = tf.tile(layer_mean, running_shape)
        layer_var = tf.tile(layer_var,  running_shape)

        norm_mean_list = [batch_mean, ins_mean, layer_mean]
        norm_var_list = [batch_var, ins_var, layer_var]


        if is_training:
            tmp_mean_back =  norm_mean_list[0]*rand_backward_normal[0] + norm_mean_list[1]*rand_backward_normal[1] +\
                             norm_mean_list[2]*rand_backward_normal[2]
            tmp_mean_forw = norm_mean_list[0]*rand_forward_normal[0] + norm_mean_list[1]*rand_forward_normal[1] +\
                            norm_mean_list[2]*rand_forward_normal[2]
            switchable_mean = tmp_mean_back + tf.stop_gradient(tmp_mean_forw - tmp_mean_back)
            tmp_var_back = norm_var_list[0]*rand_backward_normal[0] + norm_var_list[1]*rand_backward_normal[1] +\
                           norm_var_list[2]*rand_backward_normal[2]
            tmp_var_forw = norm_var_list[0]*rand_forward_normal[0] + norm_var_list[1]*rand_forward_normal[1] +\
                           norm_var_list[2]*rand_forward_normal[2]
            switchable_var = tmp_var_back + tf.stop_gradient(tmp_var_forw - tmp_var_back)

        else:
            switchable_mean = norm_mean_list[0]*rand_eval[0] + norm_mean_list[1]*rand_eval[1] +\
                              norm_mean_list[2]*rand_eval[2]
            switchable_var = norm_var_list[0]*rand_eval[0] + norm_var_list[1]*rand_eval[1] +\
                             norm_var_list[2]*rand_eval[2]

        x = (x - switchable_mean) / (tf.sqrt(switchable_var + BATCH_NORM_EPSILON))
        x = x * gamma + beta
        return x


def lower_bound_scheduler(step, branch_numbers, train_steps):
  base_bound = tf.constant(1.0/branch_numbers)
  decay_steps = tf.constant(5.0*train_steps//6.0)
  ratio = tf.math.maximum(0.0, (1.0-step/decay_steps))
  return ratio*base_bound
