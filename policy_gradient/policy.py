import tensorflow as tf
import numpy as np

class CategoricalPolicy(object):
    def __init__(self, in_dim, out_dim, hidden_dim, optimizer, session):
        self._observations = tf.placeholder(tf.float32, shape=[None, in_dim], name="observations")
        self._actions = tf.placeholder(tf.int32, name="actions")
        self._advantages = tf.placeholder(tf.float32, name="advantages")
        self._opt = optimizer
        self._sess = session

        # policy network
        def weights_tanh(shape):
            init_scale = np.sqrt(6./(shape[0]+shape[1]))
            return tf.Variable(tf.random_uniform(shape, -init_scale, init_scale))
        def weights_sigmoid(shape):
            init_scale = 4 * np.sqrt(6./(shape[0]+shape[1]))
            return tf.Variable(tf.random_uniform(shape, -init_scale, init_scale))
        def bias(shape):
            return tf.Variable(tf.zeros(shape))
        w1, b1 = weights_tanh([in_dim, hidden_dim]), bias([hidden_dim])
        a1 = tf.tanh(tf.matmul(self._observations, w1) + b1)
        w2, b2 = weights_sigmoid([hidden_dim, out_dim]), bias([out_dim])
        probs = tf.nn.softmax(tf.matmul(a1, w2) + b2)

        act_op = probs[0, :]

        action_idxs_flattened = tf.range(0, tf.shape(probs)[0]) * tf.shape(probs)[1]
        action_idxs_flattened += self._actions
        probs_vec = tf.gather(tf.reshape(probs, [-1]), action_idxs_flattened)
        log_prob = tf.log(probs_vec + 1e-8)

        surr_loss = -tf.reduce_mean(tf.mul(log_prob, self._advantages))
        grads_and_vars = self._opt.compute_gradients(surr_loss)
        train_op = self._opt.apply_gradients(grads_and_vars, name="train_op")

        self._act_op = act_op
        self._loss_op = surr_loss
        self._train_op = train_op

    def act(self, observation):
        assert observation.shape[0] == 1
        action_probs = self._sess.run(self._act_op, feed_dict={self._observations: observation})

        cs = np.cumsum(action_probs)
        idx = sum(cs < np.random.rand())
        return idx

    def train(self, observations, actions, advantages):
        loss, _ = self._sess.run([self._loss_op, self._train_op], 
            feed_dict={self._observations:observations, self._actions:actions, self._advantages:advantages})
        return loss
