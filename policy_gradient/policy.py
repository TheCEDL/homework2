import tensorflow as tf
import numpy as np

class CategoricalPolicy(object):
    def __init__(self, in_dim, out_dim, hidden_dim, optimizer, session):

        # Placeholder Inputs
        self._observations = tf.placeholder(tf.float32, shape=[None, in_dim], name="observations")
        self._actions = tf.placeholder(tf.int32, name="actions")
        self._advantages = tf.placeholder(tf.float32, name="advantages")

        self._opt = optimizer
        self._sess = session
        
        """
        Problem 1:
        
        1. Use TensorFlow to construct a 2-layer neural network as stochastic policy.
           Each layer should be fully-connected and have size `hidden_dim`.
           Use tanh as the activation function of the first hidden layer, and append softmax layer after the output
           of the neural network to get the probability of each possible action.
        
        2. Assign the output of the softmax layer to the variable `probs`.
           Let's assume n_batch equals to `self._observations.get_shape()[0]`,
           then shape of the variable `probs` should be [n_batch, n_actions].
           
        Sample solution is about 2~4 lines.
        """
        # YOUR CODE HERE >>>>>>
        # probs = ???
        # <<<<<<<<

        # --------------------------------------------------
        # This operation (variable) is used when choosing action during data sampling phase
        # Shape of probs: [1, n_actions]
        
        act_op = probs[0, :]

        # --------------------------------------------------
        # Following operations (variables) are used when updating model
        # Shape of probs: [n_timestep_per_iter, n_actions]

        # 1. Find first index of action for each timestep in flattened vector form
        #    e.g., if we have n_timestep_per_iter = 2, then we'll get [0, 2]
        #    0 is the first index of action for timestep t = 0, and 2 is the first index of action for timestep t = 1
        action_idxs_flattened = tf.range(0, tf.shape(probs)[0]) * tf.shape(probs)[1]

        # 2. Add index of the action chosen at each timestep
        #    e.g., if index of the action chosen at timestep t = 0 is 1, and index of the action
        #    chosen at timestep = 1 is 0, then `action_idxs_flattened` == [0, 2] + [1, 0] = [1, 2]
        action_idxs_flattened += self._actions

        # 3. Gather the probability of action at each timestep
        #    e.g., tf.reshape(probs, [-1]) == [0.1, 0.9, 0.8, 0.2]
        #    since action_idxs_flattened == [1, 2], we'll get [0.9, 0.8], which is the probability when we choose each action
        probs_vec = tf.gather(tf.reshape(probs, [-1]), action_idxs_flattened)

        # Add 1e-8 to `probs_vec` so as to prevent log(0) error
        log_prob = tf.log(probs_vec + 1e-8)
        
        """
        Problem 2:
        
        1. Trace the code above
        2. Currently, variable `self._advantages` represents accumulated discounted rewards 
           from each timestep to the end of an episode 
        3. Compute surrogate loss and assign it to variable `surr_loss`
           
        Sample solution is about 1~3 lines.
        """
        # YOUR CODE HERE >>>>>>
        # surr_loss = ???
        # <<<<<<<<

        grads_and_vars = self._opt.compute_gradients(surr_loss)
        train_op = self._opt.apply_gradients(grads_and_vars, name="train_op")

        # --------------------------------------------------
        # This operation (variable) is used when choosing action during data sampling phase
        self._act_op = act_op
        
        # --------------------------------------------------
        # These operations (variables) are used when updating model
        self._loss_op = surr_loss
        self._train_op = train_op

    def act(self, observation):
        # expect observation to be of shape [1, observation_space]
        assert observation.shape[0] == 1
        action_probs = self._sess.run(self._act_op, feed_dict={self._observations: observation})
        
        # `action_probs` is an array that has shape [1, action_space], it contains the probability of each action
        # sample an action according to `action_probs`
        cs = np.cumsum(action_probs)
        idx = sum(cs < np.random.rand())
        return idx
    
    def train(self, observations, actions, advantages):
        loss, _ = self._sess.run([self._loss_op, self._train_op], feed_dict={self._observations:observations, self._actions:actions, self._advantages:advantages})
        return loss