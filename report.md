#problem 1
solution:
  fc_w1 = tf.get_variable("fc_w1",[in_dim, hidden_dim])
  fc_b1 = tf.get_variable("fc_b1", [hidden_dim])
  fc_w2 = tf.get_variable("fc_w2", [hidden_dim, out_dim])
  fc_b2 = tf.get_variable("fc_b2", [out_dim])
  fc1 = tf.nn.tanh(tf.matmul(self._observations, fc_w1) + fc_b1)
  fc2 = tf.matmul(fc1, fc_w2) + fc_b2
  probs = tf.nn.softmax(fc2)

#problem 2
solution:
  surr_loss = -tf.reduce_mean(log_prob*self._advantages)
  
#problem 3
solution:
  ans = np.zeros((len(x), ))
    for idx, _ in enumerate(x):
      for j in range(len(x)-idx):
        ans[idx] += x[j+idx]*discount_rate**(len(x)-idx-j-1)
  return ans
  
#problem 4
solution:
  a = r - b

#problem 5
The policy gradient has high variance.
If we subtract a baselne from policy gradient, this way can reduce variance. (with some bias)

#problem 6
We use a discount reward.
This method is the weighted sum of all rewards afterwards,
but the later rewards are less important.
One good method is to “standardize” these returns
  -subtract mean
  -divide by standard deviation
This way we’re always encouraging and discouraging roughly half of the performed actions.

#Reference
Deep Reinforcement Learning: Pong from Pixels (http://karpathy.github.io/2016/05/31/rl/)
