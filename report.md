#組員
張嘉哲、林暘竣 <br>
#problem 1
solution: <br>
fc_w1 = tf.get_variable("fc_w1",[in_dim, hidden_dim]) <br>
fc_b1 = tf.get_variable("fc_b1", [hidden_dim]) <br>
fc_w2 = tf.get_variable("fc_w2", [hidden_dim, out_dim]) <br>
fc_b2 = tf.get_variable("fc_b2", [out_dim]) <br>
fc1 = tf.nn.tanh(tf.matmul(self._observations, fc_w1) + fc_b1) <br>
fc2 = tf.matmul(fc1, fc_w2) + fc_b2 <br>
probs = tf.nn.softmax(fc2) <br>

#problem 2
solution: <br>
surr_loss = -tf.reduce_mean(log_prob*self._advantages) <br>
  
#problem 3
solution: <br>
  ans = np.zeros((len(x), )) <br>
    for idx, _ in enumerate(x): <br>
      for j in range(len(x)-idx): <br>
        ans[idx] += x[j+idx]*discount_rate**(len(x)-idx-j-1) <br>
  return ans <br>
  
#problem 4
solution: <br>
 a = r - b <br>

#problem 5
The policy gradient has high variance. <br>
If we subtract a baselne from policy gradient, this way can reduce variance. (with some bias) <br>

#problem 6
We use a discounted reward. <br>
This method is the weighted sum of all rewards afterwards,
but the later rewards are less important. <br>
One good method is to “standardize” these returns <br>
  -subtract mean <br>
  -divide by standard deviation <br>
This way we’re always encouraging and discouraging roughly half of the performed actions. <br>

#Reference
Deep Reinforcement Learning: Pong from Pixels (http://karpathy.github.io/2016/05/31/rl/) <br>
