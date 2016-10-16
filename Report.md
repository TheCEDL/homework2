# Homework2 - Policy Gradient

## Problem 1
We construct a two fully-connected layer for training.

    W1 = tf.Variable(tf.random_normal([in_dim, hidden_dim]))
    b1 = tf.Variable(tf.zeros([hidden_dim]))
    W2 = tf.Variable(tf.random_normal([hidden_dim, out_dim]))
    b2 = tf.Variable(tf.zeros([out_dim]))
    output1 = tf.nn.tanh(tf.matmul(self._observations, W1) + b1)
    probs = tf.nn.softmax(tf.matmul(output1, W2) + b2)

## Problem 2
It's very easy to implement surrogate loss using Tensorflow, but the optimizer in Tensorflow only minimize loss, so we need to add a negative sign(Since we want gradient ascent).

    surr_loss = -tf.reduce_mean(tf.mul(log_prob, self._advantages))

## Problem 3
Accumulated reward

    def discount_cumsum(x, discount_rate):
        R = [x[-1]]
        for V in x[:-1][::-1]:
            R.append(V + discount_rate*R[-1])
    return np.array(R[::-1])

## Problem 4
Just simply do accumulated reward minus baseline

    a = r - b

## Problem 5
Compare the variance and performance before and after adding baseline (with figures is better)
Performance without adding baseline converge faster(69 iteraion v.s 81 iteration)

<img src="./w69.png" width="640">   
<img src="./81.png" width="640">

* Observations:
    * Initial value of policy is highly related to the convergence time. If it's a good initial value, then it reaches avg_return >= 195 faster.
    * Because we're trying to determine the best action for a state, so we need a baseline to compare from. In theory, training with baseline should converge faster than without baseline.

## Problem 6
The reason why we need normalized advantage function is that we want to get a better policy at each time step. When accumulating rewards, the reward we get at a time step is discounted by a factor. However, the later rewards are exponentially less important for the stimulation of action at current time step. Therefore, we normalize the advantage function over time steps to stimulate the action at each time step.

## Reference paper

[Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748)

[HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION](https://arxiv.org/pdf/1506.02438v5.pdf)
