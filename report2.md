***

## Homework2 Report

105061469 Haiyang Chang

***

## Problem1
I constructed a 2-layer neural network as required. The W1,W2 cannot be zeros like b1,b2, otherwise the Average Return will remain really low. I thought this is becauses the neural network would unable to learn the features of training data since 0*in_dim=0. So I use the truncated normal distribution with its stddev being the reciprocal to the input data to make it easier.
+ W1 = tf.Variable(tf.truncated_normal([in_dim, hidden_dim],stddev=1.0))
+ W2 = tf.Variable(tf.truncated_normal([hidden_dim, out_dim],stddev=1.0))
+ b1 = tf.Variable(tf.zeros([hidden_dim]))
b2 = tf.Variable(tf.zeros([out_dim]))
+ hidden1 = tf.nn.tanh(tf.matmul(self._observations, W1) + b1)
+ probs = tf.nn.softmax(tf.matmul(hidden1, W2) + b2)

## Problem2

The smaller loss, the higher gradient.
+ surr_loss = -tf.reduce_mean(tf.mul(log_prob, self._advantages))

## Problem3

I tested the function on python to ensure  its accuracy.
+ def discount_cumsum(x, discount_rate):
  	len_x = len(x)
  	array_discount_rate = np.zeros(len_x)
  	for i in range(len_x):
		array_discount_rate[i] = x[i] * discount_rate**i
   	array_discount_rate =np.cumsum(array_discount_rate)
   	return array_discount_rate[::-1]

## Problem4

There could be other forms, but this is the simplest form.
+  a = r - b

## Problem5

Without baseline the Average Return increased to a high level quickly, but finally achieved the similar level as the performance with baseline. The baseline is to guarantee that the gradient will increase. So if the initial state of neural network is not chosen properly, the performance without baseline may not be as satisfactory as with baseline do.
+ With baseline
![withbaseline.png](https://ooo.0o0.ooo/2016/12/08/5849ac501e81f.png)
+ Without baseline
![withoutbaseline.png](https://ooo.0o0.ooo/2016/12/08/5849ac501e0e4.png)

## Problem6

The reward is discounted by the discounted rate, with makes the later action seems less important than the former action. By normalizing the advantage, the training process can go on steadily.