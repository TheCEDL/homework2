# Policy gradient report

- Problem 1~4:
Implement a simple agent with REINFORCE algorithm, which uses the MC sampling and policy gradient.  
Problem 1: construct a simple two layer FC layer for policy prediction
```
h1 = tf.contrib.layers.fully_connected(self._observations, num_outputs=hidden_dim, activation_fn=tf.tanh)
h2 = tf.contrib.layers.fully_connected(h1, num_outputs=out_dim, activation_fn=None)
probs = tf.nn.softmax(h2)	
```
Use a simple two-layer perceptrons to embed from state space to action space.

Problem 2: surrogate loss
```
surr_loss = tf.reduce_mean(tf.mul(log_prob, self._advantages))
```
compute the surrogate loss and use optimizer to maximize it.
Problem 3: accumulated reward
```
def discount_cumsum(x, discount_rate):
  discounted_r = np.zeros(len(x))
  num_r = len(x)
  for i in range(num_r):
	  discounted_r[i] = x[i]*math.pow(discount_rate,i)
  discounted_r = np.cumsum(discounted_r[::-1])
  return discounted_r[::-1]
```
Use the numpy cumcum is quite simple and remember to reverse the array. 
Problem 4: 
```
a = r - b
```
where a is the advantage function, r is the accumulated reward, and b is the predicted baseline.

- Problem 5:
Here I compare the result of with/without variance reduction:  

|With baseline|Wihtout baseline|
|---|---|
|<img src="https://github.com/andrewliao11/homework2/blob/master/with_variance_reduce_max.png?raw=true" width="700">|<img src="https://github.com/andrewliao11/homework2/blob/master/without_variance_reduce_max.png?raw=true" width="700">|

