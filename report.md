# Policy gradient report

## Problem 1~4:
Implement a simple agent with REINFORCE algorithm, which uses the MC sampling and policy gradient.
	- Problem 1: construct a simple two layer FC layer for policy prediction   
	```
	h1 = tf.contrib.layers.fully_connected(self._observations, num_outputs=hidden_dim, activation_fn=tf.tanh)	
	h1 = tf.contrib.layers.fully_connected(self._observations, num_outputs=hidden_dim, activation_fn=tf.tanh)
	h2 = tf.contrib.layers.fully_connected(h1, num_outputs=out_dim, activation_fn=None)
	probs = tf.nn.softmax(h2)	
	```   
	Use a simple two-layer perceptrons to embed from state space to action space.   
	- Problem 2: surrogate loss   
	```
	surr_loss = tf.reduce_mean(tf.mul(log_prob, self._advantages))
	```   
	compute the surrogate loss and use optimizer to maximize it.   
	- Problem 3: accumulated reward   
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
	- Problem 4:    
	```
	a = r - b
	```   
	where a is the advantage function, r is the accumulated reward, and b is the predicted baseline.

## Problem 5:
Here I compare the result of with/without variance reduction:  

|With baseline|Wihtout baseline|
|---|---|
|<img src="https://github.com/andrewliao11/homework2/blob/master/with_variance_reduce_max.png?raw=true" width="700">|<img src="https://github.com/andrewliao11/homework2/blob/master/without_variance_reduce_max.png?raw=true" width="700">|
This figure implies the variance of the case with and without baseline. I run each case for 10 times and record the return reward over each iteration(if the game reaches over 195, the game terminates). The solid line denotes the average return reward through iteration. The upper line implies the max return reward in that iteration, while the lower line implies the min retrun reward in that iteration.   
P.S. The result is not quite apparent, and I guess that I should run the games for 100 iteration(fixed iteration) for the 10 gmaes and plot the figure
	
## Problem 6:
The reseaon why we need to standardize the advantage function is that when we calculate the accumulated reward, the immediate reward that we get is exponentially discounted by the discounted factor. This action in latter stage can't learn effeciently. So, If we standardize the advantage function over time steps, it's expected that we always encourage and discourage half of actions (since we substract the mean). 

## Reference 
[Deep Reinforcement Learning: Pong from Pixels](karpathy.github.io/2016/05/31/rl/)




