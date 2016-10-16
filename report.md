# Policy gradient report
Member: 廖元宏(102061137), 莊景堯(102061145)    
Implement a simple agent with REINFORCE algorithm, which uses the MC sampling and policy gradient.   

## Problem 1~4

- Problem 1: construct a simple two layer FC layer for policy prediction 
Here we use 2-layer neural network to represent the policy. Make sure you add softmax layer to represent probability distribution.
```python  
h1 = tf.contrib.layers.fully_connected(self._observations, num_outputs=hidden_dim, activation_fn=tf.tanh)   
h2 = tf.contrib.layers.fully_connected(h1, num_outputs=out_dim, activation_fn=None)
probs = tf.nn.softmax(h2)
```
Use a simple two-layer perceptron to embed state to action space

- Problem 2: surrogate loss
Since the optimizer in Tensorflow only support minimizing loss (gradient descent), so we simply add a minus sign to represent **gradient ascent**.
```python
surr_loss = -tf.reduce_mean(tf.mul(log_prob, self._advantages))
```

- Problem 3: accumulated reward
Construct a simple for-loop to calculate the accumulated discounted from the end of the game to the start.

```python
def discount_cumsum(x, discount_rate):
	discounted_r = np.zeros(len(x))
  	num_r = len(x)
  	for i in range(num_r):
	  	discounted_r[i] = x[i]*math.pow(discount_rate,i)
	discounted_r = np.cumsum(discounted_r[::-1])
  	return discounted_r[::-1]
```   

- Problem 4: Advantage function

```python
a = r - b
```
where a is the advantage function, r is the accumulated reward, and b is the predicted baseline.

## Problem 5

Here I compare the result of with/without variance reduction:  

|With baseline|Wihtout baseline|
|---|---|
|<img src="https://github.com/andrewliao11/homework2/blob/master/with_variance_reduce_max.png?raw=true" width="700">|<img src="https://github.com/andrewliao11/homework2/blob/master/without_variance_reduce_max.png?raw=true" width="700">|
<p align="center">X-axis: iteration, Y-axis: return reward</p>

This figure implies the variance of the case with and without baseline. I run each case for 10 times and record the return reward over each iteration(if the game reaches over 195, the game terminates). The solid line denotes the average return reward through iteration. The upper line implies the max return reward in that iteration, while the lower line implies the min retrun reward in that iteration.   
**P.S.** The result is not quite apparent, and I guess that I should run the games for 100 iteration(fixed iteration) for the 10 gmaes and plot the figure   
**P.S.** Why we need these variance reduction? Here, we're using REINFORCE, which is known to be with high variance (highly depends on your initial samples)   
**P.S.** Actually, the results highly depends on the initial parameter. If the initial return reward is around 30, the agent can reach 195 around 70 iterations; If the initial return reward is around 15, it'll take about 100 iterartion to reach 195.

## Problem 6

The reseaon why we need to standardize the advantage function is that when we calculate the accumulated reward, the immediate reward that we get is exponentially discounted by the discounted factor. This action in latter stage can't learn effeciently. So, If we standardize the advantage function over time steps, in this way we’re always encouraging and discouraging roughly half of the performed actions. Mathematically you can also interpret these tricks as a way of controlling the variance of the policy gradient estimator.    
**Additional survey on general advantage estimation(GAE):**   
- ***High-Dimensional Continuous Control Using Generalized Advantage Estimation*** [[ICLR 2016]](https://arxiv.org/abs/1506.02438)
	- John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel
	- In extremely high dimensional task(like continuous control in 3D environment), stability is a key point.
	- Propose an effective variance reduction scheme for policy gradients, which called generalized advantage estimation (GAE)
	-  Motivation of GAE: Supposed we have fixed length of steps, from eq.15,  we know that the bias of each advantage function is **k-dependent**. So, as k increases, the biased term becomes more ignorable, while the variance increases and vice versa. (if you found this concept is abstract, think of MC is unbiased but with high variance, while TD is biased, but with los variance)
	-  ***λ*** is a new concept included in this paper. 
		-  If λ = 0 (like eq.17), then we have low variance, and is biased
		-  If λ = 1 (like eq.18), then we have high variance, and is unbased

## Reference

- [Deep Reinforcement Learning: Pong from Pixels](karpathy.github.io/2016/05/31/rl/)
- [Deep-Reinforcement-Learning-Survey](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey)
