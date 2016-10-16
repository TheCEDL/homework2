# Homework2 - Policy Gradient Report
By NTHUCS Undergrad Mark, 董晉東, 102000002

## Problem 1: construct a neural network to represent policy
Given the hint:<br>
> 1. The hidden layer should be fully-connected and have size `hidden_dim`.<br>
> 2. tanh as the activation function of the first hidden layer.<br>
> 3. softmax as the output of the network.

The answer is below:
```python
hidden1 = tf.contrib.layers.fully_connected(self._observations, hidden_dim, activation_fn=tf.tanh)
probs = tf.contrib.layers.fully_connected(hidden1, out_dim, activation_fn=tf.nn.softmax)
```

## Problem 2: compute the surrogate loss
Given the hint:<br>
> 1. `self._advantages` represents the accumulated discounted rewards -> means R.
> 2. The loss function formula provided in the ipython notebook.

The answer is simply:
```python
surr_loss = -tf.reduce_mean(tf.mul(self._advantages, log_prob))
```
where we compute the mean over the multiplication of log probability `log_prob`and the accumulated discounted rewards `R`. Remember to add a negative sign in front to make the loss function find the maximum.

## Problem 3: implement a function that computes the accumulated discounted rewards of each timestep
We can observe that the process_path given the x of `[t, t+1, t+2]`, so the accumulated discounted reward should be:
```
discounted_reward = [t + (r^1)*(t+1) + (r^2)*(t+2), (t+1) + (r^1)*(t+2), (t+2)]
```
For example:<br>
```
given x = [1, 2, 3], rate = 0.9
discounted_reward = [1 + (0.9^1)*2 + (0.9^2)*3, 2 + 0.9*3, 3] = [5.23, 4.7, 3]
```
My implementation is pure python, viewing backwardly of the list. Item i is item i+1 multiply r plus the corresponding reward
```python
def discount_cumsum(x, discount_rate):
    # YOUR CODE HERE >>>>>>
    # The sequence of x should be [t, t+1, t+2], so the accumulated discounted reward should be
    #  [t + (rate^1)*(t+1) + (rate^2)*(t+2), (t+1) + (rate^1)*(t+2), t+2]
    # 
    # For example: x = [1, 2, 3], rate = 0.9:
    #  [1 + (0.9^1)*2 + (0.9^2)*3, 2 + (0.9^1)*3, 3] = [5.23, 4.7, 3]
    dis_rw = [x[-1]]
    for i in range(1, len(x)):
        dis_rw.insert(0, dis_rw[-i] * discount_rate + x[-i-1])
        
    return dis_rw
```

## Problem 4: use baseline to reduce the variance of our gradient estimate
We can see how to reduce variance from baseline in David Silver's slides:<br>
![reduce_variance_from_baseline](https://github.com/markakisdong/homework2/blob/master/reduce_variance_from_baseline.png)<br>
So the answer is simply:
```python
a = r - b
```

## Problem 5: remove baseline and compare the variance/performance
Theoretically, removing baseline should increase the variance and decrease the performance. However, for the performance part, this homework depends highly on the initial values, so the result is not too obvious. I ran the test for dozens of times to get a result where the policy gradient with baseline is better.<br>
The figures below are return values with and without baseline, we can see that the performance is better with baseline (converge at less iterations).<br>
<img src="https://github.com/markakisdong/homework2/blob/master/return_w_baseline.png" width="420">
<img src="https://github.com/markakisdong/homework2/blob/master/return_wo_baseline.png" width="420"><br>

And for the variance, there are huge difference between the model with baseline and the model without baseline, shown in below:<br>
<img src="https://github.com/markakisdong/homework2/blob/master/std_comparison.png" width="600"><br>
The variance of the model without baseline is much larger than that of the model with baseline.

## Problem 6: why we need to normalize the advantages
The normalization steps can help the network learns efficiently by reducing the high variances between each accumulated reward, and can learn faster and more stable.

## Reference
[David Silver's Slide - Policy Gradient](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/pg.pdf)
