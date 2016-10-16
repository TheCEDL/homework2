# CEDL homework 2
### Team member: Juan-ting Lin, Meng-li Shih, Fu-en, Wang

## Overview:

 In this homework, we will try to solve the classic control problem - CartPole.
 
 <img src="https://cloud.githubusercontent.com/assets/7057863/19025154/dd94466c-8946-11e6-977f-2db4ce478cf3.gif" width="400" height="200" />
 
 CartPole is an environment which contains a pendulum attached by an un-actuated joint to a cart, and the goal is to prevent it from falling over. You can apply a force of +1 or -1 to the cart. A reward of +1 is provided for every timestep that the pendulum remains upright.
 
## Implementation:

 In problem 1~4, we implement a simple agent and improve it with mote-carlo sampling and policy-based method
 * **Problem 1:**
 
   In problem 1, we construct a simple 2-layer fully-connected feedforward network to perform the policy prediction:
   
   ```
   x=tf.contrib.layers.fully_connected(inputs=self._observations,num_outputs=hidden_dim,
                                          weights_initializer=tf.random_normal_initializer(),
                                          biases_initializer=tf.random_normal_initializer(),
                                          activation_fn = tf.tanh)
   y=tf.contrib.layers.fully_connected(inputs=x,num_outputs=out_dim,
                                          weights_initializer=tf.random_normal_initializer(),
                                          biases_initializer=tf.random_normal_initializer())
   probs=tf.nn.softmax(y)
   ```
   
 * **Problem 2:**
 
   After constructing the policy prediction network, we need to compute the surrogate loss to obtain the policy gradient:
   
   ```
   surr_loss = tf.mul(log_prob,self._advantages)
   ```
   
 * **Problem 3:**
  
   Implement a function that computes the accumulated discounted rewards of each timestep t from t to the end of the episode:
   
   ```
   def discount_cumsum(x, discount_rate):
    sum = 0
    acc_sum = [0 for _ in range(len(x))]
    for i in range(len(x)-1,-1,-1):
        exp = len(x)-i-1
        sum += x[i]*(discount_rate**exp)
        acc_sum[i]=sum

    return acc_sum
   ```
 * **Problem 4:**
 
   Use baseline to reduce the variance of our gradient estimate. By doing so, we can imporove the shortcoming of the mote-carlo method:
   
   ```
   a = r-b
   ```
   
# Discussion:

  In problem 5~6, we need to discuss the effect of having baseline substracted
  
  * Problem 5:
    
    here we compare the result with and without substracting baseline:
    
|Substract baseline|Without substracting baseline|
|---|---|
|<img src="https://github.com/brade31919/homework2/blob/master/pic/with_baseline_10.png" width="700"> |<img src="https://github.com/brade31919/homework2/blob/master/pic/without_baseline_10.png" width="700"> |
|<img src="https://github.com/brade31919/homework2/blob/master/pic/with_baseline_std.png" width="700"> |<img src="https://github.com/brade31919/homework2/blob/master/pic/without_baseline_std.png" width="700">|

   In each case of problem 5, we re-conduct 10 training process to get statistical result. The upper part of the block is the average return plot of 10 experiments. It is quite obvious that the case without baseline substracted has larger variation during different training process. To prove our observation, we plot the standard deviation during the training process in the lower part of the block. The center curve is the mean of 10 training processes and the vertical line is the std ( variance^0.5) of each step. As the plot shows us, if we don't substract the baseline, we'll get larger std during training!
   For detailed explanation, we record all the returns and rewards in each single path and calculate its variance. We show the result in the table below
   
| |Substract baseline|Without substract baseline|
|---|---|---|
|returns|542.5|587.1|
|advantages|117|587.1|

If we compare the return of the 2 cases, there is no significant difference. Nevertheless, if we compare the advantages(a=r-b), the case with baseline substraction has a much lower variance.

  * Problem 6:
  
    In each path, the action performed in latter stages will be discounted exponantially. As a result, in spite of substracting baseline from the average returns, the actions in the latter stage have less influence on the learning process. By doing this, it's like we always encourage and discourage half of the actions. This might help us to control the policy gradient. 

   
  
   
