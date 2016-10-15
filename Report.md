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
  
  * Substract baseline from average return:
  
  * Without substracting baseline:
  
   
