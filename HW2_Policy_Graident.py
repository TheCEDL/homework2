
# coding: utf-8

# In this assignment, you will solve a classic control problem - CartPole using policy gradient methods.
#
# First, you will implement the "vanilla" policy gradient method, i.e., a method that repeatedly computes **unbiased** estimates $\hat{g}$ of $\nabla_{\theta} E[\sum_t r_t]$ and takes gradient ascent steps $\theta \rightarrow \theta + \epsilon \hat{g}$ so as to increase the total rewards collected in each episode. To make sure our code can solve multiple MDPs with different policy parameterizations, provided code follows an OOP manner and represents MDP and Policy as classes.
#
# The following code constructs an instance of the MDP using OpenAI gym.

import gym
import tensorflow as tf
import numpy as np
from policy_gradient import util
from policy_gradient.policy import CategoricalPolicy
from policy_gradient.baselines.linear_feature_baseline import LinearFeatureBaseline

np.random.seed(0)
tf.set_random_seed(0)

# CartPole-v0 is a MDP with finite state and action space.
# In this environment, A pendulum is attached by an un-actuated joint to a cart,
# and the goal is to prevent it from falling over. You can apply a force of +1 or -1 to the cart.
# A reward of +1 is provided for every timestep that the pendulum remains upright.
# To visualize CartPole-v0, please see https://gym.openai.com/envs/CartPole-v0

env = gym.make('CartPole-v0')


# ## Problem 1: construct a neural network to represent policy
#
# Make sure you know how to construct neural network using tensorflow.
#
# 1. Open **homework2/policy_gradient/policy.py**.
# 2. Follow the instruction of Problem 1.

# ## Problem 2: compute the surrogate loss
#
# If there are $N$ episodes in an iteration, then for $i$ th episode we define $R_t^i = \sum_{{t^′}=t}^T \gamma^{{t^′}-t}r(s_{t^′}, a_{t^′})$ as the accumulated discounted rewards from timestep $t$ to the end of that episode, where $\gamma$ is the discount rate.
#
# The pseudocode for the REINFORCE algorithm is as below:
#
# 1. Initialize policy $\pi$ with parameter $\theta_1$.
# 2. For iteration $k = 1, 2, ...$:
#     * Sample N episodes $\tau_1, \tau_2, ..., \tau_N$ under the current policy $\theta_k$, where $\tau_i =(s_i^t,a_i^t,R_i^t)_{t=0}^{T−1}$. Note that the last state is dropped since no action is taken after observing the last state.
#     * Compute the empirical policy gradient using formula: $$\hat{g} = E_{\pi_\theta}[\nabla_{\theta} log\pi_\theta(a_t^i | s_t^i) R_t^i]$$
#     * Take a gradient step: $\theta_{k+1} = \theta_k + \epsilon \hat{g}$.
#
#
# Note that we can transform the policy gradient formula as
#
# $$\hat{g} = \nabla_{\theta} \frac{1}{(NT)}(\sum_{i=1}^N \sum_{t=0}^T log\pi_\theta(a_t^i | s_t^i) R_t^i)$$
#
# and $L(\theta) = \frac{1}{(NT)}(\sum_{i=1}^N \sum_{t=0}^T log\pi_\theta(a_t^i | s_t^i) R_t^i)$ is called the surrogate loss.
#
# We can first construct the computation graph for $L(\theta)$, and then take its gradient as the empirical policy gradient.
#
#
# 1. Open **homework2/policy_gradient/policy.py**.
# 2. Follow the instruction of Problem 2.

sess = tf.Session()

# Construct a neural network to represent policy which maps observed state to action.
in_dim = util.flatten_space(env.observation_space)
out_dim = util.flatten_space(env.action_space)
hidden_dim = 8

opt = tf.train.AdamOptimizer(learning_rate=0.01)
policy = CategoricalPolicy(in_dim, out_dim, hidden_dim, opt, sess)

sess.run(tf.initialize_all_variables())


# # Problem 3
#
# Implement a function that computes the accumulated discounted rewards of each timestep _t_ from _t_ to the end of the episode.
#
# For example:
#
# ```python
# rewards = [1, 1, 1]
# discount_rate = 0.99
# util.discount_cumsum(rewards, discount_rate)
# ```
#
# should return:
#
# `array([ 2.9701,  1.99  ,  1.    ])`
#
# 1. Open **homework/policy_gradient/util.py**.
# 2. Implement the commented function.

# # Problem 4
#
# Use baseline to reduce the variance of our gradient estimate.
#
# 1. Fill in the function `process_paths` of class `PolicyOptimizer` below.

class PolicyOptimizer(object):
    def __init__(self, env, policy, baseline, n_iter, n_episode, path_length,
        discount_rate=.99):

        self.policy = policy
        self.baseline = baseline
        self.env = env
        self.n_iter = n_iter
        self.n_episode = n_episode
        self.path_length = path_length
        self.discount_rate = discount_rate

    def sample_path(self):
        obs = []
        actions = []
        rewards = []
        ob = self.env.reset()

        for _ in range(self.path_length):
            a = self.policy.act(ob.reshape(1, -1))
            next_ob, r, done, _ = self.env.step(a)
            obs.append(ob)
            actions.append(a)
            rewards.append(r)
            ob = next_ob
            if done:
                break

        return dict(
            observations=np.array(obs),
            actions=np.array(actions),
            rewards=np.array(rewards),
        )

    def process_paths(self, paths):
        for p in paths:
            if self.baseline != None:
                b = self.baseline.predict(p)
            else:
                b = 0

            # `p["rewards"]` is a matrix contains the rewards of each timestep in a sample path
            r = util.discount_cumsum(p["rewards"], self.discount_rate)

            """
            Problem 4:

            1. Variable `b` is the reward predicted by our baseline
            2. Use it to reduce variance and then assign the result to the variable `a`

            Sample solution should be only 1 line.
            """
            # YOUR CODE HERE >>>>>>
            # a = ???
	    a=r-b
            # <<<<<<<<

            p["returns"] = r
            p["baselines"] = b
            p["advantages"] = (a - a.mean()) / (a.std() + 1e-8) # normalize

        obs = np.concatenate([ p["observations"] for p in paths ])
        actions = np.concatenate([ p["actions"] for p in paths ])
        rewards = np.concatenate([ p["rewards"] for p in paths ])
        advantages = np.concatenate([ p["advantages"] for p in paths ])

        return dict(
            observations=obs,
            actions=actions,
            rewards=rewards,
            advantages=advantages,
        )

    def train(self):
        for i in range(1, self.n_iter + 1):
            paths = []
            for _ in range(self.n_episode):
                paths.append(self.sample_path())
            data = self.process_paths(paths)
            loss = self.policy.train(data["observations"], data["actions"], data["advantages"])
            avg_return = np.mean([sum(p["rewards"]) for p in paths])
            print("Iteration {}: Average Return = {}".format(i, avg_return))

            # CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.
            if avg_return >= 195:
                print("Solve at {} iterations, which equals {} episodes.".format(i, i*100))
                break

            if self.baseline != None:
                self.baseline.fit(paths)


n_iter = 200
n_episode = 100
path_length = 200
discount_rate = 0.99
baseline = LinearFeatureBaseline(env.spec)

po = PolicyOptimizer(env, policy, baseline, n_iter, n_episode, path_length,
                     discount_rate)

# Train the policy optimizer
po.train()


# # Verify your solutions
#
# if you solve the problems 1~4 correctly, your will solve CartPole with roughly ~ 80 iterations.

# # Problem 5
# Replacing line
#
# `baseline = LinearFeatureBaseline(env.spec)`
#
# with
#
# `baseline = None`
#
# can remove the baseline.
#
# Modify the code to compare the variance and performance before and after adding baseline.
# Then, write a report about your findings. (with figures is better)

# # Problem 6
#
# In function process_paths of class `PolicyOptimizer`, why we need to normalize the advantages? i.e., what's the usage of this line:
#
# `p["advantages"] = (a - a.mean()) / (a.std() + 1e-8)`
#
# Include the answer in your report.
