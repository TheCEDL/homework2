"""
In this assignment, you will solve a classic control problem - CartPole using policy gradient methods.
 
First, you will implement the "vanilla" policy gradient method, i.e., a method that repeatedly computes **unbiased** estimates and takes gradient ascent steps so as to increase the total rewards collected in each episode. To make sure our code can solve multiple MDPs with different policy parameterizations, provided code follows an OOP manner and represents MDP and Policy as classes.

The following code constructs an instance of the MDP using OpenAI gym.
"""
import gym
import tensorflow as tf
import numpy as np
from policy_gradient import util
from policy_gradient.policy import CategoricalPolicy
from policy_gradient.baselines.linear_feature_baseline import LinearFeatureBaseline

np.random.seed(0)
tf.set_random_seed(0)

"""
CartPole-v0 is a MDP with finite state and action space. 
In this environment, A pendulum is attached by an un-actuated joint to a cart, 
and the goal is to prevent it from falling over. You can apply a force of +1 or -1 to the cart.
A reward of +1 is provided for every timestep that the pendulum remains upright. 
To visualize CartPole-v0, please see https://gym.openai.com/envs/CartPole-v0
"""
env = gym.make('CartPole-v0')


"""
## Problem 1: construct a neural network to represent policy

Make sure you know how to construct neural network using tensorflow.

1. Open **homework2/policy_gradient/policy.py**.
2. Follow the instruction of Problem 1.
"""
# Problem 1 in policy.py
"""
## Problem 2: compute the surrogate loss

If there are N episodes in an iteration, then for i th episode we define R as the accumulated discounted rewards from timestep t to the end of that episode.

The pseudocode for the REINFORCE algorithm is as below:

1. Initialize policy with parameter.
2. For iteration $k = 1, 2, ...$:
    * Sample N episodes under the current policy. Note that the last state is dropped since no action is taken after observing the last state.
    * Compute the empirical policy gradient.
    * Take a gradient step.
 
We can first construct the computation graph, and then take its gradient as the empirical policy gradient.

1. Open **homework2/policy_gradient/policy.py**.
2. Follow the instruction of Problem 2.
"""
sess = tf.Session()

# Construct a neural network to represent policy which maps observed state to action. 
in_dim = util.flatten_space(env.observation_space)
out_dim = util.flatten_space(env.action_space)
hidden_dim = 8

opt = tf.train.AdamOptimizer(learning_rate=0.01)
policy = CategoricalPolicy(in_dim, out_dim, hidden_dim, opt, sess)

sess.run(tf.initialize_all_variables())

"""
## Problem 3

Implement a function that computes the accumulated discounted rewards of each timestep from t to the end of the episode.

For example:

```python
rewards = [1, 1, 1]
discount_rate = 0.99
util.discount_cumsum(rewards, discount_rate)
```

should return:

`array([ 2.9701,  1.99  ,  1.    ])`

1. Open **homework/policy_gradient/util.py**.
2. Implement the commented function.
"""
# Problem 3 in util.py
"""
## Problem 4

Use baseline to reduce the variance of our gradient estimate.

1. Fill in the function `process_paths` of class `PolicyOptimizer` below.
"""
import pdb
class PolicyOptimizer(object):
    def __init__(self, env, policy, baseline, n_iter, n_episode, path_length, discount_rate=.99):
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
            # `p["rewards"]` is a matrix contains the rewards of each timestep in a sample path
            r = util.discount_cumsum(p["rewards"], self.discount_rate)

            if self.baseline != None:
                b = self.baseline.predict(p)
            else:
                b = np.zeros(len(r))
            
            
            """
            Problem 4:

            1. Variable `b` is the reward predicted by our baseline
            2. Use it to reduce variance and then assign the result to the variable `a`

            Sample solution should be only 1 line.
            """
            # YOUR CODE HERE >>>>>>
            a = r - b
            # a = ???
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
        all_avg_return = []
        all_max_return = []
        all_min_return = []
        for i in range(1, self.n_iter + 1):
            paths = []
            for _ in range(self.n_episode):
                paths.append(self.sample_path())
            data = self.process_paths(paths)
            loss = self.policy.train(data["observations"], data["actions"], data["advantages"])
            all_return = [sum(p["rewards"]) for p in paths]
            avg_return = np.mean(all_return)
            print("Iteration {}: Average Return = {}".format(i, avg_return))
            
            # CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.
            all_avg_return.append(avg_return)
            all_max_return.append(max(all_return))
            all_min_return.append(min(all_return))
            #pdb.set_trace()

            if avg_return >= 195:
                print("Solve at {} iterations, which equals {} episodes.".format(i, i*100))
                break

            if self.baseline != None:
                self.baseline.fit(paths)

        import pickle
        with open('loss.pkl', 'wb') as output:
            pickle.dump([all_avg_return, all_max_return, all_min_return], output)


n_iter = 200
n_episode = 100
path_length = 200
discount_rate = 0.99
baseline = LinearFeatureBaseline(env.spec)
#baseline = None

po = PolicyOptimizer(env, policy, baseline, n_iter, n_episode, path_length,
                     discount_rate)

# Train the policy optimizer
po.train()

"""
## Verify your solutions

if you solve the problems 1~4 correctly, your will solve CartPole with roughly ~ 80 iterations.
"""
# Should print output here
"""
## Problem 5
Replacing line 

`baseline = LinearFeatureBaseline(env.spec)` 

with 

`baseline = None`

can remove the baseline.

Modify the code to compare the variance and performance before and after adding baseline.
Then, write a report about your findings. (with figures is better)
"""
# Discuss Problem 5
"""
## Problem 6

In function process_paths of class `PolicyOptimizer`, why we need to normalize the advantages? i.e., what's the usage of this line:

`p["advantages"] = (a - a.mean()) / (a.std() + 1e-8)`

Include the answer in your report.
"""
