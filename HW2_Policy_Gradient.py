# coding: utf-8
import gym
import tensorflow as tf
import numpy as np
from policy_gradient import util
from policy_gradient.policy import CategoricalPolicy
from policy_gradient.baselines.linear_feature_baseline import LinearFeatureBaseline
 
np.random.seed(0)
tf.set_random_seed(0)
 
# To visualize CartPole-v0, please see https://gym.openai.com/envs/CartPole-v0
# actions: a force of +1 or -1 to the cart
# rewards: +1 if not fallen every timestep

env = gym.make('CartPole-v0')

sess = tf.Session()
 
in_dim = util.flatten_space(env.observation_space)
out_dim = util.flatten_space(env.action_space)
hidden_dim = 8
 
opt = tf.train.AdamOptimizer(learning_rate=0.01)
policy = CategoricalPolicy(in_dim, out_dim, hidden_dim, opt, sess)
 
sess.run(tf.initialize_all_variables())
avg_rets = []

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
            if self.baseline != None:
                b = self.baseline.predict(p)
            else:
                b = 0
 
            r = util.discount_cumsum(p["rewards"], self.discount_rate)
            a = np.array(r)-b 
 
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
        global avg_rets
        for i in range(1, self.n_iter + 1):
            paths = []
            for _ in range(self.n_episode):
                paths.append(self.sample_path())
            data = self.process_paths(paths)
            loss = self.policy.train(data["observations"], data["actions"], data["advantages"])
            avg_return = np.mean([sum(p["rewards"]) for p in paths])
            print("Iteration {}: Average Return = {}".format(i, avg_return))
            avg_rets.append(avg_return)

            # CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.
            '''
            if avg_return >= 195:
                print("Solve at {} iterations, which equals {} episodes.".format(i, i*100))
                break
            '''
            if self.baseline != None:
                self.baseline.fit(paths)
        avg_rets = np.array(avg_rets)
        np.save('avg_rets_200_%.6f' % np.random.rand(), avg_rets)
        print("Average Return Mean: {}, Variance: {}".format(avg_rets.mean(), avg_rets.var()))
 
n_iter = 200
n_episode = 100
path_length = 200
discount_rate = 0.99
baseline = LinearFeatureBaseline(env.spec)
#baseline = None
 
po = PolicyOptimizer(env, policy, baseline, n_iter, n_episode, path_length, discount_rate)
 
po.train()
 
# # Problem 5
# Modify the code to compare the variance and performance before and after adding baseline.
# Then, write a report about your findings. (with figures is better)
 
# # Problem 6
#
# In function process_paths of class `PolicyOptimizer`, why we need to normalize the advantages? i.e., what's the usage of this line:
#
# `p["advantages"] = (a - a.mean()) / (a.std() + 1e-8)`
#
# Include the answer in your report.
