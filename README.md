# Homework2 - Policy Gradient 
Please complete each homework for each team, and <br>
mention who contributed which parts in your report.

# Introduction
In this assignment, we will solve the classic control problem - CartPole.

<img src="https://cloud.githubusercontent.com/assets/7057863/19025154/dd94466c-8946-11e6-977f-2db4ce478cf3.gif" width="400" height="200" />

CartPole is an environment which contains a pendulum attached by an un-actuated joint to a cart, 
and the goal is to prevent it from falling over. You can apply a force of +1 or -1 to the cart.
A reward of +1 is provided for every timestep that the pendulum remains upright.

# Setup
* OpenAI gym
* TensorFlow
* Numpy 
* Scipy
* IPython Notebook

The preferred approach for installing above dependencies is to use [Anaconda](https://www.continuum.io/downloads), which is a Python distribution that includes many of the most popular Python packages for science, math, engineering and data analysis.

1. **Install Anaconda**: Follow the instructions on the [Anaconda download site](https://www.continuum.io/downloads).
2. **Install TensorFlow**: See [anaconda section](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#anaconda-installation) of TensorFlow installation page.
3. **Install OpenAI gym**: Follow the official installation documents [here](https://gym.openai.com/docs).

# Prerequisites
If you are unfamiliar with Numpy or IPython, you should read materials from [CS231n](http://cs231n.github.io/):
* [Numpy tutorial](http://cs231n.github.io/python-numpy-tutorial/)
* [IPython tutorial](http://cs231n.github.io/ipython-tutorial/) 

Also, knowing the basics of TensorFlow is required to complete this assignment.

For introductory material on TensorFlow, see
* [MNIST For ML Beginners](https://www.tensorflow.org/versions/r0.11/tutorials/mnist/beginners/index.html) from official site
* [Tutorial Video](https://www.youtube.com/watch?v=l6K-MFgIEjc&t=3334s) from [Stanford CS224D](http://cs224d.stanford.edu/syllabus.html)

Feel free to skip these materials if you are already familiar with these libraries.

# How to Start
1. **Start IPython**: After you clone this repository and install all the dependencies, you should start the IPython notebook server from the home directory
2. **Open the assignment**: Open ``HW2_Policy_Graident.ipynb``, and it will walk you through completing the assignment.

# To-Do
* [**+20**] Construct a 2-layer neural network to represent policy
* [**+30**] Compute the surrogate loss
* [**+20**] Compute the accumulated discounted rewards at each timestep
* [**+10**] Use baseline to reduce the variance
* [**+20**] Modify the code and write a report to compare the variance and performance before and after adding baseline (with figures is better)
* [**BONUS +10**] In function `process_paths` of class `PolicyOptimizer`, why we need to normalize the advantages?
  i.e., what's the usage of this line: 
  
  `p["advantages"] = (a - a.mean()) / (a.std() + 1e-8)`
  
  Include the answer in your report

# Other
* Due on ??? before class.
