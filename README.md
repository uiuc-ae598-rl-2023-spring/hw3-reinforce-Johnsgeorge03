# HW3 - REINFORCE

## What to do

### Algorithms
Your goal is to implement the REINFORCE algorithm. The key difference between REINFORCE and DQN is that REINFORCE learns the *policy* rather than the action-value function Q. It can also be applied to systems with a continuous state space and (unlike DQN) a continuous action space, although you will focus in this assignment on its application to a system with finite state and action spaces.

### Environments

You will test your algorithms in the following provided environment:
- The simple grid-world from HW1. The environment is defined in `gridworld.py`. An example of how to use the environment is provided in `test_gridworld.py`.

You should express the reinforcement learning problem as a Markov Decision Process with an infinite time horizon and a discount factor of $\gamma = 1$.

### Results

Include a `train_gridworld.py` files that can be run to generate all necessary plots and data.

#### 1) Apply REINFORCE to the grid-world environment

Please apply REINFORCE to the easy version of the grid-world environment with a tabular policy, using softmax to compute action probabilities, using auto-differentiation to compute payoff gradients (the [categorical distribution](https://pytorch.org/docs/stable/distributions.html#categorical) may be helpful), and using the [SGD optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) without momentum to take descent steps.

More specifically, please write code to generate, at a minimum, the following results:
- A plot that contains a learning curve (the total reward versus the number of simulation steps)
- A plot of the policy for at least one trained agent (compared to the optimal policy, if known)

#### 2) Perform an additional investigation

Please do at least **one** of the following additional investigations:
* Test your algorithm on both the "easy" version of gridworld (deterministic state transition) and the "hard" version (stochastic state transition).
* Compare your results when gradient steps are implemented with the [SGD optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) and with the [Adam optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam).
* Apply REINFORCE with a tabular policy to the [discrete pendulum environment](discrete_pendulum.py), with finite state and action spaces, that you considered in HW1.
* Apply REINFORCE to the [pendulum environment](pendulum.py), with continuous state and action spaces, assuming the use of a Gaussian policy that is described by a neural network. By default, this pendulum environment uses a "dense reward" instead of the sparse reward we have assumed in prior versions of the pendulum. Beware! Even with a bug-free implementation and a good choice of hyper-parameters, it is likely that you will need a very large number of simulation steps in order for REINFORCE to produce good results (e.g., tens of millions). We will be impressed if you get this to work (and astonished if you get it to work with sparse reward).

## What to submit

### 1. Initial code

Create a [pull request (PR)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) to enable a [review of your code](#2-code-review). Your code should be functional - that is, each algorithm should be working and ready to be reviewed for improvements. Remember that you need to write your own code from scratch.

Name your PR "Initial hwX for Firstname Lastname (netid)".

**Due: 10am on Tuesday, April 18**

### 2. Code review

Review the code of at least one colleague. That is, you should:
- Choose a PR that does not already have a reviewer, and [assign yourself as a reviewer]((https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews)).
- Perform a code review. See [Resources](#resources) for guidance on how to perform code reviews (at a minimum, look at Google's best practices).
- Modify your `README.md` file to specify whose code you reviewed (using their GitHub username).

The goal of this review process is to arrive at a version of your code that is functional, reasonably efficient, and easy for others to understand. The goal is *not* to make all of our code the same (there are many different ways of doing things). The goal is also *not* to grade the work of your colleagues - your reviews will have no impact on others' grades. Don't forget to remind your colleagues to do the simple things like name their PR and training file correctly!

**Due: 10am on Wednesday, April 19**

### 3. Final code and results

Improve your own code based on reviews that you receive. **Respond to every comment and merge your PR.** If you address a comment fully (e.g., by changing your code), you mark it as resolved. If you disagree with or remain uncertain about a comment, engage in follow-up discussion with the reviewer on GitHub. Don't forget to reply to follow-ups on code you reviewed as well.

Submit your repository, containing your final code and a (very brief) report titled `hwX-netid.pdf`, to [Gradescope](https://uiuc-ae598-rl-2023-spring.github.io/resources/assignments/). The report should be formatted using either typical IEEE or AIAA conference/journal paper format and include the following, at a minimum:
- Plots discussed in [Results](#results).
- An introduction (e.g., discussion of the problem(s)), discussion of methods (e.g., including any specific details needed to understand your algorithm implementations like hyperparameters chosen), discussion of results, and general conclusions.

**Due: 10am on Thursday, April 20**

## Resources
Here are some resources that may be helpful:
* Google's [best practices for code review](https://google.github.io/eng-practices/review/reviewer/looking-for.html)
* A Microsoft [blog post on code review](https://devblogs.microsoft.com/appcenter/how-the-visual-studio-mobile-center-team-does-code-review/) and [study of the review process](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/MS-Code-Review-Tech-Report-MSR-TR-2016-27.pdf)
* A RedHat [blog post on python-specific code review](https://access.redhat.com/blogs/766093/posts/2802001)
* A classic reference on writing code: [The Art of Readable Code (Boswell and Foucher, O'Reilly, 2012)](https://mcusoft.files.wordpress.com/2015/04/the-art-of-readable-code.pdf)

Many other resources are out there - we will happily accept a PR to add more to this list!
