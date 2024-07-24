---
title: Getting Started
permalink: examples
layout: page
description: "Some simple examples of how to use MPPI-Generic"
---

{% include math_functions.md %}
{% assign use_output = false %}

## Library Class Description

This library is made up of 6 major types of classes:
* Dynamics
* Cost Functions
* Controllers
* Sampling Distributions
* Feedback Controllers
* Plants

The Dynamics and Cost Function classes are self-evident and are classes describing the $\vb{F}$,
{% if use_output == true %}$\vb{G}$,
{% endif %} $\ell$, and $\phi$ functions described [here]({{ site.url }}{{ site.baseurl }}{% link docs/mppi.md %}).
The Controller class finds the optimal control sequence $U$ that minimizes the cost
{% if use_output == true %}$\vb{J}(Y, U)$
{% else %} $\vb{J}(X, U)$
{% endif %} using algorithms such as MPPI.
The Sampling Distributions are used by the Controller class to generate the control samples used for determining the optimal control sequence.
The Feedback Controller class determines what feedback controller is used to help push the system back towards the desired trajectory.
Unless otherwise specified, the Feedback Controllers in code examples are instantiated but turned off by default.
Finally, Plants are a MPC wrapper around a given controller and are where the interface methods in and out of the controller are generally defined.
For example, a common-use case of MPPI is on a robotics platform running Robot Operating System (ROS) [[1]](#1).
The Plant would then be where you would implement your ROS subscribers to information such as state, ROS publishers of the control output, and the necessary methods to convert from ROS messages to MPPI-Generic equivalents.
Each class type has their own parameter structures which encapsulate the adjustable parameters of each specific instantiation of the class.
For example, the cartpole dynamics parameters include mass and length of the pendulum whereas a double integrator dynamics system has no additional parameters.

For those who want to use pre-existing Dynamics and Cost Functions, the MPPI-Generic library provides a large variety.
Pre-existing Dynamics include a quadcopter, 2D double integrator, cartpole, differential-drive robot [[2]](#2), various Autorally models [[3]](#3), and various models learned for a Polaris RZR X [[4]](#4).
The various Cost Functions included are mostly specific to each Dynamics model but we do provide a Dynamics-agnostic quadratic cost as well.

There are two ways to use the MPPI-Generic library: stand-alone or as a MPC controller.

## Stand-alone Usage
{% capture min_example %}{% include_relative submodules/MPPI_Paper_Example_Code/src/minimal_example.cu %}{% endcapture %}
{% highlight cpp linenos %}
{{ min_example }}
{% endhighlight %}
*Listing <a id="code-1">1</a>: Minimal example used in a stand-alone fashion*
{: class="codecaption"}

{% assign min_lines = min_example | newline_to_br | split: '<br />' %}

If you just wanted to run a single iteration of MPPI in order to get an optimal control sequence, the code in Lst [1](#code-1) provides a minimal working example.
{% highlight cpp %}
{% for line in min_lines offset:0 limit:5 %}{{ line }}{% endfor %}
{% endhighlight %}
Breaking it down, we start by including the controller, dynamics, cost function, and feedback controller headers used in this example.
Again, by default, the feedback controller will not be used but it is required for the MPPI controller.
In this example, we are using a cartpole system with a quadratic cost function to swing up the pole of the cart.
The controller we will be using is the standard MPPI controller.
{% highlight cpp %}
{% for line in min_lines offset:5 limit:3 %}{{ line }}{% endfor %}
{% endhighlight %}
Next, we define the number of timesteps $T$ and the number of samples, $M$, we will be using for this example.
{% highlight cpp %}
{% for line in min_lines offset:8 limit:7 %}{{ line }}{% endfor %}
{% endhighlight %}
We then set up aliases for each class type to make it easier to follow.

{% highlight cpp %}
{% for line in min_lines offset:16 limit:10 %}{{ line }}{% endfor %}
{% endhighlight %}
Then in the main method, we first set up the $\Delta t$ we would like to use followed by creating Dynamics, Cost, Feedback Controller, and Sampling Distribution variables.
We could set up different versions of this problem by adjusting the parameters of the Dynamics or Cost Function but for now, we will leave them to the default.
The Feedback Controller is setup with the dynamics and dt used.
We then setup the Gaussian Sampling Distribution with a standard deviation of $1.0$ in each control dimension.
{% highlight cpp %}
{% for line in min_lines offset:26 limit:9 %}{{ line }}{% endfor %}
{% endhighlight %}
The final setup step is to create the MPPI controller.
We first construct a parameters variable, `controller_params`, and ensure we set the $\Delta t$, $\lambda$, and parallelization parameters appropriately.
From there, we can then construct the controller and pass in the dynamics, cost function, feedback controller, sampling distribution, and controller params.
{% highlight cpp %}
{% for line in min_lines offset:35 limit:5 %}{{ line }}{% endfor %}
{% endhighlight %}
We create an initial state of the cartpole and then ask MPPI to compute an optimal control sequence starting from that state with `computeControl()`.
The optimal control sequence is then returned as an `Eigen::Matrix` from `getControLSequence()` and printed out to the terminal.

## MPC Usage

{% highlight cpp linenos %}
{% include_relative submodules/MPPI_Paper_Example_Code/include/mppi_paper_example/plants/cartpole_plant.hpp %}
{% endhighlight %}
*Listing <a id="code-2">2</a>: Basic Plant implementation that interacts with a virtual Cartpole Dynamics system stored internal to the Plant.*
{: class="codecaption"}

When using MPPI in a MPC fashion, we need to use a Plant wrapper around our controller.
The Plant houses methods to obtain new data such as state, calculate the optimal control sequence at a given rate using the latest information available, and provide the latest control to the external system
while providing the necessary tooling to ensure there are no race conditions.
As this is the class that provides the interaction between the algorithm and the actual system, it is a component that has to be modified for every use case.
For this example, we will implement a plant inheriting from `BasePlant` that houses the external system completely internal to the class.
Specifically, we will write our plant to run the dynamics inside `pubControl()` in order to produce a new state.
We shall then call `updateState()` at a different fixed rate from the controller re-planning rate to show that the capability of the code base.

In Lst. [2](#code-2), we can see a simple implementation of a Plant.
`SimpleCartpolePlant` instantiates a `CartpoleDynamics` variable upon creation, overwrites the required virtual methods from `BasePlant`, and sets up the dynamics update to occur within `pubControl()`.
Looking at the constructor, we pass a shared pointer to a Controller, an integer representing the controller replanning rate, and the minimum timestep we want to adjust the control trajectory by when performing multiple optimal control calculations to the base Plant constructor, and then create our stand-in system dynamics.
`pubControl()` is where we send the control to the system and so in this case, we create necessary extra variables and then pass the current state $\vb{x}\_{t}$ and control $\vb{u}\_t$ as `prev_state` and `u` respectively to the Dynamics' `step()` method to get the next state, $\vb{x}\_{t+1}$, in the variable `current_state_`.
We also update the current time to show that system has moved forward in time.
Looking at this class, a potential issue arises as it is templated on the controller which in turn might not use `CartpoleDynamics` as its Dynamics class.
This can be easily remedied by replacing any reference to `CarpoleDynamics` with `CONTROLLER_T::TEMPLATED_DYNAMICS` to make this plant work with the Dynamics used by the instantiated Controller.

Now that we have written our specialized Plant class, we can then go back to the minimal example and make some modifications to use the controller in a MPC fashion, shown in Lst. [3](#code-3).
For this example, we will just run a simple `for` loop that calls the Plant's `runControlIteration()` and `updateState()` methods to simulate a receiving a new state from the system and then calculating a new optimal control sequence from it.
The `updateState()` method calls `pubControl()` internally so the system state and the current time will get updated as this runs.
For real-time scenarios, there is also the `runControlLoop()` Plant method which can be launched in a separate thread and calls `runControlIteration()` internally at the specified replanning rate.

{% highlight cpp linenos %}
{% include_relative submodules/MPPI_Paper_Example_Code/src/minimal_mpc_example.cu %}
{% endhighlight %}
*Listing <a id="code-3">3</a>: Minimal example used in a MPC fashion*
{: class="codecaption"}

## References
<a id="1">[1]</a>
S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall,
"Robot Operating System 2: Design, architecture, and uses in the wild," *Science Robotics,*
vol. 7, no. 66, p. eabm6074, May 2022. [Online]. Available:
[https://www.science.org/doi/10.1126/scirobotics.abm6074](https://www.science.org/doi/10.1126/scirobotics.abm6074)

<a id="2">[2]</a>
J.-P. Laumond *et al., Robot Motion Planning and Control.* Springer, 1998, vol. 229.

<a id="3">[3]</a>
G. Williams, P. Drews, B. Goldfain, J. M. Rehg, and
E. A. Theodorou, "Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous
Driving," *IEEE Transactions on Robotics*, vol. 34, no. 6, pp. 1603-1622, 2018.
[Online]. Available:
[https://ieeexplore.ieee.org/document/8558663](https://ieeexplore.ieee.org/document/8558663)

<a id="4">[4]</a>
J. Gibson, B. Vlahov, D. Fan, P. Spieler, D. Pastor, A.-a. Agha-mohammadi, and E. A. Theodorou,
"A Multi-step Dynamics Modeling Framework For Autonomous Driving In Multiple Environments," in
*2023 IEEE International Conference on Robotics and Automation (ICRA).* IEEE, May 2023, pp. 7959-7965.
[Online]. Available:
[https://ieeexplore.ieee.org/document/10161330](https://ieeexplore.ieee.org/document/10161330)
