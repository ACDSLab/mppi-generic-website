---
title: Benchmarks
permalink: benchmarks
layout: page
description: "Comparisons to other implementations of MPPI"
---

{% include math_functions.md %}

<table class="tableright">
  <caption><em>Table <a id="table-1">1</a>: Problem parameters</em></caption>
  <thead><tr>
      <th style="text-align: center">Parameter</th>
      <th style="text-align: center">Value</th>
    </tr></thead>
  <tbody><tr>
      <td style="text-align: center" colspan="2"><b>Dynamics parameters</b></td>
    </tr><tr>
      <td style="text-align: center">dt</td>
      <td style="text-align: center">0.02s</td>
    </tr><tr>
      <td style="text-align: center">wheel radius</td>
      <td style="text-align: center">1.0m</td>
    </tr><tr>
      <td style="text-align: center">wheel length</td>
      <td style="text-align: center">1.0m</td>
    </tr><tr>
      <td style="text-align: center">max velocity</td>
      <td style="text-align: center">0.5m/s</td>
    </tr><tr>
      <td style="text-align: center">min velocity</td>
      <td style="text-align: center">-0.35m/s</td>
    </tr><tr>
      <td style="text-align: center">min rotation</td>
      <td style="text-align: center">-0.5rad/s</td>
    </tr><tr>
      <td style="text-align: center">max rotation</td>
      <td style="text-align: center">0.5rad/s</td>
    </tr><tr>
      <td style="text-align: center" colspan="2"><b>MPPI parameters</b></td>
    </tr><tr>
      <td style="text-align: center">$\lambda$</td>
      <td style="text-align: center">1.0</td>
    </tr><tr>
      <td style="text-align: center">control standard deviation</td>
      <td style="text-align: center">0.2</td>
    </tr><tr>
      <td style="text-align: center">number of timesteps</td>
      <td style="text-align: center">100</td>
    </tr><tr>
      <td style="text-align: center" colspan="2"><b>Cost parameters</b></td>
    </tr><tr>
      <td style="text-align: center">Dist. to goal coeff.</td>
      <td style="text-align: center">5</td>
    </tr><tr>
      <td style="text-align: center">Angular dist. to goal coeff.</td>
      <td style="text-align: center">5</td>
    </tr><tr>
      <td style="text-align: center">Obstacle Cost</td>
      <td style="text-align: center">20</td>
    </tr><tr>
      <td style="text-align: center">Map Width</td>
      <td style="text-align: center">11m</td>
    </tr><tr>
      <td style="text-align: center">Map Height</td>
      <td style="text-align: center">11m</td>
    </tr><tr>
      <td style="text-align: center">Map Resolution</td>
      <td style="text-align: center">0.1m/cell</td>
    </tr></tbody>
</table>

In order to see the improvements our library can provide, we decided to compare against three other implementations of MPPI publicly available.
The first comparison is with the MPPI implementation in AutoRally [[1]](#1).
This implementation was the starting point of our new library, MPPI-Generic, and so we want to compare to see how well we can perform to our predecessor.
The Autorally implementation is written in C++/CUDA, is compatible with ROS, features multiple dynamics models including linear basis functions, simple kinematics, and \ac{NN}-based models focused on the Autorally hardware platform.
There is only one Cost Function available but it does make use of CUDA textures for doing queries into an obstacle map.
Additionally, it has been shown to run in real-time on hardware to great success [[2]](#2), [[3]](#3), [[4]](#4).
However, the Autorally implementation is written mostly for use on the Autorally platform and as such, it has no general Cost Function, Dynamics, or Sampling Distribution APIs to extend.
In order to use it for different problems such as flying a quadrotor, the MPPI implementation would need significant modification.

The next implementation we will compare against is ROS2's MPPI [[5]](#5).
As of ROS Iron, there is a CPU implementation of MPPI in the ROS navigation stack
This CPU implementation is written in C++ and looks to make heavy use of AVX instructions or vectorized instructions to improve performance.
There is a small selection of dynamics models (Differential Drive, Ackermann, and Omni-directional) and cost functions that are focused around wheeled robots navigating through obstacle-laden environments.
This implementation will only become more widespread as ROS2 adoption continues to pick over the coming years, making it an essential benchmark.
Unfortunately, it does have some drawbacks as it is not possible to add new dynamics or cost functions without rewriting the base code itself, has no implementation of Tube-MPPI or RMPPI, and is only available in ROS2 Iron or newer.
This means that it might not be usable on existing hardware platforms that are unable to upgrade their systems.

The last implementation of MPPI we will compare against is in TorchRL [[6]](#6).
TorchRL is an open-source reinforcement learning Python library written by Meta AI, the developers of PyTorch itself.
As such, it is widely trusted and available to researchers who are already familiar with PyTorch and Python.
The TorchRL implementation works on both CPUs and GPUs and allows for custom dynamics and cost functions through the extension of base Environment class [[7]](#7).
However, while it does have GPU support, it is limited to the functionality that PyTorch provides meaning that there is no option to use CUDA textures to improve map queries or any direct control of shared memory usage on the GPU.
In addition, being written in Python makes it fairly legible and easy to extend but can come at the cost of performance when compared to C++ implementations.


In order to compare our library against these three implementations, we tried to recreate the same dynamics and cost function for each version of MPPI.
As ROS2's implementation would be the hardest to modify, we chose to use the Differential Drive dynamics model and some of the cost function components that already exist there as the baseline.
We ended up using the goal position quadratic cost, goal angle quadratic cost, and the costmap-based obstacle cost components so that we could maintain a fairly simple cost function that allows us to show the capabilities of our library.
We implemented these dynamics and cost functions in both CUDA and Python.
The CUDA implementations were an extension of our base Dynamics and Cost Function APIs which allowed them to plug into our library easily.
We decided to use the same code in the Autorally implementation as well which required some minor rewriting to account for different method names and state dimensions.
The Python implementation was an extension of the TorchRL base Environment class, and was compiled using PyTorch's JIT compiler in order to speed up performance when used in the TorchRL implementation.
We used the same parameters for sampling, dynamics, cost function tuning, and MPPI hyperparameters across all implementations, summarized in Table [1](#table-1).

<table class="tableright">
  <caption><em>Table <a id="table-2">2</a>: GPU parameters</em></caption>
  <thead><tr>
      <th style="text-align: center">Parameter</th>
      <th style="text-align: center">Value</th>
    </tr></thead>
  <tbody><tr>
      <td style="text-align: center">Dynamics thread block x dim</td>
      <td style="text-align: center">64</td>
    </tr><tr>
      <td style="text-align: center">Dynamics thread block y dim</td>
      <td style="text-align: center">4</td>
    </tr><tr>
      <td style="text-align: center">Cost thread block x dim</td>
      <td style="text-align: center">64</td>
    </tr><tr>
      <td style="text-align: center">Cost thread block y dim</td>
      <td style="text-align: center">1</td>
    </tr></tbody>
</table>
For both Autorally and MPPI-Generic, there are some further performance enhancing options available such as block size choice.
We ended up using the same block sizes for both Autorally and MPPI-Generic across all tests, shown in Table [2](#table-2).
As a result, the optimization times shown are not going to be the fastest possible performance that can be achieved on any given GPU but these tests should still serve as a useful benchmark to understand the average performance that can be achieved.
We then ran each of the Autorally, MPPI-Generic, and ROS2 implementations $10,000$ times to produce optimal trajectories with $128$, $256$, $512$, $1024$, $2048$, $4096$, $6144$, $8192$, and $16,384$ samples; the TorchRL implementation was only run $1000$ times due to it being too slow to compute, even when using the GPU.
This allowed us to produce more robust measurements of the computation times required as well as understand how well each implementation would scale as more computation was required.
While our dynamics and cost functions currently chosen would not need up to $16,394$ samples, one might want to implement more complex dynamics or cost functions that could require a large number of samples to find the optimal solution.
The comparisons were run across a variety of hardware including a Jetson Nano to see what bottlenecks each implementation might have.
The Jetson Nano was unfortunately only able to run the MPPI-Generic and Autorally MPPI implementations as the last supported PyTorch version and the lastest TorchRL libraries were incompatible, and the ROS2 implementation was unable to compile.
GPUs tested ranged from a NVIDIA GTX 1050 Ti to a NVIDIA RTX 4090.
Most tests were performed on an Intel 13900K which is one of the fastest available CPUs at the time of this writing in order to prevent the CPU being the bottleneck for the mostly GPU-based comparison; however, we did also run some tests on an AMD Ryzen 5 5600x to see the difference in performance on a lower-end CPU.
% while the CPU options were either an Intel 13900K or an AMD Ryzen 5 5600x.
The MPPI optimization times across all the hardware can be seen [here]({{ site.url }}{{ site.baseurl }}{% link docs/assets/mppi_runtimes_table.pdf %}) and the code used to do these comparisons is available here \bogdan{fill in where the MPPI example code repo will be available}.


# Results

<img src="{{ site.url }}{{ site.baseurl }}/docs/assets/mppi-comparisons/nvidia_geforce_rtx_4090_results.png" width="100%">
*Fig. <a id="fig-4090">1</a>: Optimization times for all MPPI implementations on a hardware system with a RTX 4090 and an Intel 13900K over a variety of number of samples.
It can be seen that the ROS2 CPU implementation grows linearly as the number of samples increase while GPU implementations grow more slowly.*

Going over all of the collected data would take too much room for this paper so we shall instead try to pull out interesting highlights to discuss.
Full results can be seen [here]({e site.url }}{{ site.baseurl }}{% link docs/assets/mppi_runtimes_table.pdf %}).
First, we can look at the results on the most powerful system tested, using an Intel i9-13900K and an NVIDIA RTX 4090 in Fig. [1](#fig-4090).
We see that as the number of samples increase, the ROS2 method which is CPU-bound increases in optimization times in a linear fashion.
Every other method is on the GPU and we see little reason to use small sample sizes as they have the same computation time till we reach around $1024$ samples.
We also see that as we hit $16,384$ samples, the AutoRally implementation starts to have lower optimization times than MPPI-Generic.
We will see this trend continue in Fig. [2](#fig-1040ti).

<img src="{{ site.url }}{{ site.baseurl }}/docs/assets/mppi-comparisons/nvidia_geforce_gtx_1050_ti_results.png" width="100%">
*Fig. <a id="fig-1050ti">2</a>: Optimization times for all MPPI implementations on a hardware system with a GTX 1050 Ti and an Intel 13900K over a variety of number of samples. It can be seen that MPPI-Generic and AutoRally on this older hardware does eventually start to scale linearly with the number of samples but does so at a much lower rate with our library compared to ROS2 or Pytorch.*

When looking at older and lower-end NVIDIA hardware such as the GTX 1050 Ti, we still see that our library still performs well compared to other implementations as seen in Fig. [2](#fig-1040ti).
Only when the number of samples is at $128$ does the ROS2 implementation on an Intel 13900k match the performance of the AutoRally implementation on this older GPU. MPPI-Generic is still more performant at these lower number of samples and eventually we see it scales linearly as we get to thousands of samples.
The TorchRL implementation also finally starts to show some GPU bottle-necking as we start to see optimization times increasing as we reach over $6144$ samples.
We can also see that there is a moment where the MPPI-Generic library optimization time grows to be larger than the AutoRally implementation.
That occurs when we switch from using the split kernels to the combined kernel discussed [here]({{site.url}}{{site.baseurl}}{% link docs/performance.md %}#split-kernels).
The AutoRally implementation uses a combined kernel as well but has fewer synchronization points on the GPU due to strictly requiring forward Euler integration for the dynamics.
At the small hit to performance in the combined kernel, our library allows for many more features, such as multi-threaded cost functions, use of shared memory in the cost function, and implementation of more computationally-heavy integration methods such as Runge-Kutta or backward Euler integration.
And while we see a hit to performance when using the combined kernel compared to AutoRally, we still see that the split kernel is faster for up to $2048$ samples.

<img src="{{ site.url }}{{ site.baseurl }}/docs/assets/mppi-comparisons/torchrl_hw_results.png" width="100%">
*Fig. <a id="fig-torchrl">3</a>: Optimization times for the TorchRL implementation across different CPUs and GPUs. We see that TorchRL computation times are more dependent on the CPU as the RTX 3080 with an AMD 5600X ends up slower than a GTX 1050 Ti with an Intel 13900k.*

The TorchRL implementation is notably performing quite poorly in Figs. [1](#fig-4090) and [2](#fig-1050ti) with runtimes being around $28$ ms no matter the number of samples.
Looking at TorchRL-specific results in Fig. [3](#fig-torchrl), we can see that the TorchRL implementation seems to be heavily CPU-bound.
A low-end GPU (1050 Ti) combined with a high-end CPU (Intel 13900K) can achieve better optimization times than a low-end CPU (AMD 5600X) combined with a high-end GPU (RTX 3080).

We also conducted tests on a Jetson Nano to show that even on relatively low-power and older systems, our library can still be used.
As the latest version of CUDA supporting the Jetson Nano is 10.2 and the OS is Ubuntu 18.04, both the pytorch and ROS2 MPPI implementations were not compatible.
As such, we only have results comparing our MPPI-Generic implementation to the AutoRally implementation in Fig. [4](#fig-jetson).
Here, we can see that the AutoRally implementation starts having faster compute times around $512$ samples.
Again, this is due to our library switching to the combined kernel which will be slower.
However, our library on a Jetson Nano at $2048$ samples has a roughly equivalent computation time to that of $2048$ samples of the ROS2 implementation on the Intel 13900K process, showing that our GPU parallelization can allow for real time optimization even on portable systems.

<img src="{{ site.url }}{{ site.baseurl }}/docs/assets/mppi-comparisons/nvidia_tegra_x1_results.png" width="100%">
*Fig. <a id="fig-jetson">4</a>: Optimization times for MPPI implementations on a Jetson Nano over a variety of number of samples. It can be seen that MPPI-Generic and AutoRally on this low-power hardware can still achieve sub 10 ms optimization times for even 2048 samples. The AutoRally implementation quickly surpasses our implementation in optimization times.*

In addition, we can really see the benefits of our library as we increase the computation time of the cost function.
At this point, the pytorch and ROS2 implementations have been shown to be slow in comparison to the other implementations and are thus dropped from this cost complexity comparison.


```cuda
float computeStateCost(...) {
    float cost = 1.0;
    for (int i = 0; i < NUM_COSINES; i++) {
        cost = cos(cost);
    }
    cost *= 0.0;
    // Continue to regular cost function
    ...
}
```
*Listing <a id="code1">1</a>: Computation time inflation code added to the cost function. We add a configurable amount of calls to `cos()` as this is a computationally heavy function to run.*
{: class="codecaption"}

We can artificially inflate the computation time of the cost function with Lst. [1](#code1) to judge how well the implementations scale to more complex cost functions.
In Fig [5](#fig-1050_ti-cost-complexity), we see how increasing the computation time of the cost function scales for both implementations over the same hardware and for the same number of samples.

<img src="{{ site.url }}{{ site.baseurl }}/docs/assets/mppi-comparisons/nvidia_geforce_gtx_1050_ti_diff_drive_cost_complexity_results.png" width="100%">
*Fig. <a id="fig-1050_ti-cost-complexity">5</a>: Optimization Times for MPPI-Generic and AutoRally implementations as the computation time of the cost function increases.
Using an Intel 13900K, NVIDIA GTX 1050 Ti, and 8192 samples, we can see that the our library implementation starts to outperform the AutoRally implementation when 20+ sequential cosine operations are added to the cost function.*

## Comparisons to sampling-efficent algorithms

While we have shown that our implementation of MPPI can have faster computation times and a lot of flexibility in where it can be applied, there remains a question of what is the balance point between number of samples and real time performance.
Ideally, we would like to sample all possible paths to calculate the optimal control trajectory but this is computationally infeasible.
While our work has decreased the computation time for sampling which in turn allows more samples in the same computation time, other works have tried to reduce the amount of samples needed to evaluate the optimal control trajectory.
In [[8]](#8), the authors introduce a generalization of MPC algorithms called Dynamic Mirror Descent Model Predictive Control (DMD-MPC) defined by the choice of shaping function, $S(\cdot)$, Sampling Distribution $\pi_\theta$, and Bregman Divergence $D_\Psi(\cdot, \cdot)$ which determines how close the new optimal control trajectory should remain to the previous.
Using the exponential function, Gaussian sampling, and the KL Divergence, they derive a slight modification to the MPPI update law that introduces a step size parameter $\gamma_t \geq 0$:

$$
\begin{align}
    u_{t}^{k+1} = \PP{1 - \gamma_t} u_{t}^k + \gamma_t \frac{\Expectation[V \sim \pi_\theta]{\Shape\PP{\J\PP{X,V}}v_t}}{\Expectation[V \sim \pi_\theta]{\Shape\PP{\J\PP{X,V}}}},
\end{align}
$$

where $v_t$ is the control value at time $t$ from the sampled control sequence $V$, and $\J$ is the Cost Function.
In their results, they showed that when using a low number of samples, the addition of a step size can help improve performance.
However, once the number of samples increases beyond a certain point, the optimal step size ends up being $1.0$ which is equivalent to the original MPPI update law.
Having this option is useful in cases where you have a low computational budget.
What we would like to show is that as long as you have a NVIDIA GPU from the last decade, you can have enough computational budget to just use more samples instead of needing to tune a step size.

<img src="{{ site.url }}{{ site.baseurl }}/docs/assets/dmd-mpc-comparisons/nvidia_geforce_gtx_1050_ti_dmd_results_horz.png" width="100%">
*Fig. <a id="fig-dmd-mpc">6</a>: Average Accumulated Costs (left) and Optimization Times (right) with error bars signifying one standard deviation for a variety of step sizes and number of samples for DMD-MPC.
$\times$ indicates the step size that achieves the lowest cost for a given number of samples.
This was run on a 2D double integrator dynamics with a cost shown in \eqref{eq:double_integrator_circle_cost}.
This system was run for 1000 timesteps to get the accumulated cost and this was repeated 1000 times to get the standard deviation bars shown.
When using a low number of samples, a lower DMD-MPC step size provides the lowest costs.
However, as the number of samples increase, the best step size choice becomes $\gamma_t = 1.0$ which is equivalent to the normal MPPI update law.
For our library, increasing the number of samples to the point where the step size is no longer useful is still able to be run at over 800 Hz on the NVIDIA GTX 1050 Ti.*

Looking at Fig. [6](#fig-dmd-mpc), we ran a 2D double integrator system with state $[x,y,v_x,v_y]$ and control $[a_x, a_y]$ with the cost shown in \eqref{eq:double_integrator_circle_cost}:

$$
\begin{align}
    J &= 1000 \PP{\mathbb{I}_{\left\{\PP{x^2+y^2} \leq 1.875^2\right\}} + \mathbb{I}_{\left\{\PP{x^2+y^2} \geq 2.125^2\right\}}} + 2\abs{2 - \sqrt{v_x^2 + v_y^2}} + 2\abs{4 - \PP{xv_y - yv_x}}.
    \label{eq:double_integrator_circle_cost}
\end{align}
$$

This cost function heavily penalizes the system from leaving a circle of radius $2$m with width $0.125$m but gives no cost inside the track width, has an $L_1$ cost on speed to maintain $2$m/s, and has an $L_1$ cost on the angular momentum being close to $4$m^2/s.
This all combines to encourage the system to move around the circle allowing some small deviation from the center line in a clockwise manner.
This system was simulated for 1000 timesteps and the cost was accumulated over that period. This simulation was run 1000 times to ensure consistent cost evaluations.
We see that as the number of samples increase, the optimal step size (marked with a x) increases to $1.0$. In addition, the computation time increase for using a number of samples where the step size is irrelevant is minimal (an increase of about $0.04$ ms).

## References
<a id="1">[1]</a>
B. Goldfain, P. Drews, G. Williams, and J. Gibson,
*AutoRally,* Georgia Tech AutoRally Organization. [Online]. Available: [https://github.com/AutoRally/autorally](https://github.com/AutoRally/autorally)

<a id="2">[2]</a>
B. Goldfain, P. Drews, C. You, M. Barulic, O. Velev,
P. Tsiotras, and J. M. Rehg, "Autorally: An open platform for aggressive autonomous driving," _IEEE Control
Systems Magazine,_ vol. 39, no. 1, pp. 26–55, 2019

<a id="3">[3]</a>
G. Williams, P. Drews, B. Goldfain, J. M. Rehg, and E. A. Theodorou,
"Aggressive Driving with Model Predictive Path Integral Control," in _2016 IEEE
International Conference on Robotics and Automation (ICRA)_. IEEE, 2016, pp. 1433–1440. [Online].
Available: [https://ieeexplore.ieee.org/document/7487277/](https://ieeexplore.ieee.org/document/7487277/)

<a id="4">[4]</a>
G. Williams, P. Drews, B. Goldfain, J. M. Rehg, and
E. A. Theodorou, "Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous
Driving," _IEEE Transactions on Robotics_, vol. 34, no. 6, pp. 1603-1622, 2018.
[Online]. Available:
[https://ieeexplore.ieee.org/document/8558663](https://ieeexplore.ieee.org/document/8558663)

<a id="5">[5]</a>
A. Budyakov and S. Macenski, "ROS2 Model Predictive
Path Integral Controller," Open Robotics. [Online].
Available: [https://github.com/ros-planning/navigation2/tree/iron/nav2_mppi_controller](https://github.com/ros-planning/navigation2/tree/iron/nav2_mppi_controller)

<a id="6">[6]</a>
A. Bou, M. Bettini, S. Dittert, V. Kumar, S. Sodhani, X. Yang, G. D. Fabritiis,
and V. Moens, "TorchRL: A data-driven decision-making library for PyTorch,"
in *The Twelfth International Conference on Learning Representations,*
Jan. 2024. [Online]. Available: [https://openreview.net/forum?id=QxItoEAVMb](https://openreview.net/forum?id=QxItoEAVMb)

<a id="7">[7]</a>
torchrl contributors, "mppi.py - pytorch/rl," Meta, 2023.
[Online]. Available: [https://github.com/pytorch/rl/blob/main/torchrl/modules/planners/mppi.py](https://github.com/pytorch/rl/blob/main/torchrl/modules/planners/mppi.py)

<a id="8">[8]</a>
N. Wagener, C.-A. Cheng, J. Sacks, and B. Boots, "An Online Learning Approach to Model Predictive
Control," in _Proceedings of Robotics: Science and Systems,_ Freiburg im Breisgau, Germany, Jun. 2019.
[Online]. Available: [http://arxiv.org/abs/1902.08967](http://arxiv.org/abs/1902.08967)
