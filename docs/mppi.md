---
title: MPPI
layout: page
description: "Overview of the algorithm and its variants"
---
{% assign use_output = false %}
# Problem Formulation

Consider a general nonlinear system with discrete dynamics and cost function of the following form:

$$
\newcommand{\vb}[1]{ {\bf #1} }
\newcommand{\PP}[1]{\left(#1\right)}
\newcommand{\R}{\mathbb{R}}
\newcommand{\expf}[1]{\exp\PP{#1}}
\newcommand{\normal}[1]{\mathcal{N}\PP{#1}}
$$

{% if use_output == true %}
$$
\begin{align}
\vb{x}_{t+1} &= \vb{F}\PP{\vb{x}_t, \vb{u}_t} \\
\vb{y_{t}} &= G\PP{\vb{x}_t, \vb{u}_t} \\
\vb{J}(Y, U) &= \phi(\vb{y}_{T}) + \sum_{t = 0}^{T - 1}\vb{\ell}\PP{\vb{y}_t, \vb{u}_{t}}
\end{align}
$$
{% else %}
$$
\begin{align}
\vb{x}_{t+1} &= \vb{F}\PP{\vb{x}_t, \vb{u}_t} \\
\vb{J}(X, U) &= \phi(\vb{x}_{T}) + \sum_{t = 0}^{T - 1}\vb{\ell}\PP{\vb{x}_t, \vb{u}_{t}}
\end{align}
$$
{% endif %}

where $\vb{x} \in \R^{n_x}$ is the state of dimension $n_x$,
$\vb{u} \in \R^{n_u}$ is the control of dimension $n_u$,
{% if  use_output == true %} $\vb{y} \in \R^{n_y}$ is the observation of the system in dimension $n_y$,
{% endif %} $T$ is the time horizon,
{% if use_output == true %} $Y$ is an output trajectory $$\left[\vb{y}_1, \vb{y}_2, ..., \vb{y}_T\right]$$,
{% else %} $x$ is a state trajectory $$\left[\vb{x}_1, \vb{x}_2, ..., \vb{x}_T\right]$$,
{% endif %} $U$ is a control trajectory $$[\vb{u}_0, \vb{u}_1, ..., \vb{u}_{T-1}]$$,
$\phi$ is the terminal cost, and $\vb{\ell}$ is the running cost.

{% if use_output == true %}
Looking at the above equations, we can see that there is a minor difference from the typical nonlinear control setup, shown below, in that we have our cost function using the output, $\vb{y}_t$ instead of $\vb{x}_t$.

$$
\begin{align}
    \vb{J}(X,U) &= \phi(\vb{x}_T) + \sum_{t=0}^{T-1} \vb{\ell}\PP{\vb{x}_t, \vb{u}_t}
\end{align}
$$

For the vast majority of systems, $\vb{y}_t$ can just be the true state, i.e. $\vb{y}_t = \vb{x}_t$ but we have seen that separating the two can be important for computational reasons.
It allows reuse of computationally-heavy calculations required for dynamics updates such as wheel position and forces in the cost function.
Having this split between $\vb{x}_t$ and $\vb{y}_t$ is one that allows us to improve the efficiency of the real-time code while not having to do unnecessary computations such as a Jacobian with respect to every wheel position.
{% else %}
{% endif %}

# MPPI Algorithm Overview

Model Predictive Path Integral (MPPI) is a stochastic optimal control algorithm that minimizes the cost function above through the use of sampling.
We start by sampling control trajectories, running each trajectory through the dynamics to create a corresponding state trajectory, and then evaluating each state and control trajectory through the cost function.
Each trajectory's cost is then run through the exponential transform,

$$
\begin{align}
    S(\vb{J};\lambda) = \expf{-\frac{1}{\lambda} \vb{J}},
\end{align}
$$

where $\lambda$ is the inverse temperature.
Finally, a weighted average of the trajectories is conducted to produce the optimal control trajectory.
The update law for $\mathcal{U}^{*}(t)$, the optimal trajectory at time $t$, ends up looking like

$$
\begin{align}
    \mathcal{U}^{*}_t &= \sum_{m=1}^{M} \frac{\expf{-\frac{1}{\lambda} \vb{J}\PP{X^m,V^m}}\vb{v}^m_t}{\sum_{j=1}^{M}\expf{-\frac{1}{\lambda} \vb{J}\PP{X^j,V^j}}}\\
    &= \vb{u}_t + \sum_{m=1}^{M} \frac{\expf{-\frac{1}{\lambda} \vb{J}\PP{X^m,V^m}}\epsilon^m_t}{\sum_{j=1}^{M}\expf{-\frac{1}{\lambda} \vb{J}\PP{X^j,V^j}}},
    \label{eq:mppi_update_rule}
\end{align}
$$

where $V^m$ is the $m$-th sampled control trajectory, $\vb{v}^m_t = \vb{u}_t + \epsilon^m_t$ is the control from the $m$-th sampled trajectory at time $t$ sampled around the previous optimal control, $\vb{u}_t$, with $\epsilon^m_t \sim \normal{0, \sigma^2}$.
Sampling in the control space ensures that the trajectories are dynamically feasible and allows us to use non-differentiable dynamics and cost functions.
Pseudo code for the algorithm is shown below.

{% include pseudocode.html id="mppi-pseudocode" code="
\begin{algorithm}
\caption{MPPI}
\begin{algorithmic}
\REQUIRE ${\bf F}\left(\cdot, \cdot\right)$, $\ell\left(\cdot, \cdot\right)$,
    $\phi\left(\cdot\right)$ $M$, $I$, $T$, $\lambda$, $\sigma$:
    System dynamics, running state cost, terminal cost, num. samples,
    num. iterations, time horizon, temperature, standard deviations;
\INPUT $\ {\bf x}_0$, ${\bf U}$: initial state, mean control sequence;
\OUTPUT $\ \mathcal{U}$: optimal control sequence;
\STATE
\COMMENT{Begin Cost sampling}
\FOR{$i \leftarrow 1$ {\textbf to} $I$}
    \FOR{$m \leftarrow 1$ {\textbf to} $M$}
        \STATE $J^m \leftarrow 0$;
        \STATE ${\bf x} \leftarrow 0$;
        \FOR{$t \leftarrow 0$ {\textbf to} $T-1$}
            \STATE $\epsilon^m(t) \sim \mathcal{N}\left(0, \sigma^2\right)$;
            \STATE ${\bf v}_t \leftarrow {\bf u}_t + \epsilon^m(t) $;
            \STATE ${\bf x} \leftarrow {\bf F}\left({\bf x}, {\bf v}_t\right)$;
            \STATE $J^m += \ell\left({\bf x}, {\bf v}_t\right)$;
        \ENDFOR
        \STATE $J^m += \phi\left({\bf x}\right)$;
    \ENDFOR
    \STATE
    \STATE
    \COMMENT{Compute Trajectory Weights}
    \STATE $\rho \leftarrow \min\left\{J^1, J^2, ..., J^M\right\}$;
    \STATE $\eta \leftarrow \sum_{m=1}^{M} \exp\left(-\frac{1}{\lambda}\left(J^m - \rho\right)\right)$;
    \FOR{$m \leftarrow 1$ {\textbf to} $M$}
        \STATE $w_m \leftarrow \frac{1}{\eta} \exp\left(-\frac{1}{\lambda}\left(J^m - \rho\right)\right)$;
    \ENDFOR
    \STATE
    \STATE
    \COMMENT{Control Update}
    \FOR{$t \leftarrow 1$ {\textbf to} $T-1$}
        \STATE $\mathcal{U}_t \leftarrow {\bf u}_t + \sum_{m=1}^{M}w_m \epsilon^m(t)$;
    \ENDFOR
\ENDFOR
\end{algorithmic}
\end{algorithm}
" %}

## Derivation and other algorithms
The original derivation of MPPI was done from a path-integral approach [[1]](#1).
Future papers then derived MPPI from information theory [[2]](#2), stochastic search [[3]](#3), and mirror descent [[4]](#4) approaches.
A Tube-based MPPI controller [[5]](#5) was also created in order to improve robustness to state disturbances.
It made use of a tracking controller to cause the real system to track back to a nominal system that ignored state disturbances causing large costs.
Both the real and the nominal trajectories are calculated using MPPI while the tracking controller was iterative Linear Quadratic Regulator (iLQR).
In this setup, the tracking controller would always be the one sending controls to the system and as MPPI was not aware of the tracking controller, it could end up fighting against the tracking controller.
In order to address this, Robust MPPI (RMPPI) was developed in [[6]](#6) [[7]](#7), which applied the tracking controller feedback within the samples MPPI used.
RMPPI also contains other changes that when taken together provided an upper bound on how quickly the cost function can grow due to disturbance.
Our library contains implementations of these algorithmic improvements as different controllers are the best choice in different scenarios.

## References
<a id="1">[1]</a>
G. Williams, P. Drews, B. Goldfain, J. M. Rehg, and E. A. Theodorou,
"Aggressive Driving with Model Predictive Path Integral Control," in _2016 IEEE
International Conference on Robotics and Automation (ICRA)_. IEEE, 2016, pp. 1433–1440. [Online].
Available: [https://ieeexplore.ieee.org/document/7487277/](https://ieeexplore.ieee.org/document/7487277/)

<a id="2">[2]</a>
G. Williams, P. Drews, B. Goldfain, J. M. Rehg, and
E. A. Theodorou, "Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous
Driving," _IEEE Transactions on Robotics_, vol. 34, no. 6, pp. 1603-1622, 201.
[Online]. Available:
[https://ieeexplore.ieee.org/document/8558663](https://ieeexplore.ieee.org/document/8558663)

<a id="3">[3]</a>
Z. Wang, O. So, K. Lee, and E. A. Theodorou, "Adaptive Risk Sensitive Model
Predictive Control with Stochastic Search,"
in _Proceedings of the 3rd Conference on Learning for Dynamics and Control_.
PMLR, 2021, pp. 510–522. [Online]. Available:
[https://proceedings.mlr.press/v144/wang21b.html](https://proceedings.mlr.press/v144/wang21b.html)

<a id="4">[4]</a>
N. Wagener, C.-A. Cheng, J. Sacks, and B. Boots,
"An Online Learning Approach to Model Predictive Control,"
in _Proceedings of Robotics: Science and Systems,_
Freiburg im Breisgau, Germany, Jun. 2019.
[Online]. Available: [http://arxiv.org/abs/1902.08967](http://arxiv.org/abs/1902.08967)

<a id="5">[5]</a>
G. Williams, B. Goldfain, P. Drews, K. Saigol, J. Rehg,
and E. Theodorou, "Robust Sampling Based Model
Predictive Control with Sparse Objective Information," in
_Robotics: Science and Systems XIV._ Robotics: Science
and Systems Foundation, Jun. 2018. [Online]. Available:
[http://www.roboticsproceedings.org/rss14/p42.pdf](http://www.roboticsproceedings.org/rss14/p42.pdf)

<a id="6">[6]</a>
M. Gandhi, B. Vlahov, J. Gibson, G. Williams,
and E. A. Theodorou, "Robust Model Predictive Path
Integral Control: Analysis and Performance Guarantees,"
_IEEE Robotics and Automation Letters,_ vol. 6, no. 2,
pp. 1423-1430, Feb. 2021. [Online]. Available:
[https://arxiv.org/abs/2102.09027v1](https://arxiv.org/abs/2102.09027v1)

<a id="7">[7]</a>
G. R. Williams, "Model Predictive Path Integral Control: Theoretical
Foundations and Applications to Autonomous Driving," 2019
