---
title: MPPI-Generic
layout: page
description: " "
---
# MPPI-Generic

MPPI-Generic is a C++/CUDA header-only library for conducting stochastic optimal control in real-time on a NVIDA GPU.

## Main Features

* Performance-focused code optimizations that makes near-full utilization of NVIDIA hardware
* Dynamics and Cost Function-agnostic implementations of Model Predictive Path Integral (MPPI) [[1]](#1), Tube-MPPI [[2]](@2), and Robust MPPI (RMPPI) [[3]](#3) algorithms.
* Largest variety of provided Dynamics and Cost Functions publicly available to use with MPPI, Tube-MPPI, and RMPPI
* Different sampling distributions such as Gaussian noise, Colored noise [[4]](#4), NLN noise [[5]](#5), and Smooth-MPPI's sampling scheme [[6]](#6)
* APIs to allow for researchers to write custom Dynamcis, Cost Functions, stochastic Controllers, and Sampling Distributions
that allow plug-and-play swapping with existing components.

## Table of Contents

* [The MPPI algorithm](docs/mppi.md)
* [Installation](docs/setup.md)
* [Getting started using the library](examples.md)
* [How do we achieve high performance?](docs/performance.md)
* [Comparisons to other popular implementations of MPPI](docs/benchmarks.md)

## Citation
If you use this work, please cite the following paper:
```BibTex
@misc{vlahov2024mppi,
      title={MPPI-Generic: A CUDA Library for Stochastic Optimization},
      author={Bogdan Vlahov and Jason Gibson and Manan Gandhi and Evangelos A. Theodorou},
      year={2024},
      eprint={2409.07563},
      archivePrefix={arXiv},
      primaryClass={cs.MS},
      url={https://arxiv.org/abs/2409.07563},
}
```


## Publications using MPPI-Generic
We have been developing MPPI-Generic for years to ensure that it is useful on real hardware in a variety of situations.
Below are papers that have already started using the MPPI-Generic library.

B. Vlahov, J. Gibson, D. D. Fan, P. Spieler, A.-a. Agha-mohammadi, and E. A. Theodorou,
"Low Frequency Sampling in Model Predictive Path Integral Control,"
*IEEE Robotics and Automation Letters,* pp. 1–8, 2024. [Online].
Available: [https://ieeexplore.ieee.org/document/10480553](https://ieeexplore.ieee.org/document/10480553)

A. M. Patel, M. J. Bays, E. N. Evans, J. R. Eastridge, and E. A. Theodorou,
"Model-Predictive Path-Integral Control of an Unmanned Surface Vessel with Wave
Disturbance," in *OCEANS 2023 - MTS/IEEE U.S. Gulf Coast,* Sep. 2023, pp. 1–7. [Online].
Available: [https://ieeexplore.ieee.org/document/10336978](https://ieeexplore.ieee.org/document/10336978)

J. Gibson, B. Vlahov, D. Fan, P. Spieler, D. Pastor, A.-a. Agha-mohammadi, and E. A. Theodorou,
"A Multi-step Dynamics Modeling Framework For Autonomous Driving In Multiple Environments,"
in *2023 IEEE International Conference on Robotics and Automation (ICRA).*
IEEE, May 2023, pp. 7959–7965. [Online].
Available: [https://ieeexplore.ieee.org/document/10161330](https://ieeexplore.ieee.org/document/10161330)

M. Gandhi, H. Almubarak, Y. Aoyama, and E. Theodorou,
"Safety in Augmented Importance Sampling: Performance Bounds for Robust MPPI,"
Apr. 2022. [Online].
Available: [http://arxiv.org/abs/2204.05963](http://arxiv.org/abs/2204.05963)

M. Gandhi, B. Vlahov, J. Gibson, G. Williams, and E. A. Theodorou,
"Robust Model Predictive Path Integral Control: Analysis and Performance Guarantees,"
*IEEE Robotics and Automation Letters,* vol. 6, no. 2, pp. 1423–1430, Feb. 2021. [Online].
Available: [https://arxiv.org/abs/2102.09027v1](https://arxiv.org/abs/2102.09027v1)

{% comment %}
## Todos
Things that still need to be done:
- [ ] Create a page for API extensability
- [x] Create pages with example code
- [x] Create a page showing off the benchmarks compared to other implementations
- [x] Create a page talking about performance improvements we have made
- [ ] Create a page for tips for using the Library
- [x] Add references to this page for MPPI, Tube-MPPI, RMPPI
- [x] Add a "cite this library using..." bibtex/other formats citation
- [x] Change Setup to Installation/Getting started
- [x]  Figure out how to make the page wider
- [ ] Add some common troubleshooting answers for installation page (nvidia architecture not found...)
- [ ] Add CUDA incompatiblities (CuFFT being broken in some versions of CUDA)
{% endcomment %}


## References
<a id="1">[1]</a>
G. Williams, P. Drews, B. Goldfain, J. M. Rehg, and E. A. Theodorou,
"Aggressive Driving with Model Predictive Path Integral Control," in _2016 IEEE
International Conference on Robotics and Automation (ICRA)_. IEEE, 2016, pp. 1433–1440. [Online].
Available: [https://ieeexplore.ieee.org/document/7487277/](https://ieeexplore.ieee.org/document/7487277/)

<a id="2">[2]</a>
G. Williams, B. Goldfain, P. Drews, K. Saigol, J. Rehg,
and E. Theodorou, "Robust Sampling Based Model
Predictive Control with Sparse Objective Information," in
_Robotics: Science and Systems XIV._ Robotics: Science
and Systems Foundation, Jun. 2018. [Online]. Available:
[http://www.roboticsproceedings.org/rss14/p42.pdf](http://www.roboticsproceedings.org/rss14/p42.pdf)

<a id="3">[3]</a>
M. Gandhi, B. Vlahov, J. Gibson, G. Williams,
and E. A. Theodorou, "Robust Model Predictive Path
Integral Control: Analysis and Performance Guarantees,"
_IEEE Robotics and Automation Letters,_ vol. 6, no. 2,
pp. 1423–1430, Feb. 2021. [Online]. Available:
[https://arxiv.org/abs/2102.09027v1](https://arxiv.org/abs/2102.09027v1)


<a id="4">[4]</a>
B. Vlahov, J. Gibson, D. D. Fan, P. Spieler, A.-a. Agha-mohammadi, and E. A. Theodorou,
"Low Frequency Sampling in Model Predictive Path Integral
Control," *IEEE Robotics and Automation Letters*, pp. 1–8, 2024. [Online]. Available:
[https://ieeexplore.ieee.org/document/10480](https://ieeexplore.ieee.org/document/10480)

<a id="5">[5]</a>
I. S. Mohamed, K. Yin, and L. Liu,
"Autonomous Navigation of AGVs in Unknown Cluttered Environments:
Log-MPPI Control Strategy," *IEEE Robotics and Automation Letters,*
vol. 7, no. 4, pp. 10 240–10 247, 2022. [Online]. Available:
[https://ieeexplore.ieee.org/document/9834098](https://ieeexplore.ieee.org/document/9834098)

<a id="6">[6]</a>
T. Kim, G. Park, K. Kwak, J. Bae, and W. Lee,
"Smooth Model Predictive Path Integral Control Without Smoothing,"
*IEEE Robotics and Automation Letters*, vol. 7, no. 4, pp. 10 406–10 413, 2021. [Online]. Available:
[https://ieeexplore.ieee.org/document/9835021](https://ieeexplore.ieee.org/document/9835021)
