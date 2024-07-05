---
title: MPPI-Generic
layout: page
description: " "
---
# MPPI-Generic

MPPI-Generic is a C++/CUDA header-only library for conducting stochastic optimal control in real-time on a NVIDA GPU.
It provides implementations of several Dynamics and Cost Functions as well as implementations of Model Predictive Path Integral (MPPI) [[1]](#1),
Tube-MPPI [[2]](@2), and Robust MPPI (RMPPI) [[3]](#3).
In addition, it provides extendable APIs for Dynamics, Cost Functions, Controllers, and more so that researchers can design their own custom classes to push their research forward.

## Table of Contents

* [Installation](docs/setup.md)
* [MPPI](docs/mppi.md)
* [How do we achieve high performance](docs/performance.md)


## Publications using MPPI-Generic
TODO: Fill in using bibtex if possible

## Todos
Things that still need to be done:
- [ ] Create a page for API extensability
- [ ] Create pages with example code
- [ ] Create a page showing off the benchmarks compared to other implementations
- [x] Create a page talking about performance improvements we have made
- [ ] Create a page for tips for using the Library
- [x] Add references to this page for MPPI, Tube-MPPI, RMPPI
- [ ] Add a "cite this library using..." bibtex/other formats citation
- [ ] Change Setup to Installation/Getting started
- [x]  Figure out how to make the page wider
- [ ] Add some common troubleshooting answers for installation page (nvidia architecture not found...)
- [ ] Add CUDA incompatiblities (CuFFT being broken in some versions of CUDA)


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
