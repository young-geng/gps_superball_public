GPS on SUPERball
================

This code is a reimplementation of the guided policy search algorithm and LQG-based trajectory optimization, meant to help others understand, reuse, and build upon existing work.

This particular fork of the [main repository](https://github.com/cbfinn/gps) contains the implementation of mirror descent guided policy search applied to the SUPERball tensegrity robot. More details can be found on the [project website](http://rll.berkeley.edu/drl_tensegrity/). Files of interest include the following:

* [agent_superball.py](python/gps/agent/ros/agent_superball.py)
* [gps_main.py](python/gps/gps_main.py)
* [hyperparams.py](experiments/md_multiflops/2c_nv_mp/hyperparams.py)
* [lin_gauss_init.py](python/gps/algorithm/policy/lin_gauss_init.py)

For full documentation, see [rll.berkeley.edu/gps](http://rll.berkeley.edu/gps).

The code base is **a work in progress**. See the [FAQ](http://rll.berkeley.edu/gps/faq.html) for information on planned future additions to the code.
