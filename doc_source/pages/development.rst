
Development Guide
=================
This document contains a guide for development of XICSRT.

.. warning::

    Documentation of XICSRT has only just begun. This please help us improve
    this development guide.

.. toctree::
    :hidden:

    todo

A list of needed improvements for XICSRT can be found at
:any:`List of Todo Items`. Also see the open `issues`_ on the bitbucket git
repository.

Programming Projects
--------------------
Here are a list of projects for XICSRT improvements. These are particularly
well suited for a undergraduate summer student, or anyone looking for a
nice self-contained improvement project.

Time estimates are for someone who has experience running XICSRT, has a very
strong python/numpy programming background, but who is not familiar with the
XICSRT code base. Time estimates include time for testing and verification.


.. admonition:: Add a cylindrical reflector object

  | Time Estimate: < 1 week
  | Create an object named `XicsrtOpticCrystalCylindrical`. Object should be
    very similar to :any:`XicsrtOpticCrystalSpherical` but with toroidal
    geometry. The object should be defined with a major and minor radius. Test
    against :any:`XicsrtOpticCrystal`, :any:`XicsrtOpticCrystalSpherical` and
    `XicsrtOpticCrystalToroidal`.

  |     added: 2021-01-24


.. admonition:: Add a toroidal reflector object

  | Time Estimate: **< 1 week**
  | Create an object named `XicsrtOpticCrystalToroidal`. Object should be very
    similar to :any:`XicsrtOpticCrystalSpherical` but with toroidal geometry.
    The object should be defined with a major and minor radius. Test against
    :any:`XicsrtOpticCrystal`, :any:`XicsrtOpticCrystalSpherical` and
    `XicsrtOpticCrystalCylindrical`.

  |     added: 2021-01-24


.. admonition:: Improve algorithm for isotropic emission with x & y limits

  | Time Estimate: **2 weeks**
  | An important vector distribution used in XICSRT is the isotropic
    distribution with separate x & y angular bounds (rectangular cone,
    pyramid). The function that implements this can be found in
    :any:`vector_dist_isotropic_xy`, The current algorithm uses filtering
    from an emission cone with circular cross-section. This is accurate but
    highly inefficient, especially if the x & y spread are very different.

  A more efficient algorithm is needed. This is almost certainly a solved
  problem so the first thing to do is to search the literature and look
  at other ray-tracing projects to find an existing example.

  If an example cannot be found I see three possibilities for a solution:

  1. Calculate the Joint Cumulative Distribution Function (CDF) on a plane
     of constant z. Use this to draw random points on the plane. A good
     (free) text on probability distributions can be found here:
     `probabilitycourse.com`_.
  2. Pull points on a unit-sphere only within the boundary of the rectangular-
     cone intersection. I have no idea how to approach this other than falling
     back on solution 1.
  3. Continue using a filtering scheme, but start with a different boundary
     shape than a circle that is closer to the one needed for the rectangular
     cone.

  It is important that the final algorithm is accurate to machine precision.

  |     added: 2021-01-24


.. admonition:: Improve mesh-grid pre-selection algorithm

  | Time Estimate: **2 weeks**
  | Mesh-grid optics in XICS use a mesh-refinement alorithm that uses
    a course grid to pre-select faces to test on the full mesh. The
    current algorithm is lossy, and often tests more faces than are
    actually required.

  The goal of this project is to improve the pre-selection algorithm
  to eliminate ray losses. This can likely be done while also improving
  performance and allowing coarser pre-selection grids.

  The specific methods in :any:`XicsrtOpticMesh` that need improvement is
  :any:`find_near_faces` however to achive this change will also be needed
  in :any:`_mesh_precalc<XicsrtOpticMesh>` and :any:`mesh_intersect_2`.

  Note:
    Consider how the new algorithm will perform with grids in which
    the x & y point density are very different. The current algorithm
    behaves especially poorly in terms of losses in those cases.


.. admonition:: Develop a numba accelerated version of XICSRT

  | Time Estimate: **1 month**
  | Performance of XICSRT can likely be dramatically improved by using the
    the `numba`_ package. Numba provides just-in-time compilation of python
    code and is highly integrated with numpy, making it well suited for
    inclusion in XICSRT.

  Numba can often provide acceleration by just adding the `@jit` decorator.
  To really achieve acceleration, it is likely that some code changes are
  required. When available, use the `@vectorize` or `@guvectorize` decorators.
  Consider how this code will perform on multiple cpus or gpus. Consider
  the use of `prange` when approprate.

  Development should be done in separate branch so as not to affect the
  master branch (though any code improvements that are not numba specific
  should still be made in the master branch).The new numba branch should
  contain a way to turn off numba, and care should be taken that the code
  still works seamlessly with numba turned off. Performance should be measured
  between the non-numba version, the numba version, and the numba version with
  numba turned off.

  Note:
    XICSRT is already highly vectorized and utilizes numpy array manipulations
    whenever possible. These operations are already very fast, and some are
    even optimized for multiple processors. For this reason it is unclear
    how much speed improvement is actually achievable with numba. During
    development of the numba branch please also look into optimizing the
    standard numpy code.

  Note:
    The main goals of XICSRT project are readability, easy development,
    cross-platform compatiblity, and pure python. Code changes that improve
    performance but make the code very complex should be avoided.

  |     added: 2021-01-24


.. _issues: https://bitbucket.org/amicitas/xicsrt/issues
.. _probabilitycourse.com: https://www.probabilitycourse.com
.. _numba: https://numba.pydata.org/

