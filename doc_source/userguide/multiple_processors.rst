
XICSRT on Multiple Processors
=============================

XICSRT has built-in support for raytracing over multiple processors through the
use of Python's multiprocessing library. To use this functionality one only
needs to replace the call :any:`xicsrt.raytrace()` with
:any:`xicsrt.raytrace_mp()`.

Windows Support
---------------

Multiprocessing on Windows requires that :any:`xicsrt.raytrace_mp()` is
wrapped in a :code:`name == "__main__"` test. In a python script or jupyter
notebook on Windows this means replacing the call to :any:`xicsrt.raytrace()`
with the following:

.. code:: python

   if __name__ == "__main__":
       results = xicsrt.raytrace_mp(config)


The :any:`Command Line Interface <command_line>` can be used without any
modifications.

Cluster Computing
-----------------

A command line interface to xicsrt is available to enable computations on a
computer cluster.

A single call to xicsrt can utilize multiple processors but is (currently)
limited to run on a single computational node. However, it is easy to combine
results from multiple calls to xicsrt allowing multiple nodes to be used. The
use of slurm Job Arrays is recommended.

To launch a single call to xicsrt on 16 processors using slurm the following
command can be used:

.. code:: bash

    srun -n1 -c16 xicsrt config.json --mp --numruns 16 --processes 16

For multiple parallel calls to xicsrt use the :code:`--suffix` option to give
all output files a unique name.

.. note::

  For multiple calls to xicsrt make sure that the random seed is
  either

  1. equal to :code:`None` (the default)
  2. set to a different value for each call using the :code:`--seed` argument.

**Job Array Example**

Below is a simple example of a slurm batch file to run xicsrt on 64 processors
over 4 nodes.

`job.sh`

.. code:: bash

  #!/bin/bash
  #SBATCH -J xicsrt
  #SBATCH -o ./job_%A_%a.out
  #SBATCH -e ./job_%A_%a.err
  #SBATCH --nodes=1
  #SBATCH --cpus-per-task=16
  #SBATCH --array=0-3

  srun xicsrt config.json --mp --numruns 16 --processes 16 --suffix $SLURM_ARRAY_TASK_ID &> xicsrt.log

To send this job the queue type :code:`sbatch job.sh` at the command line.