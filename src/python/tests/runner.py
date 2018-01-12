# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

print('runner.py : CHECKPOINT # 1', flush=True)
from ..mpi import MPI

print('runner.py : CHECKPOINT # 2', flush=True)
import os
import sys
import unittest
import warnings

print('runner.py : CHECKPOINT # 3', flush=True)
from .._version import __version__

from .mpi import MPITestRunner
from .mpi import MPITestInfo

from ..vis import set_backend

from .ctoast import test_ctoast

print('runner.py : CHECKPOINT # 4', flush=True)
from . import cbuffer as testcbuffer
print('runner.py : importing testcache', flush=True)
from . import cache as testcache
print('runner.py : importing testtiming', flush=True)
from . import timing as testtiming
print('runner.py : importing rng', flush=True)
from . import rng as testrng
print('runner.py : importing fft', flush=True)
from . import fft as testfft
print('runner.py : importing dist', flush=True)
from . import dist as testdist
print('runner.py : importing testqarray', flush=True)
from . import qarray as testqarray
print('runner.py : CHECKPOINT # 5', flush=True)
from . import tod as testtod
from . import psd_math as testpsdmath
from . import intervals as testintervals
from . import cov as testcov
from . import ops_pmat as testopspmat
from . import ops_dipole as testopsdipole
from . import ops_simnoise as testopssimnoise
from . import ops_polyfilter as testopspolyfilter
from . import ops_groundfilter as testopsgroundfilter
print('runner.py : CHECKPOINT # 6', flush=True)
from . import ops_gainscrambler as testopsgainscrambler
from . import ops_memorycounter as testopsmemorycounter
from . import ops_madam as testopsmadam
from . import map_satellite as testmapsatellite
from . import map_ground as testmapground
from . import binned as testbinned

print('runner.py : CHECKPOINT # 7', flush=True)
from ..tod import tidas_available
if tidas_available:
    from . import tidas as testtidas

from ..map import libsharp_available
if libsharp_available:
    from . import ops_sim_pysm as testopspysm
    from . import smooth as testsmooth

print('runner.py : CHECKPOINT # 8', flush=True)

def test(name=None):
    # We run tests with COMM_WORLD
    comm = MPI.COMM_WORLD

    set_backend()

    outdir = "toast_test_output"

    if comm.rank == 0:
        outdir = os.path.abspath(outdir)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    outdir = comm.bcast(outdir, root=0)

    if (name is None) or (name == "ctoast") :
        # Run tests from the compiled library.  This separately uses
        # MPI_COMM_WORLD.
        test_ctoast()

    # Run python tests.

    loader = unittest.TestLoader()
    mpirunner = MPITestRunner(verbosity=2)
    suite = unittest.TestSuite()

    if name is None:
        suite.addTest( loader.loadTestsFromModule(testcbuffer) )
        suite.addTest( loader.loadTestsFromModule(testcache) )
        suite.addTest( loader.loadTestsFromModule(testtiming) )
        suite.addTest( loader.loadTestsFromModule(testrng) )
        suite.addTest( loader.loadTestsFromModule(testfft) )
        suite.addTest( loader.loadTestsFromModule(testdist) )
        suite.addTest( loader.loadTestsFromModule(testqarray) )
        suite.addTest( loader.loadTestsFromModule(testtod) )
        suite.addTest( loader.loadTestsFromModule(testpsdmath) )
        suite.addTest( loader.loadTestsFromModule(testintervals) )
        suite.addTest( loader.loadTestsFromModule(testopspmat) )
        suite.addTest( loader.loadTestsFromModule(testcov) )
        suite.addTest( loader.loadTestsFromModule(testopsdipole) )
        suite.addTest( loader.loadTestsFromModule(testopssimnoise) )
        suite.addTest( loader.loadTestsFromModule(testopspolyfilter) )
        suite.addTest( loader.loadTestsFromModule(testopsgroundfilter) )
        suite.addTest( loader.loadTestsFromModule(testopsgainscrambler) )
        suite.addTest( loader.loadTestsFromModule(testopsmemorycounter) )
        suite.addTest( loader.loadTestsFromModule(testopsmadam) )
        suite.addTest( loader.loadTestsFromModule(testmapsatellite) )
        suite.addTest( loader.loadTestsFromModule(testmapground) )
        suite.addTest( loader.loadTestsFromModule(testbinned) )
        if tidas_available:
            suite.addTest( loader.loadTestsFromModule(testtidas) )
        if libsharp_available:
            suite.addTest( loader.loadTestsFromModule(testopspysm) )
            suite.addTest( loader.loadTestsFromModule(testsmooth) )
    elif name != "ctoast":
        if (name == "tidas") and (not tidas_available):
            print("Cannot run TIDAS tests- package not available")
            return
        else:
            modname = "toast.tests.{}".format(name)
            suite.addTest( loader.loadTestsFromModule(sys.modules[modname]) )

    ret = 0
    with warnings.catch_warnings(record=True) as w:
        # Cause all toast warnings to be shown.
        warnings.simplefilter("always", UserWarning)
        _ret = mpirunner.run(suite)
        if not _ret.wasSuccessful():
            ret += 1

    if ret > 0:
        sys.exit(ret)

    return ret

print('runner.py : CHECKPOINT # 9', flush=True)
