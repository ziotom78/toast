# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import ctypes as ct
from ctypes.util import find_library

import unittest

import healpy as hp

import numpy as np
import numpy.ctypeslib as npc

import quaternionarray as qa

from ..dist import Comm, Data
from ..operator import Operator
from ..tod import TOD
from ..tod import Interval

# Define portably the MPI communicator datatype

try:
    if MPI._sizeof(MPI.Comm) == ct.sizeof(ct.c_int):
        MPI_Comm = ct.c_int
    else:
        MPI_Comm = ct.c_void_p
except Exception as e:
    raise Exception('Failed to set the portable MPI communicator datatype. MPI4py is probably too old. You may need to install from a git checkout. ({})'.format(e))

libconviqt = ct.CDLL('libconviqt.so')

# Beam functions

libconviqt.conviqt_beam_new.restype = ct.c_void_p
libconviqt.conviqt_beam_new.argtypes = []

libconviqt.conviqt_beam_del.restype = ct.c_int
libconviqt.conviqt_beam_del.argtypes = [ct.c_void_p]

libconviqt.conviqt_beam_read.restype = ct.c_int
libconviqt.conviqt_beam_read.argtypes = [
    ct.c_void_p,
    ct.c_long,
    ct.c_long,
    ct.c_byte,
    ct.c_char_p,
    MPI_Comm
]

# Sky functions

libconviqt.conviqt_sky_new.restype = ct.c_void_p
libconviqt.conviqt_sky_new.argtypes = []

libconviqt.conviqt_sky_del.restype = ct.c_int
libconviqt.conviqt_sky_del.argtypes = [ct.c_void_p]

libconviqt.conviqt_sky_read.restype = ct.c_int
libconviqt.conviqt_sky_read.argtypes = [
    ct.c_void_p,
    ct.c_long,
    ct.c_byte,
    ct.c_char_p,
    ct.c_double,
    MPI_Comm
]

# Detector functions

libconviqt.conviqt_detector_new.restype = ct.c_void_p
libconviqt.conviqt_detector_new.argtypes = []

libconviqt.conviqt_detector_new_with_id.restype = ct.c_void_p
libconviqt.conviqt_detector_new_with_id.argtypes = [ct.c_char_p]

libconviqt.conviqt_detector_del.restype = ct.c_int
libconviqt.conviqt_detector_del.argtypes = [ct.c_void_p]

libconviqt.conviqt_detector_set_epsilon.restype = ct.c_int
libconviqt.conviqt_detector_set_epsilon.argtypes = [
    ct.c_void_p,
    ct.c_double
]

libconviqt.conviqt_detector_get_epsilon.restype = ct.c_int
libconviqt.conviqt_detector_get_epsilon.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_double)
]

libconviqt.conviqt_detector_get_id.restype = ct.c_int
libconviqt.conviqt_detector_get_id.argtypes = [
    ct.c_void_p,
    ct.c_char_p
]

# Pointing functions

libconviqt.conviqt_pointing_new.restype = ct.c_void_p
libconviqt.conviqt_pointing_new.argtypes = []

libconviqt.conviqt_pointing_del.restype = ct.c_int
libconviqt.conviqt_pointing_del.argtypes = [ct.c_void_p]

libconviqt.conviqt_pointing_alloc.restype = ct.c_int
libconviqt.conviqt_pointing_alloc.argtypes = [
    ct.c_void_p,
    ct.c_long
]

libconviqt.conviqt_pointing_data.restype = ct.POINTER(ct.c_double)
libconviqt.conviqt_pointing_data.argtypes = [ct.c_void_p]

# Convolver functions

libconviqt.conviqt_convolver_new.restype = ct.c_void_p
libconviqt.conviqt_convolver_new.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_byte,
    ct.c_long,
    ct.c_long,
    ct.c_long,
    ct.c_long,
    ct.c_long,
    MPI_Comm
]

libconviqt.conviqt_convolver_convolve.restype = ct.c_int
libconviqt.conviqt_convolver_convolve.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_byte
]

libconviqt.conviqt_convolver_del.restype = ct.c_int
libconviqt.conviqt_convolver_del.argtypes = [ct.c_void_p]


class OpSimConviqt(Operator):
    """
    Operator which uses libconviqt to generate beam-convolved timestreams.

    This passes through each observation and ...

    Args:
        
    """

    def __init__(self, lmax, beamlmax, beammmax, detectordata, pol=True, fwhm=4.0, nbetafac=6000, mcsamples=0, lmaxout=6000, order=13, calibrate=True, flavor=None, dxx=True):
        """
        Set up on-the-fly signal convolution. Inputs:
        lmax : sky maximum ell (and m). Actual resolution in the Healpix FITS file may differ.
        beamlmax : beam maximum ell. Actual resolution in the Healpix FITS file may differ.
        beammmax : beam maximum m. Actual resolution in the Healpix FITS file may differ.
        detectordata : list of (detector_name, detector_sky_file, detector_beam_file, epsilon, psipol[radian]) tuples
        pol(True) : boolean to determine if polarized simulation is needed
        fwhm(5.0) : width of a symmetric gaussian beam [in arcmin] already present in the skyfile (will be deconvolved away).
        nbetafac(6000) : conviqt resolution parameter (expert mode)
        mcsamples(0) : reserved input for future Monte Carlo mode
        lmaxout(6000) : Convolution resolution
        order(5) : conviqt order parameter (expert mode)
        calibrate(True) : Calibrate intensity to 1.0, rather than (1+epsilon)/2
        dxx(True) : The beam frame is either Dxx or Pxx. Pxx includes the rotation to polarization sensitive basis, Dxx does not.
                    When Dxx=True, detector orientation from attitude quaternions is corrected for the polarization angle.
        """
        # We call the parent class constructor, which currently does nothing
        super().__init__()

        self._lmax = lmax
        self._beamlmax = beamlmax
        self._beammmax = beammmax
        self._detectordata = {}
        for entry in detectordata:
            self._detectordata[entry[0]] = entry[1:]
        self._pol = pol
        self._fwhm = fwhm
        self._nbetafac = nbetafac
        self._mcsamples = mcsamples
        self._lmaxout = lmaxout
        self._order = order
        self._calibrate = calibrate
        self._dxx = dxx

        self._flavor = flavor        
        

    def exec(self, data):
        """
        Calling exec will perform the convolution over the communicator, one detector at a time.
        All MPI tasks must have the same list of detectors.
        """
        # the two-level pytoast communicator
        #comm = data.comm
        # the global communicator
        #cworld = comm.comm_world
        # the communicator within the group
        #cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        #crank = comm.comm_rank

        xaxis, yaxis, zaxis = np.eye(3)
        nullquat = np.array([0,0,0,1], dtype=np.float64)

        for obs in data.obs:
            tod = obs['tod']
            intrvl = obs['intervals']

            comm_ptr = MPI._addressof(tod.mpicomm)
            comm = MPI_Comm.from_address(comm_ptr)

            for det in tod.local_dets:

                try:
                    skyfile, beamfile, epsilon, psipol = self._detectordata[det]
                except:
                    raise Exception('ERROR: conviqt object not initialized to convolve detector {}. Available detectors are {}'.format(det, self._detectordata.keys()))
                    
                sky = libconviqt.conviqt_sky_new()
                err = libconviqt.conviqt_sky_read(sky, self._lmax, self._pol, skyfile.encode(), self._fwhm, comm)
                if err != 0: raise Exception('Failed to load ' + skyfile)

                beam = libconviqt.conviqt_beam_new()
                err = libconviqt.conviqt_beam_read(beam, self._beamlmax, self._beammmax, self._pol, beamfile.encode(), comm)
                if err != 0: raise Exception('Failed to load ' + beamfile)

                detector = libconviqt.conviqt_detector_new_with_id(det.encode())
                libconviqt.conviqt_detector_set_epsilon(detector, epsilon)
                
                # We need the three pointing angles to describe the pointing. read_pntg returns the attitude quaternions.
                flags = tod.read_flags(detector=det, local_start=0, n=tod.local_samples)
                pdata, pflags = tod.read_pntg(detector=det, local_start=0, n=tod.local_samples)

                pdata = pdata.reshape(-1,4).copy()
                pdata[ pflags != 0 ] = nullquat
                
                vec_dir = qa.rotate( pdata, zaxis ).T.copy()
                
                theta, phi = hp.vec2dir(*vec_dir)
                theta[ pflags != 0 ] = 0
                phi[ pflags != 0 ] = 0

                vec_orient = qa.rotate( pdata, xaxis ).T.copy()

                ypa = vec_orient[0]*vec_dir[1] - vec_orient[1]*vec_dir[0]
                xpa = -vec_dir[2]*(vec_orient[0]*vec_dir[0] + vec_orient[1]*vec_dir[1]) + vec_orient[2]*(vec_dir[0]**2 + vec_dir[1]**2)

                psi = np.arctan2(ypa, xpa)

                # Is the psi angle in Pxx or Dxx? Pxx will include the detector polarization angle, Dxx will not.

                if self._dxx:
                    psi -= psipol

                pnt = libconviqt.conviqt_pointing_new()

                err = libconviqt.conviqt_pointing_alloc( pnt, tod.local_samples*5)
                if err != 0: raise Exception('Failed to allocate pointing array')

                ppnt = libconviqt.conviqt_pointing_data(pnt)

                for row in range(tod.local_samples):
                    ppnt[row*5 + 0] = phi[row]
                    ppnt[row*5 + 1] = theta[row]
                    ppnt[row*5 + 2] = psi[row]
                    ppnt[row*5 + 3] = 0 # This column will host the convolved data upon exit
                    ppnt[row*5 + 4] = 0 # libconviqt will assign the running indices to this column.

                convolver = libconviqt.conviqt_convolver_new(sky, beam, detector, self._pol, self._lmax, self._beammmax, self._nbetafac, self._mcsamples, self._lmaxout, self._order, comm)

                if convolver is None: raise Exception("Failed to instantiate convolver")

                err = libconviqt.conviqt_convolver_convolve(convolver, pnt, self._calibrate)
                if err != 0: raise Exception('Convolution FAILED!')

                # The pointer to the data will have changed during the convolution call ...

                ppnt = libconviqt.conviqt_pointing_data( pnt )

                convolved_data = np.zeros(tod.local_samples)
                for row in range(tod.local_samples):
                    convolved_data[row] = ppnt[row*5+3]

                libconviqt.conviqt_convolver_del(convolver)

                tod.write(detector=det, flavor=self._flavor, local_start=0, data=convolved_data, flags=flags|pflags)

                libconviqt.conviqt_pointing_del(pnt)
                libconviqt.conviqt_detector_del(detector)
                libconviqt.conviqt_beam_del(beam)
                libconviqt.conviqt_sky_del(sky)

        return

