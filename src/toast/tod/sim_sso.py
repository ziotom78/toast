# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from ..utils import Logger

from ..timing import function_timer, Timer

from ..op import Operator

import ephem

from toast.mpi import MPI

import toast.qarray as qa

import healpy as hp

from scipy.constants import au as AU

from scipy.interpolate import RectBivariateSpline


def to_JD(t):
    # Unix time stamp to Julian date
    # (days since -4712-01-01 12:00:00 UTC)
    return t / 86400.0 + 2440587.5


def to_MJD(t):
    # Convert Unix time stamp to modified Julian date
    # (days since 1858-11-17 00:00:00 UTC)
    return to_JD(t) - 2400000.5


def to_DJD(t):
    # Convert Unix time stamp to Dublin Julian date
    # (days since 1899-12-31 12:00:00)
    # This is the time format used by PyEphem
    return to_JD(t) - 2415020


class OpSimSSO(Operator):
    """Operator which generates Solar System Object timestreams.
    
    Args:
        name (str): Name of the SSO, must be recognized by pyEphem
        out (str): accumulate data to the cache with name
            <out>_<detector>.  If the named cache objects do not exist,
            then they are created.
        report_timing (bool):  Print out time taken to initialize,
             simulate and observe
    """

    def __init__(self, name, out="sso", report_timing=False):
        # Call the parent class constructor
        super().__init__()

        self.name = name
        self.sso = getattr(ephem, name)()
        self._out = out
        self._report_timing = report_timing
        return

    @function_timer
    def exec(self, data):
        """Generate timestreams.

        Args:
            data (toast.Data): The distributed data.

        Returns:
            None

        """

        log = Logger.get()
        group = data.comm.group
        for obs in data.obs:
            try:
                obsname = obs["name"]
            except Exception:
                obsname = "observation"

            site_lon = self._get_from_obs("site_lon", obs)
            site_lat = self._get_from_obs("site_lat", obs)
            site_alt = self._get_from_obs("site_alt", obs)

            observer = ephem.Observer()
            observer.lon = site_lon
            observer.lat = site_lat
            observer.elevation = site_alt  # In meters
            observer.epoch = "2000"
            observer.temp = 0  # in Celcius
            observer.compute_pressure()

            prefix = "{} : {} : ".format(group, obsname)
            tod = self._get_from_obs("tod", obs)
            comm = tod.mpicomm
            rank = 0
            if comm is not None:
                rank = comm.rank
            site = self._get_from_obs("site_id", obs)

            if comm is not None:
                comm.Barrier()
            if rank == 0:
                log.info("{}Setting up SSO simulation".format(prefix))

            # Get the observation time span and compute the horizontal
            # position of the SSO
            times = tod.local_times()
            sso_az, sso_el, sso_dist = self._get_sso_position(times, observer)

            tmr = Timer()
            if self._report_timing:
                if comm is not None:
                    comm.Barrier()
                tmr.start()

            self._observe_sso(sso_az, sso_el, sso_dist, tod, comm, prefix)

            del sso_az, sso_el, sso_dist

        if self._report_timing:
            if comm is not None:
                comm.Barrier()
            if rank == 0:
                tmr.stop()
                tmr.report("{}Simulated and observed SSO signal" "".format(prefix))
        return

    def _get_beam_map(self, det):
        """
        Construct a 2-dimensional interpolator for the beam
        """
        # FIXME: for now, just construct a symmetric Gaussian
        n = 301
        w = np.radians(3)
        x = np.linspace(-w, w, n)
        y = np.linspace(-w, w, n)
        xgrid, ygrid = np.meshgrid(x, y)
        fwhm_x = 3.0  # arc min
        fwhm_y = 3.0  # arc min
        theta = np.radians(0)  # orientation of the beam ellipsis
        xsigma = np.radians(fwhm_x / 60) / 2.355
        ysigma = np.radians(fwhm_y / 60) / 2.355
        # Rotate the coordinates by -theta
        rot = np.vstack(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )
        xx, yy = np.dot(rot, np.vstack([xgrid.ravel(), ygrid.ravel()]))
        model = amp * np.exp(-0.5 * (xx ** 2 / xsigma ** 2 + yy ** 2 / ysigma ** 2))
        model = model.reshape([n, n])
        beam = RectBivariateSpline(x, y, model)
        r = np.sqrt(w ** 2 + w ** 2)
        return beam, radius

    def _get_from_obs(self, name, obs):
        """ Extract value for name from observation.

        If name is not defined in observation, raise an exception.

        """
        if name not in obs:
            raise RuntimeError(
                "Error simulating SSS: observation " 'does not define "{}"'.format(name)
            )
        return obs[name]

    @function_timer
    def _get_sso_position(self, times, observer):
        """
        Calculate the SSO horizontal position
        """
        # FIXME: we could parallelize here and also interpolate the
        # SSO position from a low sample rate vector
        """
        tmin = times[0]
        tmax = times[-1]
        tmin_tot = tmin
        tmax_tot = tmax
        if comm is not None:
            tmin_tot = comm.allreduce(tmin, op=MPI.MIN)
            tmax_tot = comm.allreduce(tmax, op=MPI.MAX)
        """
        sso_az = np.zeros(times.size)
        sso_el = np.zeros(times.size)
        for i, t in enumerate(times):
            observer.date = to_DJD(t)
            self.sso.compute(observer)
            sso_az[i] = self.sso.az
            sso_el[i] = self.sso.alt
            sso_dist[i] = self.sso.earth_distance * AU
        return sso_az, sso_el, sso_dist

    @function_timer
    def _observe_sso(self, sso_az, sso_el, sso_dist, tod, comm, prefix):
        """
        Observe the SSO with each detector in tod
        """
        log = Logger.get()
        rank = 0
        if comm is not None:
            rank = comm.rank
        tmr = Timer()
        if self._report_timing:
            if comm is not None:
                comm.Barrier()
            tmr.start()

        nsamp = tod.local_samples[1]

        if rank == 0:
            log.info("{}Observing the SSO signal".format(prefix))

        for det in tod.local_dets:
            # Cache the output signal
            cachename = "{}_{}".format(self._out, det)
            if tod.cache.exists(cachename):
                ref = tod.cache.reference(cachename)
            else:
                ref = tod.cache.create(cachename, np.float64, (nsamp,))

            try:
                # Some TOD classes provide a shortcut to Az/El
                az, el = tod.read_azel(detector=det)
            except Exception as e:
                azelquat = tod.read_pntg(detector=det, azel=True)
                # Convert Az/El quaternion of the detector back into
                # angles for the simulation.
                theta, phi = qa.to_position(azelquat)
                # Azimuth is measured in the opposite direction
                # than longitude
                az = 2 * np.pi - phi
                el = np.pi / 2 - theta

            beam, radius = self._get_beam_map(det)

            # Interpolate the beam map at appropriate locations
            x = (az - sso_az) * np.cos(el)
            y = el - sso_el
            r = np.sqrt(x ** 2 + y ** 2)
            good = r < radius
            sig = beam(x[good], y[good], grid=False)
            ref[:][good] += sig

            del ref, sig, beam

        if self._report_timing:
            if comm is not None:
                comm.Barrier()
            if rank == 0:
                tmr.stop()
                tmr.report("{}OpSimSSO: Observe signal".format(prefix))
        return
