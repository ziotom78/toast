# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os

import numpy as np

from ..timing import function_timer, Timer
from ..utils import Logger, Environment

from ..map import DistPixels
from ..todmap import OpSimPySM, OpSimScan, OpMadam, OpLocalPixels


def add_sky_map_args(parser):
    """ Add the sky arguments
    """

    parser.add_argument("--input-map", required=False, help="Input map for signal")

    # The nside may already be added
    try:
        parser.add_argument(
            "--nside", required=False, default=512, type=np.int, help="Healpix NSIDE"
        )
    except argparse.ArgumentError:
        pass
    # The coordinate system may already be added
    try:
        parser.add_argument(
            "--coord", required=False, default="C", help="Sky coordinate system [C,E,G]"
        )
    except argparse.ArgumentError:
        pass

    return


def add_pysm_args(parser):
    """ Add the sky arguments
    """

    parser.add_argument(
        "--pysm-model",
        required=False,
        help="Comma separated models for on-the-fly PySM "
        'simulation, e.g. "s1,d6,f1,a2"',
    )

    parser.add_argument(
        "--pysm-apply-beam",
        required=False,
        action="store_true",
        help="Convolve sky with detector beam",
        dest="pysm_apply_beam",
    )
    parser.add_argument(
        "--no-pysm-apply-beam",
        required=False,
        action="store_false",
        help="Do not convolve sky with detector beam.",
        dest="pysm_apply_beam",
    )
    parser.set_defaults(pysm_apply_beam=True)

    parser.add_argument(
        "--pysm-precomputed-cmb-K_CMB",
        required=False,
        help="Precomputed CMB map for PySM in K_CMB"
        'it overrides any model defined in pysm_model"',
    )

    # The nside may already be added
    try:
        parser.add_argument(
            "--nside", required=False, default=512, type=np.int, help="Healpix NSIDE"
        )
    except argparse.ArgumentError:
        pass
    # The coordinate system may already be added
    try:
        parser.add_argument(
            "--coord", required=False, default="C", help="Sky coordinate system [C,E,G]"
        )
    except argparse.ArgumentError:
        pass

    return


@function_timer
def scan_sky_signal(
    args, comm, data, localsm, subnpix, cache_prefix="signal", verbose=True
):
    """ Scan sky signal from a map.

    """
    if not args.input_map:
        return None

    log = Logger.get()
    timer = Timer()
    timer.start()

    if comm.world_rank == 0 and verbose:
        log.info("Scanning input map")

    npix = 12 * args.nside ** 2

    # Scan the sky signal
    if comm.world_rank == 0 and not os.path.isfile(args.input_map):
        raise RuntimeError("Input map does not exist: {}".format(args.input_map))
    distmap = DistPixels(
        comm=comm.comm_world,
        size=npix,
        nnz=3,
        dtype=np.float32,
        submap=subnpix,
        local=localsm,
    )
    distmap.read_healpix_fits(args.input_map)
    scansim = OpSimScan(distmap=distmap, out=cache_prefix)
    scansim.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    if comm.world_rank == 0 and verbose:
        timer.report_clear("Read and sample map")

    return cache_prefix


@function_timer
def simulate_sky_signal(
    args, comm, data, focalplanes, subnpix, localsm, cache_prefix, verbose=False
):
    """ Use PySM to simulate smoothed sky signal.

    """
    if not args.pysm_model:
        return None
    timer = Timer()
    timer.start()
    # Convolve a signal TOD from PySM
    op_sim_pysm = OpSimPySM(
        comm=comm.comm_rank,
        out=cache_prefix,
        pysm_model=args.pysm_model.split(","),
        pysm_precomputed_cmb_K_CMB=args.pysm_precomputed_cmb_K_CMB,
        focalplanes=focalplanes,
        nside=args.nside,
        subnpix=subnpix,
        localsm=localsm,
        apply_beam=args.pysm_apply_beam,
        coord=args.coord,
    )
    op_sim_pysm.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    if comm.world_rank == 0 and verbose:
        timer.report_clear("PySM")

    return cache_prefix
