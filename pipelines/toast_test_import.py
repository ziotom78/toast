#!/usr/bin/env python3

print('Checkpoint # 0 : importing toast.mpi', flush=True)

from toast.mpi import MPI

print('{:5} : Checkpoint # 1 : importing toast'.format(MPI.COMM_WORLD.rank),
      flush=True)

import toast

print('{:5} : Checkpoint # 2 : toast imported'.format(MPI.COMM_WORLD.rank),
      flush=True)
