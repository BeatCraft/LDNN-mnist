#!/bin/sh

#mpiexec -host localhost:2 -n 2 python3 ./mpi.py

mpiexec -np 8 python3 ./mpi.py

