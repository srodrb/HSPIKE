#!/bin/bash
nvcc -I/usr/local/cuda/include cuda_solver_test.cu -o cuSolverTest -L/usr/local/cuda/lib64 -lcusolver -lcusparse