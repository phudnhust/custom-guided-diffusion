import torch

print(torch.cuda.get_device_name(torch.device('cuda:0')))
# print(torch.cuda.memory_stats(torch.device('cuda:0')))

print(torch.cuda.get_device_name(torch.device('cuda:1')))
# print(torch.cuda.memory_stats(torch.device('cuda:1')))

GPUS_PER_NODE = 8
SETUP_RETRY_COUNT = 3

from mpi4py import MPI
import os
string = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
print(string)
