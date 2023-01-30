import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def train(gpu,ngpus_per_node):
    print(f"gpu:{gpu}, total_gpu:{ngpus_per_node}")
    # step1 기본 setting
    torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://141.223.163.114:2254',
            world_size=ngpus_per_node, #전체 gpu 개수
            rank=gpu)
    # step2 model 정의 -> to(gpu)
    model = nn.Linear(3,5).to(gpu)
    # step3 DDP로 묶기
    model = DDP(model, device_ids = [gpu])
    print(model)
    print("Done")


def main():
    ngpus_per_node = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, nprocs = ngpus_per_node, args=(ngpus_per_node, ))

if __name__ == '__main__':
    main()