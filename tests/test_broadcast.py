import horovod.torch as hvd
import time
import random

hvd.init()

def broadcast(val, name=None):
    val = torch.tensor(val)
    hvd.broadcast_async(val, root_rank=0, name=name)

def train():
    global step

    for _ in range(5):
        step += 1
        l = random.randint(0, 3)
        time.sleep(l)
        print(step)

def main():
    global step
    step = 0
    for _ in range(4):
        train()

if __name__ == '__main__':
    main()