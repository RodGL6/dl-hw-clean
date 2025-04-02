import numpy as np

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs/test_logs")
for i in range(100):
    writer.add_scalar("test_sine", np.sin(i / 10), i)
writer.flush()
writer.close()
