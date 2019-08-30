
初始化在训练入口文件的开头部分加入下面的代码
import horovod.torch as hvd
hvd.init()


绑定 GPU 分布式训练默认采用单 GPU 单进程模式，所以需要设置可见的 GPU 设备，这里可以通过 local rank 获取到。
torch.cuda.set_device(hvd.local_rank())



根据 GPU 数切分数据，GPU 数可以通过 hvd.size() 获取到。
train_sampler = torch.utils.data.distributed.DistributedSampler(
  train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)




对 optimizer 进行包装，实现 gradient 的 allgather 或 allreduce 操作。
optimizer = optim.SGD(model.parameters())

optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())



对变量进行初始化。
hvd.broadcast_parameters(model.state_dict(), root_rank=0)




修改代码只在一个 worker 上输出日志。
if hvd.rank() == 0:
  print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
    test_loss, 100. * test_accuracy))
