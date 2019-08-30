初始化在训练入口文件的开头部分加入下面的代码。

import horovod.tensorflow as hvd
hvd.init()


绑定 GPU 分布式训练默认采用单 GPU 单进程模式，所以需要在 config 中设置可见的 GPU 设备，这里可以通过 local rank 获取到。


config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())



根据 GPU 数（或进程数）设置学习率，GPU 数可以通过 hvd.size() 获取到。

# Build model...
loss = ...
opt = tf.train.AdagradOptimizer(0.01 * hvd.size())



对 optimizer 进行包装，实现 gradient 的 allgather 或 allreduce 操作。

opt = hvd.DistributedOptimizer(opt)





对变量进行初始化。
使用 MonitoredTrainingSession 的情况下：

hooks = [hvd.BroadcastGlobalVariablesHook(0)]

with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                      hooks=hooks) as mon_sess:

									  
不使用 MonitoredTrainingSession 的情况下：

sess = tf.Session()
sess.run(tf.global_variables_initializer())
hvd.broadcast_global_variables(0)



修改代码只在一个 worker 上保存 checkpoints


checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
