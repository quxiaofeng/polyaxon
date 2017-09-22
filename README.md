[![Build Status](https://travis-ci.org/polyaxon/polyaxon.svg?branch=master)](https://travis-ci.org/polyaxon/polyaxon)
[![PyPI version](https://badge.fury.io/py/polyaxon.svg)](https://badge.fury.io/py/polyaxon)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/polyaxon/polyaxon)

# Polyaxon

（注：Polyaxon 多轴突神经元，并不罕见但不太为人所知的一种神经元。）

针对端到端的模型和实验设计的，基于 TensorFlow 的深度学习和强化学习库。

# 设计目标

Polyaxon 的设计目标：

 * 模块化：用模块化的、易于理解的模块创建计算图。以利于在稍后的应用中重用。
 
 * 易用：训练模型非常简单，可以用来快速做实验。

 * 配置灵活：可以用 YAML 或 Json 文件来创建模型和设计实验，也可以用 Python 文件。

 * 易扩展：由于完善的模块化和充分的代码文档，用现有模块来构建或者拓展现有模块，都非常容易。

 * 高性能：Polyaxon 低层基于 `tensorflow` 代码库，可以充分利用了其分布式学习特性。

 * 多种数据预处理接口：Polyaxon 提供了多种数据处理接口，支持各种数据输入。


# 快速入门指南

## 基本的线性回归例程

```python
X = np.linspace(-1, 1, 100)
y = 2 * X + np.random.randn(*X.shape) * 0.33

# Test a data set
X_val = np.linspace(1, 1.5, 10)
y_val = 2 * X_val + np.random.randn(*X_val.shape) * 0.33


def graph_fn(mode, inputs):
    return plx.layers.SingleUnit(mode)(inputs['X'])


def model_fn(features, labels, mode):
    model = plx.models.Regressor(
        mode, graph_fn=graph_fn, loss_config=plx.configs.LossConfig(module='mean_squared_error'),
        optimizer_config=plx.configs.OptimizerConfig(module='sgd', learning_rate=0.009),
        eval_metrics_config=[],
        summaries='all', name='regressor')
    return model(features, labels)


estimator = plx.estimators.Estimator(model_fn=model_fn, model_dir="/tmp/polyaxon_logs/linear")

estimator.train(input_fn=numpy_input_fn(
    {'X': X}, y, shuffle=False, num_epochs=10000, batch_size=len(X)))
```


## 强化学习例程

```python
env = plx.envs.GymEnvironment('CartPole-v0')

def graph_fn(mode, inputs):
    return plx.layers.FullyConnected(mode, num_units=512)(inputs['state'])

def model_fn(features, labels, mode):
    model = plx.models.DQNModel(
        mode, 
        graph_fn=graph_fn, 
        loss_config=plx.configs.LossConfig(module='huber_loss'),
        num_states=env.num_states, 
        num_actions=env.num_actions,
        optimizer_config=plx.configs.OptimizerConfig(module='sgd', learning_rate=0.01),
        exploration_config=plx.configs.ExplorationConfig(module='decay'),
        target_update_frequency=10, 
        dueling='mean', 
        summaries='all')
    return model(features, labels)

memory = plx.rl.memories.Memory(
    num_states=env.num_states, num_actions=env.num_actions, continuous=env.is_continuous)
agent = plx.estimators.Agent(
    model_fn=model_fn, memory=memory, model_dir="/tmp/polyaxon_logs/dqn_cartpole")

agent.train(env)
```


## 分类问题例程

```python
X_train, Y_train, X_test, Y_test = load_mnist()

config = {
    'name': 'lenet_mnsit',
    'output_dir': output_dir,
    'eval_every_n_steps': 10,
    'train_steps_per_iteration': 100,
    'run_config': {'save_checkpoints_steps': 100},
    'train_input_data_config': {
        'input_type': plx.configs.InputDataConfig.NUMPY,
        'pipeline_config': {'name': 'train', 'batch_size': 64, 'num_epochs': None,
                            'shuffle': True},
        'x': X_train,
        'y': Y_train
    },
    'eval_input_data_config': {
        'input_type': plx.configs.InputDataConfig.NUMPY,
        'pipeline_config': {'name': 'eval', 'batch_size': 32, 'num_epochs': None,
                            'shuffle': False},
        'x': X_test,
        'y': Y_test
    },
    'estimator_config': {'output_dir': output_dir},
    'model_config': {
        'summaries': 'all',
        'model_type': 'classifier',
        'loss_config': {'name': 'softmax_cross_entropy'},
        'eval_metrics_config': [{'name': 'streaming_accuracy'},
                                {'name': 'streaming_precision'}],
        'optimizer_config': {'name': 'Adam', 'learning_rate': 0.002,
                             'decay_type': 'exponential_decay', 'decay_rate': 0.2},
        'graph_config': {
            'name': 'lenet',
            'definition': [
                (plx.layers.Conv2d, {'num_filter': 32, 'filter_size': 5, 'strides': 1,
                                     'regularizer': 'l2_regularizer'}),
                (plx.layers.MaxPool2d, {'kernel_size': 2}),
                (plx.layers.Conv2d, {'num_filter': 64, 'filter_size': 5,
                                     'regularizer': 'l2_regularizer'}),
                (plx.layers.MaxPool2d, {'kernel_size': 2}),
                (plx.layers.FullyConnected, {'n_units': 1024, 'activation': 'tanh'}),
                (plx.layers.FullyConnected, {'n_units': 10}),
            ]
        }
    }
}
experiment_config = plx.configs.ExperimentConfig.read_configs(config)
xp = plx.experiments.create_experiment(experiment_config)
xp.continuous_train_and_evaluate()
```

## 回归问题例程

```python
X, y = generate_data(np.sin, np.linspace(0, 100, 10000, dtype=np.float32), time_steps=7)

config = {
    'name': 'time_series',
    'output_dir': output_dir,
    'eval_every_n_steps': 5,
    'run_config': {'save_checkpoints_steps': 100},
    'train_input_data_config': {
        'input_type': plx.configs.InputDataConfig.NUMPY,
        'pipeline_config': {'name': 'train', 'batch_size': 64, 'num_epochs': None,
                            'shuffle': False},
        'x': X['train'],
        'y': y['train']
    },
    'eval_input_data_config': {
        'input_type': plx.configs.InputDataConfig.NUMPY,
        'pipeline_config': {'name': 'eval', 'batch_size': 32, 'num_epochs': None,
                            'shuffle': False},
        'x': X['val'],
        'y': y['val']
    },
    'estimator_config': {'output_dir': output_dir},
    'model_config': {
        'model_type': 'regressor',
        'loss_config': {'name': 'mean_squared_error'},
        'eval_metrics_config': [{'name': 'streaming_root_mean_squared_error'},
                                {'name': 'streaming_mean_absolute_error'}],
        'optimizer_config': {'name': 'Adagrad', 'learning_rate': 0.1},
        'graph_config': {
            'name': 'regressor',
            'definition': [
                (plx.layers.LSTM, {'num_units': 7, 'num_layers': 1}),
                # (Slice, {'begin': [0, 6], 'size': [-1, 1]}),
                (plx.layers.FullyConnected, {'n_units': 1}),
            ]
        }
    }
}
experiment_config = plx.configs.ExperimentConfig.read_configs(config)
xp = plx.experiments.create_experiment(experiment_config)
xp.continuous_train_and_evaluate()
```

## 分布式实验例程

```python
def create_experiment(task_type, task_index=0):

    def graph_fn(mode, inputs):
        x = plx.layers.FullyConnected(mode, num_units=32, activation='tanh')(inputs['X'])
        return plx.layers.FullyConnected(mode, num_units=1, activation='sigmoid')(x)

    def model_fn(features, labels, mode):
        model = plx.models.Regressor(
            mode, graph_fn=graph_fn, loss_config=plx.configs.LossConfig(module='absolute_difference'),
            optimizer_config=plx.configs.OptimizerConfig(module='sgd', learning_rate=0.5, decay_type='exponential_decay', decay_steps=10),
            summaries='all', name='xor')
        return model(features, labels)

    os.environ['task_type'] = task_type
    os.environ['task_index'] = str(task_index)

    cluster_config = {
            'master': ['127.0.0.1:9000'],
            'ps': ['127.0.0.1:9001'],
            'worker': ['127.0.0.1:9002'],
            'environment': 'cloud'
        }

    config = plx.configs.RunConfig(cluster_config=cluster_config)

    estimator = plx.estimators.Estimator(model_fn=model_fn, model_dir="/tmp/polyaxon_logs/xor", config=config)

    return plx.experiments.Experiment(estimator, input_fn, input_fn)
```

# 安装

最新版 Polyaxon 的安装方法：`pip install polyaxon`

也可以下载源代码后，在源码目录下执行：`python setup.py install`

还可以直接克隆此代码库（git repo） `git clone https://github.com/polyaxon/polyaxon.git`，然后在 docker 里面执行：
 
 * `cmd/rebuild` 创建 docker 容器。
 * `cmd/py` 启动带有所有依赖的 python3 命令行。
 * `cmd/jupyter` 启动 jupyter notebook 服务。
 * `cmd/tensorboard` 启动 tensorboard 服务。
 * `cmd/test` 运行测试。 

# 代码范例

这里有一些例程：[例程](examples)，稍后也会继续上传更多的例程，同时也欢迎大家贡献例程。

# 项目进展

Polyaxon 还处于没有正式发布的“内部测试（alpha）”状态。所有的接口、开发接口、数据结构都有可能剧烈变动，请做好准备。
我们会尽可能就具有潜在颠覆性的修改进行沟通。

# 参与开发

如果想要参与开发，为 Polyxon 贡献代码，请遵照以下指导文件： *[为 Polyaxon贡献力量](CONTRIBUTING.md)*.

# 协议

MIT 协议

# 致谢

本项目受很多不同的项目启发，包括 `tensorflow.contrib.learn`，`keras`，`sonnet`，`seq2seq` 以及许多其他杰出的开源项目。具体可参考 [致谢](ACKNOWLEDGEMENTS).

这个库的目的就是给工程师和研究人员提供一个能够开发和实验端到端的深度学习解决方法的工具。

为了能够全面的控制 API 设计以及其它一些决断，才选择设计这么一个新的库。
