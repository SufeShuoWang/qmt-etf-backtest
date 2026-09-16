# XGBoost 与形态策略运行

在项目根目录使用目标电脑原有 Python 环境。项目只有回测和模拟盘自动运行两个入口。

## 回测

XGBoost：

```powershell
python run_backtest.py private_strategy/xgboost_example/experiment.yaml
```

形态策略使用同一个命令，更换实验路径：

```powershell
python run_backtest.py private_strategy/shape_choice/experiment.yaml
```

XGBoost 完成后，将结果目录中的 `model_bundle.ubj` 复制到模拟盘 YAML 已指定的 bundle 路径。
沿用现有模型文件时无需重新训练。回测区间、资金、证券池由实验配置决定，参数在对应策略 Python 文件中。

## 模拟盘自动运行

```powershell
python run_paper_trading.py --config qmt_example/configs/live/xgboost_example_paper.yaml
```

保持进程运行，现有调度自动处理每日信号、下一交易日下单、撤单、回调、对账和快照。
同一账户只启动一个进程。停止使用 Ctrl+C，再次启动沿用原有数据库状态。
新信号按配置时间生成，不会补造昨日信号；错过新单窗口时沿用原有跳过规则。

保留目标电脑现有配置、账户与数据库。单个策略、初始资金和模型路径通过 YAML 的 `strategy` 管理，
修改后停止并重新启动进程生效。
