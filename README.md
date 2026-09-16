# ETF 日频回测与 MiniQMT 模拟盘

项目只保留两个运行入口。使用目标电脑原有 Python 环境、MySQL 数据、MiniQMT 账户及配置。

阅读代码可先看 [核心流程阅读指南](docs/核心流程阅读指南.md)，按回测主线、策略接口和模拟盘闭环进入关键函数；需要查找具体定义时使用 [类与函数索引](docs/类与函数索引.md)。业务代码中的类和函数已有中文作用说明，测试代码保持原样。

## 1. 回测

在项目根目录运行：

```powershell
python run_backtest.py private_strategy/beginner_example/experiment.yaml
```

更换实验配置路径即可运行其他 Rule 或 Model。也可以在 IDE 中直接运行 `run_backtest.py`，
使用其中的 `EXPERIMENT_PATH`。可选 `--system` 指定系统配置；默认仍为
`qmt_example/configs/system.yaml`。

实验 YAML 设置日期、资金、证券池和 `case: rule/model`；策略逻辑在同目录 `rule.py` 或
`model.py`。新建策略时复制一个现有示例目录后编辑，不需要单独的创建命令。

Rule 根据前复权行情及只读持仓、份额、汇金、指数数据返回目标权重。Model 支持 Torch 和
XGBoost，一次回测训练一次，再逐日预测并分配组合。两者共用 D+1 收盘执行、交易规则及账户核算。

模型设备在各策略 `model.py` 的 `TorchTrainingConfig` 或 `XGBoostTrainingConfig` 中选择：
`device="cpu"`（默认）、`device="cuda"` 或 `device="cuda:0"`。模拟盘通过 YAML 的
`strategy.model.device` 独立选择推理设备。具体示例见 [Model 设备配置](docs/MODEL_API.md#选择-cpu-或-gpu)。

结果保存在系统配置的 `runs_dir` 中，每次生成新目录，包括净值、持仓、订单、成交、指标、
最终账户及收益/回撤/现金图。Model 另保存 `predictions.csv` 和 `.pt` 或 `.ubj` 模型文件。

## 2. 模拟盘每日自动运行

```powershell
python run_paper_trading.py --config qmt_example/configs/live/xgboost_example_paper.yaml
```

配置使用 `config_version: "4.0"`，一个 `strategy` 对象选择 Rule 或 Model。
`strategy.initial_capital` 是策略初始本金；现金、持仓及未成交订单预留独立记账，
下单只检查策略额度，不查询券商可用现金。旧版 `strategies`、`enabled`、资金池配置已删除。
Model 使用配置指定的固定模型文件；从回测结果复制模型到已有配置指定的位置即可。

进程保持运行后，按配置和交易日历自动完成：

1. 收盘后生成下一交易日使用的信号。
2. 下一交易日按既定时间连接 MiniQMT、恢复对账，先卖后买，核实成交并处理撤单。
3. 日终对账，保存该策略现金、持仓和资产快照。

首次启动必须在所需信号日的信号时间之前保持运行，或数据库中已存在对应待执行目标；
系统不会为了追赶启动时间而补造昨日信号，也不会在停止新单时间之后开启新批次。
Ctrl+C 停止进程；运行中的券商会话和账户锁由原生命周期逻辑释放。再次启动沿用既有记录恢复。

数据库表与现有记录沿用目标电脑现状。程序不提供建表、升级、迁移或清库命令；正常自动交易
仍按原逻辑读写既有 `live_*` 状态表，历史行情表只读。密码、连接参数、调度时间和依赖保持原样。
本程序需保持进程运行；不安装 Windows 定时任务或自动开机服务。

同一账户已有同一策略账本时直接恢复；已有其他策略账本则拒绝启动，避免重新发放本金。
切换策略 ID 前先结束旧策略及未完成订单，再使用独立状态库；详见开发说明中的单策略资金约定。

## 配置与代码位置

| 内容 | 位置 |
|---|---|
| MySQL、费用、滑点、资源与输出目录 | `qmt_example/configs/system.yaml` |
| 模拟盘账户、策略资金、模型路径、调度与风控 | `qmt_example/configs/live/*.yaml` |
| 策略与实验配置 | `private_strategy/<策略>/` |
| 模拟盘组件装配 | `etf_backtest/live/service.py` |
| 自动调度与业务执行 | `etf_backtest/live/engine.py`、`scheduler.py`、`jobs.py` |

Rule 接口见 [RULE_API.md](docs/RULE_API.md)，Model 接口见 [MODEL_API.md](docs/MODEL_API.md)。
维护位置见 [README_DEVELOPER.md](README_DEVELOPER.md)。

本项目采用 [MIT License](LICENSE)。
