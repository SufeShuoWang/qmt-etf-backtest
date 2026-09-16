# 项目维护说明

## 先打开哪个文件

日常修改从你实际使用的 `private_strategy/<策略目录>/` 开始。回测命令的实验路径、
模拟盘配置中的 `experiment_path` 决定使用哪个目录。

| 想修改什么 | 先打开哪里 | 修改位置 |
|---|---|---|
| 回测起止日期、初始资金、证券池、规则/模型模式 | 策略目录的 `experiment.yaml` | `start_date`、`end_date`、`initial_cash`、`universe`、`case` |
| 规则策略的买卖条件 | 策略目录的 `rule.py` | `Strategy.generate_weights()` |
| 规则策略回看长度、调仓周期、目标仓位、常量 | 同一个 `rule.py` | `RuleSettings` |
| 模型特征、特征顺序、历史长度 | 策略目录的 `model.py` | `Features.build_features()`、`feature_names`、`required_history_trading_days` |
| 模型结构和参数 | 同一个 `model.py` | `Model`；Torch 的网络定义在 `create()` |
| 训练区间、训练参数、预测后的组合分配 | 同一个 `model.py` | `ModelSettings` 及其 training、portfolio 设置 |
| 模拟盘单个策略、虚拟资金、模型路径、运行时刻 | 实际传给模拟盘命令的 YAML | `strategy`、`signal`、`execution`、`eod` |

改特征、模型结构或与模型绑定的组合参数后，先通过回测生成相匹配的模型文件，
再让模拟盘配置指向它；模拟盘不会自动训练模型。具体命令见 [README.md](README.md)。

## 配置怎样传到运行代码

回测和模拟盘信号计算共用 `application/strategy_source.py → build_backtest_config()`：
先从 `rule.py` 或 `model.py` 转换策略参数，再调用
`experiments/config.py → UserExperimentConfig.build_case()` 合并实验与系统设置。
两个运行入口不再各写一遍合并过程，回测准备阶段也不重复构建策略参数。

| 参数类别 | 修改来源 | 最终使用位置 |
|---|---|---|
| 回测日期、证券池、初始现金 | 策略目录的 `experiment.yaml` | 运行配置 → `experiment.py` / `core/engine.py` |
| 规则回看长度、调仓周期、目标仓位 | 策略目录的 `rule.py → RuleSettings` | 运行配置和用户规则 |
| 模型训练/验证区间、训练参数、组合分配 | 策略目录的 `model.py → ModelSettings` | `LoadedModelComponents.create_workflow()` 与模型组合分配 |
| 费用、回测滑点、成交量参与比例、结果目录 | 实际使用的系统 YAML | 运行配置 → 费用、成交和输出模块 |
| 模拟盘策略资金、调度起点 | 模拟盘 YAML 的 `strategy` | `StrategySpec` 与虚拟策略账户、信号调度 |
| 模拟盘每日运行时刻、撤单时限、报价设置 | 模拟盘 YAML 的 `signal`、`execution`、`eod` | `LiveConfig` 直接传给调度器与交易任务 |
| 模拟盘订单金额、总仓位等风控限制 | 模拟盘 YAML 的 `risk` | `jobs.py` → `LiveRiskManager` |
| 模拟盘固定模型后端、模型文件 | 模拟盘 YAML 的 `strategy.model` | `LoadedModelComponents.load_inference_bundle()` |

回测的系统文件由 `--system` 决定，省略时沿用入口原有默认路径；
模拟盘通过 `account.system_path` 选择系统文件。
模拟盘启动时，费用与单个策略共用一次读取并校验的系统设置，没有新增跨运行缓存；
状态数据库的配置解析和连接方式仍沿用原实现。

容易混淆的参数要分别修改：

- `experiment.yaml.initial_cash` 是回测初始资金；模拟盘使用 `strategy.initial_capital`。
- 模型组合参数决定生成的目标仓位；模拟盘 `risk` 决定该目标能否通过交易风控，两者都生效。
- 模型训练/验证区间来自 `ModelSettings`；测试区间沿用实验的回测起止日期。
- 模拟盘 YAML 选择已有模型文件，不会因为修改部署路径而重新训练模型。

配置模型仍负责拒绝未知字段、非法金额、日期冲突及不合法时间顺序。
增加共用运行参数时，修改其所属配置模型及 `UserExperimentConfig.build_case()` 的映射；
只有新增策略专属参数时才需要调整 `build_backtest_config()` 中的对应分支。
模拟盘 YAML 使用 `config_version: "4.0"` 和单个 `strategy` 对象；旧版策略列表不再接受。

### 单策略资金与已有账本

模拟盘只配置 `strategy.initial_capital` 作为策略初始本金，不配置 `account.capital_pool`
或 `enabled`。策略可用现金等于账本现金减去未成交买单及其手续费预留；下单不查询券商
账户可用现金。成交、手续费和持仓仍写入策略账本，重启不会重新发放初始本金。

保留 `strategy_id` 用于历史订单和持仓归属。现有状态库只有同一策略账本时可继续使用；
若该账户已有其他策略账本，启动会报错，不会自动删除、合并或重置资金。
切换策略 ID 时须先结束旧策略及其未完成订单，再使用独立的 `state_database`。
数据库表结构和旧状态枚举保留用于历史数据兼容；`live_account.capital_pool` 仅写入初始本金，
不再参与资金分配或下单校验。没有执行数据库迁移。

原 Rule+Model 联合配置已删除。运行规则策略使用 `beginner_example_paper.yaml`；
运行模型策略使用 `xgboost_example_paper.yaml`，同一账户同时只启动一个进程。

## 策略数据从哪里来、在哪里改

回测和模拟盘都通过 `application/daily_decision.py → evaluate_daily_decision()` 进入策略。
只有到达策略调度日才准备上下文和历史行情：

1. `data/portal.py` 按信号日筛选已经加载的数据；各历史查询共用 `_checked_cutoff()` 检查日期覆盖。
2. `strategy/context.py → StrategyContext.from_portal()` 组装份额、汇金和指数数据，
   构造器继续校验日期、证券范围与数值，生成只读上下文。
3. `strategy/rule.py → RuleMarketData` 引用这个上下文，并提供 `bars()`、
   `share_on()`、`latest_huijin_ratio()` 等查询；不重新保存一套账户和辅助数据。

| 要改的数据 | 实际修改位置 |
|---|---|
| 数据源的读取与记录转换 | `data/mysql.py`；仅新增字段且现有加载内容不足时需要调整 |
| 前复权行情、份额、汇金、指数的日期筛选 | `data/portal.py` 对应查询方法 |
| 辅助字段的声明、组装和合法性检查 | `strategy/context.py` 的字段、`from_portal()` 和对应 `_freeze_...` |
| 用户规则需要的便捷查询 | `strategy/rule.py` 的 `RuleMarketData`；已有字段直接用现有方法 |
| 回测账户转换为策略现金与持仓 | `strategy/context.py → AccountView.from_account()` |
| 模拟盘虚拟账户的估值与持仓转换 | `live/account_adapter.py → adapt_virtual_account()`；账户视图和权重共用 `_account_state()` |

新增辅助字段时，不需要在回测主流程和模拟盘主流程各加一套传递逻辑。
原有直接构造 `StrategyContext`、`RuleMarketData` 的方式继续保留。

日期口径不能混用：ETF 行情按最近 N 个交易日筛选；指数按截至 D 的最近 N 条已有记录筛选。
份额和指数可包含 D 日，汇金报告期必须严格早于 D；
合并汇金比例采用最新同一报告期，不能直接把各主体不同报告期的最新值相加。

虚拟账户直接用自己的现金、持仓和原始收盘价计算净资产与权重，不再构造中间券商资产对象。
策略账户仍只暴露现金和数量，前复权行情与原始估值价格保持分离。

## 回测按什么顺序运行

从项目根目录阅读以下路径：

1. `run_backtest.py → main()`：接收实验路径和系统配置路径。
2. `etf_backtest/experiment.py → run_experiment()`：调用 `prepare_experiment()`，
   读取用户策略和配置，再通过 `build_backtest_config()` 合并运行参数；加载与参数组装逻辑位于 `application/strategy_source.py`。
3. `application/runtime_factory.py → build_backtest_runtime()`：确定证券范围，
   读取行情和日历，准备 `DailyDataPortal` 与交易规则。
4. 模型模式调用 `experiment.py → _build_model()`，构建样本，通过
   `strategy/model.py → LoadedModelComponents.create_workflow()` 选择后端并训练一次；
   规则模式直接使用用户规则。
5. `experiment.py → _run_backtest()`：创建账户、费用和成交组件，调用
   `core/engine.py → BacktestEngine.run()`，最后检查净值日期是否完整。
   逐日引擎通过 `application/daily_decision.py → evaluate_daily_decision()` 产生新目标。
6. `experiment.py → _write_results()`：统一准备指标、来源信息和图表，
   通过 `output/writer.py` 写出结果；规则和模型共用一次结果写出；模型临时文件一直保留到结果写出结束。

只看运行顺序时，从 `experiment.py` 顶部的 `run_experiment()` 开始；
账户初始化与回测组件在同文件的 `_run_backtest()`，报告准备在 `_write_results()`。
实际成交、持仓变化和逐日信号仍在 `core/engine.py`，不放入入口函数。

策略读取复权价，成交和估值使用原始价。最后一个行情日不再产生无法执行的新目标。

## 模拟盘按什么顺序运行

1. `run_paper_trading.py → main()` 加载模拟盘配置。
2. `live/service.py → build_production_runtime()` 组装用户策略、固定模型、数据入口、
   券商、回调处理和调度器。已有模型通过
   `strategy/model.py → LoadedModelComponents.load_inference_bundle()` 加载，不触发训练。
3. `live/engine.py → run_forever()` 维持账户锁，循环调用
   `live/scheduler.py → tick()`。其中直接列出调仓、日终、生成信号的时间表；
   持久化记录用于去重。常驻循环只在交易引擎中，调度器不另起循环。
   调仓与日终必须经过交易引擎的券商会话，不能由调度器直接调用交易任务。
4. 信号任务进入 `live/jobs.py → prepare_signal()`，由
   `live/signals.py → SignalService.evaluate()` 准备历史数据和账户视图，
   调用共用的 `evaluate_daily_decision()`，然后保存决策及目标仓位。
5. 调仓和日终任务先经过 `live/engine.py → _run_broker_job()`：
   建立券商会话，执行 `startup_reconcile()`，然后进入具体任务，结束后断开会话。

具体交易顺序集中在 `live/jobs.py`：

| 任务 | 按顺序阅读的函数 |
|---|---|
| 调仓 | `execute_pending_target()` → `_execute_pending_target()` → `_execute_phase()`：卖出 → 等待/对账 → 买入 → 等待/对账 |
| 生成与提交订单 | `_execute_strategy_side()` → 执行规划、价格策略和风控 → `_submit_intent()` |
| 成交等待与撤单确认 | `_wait_for_phase()` / `_cancel_open_orders()` → `_wait_for_order_closure()`：查询 → 对账 → 检查本地活动订单 |
| 日终 | `eod()` → `_reconcile_eod()` → `_snapshot_eod()` → `_snapshot_strategy()` |
| 券商事实入账 | `live/broker/callbacks.py` 和 `live/reconciliation.py` → `repository.record_strategy_trade_if_absent()` |

`_run_job()` 管理任务锁、去重和结束状态；`_run_strategy_step()` 管理单个策略步骤。
`repository.finish_job_run()` 无额外参数表示成功，`error=异常` 表示失败，
`skip_reason=原因` 表示跳过。账户安全异常会中止整个批次，普通策略异常按原流程记录处理。
`_execute_phase()` 共用卖出/买入的任务记录与等待处理，卖出完成后额外对账；买入前重新取时间与报价。

## 一笔模拟盘订单怎样结束

1. `jobs.py → _execute_strategy_side()` 生成目标订单、检查风控并保存意图；
   只有允许提交且仍为 `PLANNED` 的意图进入 `_submit_intent()`。
2. `_submit_intent()` 先标记 `SUBMITTING`，再调用券商。
   接受后绑定券商订单号，明确拒绝则记录 `REJECTED`；
   提交异常、结果不明或绑定失败，通过 `_mark_submission_unknown()` 记录不确定状态并暂停账户。
3. `broker/callbacks.py → BrokerEventConsumer` 处理单条回报，核验本地订单身份后保存订单或成交。
   回报处理与主动查询共用现有的成交去重入账方法，不各记一套账。
4. `reconciliation.py → ReconciliationService.reconcile()` 用主动查询的订单与成交核对数量、身份和状态，
   对已确认终态的意图标记 `COMPLETED` 或 `INCOMPLETE`。
5. `jobs.py → _wait_for_order_closure()` 共用轮询流程：查询订单与成交 → 对账 → 检查本地活动订单。
   没有活动订单时结束等待，并不表示每笔订单都已全部成交。
   成交等待到期后调用撤单；撤单确认到期仍有活动订单时暂停账户并报错。

订单数量、限价和备注标识的匹配集中在 `reconciliation.py → order_terms_match()`。
证券和买卖方向仍在各自的身份检查入口核验：
回报遇到非法方向会抛出异常，对账将方向不匹配归入对账报告；原有差异保留。
订单、成交的归属查找及事务边界保持原样。

| 订单停在哪一步 | 先读哪个函数 | 对照的现有状态或原因 |
|---|---|---|
| 生成后没有提交 | `jobs.py → _execute_strategy_side()` | 意图是否为 `PLANNED`，风控或时间窗的拒绝原因 |
| 提交结果不明、接受后本地绑定失败 | `jobs.py → _submit_intent()`、`_mark_submission_unknown()` | `SUBMIT_UNKNOWN`、`SUBMIT_RESULT_UNKNOWN` |
| 部分成交后一直等待 | `jobs.py → _wait_for_order_closure()`，再看 `reconciliation.py` | 活动订单、成交数量差异、`RECONCILIATION_UNRESOLVED` |
| 撤单未接受或确认超时 | `jobs.py → _cancel_open_orders()` | `CANCEL_RESULT_UNKNOWN`、`CANCEL_CONFIRM_TIMEOUT` |
| 回报身份异常或无法入账 | `broker/callbacks.py → _persist_order()` / `_persist_trade()` | `LOCAL_CALLBACK_*_IDENTITY_MISMATCH`、`BROKER_CALLBACK_PERSISTENCE_ERROR` |
| 重启后不继续下单 | `jobs.py → _resume_persisted_intents()` 和启动对账 | `SUBMITTING` / `SUBMIT_UNKNOWN` 不会被直接重新提交；`PLANNED` 仍受提交窗口限制 |

上述表格用于定位代码和已有记录，不需要手工修改数据库状态。

## 出问题先看哪里

回测和模拟盘入口都会把启动错误写到终端。模拟盘日志包含时间、级别、模块名；
当前入口没有配置固定日志文件，先查看启动终端或目标电脑已有的输出重定向位置。

| 现象 | 先看什么 | 对应代码 |
|---|---|---|
| 配置/策略加载失败，尚未开始回测 | 终端异常类型和消息；此时可能没有输出目录 | `application/strategy_source.py`、`experiments/config.py`、`strategy/loader.py`、`strategy/model.py` |
| 回测运行失败 | 若已生成，本次输出目录的 `run.json` 中 `error_type`、`error_message` | `experiment.py`；再按报错转到对应模块 |
| 行情缺失、日期不齐、停牌或价格上下限错误 | 终端或 `run.json` 中指出的证券、日期和字段 | `data/mysql.py` → `data/portal.py`；规则边界看 `core/effective_rules.py` |
| 模型无法加载 | 终端的模型兼容性异常，按字段核对用户模型与模型文件 | `strategy/model_contracts.py` 及对应训练后端模块 |
| 回测没有交易或结果异常 | `orders.csv`、`trades.csv`、`daily_positions.csv`、`daily_nav.csv` | `application/daily_decision.py`、用户策略、`core/engine.py` |
| 模拟盘到时间没有运行 | 终端调度日志；已有 `live_job_run` 的任务日期、status、error_type、error_message | `live/scheduler.py → tick()`、`live/jobs.py → _run_job()` |
| 模拟盘有信号但没有订单 | 已有 `live_decision` 的决策状态、`live_target_position` 的目标；未到调仓日或不调仓也会保存决策 | `live/signals.py`、`live/jobs.py → _execute_pending_target()` |
| 订单被拒绝或提交结果不明确 | 终端券商日志、已有 `live_order_intent` 的 status 和 reject_reason | `live/execution/`、`live/risk.py`、`live/broker/` |
| 账户暂停或现金、持仓不一致 | 已有 `live_account.pause_reason`，结合券商订单、成交和虚拟策略账本记录 | `live/reconciliation.py`、`live/persistence/repository.py` |

回测输出根目录由系统配置的 `runs_dir` 决定，默认是 `runs/`；每次运行有独立 run_id。
模拟盘排查时先确定 account_id、strategy_id、任务日期，再沿 decision_id、intent_id 查记录。
以上状态表只用于说明已有记录的位置，不需要新增、删除或手工修改数据库记录。

## 修改框架功能的定位表

| 功能 | 文件 |
|---|---|
| 证券池、上市生命周期 | `etf_backtest/universe/resolver.py` |
| 特征、训练样本、标签、评估 | `etf_backtest/strategy/model_data.py` |
| 回测账户初始化、运行顺序、报告准备 | `etf_backtest/experiment.py` 的 `_run_backtest()`、`run_experiment()`、`_write_results()` |
| 模型训练后端选择、固定模型加载 | `etf_backtest/strategy/model.py` 的 `LoadedModelComponents` |
| 每天三项任务的调度顺序 | `etf_backtest/live/scheduler.py` 的 `tick()`；执行时间仍在模拟盘 YAML |
| 券商连接、账户锁、常驻循环 | `etf_backtest/live/engine.py` |
| 模型说明信息、保存格式与公共检查 | `etf_backtest/strategy/model_contracts.py` |
| Torch / XGBoost 训练与模型文件 | `etf_backtest/strategy/model_training.py` / `xgboost_training.py` |
| 预测、组合分配、策略可见数据 | `etf_backtest/strategy/model_runtime.py`、`portfolio.py`、`context.py`、`rule.py` |
| 目标股数、交易限制、费用、滑点 | `etf_backtest/core/sizing.py`、`etf_rules.py`、`fee.py`、`fill.py` |
| 回测成交与账户记账 | `etf_backtest/core/fill.py`、`account.py`、`position.py` |
| 模拟盘虚拟账本、任务和订单状态 | `etf_backtest/live/persistence/repository.py` |
| 回测指标、图表、文件输出 | `etf_backtest/evaluation/`、`etf_backtest/output/writer.py` |

## 修改时要保留的约定

- D 日生成目标，最早下一合法交易日执行；模拟盘遵循自己的配置时间窗。
- Rule 返回空映射或 NO_REBALANCE 表示保持持仓，显式零表示清仓，省略证券表示保持数量。
- 模型回测训练一次，模拟盘使用固定模型；已有 Torch/XGBoost 文件格式保持兼容。
- 单策略独立记账、先卖后买、成交去重、撤单确认、锁和恢复是业务规则。
- 环境、数据库、账号密码及现有表结构不因代码整理而修改。
- 文件哈希集中在 `file_utils.py`；模型身份、资源身份、订单标识和任务锁仍沿用原有含义。

历史代码保存在项目相邻的 `main_backtest_baselines/` 压缩备份中，不依赖 Git。
其中的旧代码副本和核对材料不参与运行。测试目录未随内部接口精简同步，
部分旧测试仍引用已删除接口；不能据此判断目标环境当前是否可运行。
已完成的是本地静态核对与离线比对，精简后的代码没有重新进行目标环境端到端验收。


## 本轮完整精简流程与停止标准（2026-09-09）

本轮基线：核心包 79 个 Python 文件、18,655 行（含注释和空行）。
按以下顺序连续完成，不以必须删除多少行为目标：

1. 保存本轮源码基线，核对两个入口、用户策略接口和配置边界。
2. 检查内部无调用代码、纯转发包装及重复转换；保留框架自动调用的校验和公开策略接口。
3. 检查模型特征、训练、预测、保存和加载；合并确实相同的处理，保留两种后端差异。
4. 整理回测报告组装和输出字段，保持文件名、列顺序、数值格式及失败处理。
5. 复核配置到信号、信号到任务的调用链；保留交易保护，不修改数据库及账本逻辑。
6. 更新本说明，完成引用、语法和针对改动的离线对照，记录实际变化及验证限制。

预期效果：两个运行入口，明确的策略修改位置，集中维护的公共逻辑，
可沿调用顺序定位问题。没有新命令、新依赖或新增生产代码文件。
数据库、环境、账号密码、现有 YAML 和默认值不因本轮整理而改变。

停止标准：本轮检查范围内能够确认的冗余已处理；剩余代码用于实际功能、
用户接口兼容或必要保护。到此结束整体精简，后续围绕具体需求修改，
不再为了行数继续抽象、合并模块或删除保护。


### 执行结果：六步已完成

- 基线已保存到原有压缩备份的 `before_final_cleanup/`，没有增加生产代码文件。
- 内部引用检查未发现可直接删除的私有死代码；Pydantic 自动调用的校验全部保留。
- Torch 的两个加载入口共用 `model_training.py → _load_payload()`；
  训练、推理各自的格式与兼容性要求仍在各入口核验。XGBoost 保留原有实现。
- `experiment.py → _write_results()` 统一规则和模型的成功输出，
  用标准库 `ExitStack` 管理模型临时目录；数据身份只构造一次。
- `output/writer.py` 的 `_DAILY_FIELDS`、`_ORDER_FIELDS`、
  `_APPROVAL_FIELDS`、`_TRADE_FIELDS` 同时用于字段取值和 CSV 表头，
  减少字段漏改。需要计算的持仓列、指标公式和最终账户组装继续显式保留。
- 两个命令、共用信号入口和模拟盘调度链已复核；本轮未修改模拟盘与数据库模块。

核心代码由 18,655 行变为 18,625 行，净减少 30 行；仍为 79 个 Python 文件。
实际修改 3 个生产代码文件及本说明。行数包含空行和注释，不含本说明与离线核对脚本。

针对本轮改动的 61 组离线对照通过：13 组输出文件/失败处理对照，
22 组报告组装/异常/临时目录清理对照，26 组 Torch 加载入口对照。
输出对照使用实际输出代码及合成数据；Torch 加载对照替代了底层文件读取，
没有运行实际模型训练。全部核心文件语法通过，
其余生产源码、入口、用户策略及已备份配置与本轮基线一致。
未连接数据库、未连接券商、未重新运行目标电脑端到端验收。

可执行核对脚本与基线放在同一 ZIP 的 `before_final_cleanup/check_final.py`；
如需重跑，将它解压到临时位置，从项目根目录用现有 Python 执行即可。
历史测试目录按本轮要求未整理，仍存在前文说明的旧接口引用。

本轮结论：已达到本说明的停止标准。剩余代码主要承担行情与交易规则、
两种模型后端、账户/订单生命周期和必要校验。本轮未发现更多能够明确保持行为、
同时让代码更易读的删减；这不代表数学意义上的最少代码。结束整体精简，后续按具体需求维护。
