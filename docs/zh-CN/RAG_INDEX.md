# RAG 索引

本文档帮助检索系统把问题路由到正确的 GIGNN 文档分块。

## 检索策略

宽泛问题用 `llms.txt` 作为首个检索命中。长上下文回答或代理引导用 `llms-full.txt`。精确回答用下面的专题文档。

按标题分块。保留文件路径、标题与邻近表格作为元数据。

建议元数据字段：

| 字段 | 示例 |
| --- | --- |
| `project` | `GIGNN` |
| `doc_path` | `docs/API.md` |
| `heading` | `GIGNNAutoClassifier` |
| `audience` | `user`、`developer`、`agent`、`release`、`security` |
| `topic` | `api`、`graph`、`geodetector`、`validation`、`plotting`、`reports` |
| `stability` | `public`、`advanced`、`internal`、`operational` |

## 查询路由表

| 查询意图 | 最佳文档 | 关键词 |
| --- | --- | --- |
| GIGNN 是什么？ | `llms.txt`、`README.md`、`docs/FEATURE_OVERVIEW.md` | overview、purpose、capabilities、GIS、RS、GNN |
| GIGNN 完整实现了吗？ | `docs/IMPLEMENTATION_STATUS.md`、`docs/API.md`、`tests/test_model_training.py` | implementation status、notebook core、paper reproduction、NaturalBreaks、edge_attr |
| 端到端怎么用？ | `docs/USER_GUIDE.md`、`examples/quickstart.py`、`README.md` | quickstart、workflow、fit、predict、simulate |
| 有哪些 API？ | `docs/API.md`、`llms-full.txt` | API、signature、method、return、exception |
| `GIGNNAutoClassifier` 做什么？ | `docs/API.md`、`docs/FEATURE_OVERVIEW.md`、`docs/REPORTS.md` | auto、grid、selector、graph_report、model_selection |
| `GIGNNClassifier` 做什么？ | `docs/API.md`、`docs/FEATURE_OVERVIEW.md`、`docs/SECURITY.md` | classifier、predict_proba、score、save、load |
| 什么输入合法？ | `docs/DATA_CONTRACT.md`、`docs/API.md` | X、y、geometry、edge_index、labels、finite、numeric |
| 图怎么构建？ | `docs/DATA_CONTRACT.md`、`docs/FEATURE_OVERVIEW.md`、`docs/API.md` | knn、distance、contiguity、edge_index、adjacency |
| 地理探测器在这里怎么工作？ | `docs/FEATURE_OVERVIEW.md`、`docs/API.md` | q statistic、discretization、interaction、selector |
| 产出哪些报告？ | `docs/REPORTS.md`、`docs/API.md` | discovery_report、graph_report、SimulationReport |
| 结果怎么绘图？ | `docs/PLOTTING.md`、`docs/API.md` | plot、Figure、Axes、output_path、map、histogram |
| 怎么安装或测试？ | `docs/COMMANDS.md`、`docs/CONTRIBUTING.md`、`README.md` | install、pytest、build、black、quickstart |
| 怎么发布？ | `docs/RELEASE.md`、`docs/RUNBOOK.md`、`docs/COMMANDS.md` | release、wheel、sdist、PyPI、artifact |
| 运维上什么会失败？ | `docs/RUNBOOK.md`、`docs/ENV.md`、`docs/SECURITY.md` | troubleshooting、environment、PyTorch、checkpoint |
| 加载检查点安全吗？ | `docs/SECURITY.md`、`docs/DATA_CONTRACT.md`、`docs/API.md` | checkpoint、weights_only、file path、trusted |
| 代理应该怎么改本仓库？ | `docs/AGENTS.md`、`llms.txt`、`.spec/docs/*.md` | agent、task routing、validation、docs sync |
| 参考论文与 notebook 证据在哪？ | `research/README.md`、`docs/IMPLEMENTATION_STATUS.md` | research evidence、notebook、paper、implementation status |

## 文档卡片

### `llms.txt`

受众：LLM、编码代理、爬虫。

用于：

- 短上下文注入。
- 初始模型锚定。
- 快速路由到权威文档。

避免用于：

- 精确报告字段。
- 详细示例。
- 长解释。

### `llms-full.txt`

受众：长上下文 LLM、代理引导、RAG 全上下文模式。

用于：

- 完整项目摘要。
- 能力与 API 总览。
- 代理任务路由。
- 安全与验证提醒。

有专题文档可用时，避免作为字段级精确细节的唯一来源。

### `README.md`

受众：人类用户与包评估者。

用于：

- 项目定位。
- 安装路径。
- 快速上手。
- 文档导航。

### `docs/IMPLEMENTATION_STATUS.md`

受众：用户、维护者、代理。

用于：

- 回答 notebook GIGNN 核心是否已实现。
- 区分 notebook 核心实现与完整论文数值复现。
- 查找把 notebook 要求链接到源码与测试的审计矩阵。

### `docs/FEATURE_OVERVIEW.md`

受众：用户、维护者、代理。

用于：

- 能力层级。
- 理解数据、地理探测器、图、模型、报告、绘图与持久化如何组合。

最佳分块标题：

- `Data Preparation`
- `GeoDetector Feature Discovery`
- `Graph Construction`
- `GNN Model and Training`
- `Estimators`
- `Reports and Diagnostics`
- `Plotting`
- `Persistence and Safety Limits`

### `docs/USER_GUIDE.md`

受众：库用户。

用于：

- 端到端工作流。
- 如何准备输入。
- 自动估计器与直接估计器的使用。
- 模拟、诊断、绘图、保存/加载。

### `docs/API.md`

受众：用户、开发者、代理。

用于：

- 公开 API 名称。
- 方法签名。
- 返回值。
- 估计器学习属性。
- 异常。
- 高级模块。

顶层、`gignn.geodetector`、`gignn.graph` 设 `stability=public`；`validation`、`training`、`models`、`io` 设 `stability=advanced`。

### `docs/DATA_CONTRACT.md`

受众：用户、开发者、代理。

用于：

- 特征表规则。
- 标签规则。
- 几何要求。
- 边索引语义。
- 预测图行为。
- 资源上限。

校验相关问题优先使用本文档。

### `docs/ARCHITECTURE.md`

受众：维护者与编码代理。

用于：

- 包布局。
- 模块边界。
- 内部数据流。
- 依赖边界。
- 持久化边界。

### `docs/REPORTS.md`

受众：用户、报告消费者、notebook 作者。

用于：

- 报告 dataclass 字段。
- `discovery_report_`、`graph_report_`、`model_selection_report_`、`generalization_report_`。
- `SimulationReport` 字段。
- 数据集与诊断报告字段。

### `docs/PLOTTING.md`

受众：notebook 用户与可视化工作流。

用于：

- 必需绘图列。
- 返回对象。
- `output_path` 行为。
- 图保存行为。

### `docs/EXAMPLES.md`

受众：新用户与冒烟测试代理。

用于：

- 内置 quickstart 行为。
- 预期输出形态。
- 未来示例类别。

### `docs/COMMANDS.md`

受众：开发者、发布代理、CI 作者。

用于：

- 安装命令。
- 测试命令。
- 构建命令。
- 干净 wheel 验证。
- 包元数据事实。

### `docs/CONTRIBUTING.md`

受众：贡献者与编码代理。

用于：

- 开发工作流。
- 测试地图。
- 文档维护矩阵。
- PR 检查清单。

### `docs/RELEASE.md`

受众：维护者与发布代理。

用于：

- 手动发布检查清单。
- 版本来源。
- 构建产物。
- 产物检查。
- 发布策略。

### `docs/RUNBOOK.md`

受众：维护者与排障代理。

用于：

- 运行模型。
- 标准验证。
- 干净 wheel 冒烟。
- 常见问题。
- 恢复与升级。

### `docs/SECURITY.md`

受众：安全审查者、服务包装作者、代理。

用于：

- 信任边界。
- 检查点加载安全。
- 文件路径指引。
- 不可信数据指引。
- 运行时机密。

### `docs/AGENTS.md`

受众：编码代理。

用于：

- 任务专属阅读顺序。
- 文件所有权。
- 文档同步规则。
- 验证命令。
- 安全须知。

## 分块建议

- 宽检索按 `##` 标题切分 Markdown。
- `docs/API.md`、`docs/FEATURE_OVERVIEW.md`、`docs/REPORTS.md`、`docs/PLOTTING.md` 按 `###` 标题切分。
- 表格与前导标题保持在一起。
- 代码块与紧邻的前导解释段落保持在一起。
- 以 `doc_path` 与标题路径为元数据存储。
- API 文档用更小分块，总览文档用更大分块。

## 反幻觉规则

检索回答不应声称：

- GIGNN 有 HTTP 端点。
- GIGNN 需要环境变量。
- GIGNN 自动发布到 PyPI。
- GIGNN 下载或预处理原始遥感影像。
- GIGNN 检查点加载在服务上下文中对任意不可信路径安全。
- `natural_breaks`/`fisher_jenks` 回退到分位数分箱，或 `maximum_breaks` 与 Fisher-Jenks 相同（它是独立的最宽间隙法）。
- 包完整复现每个论文实验、外部数据集、基线或 notebook 文件系统产物。

不确定时，除文档外再检索源代码或测试。
