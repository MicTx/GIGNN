# GIGNN Agent Notes

## Spec Source
- 当前 `$spec` 真源是 `.spec/specs/`。旧 `.trae` 任务包已迁移到 `.spec/specs/archive/` 的标准归档包，原始文件保存在各包 `legacy/` 中；不要再把 `.trae` 当作历史入口或活跃任务来源。
- 新建任务包后，后续阶段应视为 `spec:update/status/check/run`，不能因为模板未补齐又重复 `spec:new`。
- 仓库双态：私有 Gitea 为真源态，`patent/` 与 `.spec/` 正常跟踪，不要 untrack；GitHub 为发布态，发布时按 `docs/RELEASE.md` 的净化导出流程生成无 `patent/`、`.spec/`、`tmp/` 的新历史仓库。

## Paper Alignment
- 本项目目标是论文 `Geographically informed graph neural networks` 的核心算法开源库；论文 PDF、运行时代码、测试证据优先于 README/docs 的宽泛描述。
- 回答“是否彻底实现 GIGNN”时，先读 `docs/IMPLEMENTATION_STATUS.md`。当前口径：notebook GIGNN 核心链路已实现；完整论文数值复现、外部数据和所有 baseline 不在默认包能力内。
- 严格对齐 GIGNN 时，不能把“GeoDetector + 普通 GraphSAGE”当作完成。必须证明 geographically informed process 进入 GNN framework。
- 2026-06-19 起，`calculate_gaussian_edge_attributes` 之后还必须保持 `edge_attr`/`edge_weight` 进入 PyG `Data`、model forward 和 message passing；回归检查优先跑 `tests/test_graph_builders.py` 与 `tests/test_model_training.py`。
- 2026-06-19 起，`natural_breaks/fisher_jenks` 为库内向量化 Fisher-Jenks DP（精确全局最优，非 quantile fallback）；`maximum_breaks` 是最宽间隙中点法，与 Jenks 不同，不能混写。回归检查优先跑 `tests/test_geodetector.py`。
- `k_neighbors` 当前是 KNN 构图参数，不自动等同论文 Eq. (4) 的 k-order neighbor aggregation；实现或文档必须分清两者。
- 论文对齐默认（GeoDetectorSelector：`natural_breaks`/5 层/`strategy="interaction"`/`top_interactions=10`；GIGNNClassifier：`edge_weight_mode="product"`/`normalize_edge_weights=False`/`inference="full"`/`hidden_channels=(128,)`）是论文所述机制；GIGNNAutoClassifier 保持 `strategy="factor"` 的通用扩展语义，二者不要混写。
- 偏差符号以论文图 10 为准：`bias = pred_code − obs_code`，正值为高估；notebook 的 `obs − pred` 加 "Overestimated" 标签与论文定义相反，属已知分歧。
- notebook 边属性（`edge_attr`/`edge_attr_diff`）从未进入其聚合器；包按论文所述让边权进入消息聚合 —— 叙述时不能声称「复刻 notebook」。

## Verification Habits
- 对论文对齐类任务，先建立“论文要求 -> 当前代码证据 -> 缺口 -> 验收命令”的矩阵，再动核心算法。
- 提交前至少跑 spec status/check，并用 `git diff --name-only` 对照任务包 `boundary`；不要把未跟踪论文 PDF 或构建产物顺手纳入提交。

## Directory Hygiene
- 正式研究证据统一放在 `research/`：论文进 `research/papers/`，notebook 进 `research/notebooks/`；`tmp/` 只允许放临时草稿和本地实验产物，不承载被文档引用的真源。
- 根目录保持极简：运行时代码在 `gignn/`，测试在 `tests/`，示例在 `examples/`，文档在 `docs/`，任务记录在 `.spec/`。不要提交 `build/`、`dist/`、`*.egg-info/`、缓存或系统噪音文件。
