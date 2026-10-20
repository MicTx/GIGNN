# 报告

GIGNN 报告是拟合或诊断后暴露的纯 dataclass 与 pandas 表。它们面向检查、notebook 与下游文档。

面向发表级可复现性时，`export_research_artifacts` 把这些报告收集进带 `raw/`、`preprocessing/`、`process/`、`results/` 证据层的目录包。包与模型检查点分离：检查点恢复拟合的模型，研究产物记录一次运行是如何计算的。

## 自动工作流报告

`GIGNNAutoClassifier.fit` 创建五份主报告与一份由 `simulate` 返回的模拟报告。

## `SpatialRelationshipReport`

属主：`GIGNNAutoClassifier.discovery_report_`

字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `selected_features` | `list[str]` | `GeoDetectorSelector` 选出的特征。 |
| `factor_scores` | `pd.DataFrame` | 按因子名索引的因子 q 分与 p 值。 |
| `interaction_scores` | `pd.DataFrame` | 成对地理探测器交互分。 |
| `discretizer` | `str` | 选定的离散化方法。 |
| `max_bins` | `int` | 选定的最大分箱数。 |

典型用法：

```python
model.discovery_report_.selected_features
model.discovery_report_.factor_scores
model.discovery_report_.interaction_scores
```

## `GraphSelectionReport`

属主：`GIGNNAutoClassifier.graph_report_`

字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `selected_strategy` | `str` | 选定的图策略，如 `edge_index`、`knn`、`distance`、`contiguity`。 |
| `selected_params` | `dict[str, object]` | 选定图策略使用的参数。 |
| `candidates` | `pd.DataFrame` | 候选图指标、切分边审计字段、失败原因与验证分。 |
| `selected_edge_index` | `np.ndarray` | 最终模型使用的全量数据部署边索引。 |

候选表包含全量图指标加上可用的 `train_*`、`validation_*`、`test_*` 边数、节点数与规范化 SHA-256 边哈希。切分哈希描述本地选择/评估图；不意味着最终全量部署图独立于留出几何。

## `ModelSelectionReport`

属主：`GIGNNAutoClassifier.model_selection_report_`

字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `best_params` | `dict[str, object]` | 按验证准确率选出的最佳模型参数。 |
| `candidates` | `pd.DataFrame` | 候选模型/图组合与分数。 |

典型用法：

```python
model.model_selection_report_.best_params
model.model_selection_report_.candidates.sort_values("validation_accuracy", ascending=False)
```

## `GeneralizationReport`

属主：`GIGNNAutoClassifier.validation_report_` 与 `GIGNNAutoClassifier.generalization_report_`

字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `accuracy` | `float` | 报告自身切分上的准确率。 |
| `macro_f1` | `float` | 宏平均 F1。 |
| `weighted_f1` | `float` | 加权 F1。 |
| `per_class` | `pd.DataFrame` | 来自 scikit-learn 分类报告的逐类指标。 |

`validation_report_` 产自用于候选选择的验证切分。`generalization_report_` 仅产自选择之后的独立测试切分，不影响选定候选。外部观测/预测对比请用 `generalization_diagnostics`。

## `SimulationReport`

由 `GIGNNAutoClassifier.simulate(...)` 返回。

字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `baseline_predictions` | `np.ndarray` | 基线特征的预测标签。 |
| `scenario_predictions` | `np.ndarray` | 情景特征的预测标签。 |
| `baseline_probabilities` | `np.ndarray` | 基线特征的类概率。 |
| `scenario_probabilities` | `np.ndarray` | 情景特征的类概率。 |
| `probability_delta` | `np.ndarray` | `scenario_probabilities - baseline_probabilities`。 |
| `changed_mask` | `np.ndarray` | 预测标签变化的布尔掩码。 |

基线与情景特征都用拟合的 `GeoDetectorSelector` 变换。

## 数据集报告

### `StandardizedSpatialDataset`

由 `standardize_spatial_dataset` 返回。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `frame` | `pd.DataFrame` | 重置索引后的完整帧副本。 |
| `features` | `pd.DataFrame` | 校验过的数值特征表。 |
| `target` | `pd.Series` | 目标标签。 |
| `ids` | `pd.Series | None` | 可选的空间单元标识。 |
| `geometry` | `pd.Series | None` | 可选的几何对象。 |

### `SpatialDatasetReport`

与 `StandardizedSpatialDataset` 一同返回。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id_column` | `str | None` | ID 列名。 |
| `target_column` | `str` | 目标列名。 |
| `feature_columns` | `list[str]` | 使用的特征列。 |
| `geometry_column` | `str | None` | 几何列名。 |
| `missing_filled` | `dict[str, int]` | 填充前的缺失特征计数。 |
| `row_count` | `int` | 行数。 |

## 诊断报告

### `GeneralizationDiagnostics`

由 `generalization_diagnostics` 返回。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `accuracy` | `float` | 观测与预测标签间准确率。 |
| `bias_frame` | `pd.DataFrame` | 行级观测、预测、偏差、偏差类型与校验标签。 |
| `classification_report` | `pd.DataFrame` | 转换为 DataFrame 的 scikit-learn 分类报告。 |
| `bias_moran` | `MoranResult \| None` | 偏差在 `edge_index` 上的 Moran's I；未传边索引时为 `None`。 |

`bias_frame` 列：

| 列 | 说明 |
| --- | --- |
| `observed` | 原始观测标签。 |
| `prediction` | 原始预测标签。 |
| `Bias` | `predicted_code - observed_code`；正值表示模型高估（论文图 10）。 |
| `Bias Type` | `Correct`、`Overestimated (...)` 或 `Underestimated (...)`。 |
| `Validation` | 如 `ObservedLabel-PredictedLabel` 的文本对。 |

`MoranResult` 字段语义同 `esda.Moran`：`statistic`、`expected`、`variance_norm`、`z_norm`、`p_norm`（双尾，正态假设），以及置换推断的 `p_sim`（折叠单尾伪 p）、`z_sim`、`permutations`；取值为常数时各统计量为 NaN。

### `CrossValidationReport`

由 `cross_validate` 返回（直推式 k 折，论文 §5.2.1）。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `fold_metrics` | `pd.DataFrame` | 各折的 `fold`、`n_test`、`accuracy`、`f1_macro`。 |
| `out_of_fold_observed` | `pd.Series` | 全部标签（每个节点恰被留出一次）。 |
| `out_of_fold_predictions` | `pd.Series` | 汇合的留出预测。 |
| `bias_moran` | `MoranResult \| None` | 折外偏差的 Moran's I；需要 `edge_index`。 |

标签顺序有领域含义时提供 `class_order`。

## 训练报告

### `TrainingResult`

属主：`GIGNNClassifier.training_result_`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `losses` | `list[float]` | 每个 epoch 一个损失值。 |

`TrainingResult.losses` 可转换为 DataFrame 供 `plot_training_curves` 使用。

## 研究产物包

`export_research_artifacts(model, output_dir, ...)` 与 `GIGNNAutoClassifier.export_research_artifacts(...)` 返回 `ResearchArtifactBundle`。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `root` | `Path` | 产物包根目录。 |
| `manifest_path` | `Path` | `manifest.json` 路径。 |
| `files` | `dict[str, Path]` | 逻辑产物名到已写文件的映射。 |

包分层：

| 层 | 用途 |
| --- | --- |
| `raw/` | 输入模式、行/列摘要、SHA-256 指纹，`include_raw=True` 时的可选原始 CSV。 |
| `preprocessing/` | 特征描述、缺失计数、填充摘要、目标分布、选定列与训练/验证/测试索引。 |
| `process/` | 地理探测器因子与交互分、离散化分箱、图候选、选定边表、模型候选、训练损失。 |
| `results/` | 验证指标、独立测试指标、逐类测试报告、预测、概率、可选模拟输出与图表清单。 |

原始表导出是可选的。默认只记录模式、摘要与指纹，避免敏感研究数据被意外复制。
