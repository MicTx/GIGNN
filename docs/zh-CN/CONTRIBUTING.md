# 贡献指南

在开发、测试与准备 GIGNN 变更时使用本指南。

## 前置要求

| 要求 | 版本 / 说明 |
| --- | --- |
| Python | `>=3.10`；本地验证常用 Python 3.11。 |
| 包管理器 | 本地搭建可用 `pip` 或 `uv`。 |
| 核心 ML 依赖 | `torch>=2.0` 与 `torch-geometric>=2.4`。 |
| 可选 GIS 依赖 | GeoPandas 与 Shapely 工作流请安装 `.[gis]`。 |

## 环境搭建

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

可选 GIS 辅助组件：

```bash
python -m pip install -e '.[gis,dev]'
```

若 PyTorch 或 PyTorch Geometric 自动安装失败，先为平台安装官方 wheel 组，再重跑 GIGNN 安装命令。

## 开发工作流

1. 变更范围保持在一个功能、一个缺陷修复或一次文档更新内。
2. 保持既有公开 API 行为不变，除非该变更显式更新 API 文档与测试。
3. 为行为变更新增或更新测试。
4. 用户可见的行为、输入、输出、报告或命令变更时更新文档。
5. 提交前运行验证命令。

## 验证命令

| 命令 | 用途 |
| --- | --- |
| `.venv/bin/python -m pytest` | 运行单元与集成测试。 |
| `.venv/bin/python -m black --check gignn tests examples` | 检查格式。 |
| `.venv/bin/python examples/quickstart.py` | 演练端到端示例。 |
| `.venv/bin/python -m build` | 面向发布的变更验证包构建。 |

## 测试地图

| 领域 | 测试 |
| --- | --- |
| 导入与公开导出 | `tests/test_import.py` |
| 校验与标签 | `tests/test_validation.py` |
| 地理探测器 | `tests/test_geodetector.py` |
| 图构建与 PyG 数据 | `tests/test_graph_builders.py` |
| 模型与训练工具 | `tests/test_model_training.py`、`tests/test_training.py` |
| 直接估计器 | `tests/test_estimator.py` |
| 自动工作流 | `tests/test_auto.py` |
| 泛化诊断 | `tests/test_generalization_standard.py` |
| 绘图 | `tests/test_plotting.py` |

## 文档维护

| 文档 | 更新时机 |
| --- | --- |
| `README.md` | 项目定位、安装路径、快速上手或导航变更。 |
| `docs/FEATURE_OVERVIEW.md` | 能力层或用户可见行为变更。 |
| `docs/API.md` | 公开类、函数、签名、返回、异常或报告变更。 |
| `docs/DATA_CONTRACT.md` | 输入形状、校验规则、标签、图语义或资源上限变更。 |
| `docs/REPORTS.md` | dataclass 字段或报告表语义变更。 |
| `docs/PLOTTING.md` | 绘图输入、输出或保存行为变更。 |
| `docs/COMMANDS.md` | 构建、测试、安装或包命令变更。 |
| `docs/RELEASE.md` | 发布检查清单或产物策略变更。 |
| `docs/SECURITY.md` | 信任边界、检查点加载或不可信输入处理变更。 |

## 代码风格

- 遵循 PEP 8。
- 公开函数与方法使用类型注解。
- 报告与 DTO 优先使用不可变 dataclass。
- 在模块边界校验输入。
- 避免修改调用方传入的 DataFrame 或数组。
- 公开错误保持可操作，酌情抛出 `ValidationError` 或 `NotFittedError`。

## PR 检查清单

- 变更范围清晰。
- 测试覆盖新增或变更的行为。
- 公开 API 变更已有文档。
- `.venv/bin/python -m pytest` 通过。
- 格式检查通过或记录了明确理由。
- 打包或面向发布的变更通过包构建。
- 无本地产物、私有路径、notebook 或任务文件被意外暂存。
- 除非显式要求，不执行 PyPI 发布。
