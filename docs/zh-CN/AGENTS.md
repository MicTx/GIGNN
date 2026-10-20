# 代理指南

本指南面向在 GIGNN 上工作的编码代理、文档代理与自动审查器。

## 从这里开始

多数任务按以下顺序阅读：

1. `llms.txt` 获取项目短上下文。
2. `README.md` 获取面向人的定位与导航。
3. `docs/IMPLEMENTATION_STATUS.md`（任务问 GIGNN 是否完整实现或与论文/notebook 对齐时）。
4. `docs/FEATURE_OVERVIEW.md` 获取能力层级。
5. `docs/API.md` 获取公开 API 与返回类型。
6. `docs/DATA_CONTRACT.md` 获取校验、图、预测与资源上限。
7. 路由表中的任务专属文档。

文档与实现不一致时，以源代码与测试为最终权威。

## 任务路由

| 用户任务 | 先读 | 通常触碰 |
| --- | --- | --- |
| 解释 GIGNN 做什么 | `llms.txt`、`docs/FEATURE_OVERVIEW.md`、`README.md` | 通常不改。 |
| 审计 GIGNN 是否完整实现 | `docs/IMPLEMENTATION_STATUS.md`、`research/notebooks/space-prediction-generalization.ipynb`、`tests/test_model_training.py` | 文档、测试、SPEC 包；仅当证据显示缺口时改代码。 |
| 新增或变更公开 API | `docs/API.md`、`docs/DATA_CONTRACT.md`、相关测试 | 代码、测试、`docs/API.md`、`docs/FEATURE_OVERVIEW.md`、`llms*.txt`。 |
| 变更自动工作流 | `gignn/auto.py`、`tests/test_auto.py`、`docs/API.md`、`docs/REPORTS.md` | 自动代码、测试、报告/数据契约文档。 |
| 变更直接分类器 | `gignn/estimators.py`、`tests/test_estimator.py`、`docs/API.md` | 估计器代码、检查点文档、测试。 |
| 变更地理探测器行为 | `gignn/geodetector/*.py`、`tests/test_geodetector.py`、`docs/FEATURE_OVERVIEW.md` | 地理探测器文档与测试。 |
| 变更图语义 | `gignn/graph/*.py`、`tests/test_graph_builders.py`、`docs/DATA_CONTRACT.md` | 图文档与测试。 |
| 变更泛化诊断 | `gignn/generalization.py`、`tests/test_generalization_standard.py`、`docs/REPORTS.md` | 泛化文档与测试。 |
| 变更绘图 | `gignn/plotting.py`、`tests/test_plotting.py`、`docs/PLOTTING.md` | 绘图文档与测试。 |
| 变更检查点持久化 | `gignn/io/persistence.py`、`gignn/estimators.py`、`tests/test_estimator.py`、`docs/SECURITY.md` | 安全文档、测试。 |
| 变更包元数据或构建 | `pyproject.toml`、`docs/COMMANDS.md`、`docs/RELEASE.md`、`docs/RUNBOOK.md` | 命令/发布/运维文档。 |
| 仅更新文档 | `llms.txt`、`README.md`、相关文档 | 文档与 spec 任务包。 |

## 源码权威

证据冲突时按以下顺序：

1. `tests/` 下的测试。
2. `gignn/` 下的当前实现。
3. `examples/quickstart.py`。
4. `docs/IMPLEMENTATION_STATUS.md`（当前 notebook/论文对齐边界）。
5. `README.md` 与 `docs/*.md`。
6. `.spec/specs/` 下的当前 spec 任务包。

遗留 Trae 任务包已迁入 `.spec/specs/archive/`；原始 Trae 文件保存在各迁移包的 `legacy/` 目录。不要把 `.trae` 当作活动或历史来源。

除非用户显式询问历史，不要把归档 spec 当作当前实现状态。

## 实现边界

GIGNN 是库包。

除非带测试实现，不要引入或记录以下内容：

- HTTP API 端点。
- Web 服务器或 worker 运行时。
- 数据库持久化。
- 必需的环境变量。
- 自动 PyPI 发布。
- 原始遥感影像获取或预处理流水线。
- 服务级认证、限流、CSRF 或部署控制。

## 公开 API 边界

稳定用户入口：

- 顶层 `gignn` 导出
- `gignn.geodetector`
- `gignn.graph`

高级集成面有文档但 `1.0` 前较不稳定：

- `gignn.validation`
- `gignn.training`
- `gignn.models`
- `gignn.io`

## 验证要求

变更后运行最相关的命令集。

纯文档最低要求：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import re
root = Path('.')
missing = []
for path in [root/'README.md', root/'llms.txt', root/'llms-full.txt', *sorted((root/'docs').glob('*.md'))]:
    text = path.read_text()
    for match in re.finditer(r'\[[^\]]+\]\(([^)]+)\)', text):
        target = match.group(1)
        if '://' in target or target.startswith('#'):
            continue
        rel = target.split('#', 1)[0]
        if rel and not (path.parent / rel).resolve().exists():
            missing.append(f'{path} -> {target}')
if missing:
    raise SystemExit('\n'.join(missing))
print('All local markdown links resolve')
PY
```

标准包验证：

```bash
.venv/bin/python -m pytest
.venv/bin/python examples/quickstart.py
.venv/bin/python -m build
```

格式检查：

```bash
.venv/bin/python -m black --check gignn tests examples
```

## 文档同步规则

变更代码时按主题更新文档：

| 代码变更 | 文档同步 |
| --- | --- |
| 新顶层导出 | `docs/API.md`、`docs/FEATURE_OVERVIEW.md`、`llms.txt`、`llms-full.txt`。 |
| 新估计器行为 | `docs/API.md`、`docs/USER_GUIDE.md`、`docs/DATA_CONTRACT.md`。 |
| 新报告字段 | `docs/REPORTS.md`、`docs/API.md`，相关时更新示例。 |
| 新图行为 | `docs/DATA_CONTRACT.md`、`docs/FEATURE_OVERVIEW.md`。 |
| 新 GIGNN 核心对齐证据 | `docs/IMPLEMENTATION_STATUS.md`、`docs/RAG_INDEX.md`、`llms*.txt`、`Agent.md`。 |
| 新绘图参数 | `docs/PLOTTING.md`、`docs/API.md`。 |
| 新命令或依赖 | `docs/COMMANDS.md`、`docs/CONTRIBUTING.md`、`docs/RELEASE.md`。 |
| 新安全边界 | `docs/SECURITY.md`、`docs/RUNBOOK.md`。 |

新增或重命名文档时更新 `docs/RAG_INDEX.md`。

## 代理安全须知

- 不要用系统 Python 3.9 做验证；用 `.venv/bin/python` 或 Python `>=3.10`。
- 未检查 `git status` 前不要删除或覆盖用户工作。
- 除非显式要求，不要提交生成的构建产物。
- 把检查点文件与本地数据文件当不可信输入。
- 不要绕过检查点模式/资源校验。
- 未发生显式上传前，不要声称发布已发布。

## 审查清单

收尾前：

- 公开 API 文档与 `gignn/__init__.py` 及子包 `__all__` 声明一致。
- 数据契约与校验及图构建测试一致。
- 报告文档与 `gignn/reports.py`、`gignn/generalization.py` 的 dataclass 一致。
- 安全文档与代码中检查点及路径行为一致。
- README 链接到每份面向用户的文档。
- `llms.txt` 与 `llms-full.txt` 反映任何变更后的能力边界。
