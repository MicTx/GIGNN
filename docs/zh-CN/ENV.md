# 环境变量

GIGNN 是一个 Python 库包，不是部署的服务。当前代码库在正常库使用场景下不需要运行时环境变量。

## 必需变量

| 变量 | 必需 | 说明 |
| --- | --- | --- |
| 无 | 否 | 包运行时不要求任何环境变量。 |

## 已检查的环境来源

| 来源 | 状态 |
| --- | --- |
| `.env.example` | 不存在。 |
| `.env.template` | 不存在。 |
| `.env.sample` | 不存在。 |
| Docker 环境文件 | 不存在。 |

## Python 环境

使用 Python `>=3.10`。项目通常在本地 `.venv`（Python 3.11）中验证。

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

macOS 系统 `python3` 可能是 Python 3.9，不满足本项目要求。

## 依赖上下文

PyTorch 与 PyTorch Geometric 是核心依赖。若自动依赖解析失败，请先安装平台对应的官方 wheel 组，再重新安装 GIGNN。

可选 GIS 辅助组件安装：

```bash
python -m pip install -e '.[gis,dev]'
```

## 服务包装层

若其他应用把 GIGNN 包装在 API 或定时任务之后，该包装层可能需要自己的环境变量（文件路径、存储、认证、资源上限）。这些变量不在当前 GIGNN 包契约范围内。
