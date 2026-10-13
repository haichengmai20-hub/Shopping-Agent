# 2026-09-25 Windows 本机环境准备

状态：依赖安装、本地前端和隔离后端启动检查通过；真实模型 API 尚待配置与验收。代码基线：`8cbd4d6b94eb9713952d3bd8f050da176accdca1`，本次改动未提交。

## 需求与变更

为 Windows 本机准备 Globex 项目，在用户提供模型 API 前先完成可独立执行的依赖、配置和启动检查。保留原有 `data/`，首次创建被 Git 忽略的 `.env`，将示例 `LLM_API_KEY` 占位符清空。

- 在工作区 `D:\RL及harness\.tooling\uv\bin\uv.exe` 安装 uv，指定已有 Python 3.13 创建项目 `.venv`；按 `uv.lock` 安装 110 个包。
- 前端锁文件的包地址指向不可达的 `registry.anpm.alibaba-inc.com`，安装时临时将下载地址映射到官方 npm 仓库，未修改 `frontend/package-lock.json`。
- `app/infrastructure/persistence/json_file_stores.py` 原先直接导入 Unix 的 `fcntl`，原生 Windows 无法导入后端。改用跨平台 `portalocker` 文件锁，并在 `pyproject.toml`、`uv.lock` 声明依赖；保留同一 `.lock` 文件与独占锁语义。

## 验证与边界

命令和实际结果见[本机验证记录](../../../eval/verification/windows-setup-20260925/README.md)。前端构建通过、本地页面可访问、文件锁回归 15 项通过；使用假密钥及不可达的本地模型端点，在独立数据目录启动后端，`/health` 与 `/docs` 返回 200，SQLite 初始化成功。向量与知识库在假端点下预期降级。未调用真实模型，未验证实际推荐质量、向量召回或专用精排。

## 启用与回滚

用户在本地 `.env` 填写可用模型配置后，进入工程根目录启动：

```powershell
& '.\.venv\Scripts\python.exe' -m uvicorn app.presentation.server:app --host 127.0.0.1 --port 8000 --workers 1
npm.cmd --prefix frontend run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

两条命令分别在两个终端执行。首次向量建库可能调用大量 embedding 请求；确认服务额度和模型权限后再做真实选购。若需回退本次源码兼容改动，只恢复 `json_file_stores.py`、`pyproject.toml` 与 `uv.lock` 的本次改动；保留已有 `data/` 和私有 `.env`，不要把本地数据删除作为回滚手段。Linux 路径本次未实测。
