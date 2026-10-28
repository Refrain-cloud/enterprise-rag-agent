# 项目：北辰智造企业知识库 RAG Agent

面向企业内部知识库的问答 Agent。知识库位于 `data/kb/`，公司设定为北辰智造科技（197 人，工业软件 / MES）。项目必须配套可量化的评估集（`eval/`），不以"能跑通 demo"为完成标准。

当前阶段：M1 数据底座（解析、元数据、种子评估集、baseline）。

## 技术选型

- Python 3.14，Windows + bash（命令使用 Unix 风格路径）
- 生成模型：Qwen（DashScope OpenAI 兼容接口），具体模型由 `configs/` 决定
- Embedding：`text-embedding-v3`
- 编排：LangGraph，**只负责路由与状态流转**
- 检索：Chroma（向量）+ BM25（关键词）+ 元数据过滤；表格数据用 DuckDB / pandas
- 服务：FastAPI + 简易 Web Demo（角色切换、引用展示）
- 所有密钥从环境变量读取，禁止硬编码

## 数据与元数据

- `data/kb/_manifest.json`（273 条）提供 doc_id、部门、密级、版本、生效日期、状态、备注等字段，是元数据的起点。
- 清单**不能盲信**：同主题新旧版本可能同时标为 `active`（如 HR-001 与 HR-023、FN-001 与 FN-012）。权威版本需结合 `effective_date`、文档正文中的废止声明和 `data/kb/policy/PO-008_制度修订记录.csv` 综合判定。
- 状态取值：`active` / `outdated` / `draft` / `superseded`；密级：`public` / `internal` / `confidential`；可见范围：`company` / `dept:XX`。
- `data/kb/_共享盘索引.txt` 记录文件大小与修改时间，可用于核对。
- 清单的 `note` 字段标注了部分评测陷阱（如 MS-006 批注稿 750 vs 800），可作为出题线索，但结论仍需回原文验证。

## 核心原则

1. **评估优先**：任何分块、检索、提示词、Agent 逻辑改动，都必须先跑评估集并对比指标（Recall@5 / MRR / nDCG、答案正确性、引用准确率、权限合规、拒答正确率、延迟与成本）。禁止凭感觉调参。
2. **权限在检索层**：按角色（普通员工 / HR / 财务 / 销售 / 管理层）和文档密级过滤，未授权内容不得进入上下文；不能只靠系统提示词让模型拒答。
3. **版本权威性**：结合元数据（状态、生效日期、版本）、正文废止声明和 `PO-008` 判定权威版本。草稿、副本、已废止版本不得作为最终依据；回答需说明所依据的版本。
4. **引用强制**：知识类回答必须给出来源文件名（必要时到章节/行）；知识库无法确证时明确拒答。
5. **结构化问题走工具**：CSV / XLSX 的统计、汇总、对比用 SQL / pandas，不做向量检索硬答。

## 目录结构

```
data/kb/        只读原始知识库
src/
  ingest/       解析、元数据、去重、OCR
  index/        Chroma、BM25、DuckDB 索引构建
  retrieval/    混合检索、rerank、元数据/版本/密级过滤
  security/     角色与权限
  tools/        SQL / pandas 查询工具
  agent/        LangGraph 状态图（路由、改写、多步检索、自检）
  api/          FastAPI
  ui/           Web Demo
eval/           评估集、judge、跑分脚本与报告
configs/        模型与检索配置
build/          派生索引与中间产物（不入库）
```

## 开发命令

```bash
python -m pip install -r requirements.txt
# 以下命令随 M1 实现补充：解析入库、建索引、跑评估、启动 API
```

## Git 工作流

- 每完成一个**可验证的功能**（测试或评估跑通）就提交一次；不要把多个无关功能混在一个提交里。
- 提交信息使用前缀：`feat:` 新功能、`fix:` 修复、`eval:` 评估集与指标、`docs:` 文档、`chore:` 工程配置。
- 提交前至少跑一次相关测试或评估脚本，不提交跑不起来的代码。
- 不提交：`.env`、密钥、`build/` 下的索引与缓存、`__pycache__`、`.venv`、`.claude/settings.local.json`。
- `data/` 是只读知识库快照，可以入库，但不得修改、移动或删除。

## 禁止事项

- 不修改 `data/` 下的原始文件；派生结果写入 `build/`。
- 不硬编码 API key、模型地址或个人绝对路径。
- 不为通过评估而修改评估集的 ground truth 或 gold 文档；发现矛盾应修正数据或实现。
- 不把权限控制只写在提示词里。
- 不把详细架构、评估报告、临时进度写进本文件。
