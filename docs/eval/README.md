# 评估报告存档

`build/` 不入库，本目录保存冻结基线的**导出摘要**（汇总指标、分类指标与逐题得分），用于在仓库内直接核对 README 引用的数字。完整 JSON 与日志（含回答原文、判定理由）仍在本地 `build/eval/`，可由下方命令重新生成。

数字均为单次采样结果；回答指纹与判定指纹以导出文件内的记录为准。三个集合用途不同，**不跨集合比较**。

| 文件 | 集合 | 题数 | 关键结果 | 生成时间 |
|---|---|---|---|---|
| [2026-11-03_开发集121题.md](2026-11-03_开发集121题.md) | 开发集 v5 | 121 | 平均分 1.942；Recall 1.000；MRR 0.849；引用 0.991 | 2026-11-03T09:10:06 |
| [2026-11-02_留出集20题.md](2026-11-02_留出集20题.md) | 留出集 v1 | 20 | 平均分 1.85；Recall 1.000；MRR 0.971 | 2026-11-02T19:53:38 |
| [2026-11-02_同义改写10题.md](2026-11-02_同义改写10题.md) | 同义改写 v1 | 10 | 平均分 1.90；Recall 1.000；MRR 0.853 | 2026-11-02T19:58:00 |
| [2026-11-02_提示注入7题.md](2026-11-02_提示注入7题.md) | 对抗集 v1 | 7 | 通过 7/7；judge 均分 2.0 | 2026-11-02T20:18:23 |
| [2026-11-02_judge稳定性20题.md](2026-11-02_judge稳定性20题.md) | 稳定性验证 | 20 × 3 | 分数完全一致 1.000；121 题 ×3 一致率 0.9835 | 2026-11-02T20:06:31 |

## judge 一致性口径

- 20 题稳定性验证：已存回答重复判分 3 次，逐题得分完全一致率 1.000。
- 121 题全量 judge×3：得分一致率 **0.9835**（119/121）。不一致的两题是 `fact_10`（`[1, 2, 2]`）与 `agg_13`（`[2, 1, 2]`），众数均为 2；该字段来自 `build/eval/report.json` 的 `judge_score_agreement`。
- judge 对复合事实点（`key_facts` 用逗号串联多个结论）仍存在假阴性，属已知局限，详见根目录 README。

## 重新生成

```bash
# 开发集 121 题（出稿验收用 judge×3）
PYTHONIOENCODING=utf-8 python -m eval.run_eval --judge-repeats 3

# 留出集 / 同义改写（冻结后只跑一次）
PYTHONIOENCODING=utf-8 python -m eval.run_eval \
  --questions eval/questions.holdout.v1.jsonl \
  --report-stem report_holdout_v1 \
  --results-name results_holdout_v1.jsonl \
  --judge-repeats 1

PYTHONIOENCODING=utf-8 python -m eval.run_eval \
  --questions eval/questions.paraphrase.v1.jsonl \
  --report-stem report_paraphrase_v1 \
  --results-name results_paraphrase_v1.jsonl \
  --judge-repeats 1

# 提示注入 7 题
PYTHONIOENCODING=utf-8 python -m eval.run_injection_eval --stem injection_v1

# judge 稳定性：复用已存回答做 20 题 ×3 判分
PYTHONIOENCODING=utf-8 python -m eval.run_eval \
  --results build/eval/results_m3.jsonl \
  --judge-stability 3 \
  --ids fact_02,fact_10,fact_15,ver_01,ver_05,ver_11,agg_01,agg_06,agg_11,perm_02,perm_05,perm_10,multi_02,multi_07,multi_10,ref_02,ref_08,chat_03,chat_06,term_01 \
  --stability-stem report_stability_v1
```

报告写入 `build/eval/report*.{json,md}`。注意：生成不可完全复现（措辞漂移），跨运行比较以检索指标、路由与固定回答指纹的判定为准。
