# MACF Reproduction (Reconstructed)

本仓库是对论文 **Multi-Agent Collaborative Filtering: Orchestrating Users and Items for Agentic Recommendations** 的可运行复现（reconstructed reproduction）。

## 透明性声明
- 本实现**不**声称访问作者的隐藏 prompt。
- 所有 prompt 文本都在 `prompts.py` 中标记为 **RECONSTRUCTED / INFERRED FROM PAPER**。

## 与论文对齐的核心点
- 3 类 agent：Orchestrator / User Agent / Item Agent
- 4 个工具接口：
  - `GetSimilarUsers(user_id, n)`
  - `GetRelevantItems(user_id, query, n)`
  - `RetrieveByQuery(query, k)`
  - `RetrieveByItem(item_id, k)`
- DAR（动态招募）、PCI（个性化协作指令）、ATU（自适应工具使用）
- 多轮讨论 `Tmax=5`
- 最终推荐 `Top-K=10`

## LLM Policy（Qwen3-8B，Think模式关闭）
- 默认配置：
  - `provider: qwen_local`
  - `model: Qwen/Qwen3-8B`
  - `temperature: 0.3`
  - `enable_thinking: false`

## 推荐使用方式（避免 `python -m macf.main` 导入问题）
你可以直接使用仓库根目录脚本 `retrieval_baselines.py`，无需手动设置 `PYTHONPATH`。

```bash
python retrieval_baselines.py \
  --metadata amazon_beauty/metadata.csv \
  --query-data amazon_beauty/query_data1.csv \
  --output-dir amazon_beauty/baseline_eval_full_query_only \
  --preference-only
```

说明：
- 会实时打印处理进度与“已处理用户平均指标”（HR/NDCG @10/@20/@40）。
- 结果会写入：`<output-dir>/macf_eval_result.json`。

## 测试
```bash
cd macf_reproduction
PYTHONPATH=src pytest -q
```
