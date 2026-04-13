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
- 对应实现：`src/macf/llm.py` 中 `QwenLocalBackend`。
- 若环境缺少 `transformers/torch` 或模型不可下载，会自动回退到 deterministic fallback（保证流程可运行）。

## Amazon Beauty 评测（含实时进度和平均指标打印）
支持：
- `amazon_beauty/query_data1.csv`
- `amazon_beauty/metadata.csv`

评测时会实时打印：
1. 当前处理进度（第几个用户 / 总用户数）
2. **已处理用户的平均** HR/NDCG（@10/@20/@40）

## 可直接运行命令
```bash
cd macf_reproduction
PYTHONPATH=src python -m macf.main \
  --mode evaluate \
  --config config/default.yaml \
  --query-csv amazon_beauty/query_data1.csv \
  --metadata-csv amazon_beauty/metadata.csv
```

## 测试
```bash
cd macf_reproduction
PYTHONPATH=src pytest -q
```
