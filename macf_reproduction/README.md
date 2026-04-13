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
- LLM 配置抽象：`model=gpt-4o`, `temperature=0.3`

## 新增：适配 `amazon_beauty` 数据评测
支持使用：
- `amazon_beauty/query_data1.csv`
- `amazon_beauty/metadata.csv`

流程：
1. 读取 query 与用户历史交互物品；
2. 在 metadata 全库上执行 agent 协作检索与排序；
3. 计算 target 物品在 Top@10/20/40 的命中与位次；
4. 输出 Hit Rate 与 NDCG。

## 运行

### 1) Demo（内置 mock 数据）
```bash
cd macf_reproduction
PYTHONPATH=src python -m macf.main --mode demo --config config/default.yaml --query "hydrating skincare for sensitive skin"
```

### 2) Amazon Beauty 评测
```bash
cd macf_reproduction
PYTHONPATH=src python -m macf.main \
  --mode evaluate \
  --config config/default.yaml \
  --query-csv amazon_beauty/query_data1.csv \
  --metadata-csv amazon_beauty/metadata.csv
```

输出为结构化 JSON，包括：
- `metrics.hit@10, metrics.ndcg@10`
- `metrics.hit@20, metrics.ndcg@20`
- `metrics.hit@40, metrics.ndcg@40`
- 每条样本的 `rank`（target 排名，未命中为 null）

## 说明：哪些是重建假设
- prompt 的精确措辞
- CSV 字段名兼容映射
- 文本检索与相似度打分函数（token overlap）

## 测试
```bash
cd macf_reproduction
PYTHONPATH=src pytest -q
```
