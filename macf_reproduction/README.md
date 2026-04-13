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

## LLM Policy（已切换为 Qwen3-8B）
- 默认配置已改为：
  - `provider: qwen_local`
  - `model: Qwen/Qwen3-8B`
  - `temperature: 0.3`
  - `enable_thinking: true`
- 对应实现：`src/macf/llm.py` 中 `QwenLocalBackend`。
- 若环境缺少 `transformers/torch` 或模型不可下载，会自动回退到 deterministic fallback（保证流程可运行）。

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

## Qwen 官方调用对齐说明
仓库中的 `QwenLocalBackend` 采用与官方示例一致的核心流程：
- `AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")`
- `AutoModelForCausalLM.from_pretrained(..., torch_dtype="auto", device_map="auto")`
- `apply_chat_template(..., enable_thinking=True)`
- `model.generate(..., max_new_tokens=...)`
- 兼容 `</think>` 分段解析（token id `151668`）

## 说明：哪些是重建假设
- prompt 的精确措辞
- CSV 字段名兼容映射
- 文本检索与相似度打分函数（token overlap）

## 测试
```bash
cd macf_reproduction
PYTHONPATH=src pytest -q
```
