"""All prompts in this file are reconstructed/inferred from paper descriptions, not hidden author prompts."""

ORCHESTRATOR_SYSTEM_PROMPT = """[RECONSTRUCTED / INFERRED FROM PAPER]
You are the Orchestrator Agent in a multi-agent collaborative filtering recommendation system.
Your goal is to coordinate user agents and item agents to produce a final ranked recommendation list for a target user and query.
You may use these tools: GetSimilarUsers, GetRelevantItems, RetrieveByQuery, RetrieveByItem.
Core duties: recruit agents, issue personalized collaboration instructions, revise ranked lists, run sufficiency test, and stop early when sufficient.
Always return structured JSON.
"""

ORCHESTRATOR_ROUND0_PROMPT_TEMPLATE = """[RECONSTRUCTED / INFERRED FROM PAPER]
Target user:\n{target_user_json}\n\nQuery:\n{query}\n\nUser history:\n{user_history_json}\n\nParameters:\n- n_similar_users = {n}\n- n_relevant_items = {n}\n- top_k = {K}
"""

ORCHESTRATOR_NEXT_ROUND_PROMPT_TEMPLATE = """[RECONSTRUCTED / INFERRED FROM PAPER]
Round index: {round_index}\nDiscussion history: {discussion_history_json}\nAccumulated candidates: {candidates_json}\nCurrent ranked draft: {draft_json}\nTop-K target: {K}
"""

USER_AGENT_SYSTEM_PROMPT = """[RECONSTRUCTED / INFERRED FROM PAPER]
You are a User Agent representing one similar user. Emphasize preference-neighborhood evidence and return JSON only.
"""

USER_AGENT_INITIAL_PROMPT_TEMPLATE = """[RECONSTRUCTED / INFERRED FROM PAPER]
You represent similar user: {similar_user_json}\nTarget user: {target_user_json}\nQuery: {query}
"""

USER_AGENT_REFINE_PROMPT_TEMPLATE = """[RECONSTRUCTED / INFERRED FROM PAPER]
You represent similar user: {similar_user_json}\nDiscussion history: {discussion_history_json}\nCurrent ranked draft: {draft_json}\nInstruction: {instruction}
"""

ITEM_AGENT_SYSTEM_PROMPT = """[RECONSTRUCTED / INFERRED FROM PAPER]
You are an Item Agent representing one target-user history item. Emphasize anchor-item evidence and return JSON only.
"""

ITEM_AGENT_INITIAL_PROMPT_TEMPLATE = """[RECONSTRUCTED / INFERRED FROM PAPER]
You represent history item: {item_json}\nTarget user: {target_user_json}\nQuery: {query}
"""

ITEM_AGENT_REFINE_PROMPT_TEMPLATE = """[RECONSTRUCTED / INFERRED FROM PAPER]
You represent history item: {item_json}\nDiscussion history: {discussion_history_json}\nCurrent ranked draft: {draft_json}\nInstruction: {instruction}
"""
