from consolidation import RulePool

pool = RulePool("/tmp/test_rules.jsonl")

# 第一轮:加三条规则
pool.apply_ops("ADD 1: Always identify the bridge entity first.\n"
               "ADD 2: For comparison questions, the answer is often yes/no.\n"
               "ADD 3: Verify dates before concluding.", q_index=10)
print("第一轮后(期望3条,count都是2):")
for r in pool.rules: print(f"  count={r['count']}  {r['text'][:40]}")

# 第二轮:AGREE 1(+1→3), REMOVE 2(-1→1), EDIT 3(改写+1→3), ADD 新的(=2), UPVOTE alias 测试
pool.apply_ops("AGREE 1: Always identify the bridge entity first.\n"
               "REMOVE 2: For comparison questions, the answer is often yes/no.\n"
               "EDIT 3: Cross-check dates across all retrieved documents before finalizing.\n"
               "ADD 4: Prefer exact-title search over keyword search.", q_index=20)
print("\n第二轮后(期望: 规则1 count=3, 规则2 count=1, 规则3被改写且count=3, 新规则count=2):")
for r in pool.rules: print(f"  count={r['count']}  {r['text'][:50]}")

# 第三轮:把规则2 REMOVE 到死(count=1, -1 → 0 → 删除)
pool.apply_ops("REMOVE 4: For comparison questions, the answer is often yes/no.", q_index=30)
# 注意:排序后规则顺序变了,这里只验证"count<=0被删"的行为
print("\n第三轮后(应有一条规则被删除,总数减1):")
for r in pool.rules: print(f"  count={r['count']}  {r['text'][:50]}")

print("\nrender() 输出(注入做题prompt的形态):")
print(pool.render())
