def bpe_encode(text, token_to_id, merges):
    if not text:
        return []

    # 1. 字符转初始 token id 序列
    seq = [token_to_id[ch] for ch in text]

    while True:
        # 2. 找当前相邻 pair 中 merges[pair] 最小的
        best_pair = None
        best_rank = None

        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            if pair in merges:
                rank = merges[pair]  # new_id 即 rank
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

        # 没有可合并 pair，结束
        if best_pair is None:
            return seq

        # 3. 对本轮获胜 pair 做一次全局不重叠替换
        a, b = best_pair
        new_id = merges[best_pair]

        out = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1

        seq = out