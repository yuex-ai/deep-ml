def most_frequent_pair(sequences):
    """
    Args:
        sequences: list[list[int]] - list of token ID sequences
    Returns:
        tuple(int, int) or None - the most frequent adjacent pair, with ties
        broken by first appearance. Returns None if no pair exists.
    """
    counts={}
    first_seen={}
    scan_pos=0
    for seq in sequences:
        for i in range(len(seq)-1):
            pair=(seq[i],seq[i+1])
            if pair not in first_seen:
                first_seen[pair]=scan_pos
            counts[pair]=counts.get(pair,0)+1
            scan_pos+=1
    if not counts:
        return None
    return max(
        counts,
        key=lambda pair:(counts[pair],-first_seen[pair])
    )

