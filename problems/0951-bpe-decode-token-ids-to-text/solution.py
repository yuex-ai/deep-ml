def bpe_decode(ids, vocab):
    """
    Args:
        ids: list[int] - token IDs to decode
        vocab: dict[int, str] - mapping from token ID to token string
    Returns:
        str - the decoded text
    """
    if len(ids)<1:
        return str('')
    out=[]
    for token_id in ids:
        token=vocab[token_id]
        if token.startswith("G"):
            out.append(" "+token[1:])
        else:
            out.append(token)
    return "".join(out)
