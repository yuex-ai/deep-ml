def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.

    Args:
        corpus: Dictionary mapping space-separated token sequences to frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.

    Returns:
        List of tuples, where each tuple contains the two tokens merged.
    """
    word_freq={}
    for letter,freq in corpus.items():
        out=[]
        out=tuple(letter.split())
        word_freq[out]=word_freq.get(out,0)+freq
    
    merges=[]
    for _ in range(num_merges):
        p={}
        for tokens,freq in word_freq.items():
            for i in range(len(tokens)-1):
                pair=(tokens[i],tokens[i+1])
                p[pair]=p.get(pair,0)+freq
        best_pair=None
        best_freq=-1
        for tokens,freq in p.items():
            if best_pair is None or best_freq<freq:
                best_pair=tokens
                best_freq=freq
        a,b=best_pair
        merges.append(best_pair)
        new_word_freq={}
        for tokens,freq in word_freq.items():
            new_tokens=[]
            i=0
            while i<len(tokens):
                if i <len(tokens)-1 and tokens[i]==a and tokens[i+1]==b:
                    new_tokens.append(a+b)
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            new_tokens=tuple(new_tokens)
            new_word_freq[new_tokens]=new_word_freq.get(new_tokens,0)+freq
        word_freq=new_word_freq
    return merges


    

            

                




    
