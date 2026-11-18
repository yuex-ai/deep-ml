def replace_pair(sequences, pair, new_id):
    # Your code here
    result=[]
    for seq in sequences:
        newids=[]
        i=0
        while i <len(seq):
            if i<len(seq)-1 and seq[i]==pair[0] and seq[i+1]==pair[1]:
                newids.append(new_id)
                i+=2
            else:
                newids.append(seq[i])
                i+=1
        result.append(newids)
    return result