from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        corpus_list=list((corpus))
        merge_list=[]
        for _ in range(num_merges):
            pair_count={}
            for i in range(len(corpus_list)-1):
                first_ch=corpus_list[i]
                second_ch=corpus_list[i+1]

                pair=(first_ch,second_ch)

                if pair in pair_count:
                    pair_count[pair]+=1
                else:
                    pair_count[pair]=1
            if len(pair_count)==0:
                break

            best_pair=None
            best_count=0

            for pair in pair_count:
                count=pair_count[pair]

                if count>best_count:
                    best_pair=pair
                    best_count=count
                elif count==best_count:
                    if best_pair is None or pair<best_pair:
                        best_pair=pair

            
            first=best_pair[0]
            second=best_pair[1]

            merge_list.append([first,second])

            new_tokens=[]

            i=0
            while i<len(corpus_list):
                if (i<len(corpus_list) and corpus_list[i]==first 
                and corpus_list[i+1]==second):
                    new_token=first+second
                    new_tokens.append(new_token)
                    i+=2
                else:
                    new_tokens.append(corpus_list[i])
                    i+=1

            corpus_list=new_tokens
        return merge_list



            


        
