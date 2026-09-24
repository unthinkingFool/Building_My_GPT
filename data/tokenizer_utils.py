from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        result=[]

        for num in numbers:
            text=str(num)
            tokens=[]

            i=0
            while i<len(text):
                longest_match=None
                for end in range(len(text),i,-1):
                    candidate=text[i:end]

                    if candidate in vocab:
                        longest_match=candidate
                        break
                
                if longest_match is None:
                    longest_match=text[i]
                    i+=1

                else:
                    i+=len(longest_match)

                tokens.append(longest_match)
            
            result.append(tokens)

        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        count=0

        word=text
        

        
        i=0
        while i<len(word):
            longest_match=None

            for end in range(len(word),i,-1):
                candidate=word[i:end]

                if candidate in vocab:
                    longest_match=candidate
                    break
            if longest_match is None:
                longest_match=word[i]
                i+=1
            else:
                i+=len(longest_match)

            count+=1
        
        return count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words=text.split()

        if not words:
            return 0.0
        
        token_count=self.count_tokens(text,vocab)
        word_count=len(words)

        fertility=token_count/word_count

        return round(fertility,4)
