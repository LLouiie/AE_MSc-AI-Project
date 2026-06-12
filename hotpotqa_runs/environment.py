import re
import string
from typing import Tuple, Dict, List
from difflib import SequenceMatcher
 
 
class DistractorDocstore:
    """
    A simple docstore that searches within the 10 given HotpotQA distractor documents.
    Replaces LangChain's DocstoreExplorer(Wikipedia()).
    """
    def __init__(self, titles: List[str], sentences: List[List[str]]):
        # Build a dict: title -> paragraph (joined sentences)
        self.docs = {}
        for title, sents in zip(titles, sentences):
            self.docs[title] = " ".join(sents)
        
        self.last_searched_title = None
        self.last_searched_sentences = []
        self.lookup_index = 0
    
    def search(self, query: str) -> str:
        """Search for a document by title. Uses fuzzy matching."""
        # Exact match first
        if query in self.docs:
            self.last_searched_title = query
            self.last_searched_sentences = self.docs[query].split('. ')
            self.lookup_index = 0
            return self.docs[query]
        
        # Fuzzy match: find the best matching title
        best_match = None
        best_score = 0
        for title in self.docs:
            score = SequenceMatcher(None, query.lower(), title.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = title
        
        if best_match and best_score > 0.5:
            self.last_searched_title = best_match
            self.last_searched_sentences = self.docs[best_match].split('. ')
            self.lookup_index = 0
            return self.docs[best_match]
        
        # No match found, suggest similar titles
        similar = [t for t in self.docs if query.lower() in t.lower() or t.lower() in query.lower()]
        if similar:
            return f"Could not find [{query}]. Similar: {similar}"
        return f"Could not find [{query}]. Similar: {list(self.docs.keys())[:5]}"
    
    def lookup(self, keyword: str) -> str:
        """Find the next sentence containing keyword in the last searched document."""
        if self.last_searched_title is None:
            raise ValueError("No document has been searched yet.")
        
        for i in range(self.lookup_index, len(self.last_searched_sentences)):
            if keyword.lower() in self.last_searched_sentences[i].lower():
                self.lookup_index = i + 1
                result_num = sum(1 for j in range(i + 1) 
                               if keyword.lower() in self.last_searched_sentences[j].lower())
                return f"(Result {result_num} / {sum(1 for s in self.last_searched_sentences if keyword.lower() in s.lower())}) {self.last_searched_sentences[i]}"
        
        return f"No more results found for '{keyword}' in the current document."
 
 
class QAEnv:
    def __init__(self,
                 question: str,
                 key: str,
                 context: Dict,
                 max_steps: int = 6):
        
        self.question = question
        self.key = key
        self.max_steps = max_steps
        self.explorer = DistractorDocstore(
            titles=list(context['title']),
            sentences=list(context['sentences'])
        )
        self.reset()
 
    def reset(self):
        self.curr_step = 0
        self.terminated = False
        self.answer = ''
        # Reset the docstore lookup state
        self.explorer.last_searched_title = None
        self.explorer.last_searched_sentences = []
        self.explorer.lookup_index = 0
 
    def step(self, action: str) -> Tuple[str, bool, bool, bool, int]:
        action_type, argument = parse_action(action)
 
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                observation = 'Answer is CORRECT'
            else: 
                observation = 'Answer is INCORRECT'
            self.terminated = True
 
        elif action_type == 'Search':
            try:
                observation = self.explorer.search(argument).strip('\n').strip()
            except Exception as e:
                print(e)
                observation = f'Could not find that page, please try again.'
                    
        elif action_type == 'Lookup':
            try:
                observation = self.explorer.lookup(argument).strip('\n').strip()
            except ValueError as e:
                observation = str(e)
 
        else:
            observation = 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'
 
        reward = self.is_correct()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
 
        self.curr_step += 1
 
        return observation, reward, terminated, truncated, self.curr_step
 
    def is_correct(self) -> bool:
        return EM(self.answer, self.key)
    
    def is_terminated(self) -> bool:
        return self.terminated
 
    def is_truncated(self) -> bool:
        return self.curr_step >= self.max_steps
 
 
def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    
    else:
        return None, None
 
 
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
 
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
 
    def lower(text):
        return text.lower()
 
    return white_space_fix(remove_articles(remove_punc(lower(s))))
 
 
def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)