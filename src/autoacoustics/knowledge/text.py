"""Deterministic lexical recall, including CJK bigrams; no embedding service."""
import re,unicodedata
from collections import Counter

STOP_EN={'what','why','how','when','where','is','are','the','and','a','an','to','of','for','in','on','please','this','that','seat','motor','current','analysis'}
STOP_ZH=('请问','为什么','如何','怎么','什么','多少','是多少','是否','请','本次','当前','这些','这段','座椅电机','座椅','电机','记录','分析','的','要','了','呢','吗')

def normalized(value):return unicodedata.normalize('NFKC',str(value)).casefold().strip()

def tokens(value,*,question=False,limit=None):
    text=normalized(value)
    if question:
        for word in STOP_ZH:text=text.replace(word,' ')
    output=[]
    for part in re.findall(r'[a-z][a-z0-9_]*|\d+(?:\.\d+)?|[\u3400-\u9fff]+',text):
        if re.fullmatch(r'[\u3400-\u9fff]+',part):
            output.extend(part[index:index+2] for index in range(len(part)-1))
        elif part not in STOP_EN:output.append(part)
    result=Counter(output)
    if limit is not None:result=Counter(dict(result.most_common(limit)))
    return result

def chunks(text,max_chars=800):
    """Non-overlapping original-text spans, never rewritten source paragraphs."""
    for match in re.finditer(r'\S(?:.*?\S)?(?=\n\s*\n|\Z)',text,re.S):
        left,right=match.span()
        while left<right:
            end=min(right,left+max_chars)
            if end<right:
                boundary=max(text.rfind('\n',left,end),text.rfind('。',left,end),text.rfind(' ',left,end))
                if boundary>left+max_chars//2:end=boundary+1
            yield left,end,text[left:end]
            left=end
