"""Current active versions only. Lexical evidence is never a diagnosis."""
import hashlib,math
from .models import Evidence,SearchFilters,KnowledgeError
from .text import tokens,normalized

class SearchEngine:
    def __init__(self,store):self.store=store
    def search_local(self,question,filters=None,limit=6):
        if not isinstance(question,str) or not question.strip():return ()
        if len(question)>4000:raise KnowledgeError('问题超过4000字符，请缩短问题。')
        filters=filters or SearchFilters();limit=max(0,min(6,int(limit)))
        query=tokens(question,question=True,limit=128)
        if not query or not limit:return ()
        placeholders=','.join('?' for _ in query)
        sql=f'''SELECT t.source_id,t.version,t.ordinal,t.term,t.count,c.start,c.end,c.body
                FROM terms t JOIN chunks c USING(source_id,version,ordinal)
                JOIN sources s ON s.source_id=t.source_id AND s.current_version=t.version
                WHERE s.active=1 AND s.deleted=0 AND t.term IN ({placeholders})'''
        with self.store._lock:rows=self.store._db.execute(sql,tuple(query)).fetchall()
        groups={}
        for row in rows:
            key=(row['source_id'],row['version'],row['ordinal'])
            entry=groups.setdefault(key,{'row':row,'hits':set(),'weight':0.})
            entry['hits'].add(row['term']);entry['weight']+=1+math.log1p(row['count'])
        candidates=[]
        for key,item in groups.items():
            coverage=len(item['hits'])/len(query)
            if coverage<.5:continue
            record=self.store.get_source(key[0],key[1]);source=record.entry
            if record.kind=='knowledge' and 'watermark_only_or_missing_body' in source.metadata.get('flags',()):continue
            if filters.kind!='all' and record.kind!=filters.kind:continue
            if not set(map(normalized,filters.tags)).issubset(set(map(normalized,source.tags))):continue
            for field in ('equipment','conditions'):
                selected=normalized(getattr(filters,field));actual=normalized(getattr(source,field))
                if selected and (record.kind=='case' or actual) and actual!=selected:break
            else:
                row=item['row'];snippet=row['body'];content_hash=hashlib.sha256(snippet.encode('utf-8')).hexdigest()
                # An image-only page has provenance but no text evidence. Its
                # title must not produce an empty remote citation.
                if not snippet.strip():continue
                evidence=Evidence(('K' if record.kind=='knowledge' else 'C')+'_'+key[0]+f'_v{key[1]}_p{key[2]}',key[0],key[1],source.title,snippet,record.kind,content_hash,
                    source.source_path,row['start'],row['end'],coverage+item['weight']*.01,
                    {**(dict(source.metadata) if record.kind=='knowledge' else {}),
                     'source_locator':source.source_locator if record.kind=='knowledge' else '',
                     'source_content_hash':record.content_hash,'source_state':record.source_state,'tags':source.tags,
                     'equipment':source.equipment,'conditions':source.conditions,'source_version':source.source_version})
                candidates.append(evidence)
        candidates.sort(key=lambda item:(-item.score,item.source_id,item.start))
        seen=set();output=[]
        for evidence in candidates:
            if evidence.content_hash in seen:continue
            seen.add(evidence.content_hash);output.append(evidence)
            if len(output)>=limit:break
        return tuple(output)
