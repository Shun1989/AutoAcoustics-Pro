from __future__ import annotations
from dataclasses import dataclass,field
from pathlib import Path
from typing import Mapping
from ..model import FrozenMap,AcousticError

class KnowledgeError(AcousticError):
    code='knowledge_error'

@dataclass(frozen=True)
class KnowledgeEntry:
    title:str
    body:str
    tags:tuple[str,...]=()
    equipment:str=''
    conditions:str=''
    source_path:Path|None=None
    source_version:str=''
    entry_id:str=''
    updated_at:str=''
    source_locator:str=''
    metadata:Mapping=field(default_factory=FrozenMap)
    def __post_init__(self):
        if not isinstance(self.source_locator,str):raise KnowledgeError('知识来源位置须为文字。')
        object.__setattr__(self,'metadata',FrozenMap(self.metadata))
        image_page=bool(self.source_locator and (self.metadata.get('render_path') or self.metadata.get('page')))
        if not isinstance(self.title,str) or not self.title.strip() or not isinstance(self.body,str) or (not self.body.strip() and not image_page):raise KnowledgeError('知识条目须填写标题和正文；图像页须保留原页图。')
        object.__setattr__(self,'tags',tuple(dict.fromkeys(str(tag).strip() for tag in self.tags if str(tag).strip())))
        if self.source_path is not None:object.__setattr__(self,'source_path',Path(self.source_path))

@dataclass(frozen=True)
class HistoryCase:
    title:str
    equipment:str=''
    conditions:str=''
    observation:str=''
    actions:str=''
    outcome:str=''
    tags:tuple[str,...]=()
    source_path:Path|None=None
    source_version:str=''
    source_record_id:str=''
    analysis_ref:str=''
    analysis_snapshot:Mapping=field(default_factory=FrozenMap)
    entry_id:str=''
    updated_at:str=''
    def __post_init__(self):
        if not isinstance(self.title,str) or not self.title.strip():raise KnowledgeError('历史案例须填写标题。')
        for name in ('equipment','conditions','observation','actions','outcome'):
            if not isinstance(getattr(self,name),str):raise KnowledgeError('案例字段须为文字。')
        object.__setattr__(self,'tags',tuple(dict.fromkeys(str(tag).strip() for tag in self.tags if str(tag).strip())))
        object.__setattr__(self,'analysis_snapshot',FrozenMap(self.analysis_snapshot))
        if self.source_path is not None:object.__setattr__(self,'source_path',Path(self.source_path))

@dataclass(frozen=True)
class SearchFilters:
    kind:str='all'
    equipment:str=''
    conditions:str=''
    tags:tuple[str,...]=()
    def __post_init__(self):
        if self.kind not in ('all','knowledge','case'):raise KnowledgeError('检索范围须为知识、案例或全部。')
        object.__setattr__(self,'tags',tuple(self.tags))

@dataclass(frozen=True)
class SaveResult:
    source_id:str
    version:int
    status:str

@dataclass(frozen=True)
class SourceRecord:
    source_id:str
    version:int
    kind:str
    entry:KnowledgeEntry|HistoryCase
    content_hash:str
    active:bool=True
    deleted:bool=False
    @property
    def text(self):
        if self.kind=='knowledge':return self.entry.body
        item=self.entry
        return '\n'.join(f'{label}：{value}' for label,value in [('设备',item.equipment),('工况',item.conditions),('观察',item.observation),('已采取措施',item.actions),('处理结果',item.outcome)] if value)
    @property
    def source_state(self):
        path=self.entry.source_path
        return 'manual' if path is None else 'available' if path.is_file() else 'missing'

@dataclass(frozen=True)
class Evidence:
    evidence_id:str
    source_id:str
    version:int
    title:str
    snippet:str
    kind:str
    content_hash:str
    source_path:Path|None=None
    start:int=0
    end:int=0
    score:float=0.
    metadata:Mapping=field(default_factory=FrozenMap)
    def __post_init__(self):object.__setattr__(self,'metadata',FrozenMap(self.metadata))

@dataclass(frozen=True)
class PurgeImpact:
    source_id:str
    version_count:int
    session_ids:tuple[str,...]=()

@dataclass(frozen=True)
class ImportRow:
    row:int
    entry:KnowledgeEntry|HistoryCase|None=None
    status:str='ready'
    error:str=''

@dataclass(frozen=True)
class ImportPreview:
    path:Path
    kind:str
    encoding:str
    rows:tuple[ImportRow,...]
    columns:tuple[str,...]=()

@dataclass(frozen=True)
class ImportOutcome:
    row:int
    status:str
    source_id:str=''
    version:int=0
    error:str=''
