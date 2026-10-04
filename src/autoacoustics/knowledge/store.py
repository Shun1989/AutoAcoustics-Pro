"""SQLite versions and citation snapshots. Never guess or overwrite an old DB."""
from __future__ import annotations
from dataclasses import asdict
from contextlib import contextmanager,nullcontext
from datetime import datetime,timezone
import hashlib,json,sqlite3,threading
from pathlib import Path
from uuid import uuid4
from ..model import to_plain
from .models import KnowledgeEntry,HistoryCase,SourceRecord,SaveResult,PurgeImpact,KnowledgeError
from .text import chunks,tokens

SCHEMA_VERSION=1
SCHEMA_COLUMNS={
    'sources':{'source_id','kind','source_key','current_version','active','deleted'},
    'versions':{'source_id','version','payload','content_hash','created_at'},
    'chunks':{'source_id','version','ordinal','start','end','body'},
    'terms':{'term','source_id','version','ordinal','count'},
    'sessions':{'session_id','title','created_at'},
    'turns':{'turn_id','session_id','question','answer','evidence_json','scope_fingerprint','status','model','provider_name','created_at'},
    'settings':{'name','value'},
}

def now():return datetime.now(timezone.utc).isoformat()
def canonical(value):return json.dumps(to_plain(value),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()

class KnowledgeStore:
    def __init__(self,path,readonly=False):
        self.path=Path(path).resolve();self.readonly=readonly;self._lock=threading.RLock();self._db=None;self._batch_depth=0
        if readonly and not self.path.is_file():raise KnowledgeError('只读知识库文件不存在。')
        if not readonly:self.path.parent.mkdir(parents=True,exist_ok=True)
        try:
            self._db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,check_same_thread=False) if readonly else sqlite3.connect(self.path,check_same_thread=False)
            self._db.row_factory=sqlite3.Row;self._db.execute('PRAGMA foreign_keys=ON');self._db.execute('PRAGMA busy_timeout=5000')
            current=self._db.execute('PRAGMA user_version').fetchone()[0]
            tables={row[0] for row in self._db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if current not in (0,SCHEMA_VERSION):raise KnowledgeError(f'不支持知识库版本 {current}；原文件已保留。')
            if current==0 and tables:
                # A pre-release v0 with this exact public structure may migrate.
                if not set(SCHEMA_COLUMNS).issubset(tables):raise KnowledgeError('此数据库没有本应用的来源版本结构；请另行映射导入，原库已保留。')
            if tables:
                for table,expected in SCHEMA_COLUMNS.items():
                    actual={row['name'] for row in self._db.execute(f'PRAGMA table_info({table})')}
                    if not expected.issubset(actual):raise KnowledgeError('知识库表结构不完整或并非本应用版本；原文件已保留，请从备份恢复或映射导入。')
            if current==0:
                if readonly:raise KnowledgeError('旧结构需先在可写副本中迁移，原库已保留。')
                self._initialize()
            self.fts_available='source_fts' in {row[0] for row in self._db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not readonly:self._db.execute('PRAGMA journal_mode=WAL');self._db.execute('PRAGMA secure_delete=ON')
        except (sqlite3.Error,OSError,KnowledgeError) as error:
            self.close()
            if isinstance(error,KnowledgeError):raise
            raise KnowledgeError('知识库无法打开，请检查路径或使用备份恢复；原文件未被重建。') from error

    def _initialize(self):
        self._db.executescript('''
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS sources(source_id TEXT PRIMARY KEY,kind TEXT NOT NULL,source_key TEXT UNIQUE NOT NULL,current_version INTEGER NOT NULL,active INTEGER NOT NULL DEFAULT 1,deleted INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS versions(source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,version INTEGER NOT NULL,payload TEXT NOT NULL,content_hash TEXT NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(source_id,version));
            CREATE TABLE IF NOT EXISTS chunks(source_id TEXT NOT NULL,version INTEGER NOT NULL,ordinal INTEGER NOT NULL,start INTEGER NOT NULL,end INTEGER NOT NULL,body TEXT NOT NULL,PRIMARY KEY(source_id,version,ordinal),FOREIGN KEY(source_id,version) REFERENCES versions(source_id,version) ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS terms(term TEXT NOT NULL,source_id TEXT NOT NULL,version INTEGER NOT NULL,ordinal INTEGER NOT NULL,count INTEGER NOT NULL,PRIMARY KEY(term,source_id,version,ordinal),FOREIGN KEY(source_id,version,ordinal) REFERENCES chunks(source_id,version,ordinal) ON DELETE CASCADE);
            CREATE INDEX IF NOT EXISTS terms_source ON terms(source_id,version);
            CREATE TABLE IF NOT EXISTS sessions(session_id TEXT PRIMARY KEY,title TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS turns(turn_id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,question TEXT NOT NULL,answer TEXT NOT NULL,evidence_json TEXT NOT NULL,scope_fingerprint TEXT NOT NULL,status TEXT NOT NULL,model TEXT NOT NULL,provider_name TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS settings(name TEXT PRIMARY KEY,value TEXT NOT NULL);
            PRAGMA user_version=1;
            COMMIT;''')
        try:self._db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS source_fts USING fts5(source_id UNINDEXED,version UNINDEXED,ordinal UNINDEXED,title,body)');self._db.commit()
        except sqlite3.OperationalError:self._db.rollback()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):
        if self._db is not None:self._db.close();self._db=None
    def _writable(self):
        if self.readonly:raise KnowledgeError('此知识库为只读，不能修改。')
        if self._db is None:raise KnowledgeError('知识库已关闭。')

    @contextmanager
    def source_transaction(self):
        """One atomic corpus import, using this connection and its write lock."""
        self._writable()
        with self._lock:
            if self._batch_depth:raise KnowledgeError('资料导入事务不能嵌套。')
            self._batch_depth=1
            try:
                with self._db:yield
            finally:self._batch_depth=0

    def add_source(self,entry):
        if not isinstance(entry,KnowledgeEntry):raise KnowledgeError('须提供 KnowledgeEntry。')
        return self._save('knowledge',entry)
    def add_case(self,case):
        if not isinstance(case,HistoryCase):raise KnowledgeError('须提供 HistoryCase。')
        return self._save('case',case)
    def _save(self,kind,entry):
        self._writable();payload=to_plain(entry)
        if entry.source_path is not None:payload['source_path']=str(entry.source_path.resolve())
        payload['entry_id']='';payload['updated_at']=''
        # Preserve legacy payload hashes when new optional provenance is absent.
        if kind=='knowledge':
            if not entry.source_locator:payload.pop('source_locator',None)
            if not entry.metadata:payload.pop('metadata',None)
        fingerprint=digest(payload)
        if entry.entry_id:source_key=None
        elif entry.source_path is not None and (kind=='knowledge' or entry.source_record_id):
            source_key=kind+':'+str(entry.source_path.resolve()).casefold()+(':'+entry.source_record_id if kind=='case' else (':locator:'+entry.source_locator if entry.source_locator else ''))
        else:source_key=kind+':manual:'+uuid4().hex
        with self._lock,(nullcontext() if self._batch_depth else self._db):
            row=self._db.execute('SELECT * FROM sources WHERE source_id=?' if entry.entry_id else 'SELECT * FROM sources WHERE source_key=?',(entry.entry_id or source_key,)).fetchone()
            if entry.entry_id and row is None:raise KnowledgeError('待更新的来源不存在。')
            if row and row['kind']!=kind:raise KnowledgeError('不能把知识版本改为历史案例，或反向转换。')
            if row:
                previous=self._db.execute('SELECT content_hash FROM versions WHERE source_id=? AND version=?',(row['source_id'],row['current_version'])).fetchone()
                if previous['content_hash']==fingerprint:return SaveResult(row['source_id'],row['current_version'],'duplicate')
                source_id=row['source_id'];version=row['current_version']+1
                self._db.execute('UPDATE sources SET current_version=?,active=1,deleted=0 WHERE source_id=?',(version,source_id))
            else:
                source_id=uuid4().hex;version=1
                self._db.execute('INSERT INTO sources(source_id,kind,source_key,current_version) VALUES(?,?,?,?)',(source_id,kind,source_key,version))
            self._db.execute('INSERT INTO versions VALUES(?,?,?,?,?)',(source_id,version,canonical(payload),fingerprint,now()))
            record=SourceRecord(source_id,version,kind,entry,fingerprint)
            # Extracted PDF text commonly ends in a newline. The chunk regex
            # requires a non-whitespace final character; trim only its input,
            # preserving stored body and every original span offset.
            spans=list(chunks(record.text.rstrip())) or [(0,0,'')]
            for ordinal,(start,end,body) in enumerate(spans):
                self._db.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?)',(source_id,version,ordinal,start,end,body))
                self._db.executemany('INSERT INTO terms VALUES(?,?,?,?,?)',((term,source_id,version,ordinal,count) for term,count in tokens(entry.title+'\n'+body).items()))
                if self.fts_available:self._db.execute('INSERT INTO source_fts VALUES(?,?,?,?,?)',(source_id,version,ordinal,entry.title,body))
            return SaveResult(source_id,version,'updated' if row else 'added')

    def get_source(self,source_id,version=None):
        with self._lock:
            row=self._db.execute('SELECT s.*,v.version,v.payload,v.content_hash,v.created_at FROM sources s JOIN versions v ON v.source_id=s.source_id AND v.version=COALESCE(?,s.current_version) WHERE s.source_id=?',(version,source_id)).fetchone()
            if row is None:raise KnowledgeError('来源或该内容版本不存在；可能已彻底删除。')
            payload=json.loads(row['payload']);payload['entry_id']=source_id;payload['updated_at']=row['created_at']
            cls=KnowledgeEntry if row['kind']=='knowledge' else HistoryCase
            return SourceRecord(source_id,row['version'],row['kind'],cls(**payload),row['content_hash'],bool(row['active']),bool(row['deleted']))

    def list_sources(self,include_inactive=False,include_deleted=False):
        conditions=[]
        if not include_inactive:conditions.append('active=1')
        if not include_deleted:conditions.append('deleted=0')
        sql='SELECT source_id FROM sources'+(' WHERE '+' AND '.join(conditions) if conditions else '')+' ORDER BY rowid'
        with self._lock:return tuple(self.get_source(row[0]) for row in self._db.execute(sql).fetchall())
    def active_fingerprint(self):
        with self._lock:return digest([tuple(row) for row in self._db.execute('SELECT source_id,current_version FROM sources WHERE active=1 AND deleted=0 ORDER BY source_id')])
    def set_active(self,source_id,active):
        self._writable()
        with self._lock,self._db:
            if self._db.execute('UPDATE sources SET active=? WHERE source_id=? AND deleted=0',(int(bool(active)),source_id)).rowcount!=1:raise KnowledgeError('来源不存在或已删除。')
    def delete_source(self,source_id,permanent=False):
        if permanent:return self.purge_source(source_id)
        self._writable()
        with self._lock,self._db:self._db.execute('UPDATE sources SET deleted=1,active=0 WHERE source_id=?',(source_id,))
    def preview_purge(self,source_id):
        with self._lock:
            versions=self._db.execute('SELECT count(*) FROM versions WHERE source_id=?',(source_id,)).fetchone()[0]
            sessions=set()
            for row in self._db.execute('SELECT session_id,evidence_json FROM turns'):
                if any(item.get('source_id')==source_id for item in json.loads(row['evidence_json'])):sessions.add(row['session_id'])
            return PurgeImpact(source_id,versions,tuple(sorted(sessions)))
    def purge_source(self,source_id):
        self._writable();impact=self.preview_purge(source_id)
        with self._lock,self._db:
            for session in impact.session_ids:self._db.execute('DELETE FROM sessions WHERE session_id=?',(session,))
            if self.fts_available:self._db.execute('DELETE FROM source_fts WHERE source_id=?',(source_id,))
            self._db.execute('DELETE FROM sources WHERE source_id=?',(source_id,))
        with self._lock:
            self._db.execute('PRAGMA wal_checkpoint(TRUNCATE)');self._db.execute('VACUUM');self._db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        return impact
    def create_session(self,title='知识问答'):
        self._writable();identity=uuid4().hex
        with self._lock,self._db:self._db.execute('INSERT INTO sessions VALUES(?,?,?)',(identity,title,now()))
        return identity
    def list_sessions(self):
        with self._lock:return tuple(dict(row) for row in self._db.execute('SELECT * FROM sessions ORDER BY created_at DESC'))
    def delete_session(self,session_id):
        self._writable()
        with self._lock,self._db:self._db.execute('DELETE FROM sessions WHERE session_id=?',(session_id,))
    def append_turn(self,session_id,question,answer,evidence,*,scope_fingerprint,status,model='',provider_name=''):
        self._writable()
        with self._lock,self._db:
            cursor=self._db.execute('INSERT INTO turns(session_id,question,answer,evidence_json,scope_fingerprint,status,model,provider_name,created_at) VALUES(?,?,?,?,?,?,?,?,?)',
                (session_id,question,answer,canonical(evidence),scope_fingerprint,status,model,provider_name,now()))
            return cursor.lastrowid
    def commit_grounded_turn(self,session_id,question,answer,evidence,*,scope_fingerprint,status,model='',provider_name=''):
        """Cross-window source validation and answer persistence in one write lock."""
        self._writable()
        with self._lock:
            try:
                self._db.execute('BEGIN IMMEDIATE')
                for item in evidence:
                    if item['kind']=='analysis':continue
                    row=self._db.execute('SELECT s.current_version,s.active,s.deleted,v.content_hash FROM sources s JOIN versions v ON v.source_id=s.source_id AND v.version=s.current_version WHERE s.source_id=?',(item['source_id'],)).fetchone()
                    expected=item.get('metadata',{}).get('source_content_hash')
                    if not row or not row['active'] or row['deleted'] or row['current_version']!=item['version'] or row['content_hash']!=expected:
                        self._db.rollback();return None,'stale_sources'
                if session_id and not self._db.execute('SELECT 1 FROM sessions WHERE session_id=?',(session_id,)).fetchone():
                    self._db.rollback();return None,'stale_session'
                if not session_id:
                    session_id=uuid4().hex;self._db.execute('INSERT INTO sessions VALUES(?,?,?)',(session_id,question[:80],now()))
                self._db.execute('INSERT INTO turns(session_id,question,answer,evidence_json,scope_fingerprint,status,model,provider_name,created_at) VALUES(?,?,?,?,?,?,?,?,?)',
                    (session_id,question,answer,canonical(evidence),scope_fingerprint,status,model,provider_name,now()))
                self._db.commit();return session_id,'ok'
            except Exception:self._db.rollback();raise
    def history(self,session_id,scope_fingerprint=None,limit=None):
        sql='SELECT * FROM turns WHERE session_id=?';params=[session_id]
        if scope_fingerprint is not None:sql+=' AND scope_fingerprint=?';params.append(scope_fingerprint)
        sql+=' ORDER BY turn_id DESC'
        if limit is not None:sql+=' LIMIT ?';params.append(max(0,int(limit)))
        with self._lock:
            rows=self._db.execute(sql,params).fetchall();output=[]
            for row in reversed(rows):
                item=dict(row);item['evidence']=json.loads(item.pop('evidence_json'));output.append(item)
            return tuple(output)
    def save_setting(self,name,value):
        self._writable()
        if name in {'api_key','secret','token'}:raise KnowledgeError('密钥不得保存到知识数据库。')
        with self._lock,self._db:self._db.execute('INSERT INTO settings VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value=excluded.value',(name,canonical(value)))
    def load_setting(self,name,default=None):
        with self._lock:row=self._db.execute('SELECT value FROM settings WHERE name=?',(name,)).fetchone()
        return json.loads(row[0]) if row else default
