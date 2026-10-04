"""Read-only source previews; explicit case columns, no diagnostic inference."""
from dataclasses import fields
import csv,hashlib,io,json,re
from pathlib import Path
from .models import KnowledgeEntry,HistoryCase,ImportRow,ImportPreview,ImportOutcome,KnowledgeError

MAX_SOURCE_BYTES=10*1024*1024

def _read(path,encoding=None):
    path=Path(path).resolve()
    try:
        if path.stat().st_size>MAX_SOURCE_BYTES:raise KnowledgeError('资料超过10 MiB，请拆分后导入。')
        data=path.read_bytes()
    except OSError as error:raise KnowledgeError('资料无法读取，请检查源文件路径。') from error
    for candidate in ([encoding] if encoding else ['utf-8-sig','gb18030']):
        try:
            text=data.decode(candidate)
            if '\x00' in text:raise KnowledgeError('资料含二进制空字节；请导出为文本资料。')
            return path,text,candidate,hashlib.sha256(data).hexdigest()
        except (UnicodeDecodeError,LookupError):continue
    raise KnowledgeError('文本编码无法确认，请明确选择 UTF-8 或 GB18030 编码。')

def preview_import(path,kind=None,mapping=None,encoding=None):
    suffix=Path(path).suffix.lower()
    if suffix not in ('.md','.markdown','.txt','.csv','.json'):raise KnowledgeError('知识资料支持 Markdown/TXT；历史案例支持 CSV/JSON。旧数据库需单独映射。')
    actual_kind=kind or ('case' if suffix in ('.csv','.json') else 'knowledge')
    path,text,encoding,file_hash=_read(path,encoding)
    if actual_kind=='knowledge':
        if suffix not in ('.md','.markdown','.txt'):raise KnowledgeError('知识正文须为 Markdown 或 TXT。')
        heading=re.search(r'^\s*#{1,6}\s+(.+)$',text,re.M)
        title=heading.group(1).strip() if heading else path.stem
        entry=KnowledgeEntry(title,text,source_path=path,source_version=file_hash)
        return ImportPreview(path,'knowledge',encoding,(ImportRow(1,entry),))
    if actual_kind!='case':raise KnowledgeError('资料类型须为 knowledge 或 case。')
    if suffix=='.csv':
        reader=csv.DictReader(io.StringIO(text));columns=tuple(reader.fieldnames or ())
        if not mapping:raise KnowledgeError('CSV 案例须先映射标题、设备、观察等字段；应用不猜测列含义。')
        records=list(reader)
    elif suffix=='.json':
        try:records=json.loads(text)
        except json.JSONDecodeError as error:raise KnowledgeError('案例 JSON 语法无效。') from error
        if isinstance(records,dict) and set(records)=={'cases'}:records=records['cases']
        if not isinstance(records,list):raise KnowledgeError('案例 JSON 须为条目数组或仅包含 cases 数组的对象。')
        columns=tuple(dict.fromkeys(key for record in records if isinstance(record,dict) for key in record))
        mapping=mapping or {field.name:field.name for field in fields(HistoryCase) if field.name in columns}
    else:raise KnowledgeError('案例文件须为 CSV 或 JSON。')
    allowed={field.name for field in fields(HistoryCase)}-{'entry_id','updated_at','source_path','source_version','analysis_snapshot'}
    if 'title' not in mapping or any(key not in allowed for key in mapping):raise KnowledgeError('案例映射须包含 title，且只能使用明确的案例文字字段。')
    unknown=[column for column in mapping.values() if column not in columns]
    if unknown:raise KnowledgeError('映射列不存在：'+', '.join(map(str,unknown)))
    output=[]
    for row,record in enumerate(records,2 if suffix=='.csv' else 1):
        try:
            if not isinstance(record,dict):raise KnowledgeError('案例条目须为字段对象。')
            values={key:record.get(column,'') for key,column in mapping.items()}
            if 'tags' in values:
                value=values['tags'];values['tags']=tuple(x.strip() for x in value.split(',') if x.strip()) if isinstance(value,str) else tuple(value or ())
            for key in tuple(values):
                if key!='tags':
                    value=values[key]
                    if value is None:value=''
                    if not isinstance(value,(str,int,float)):raise KnowledgeError('案例字段含不支持的嵌套对象。')
                    values[key]=str(value).strip()
            values.setdefault('source_record_id',str(row))
            values.update(source_path=path,source_version=file_hash)
            output.append(ImportRow(row,HistoryCase(**values)))
        except (KnowledgeError,ValueError,TypeError) as error:output.append(ImportRow(row,status='failed',error=str(error)))
    return ImportPreview(path,'case',encoding,tuple(output),columns)

def import_preview(store,preview):
    output=[]
    for row in preview.rows:
        if row.status!='ready' or row.entry is None:
            output.append(ImportOutcome(row.row,'failed',error=row.error));continue
        try:
            saved=store.add_source(row.entry) if preview.kind=='knowledge' else store.add_case(row.entry)
            output.append(ImportOutcome(row.row,saved.status,saved.source_id,saved.version))
        except Exception as error:output.append(ImportOutcome(row.row,'failed',error=str(error)))
    return tuple(output)
