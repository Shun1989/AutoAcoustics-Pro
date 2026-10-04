import importlib,sqlite3,hashlib
import pytest

def contracts():
    return importlib.import_module('autoacoustics.knowledge.models'),importlib.import_module('autoacoustics.knowledge.store')

def test_empty_store_versioned_sources_restart_and_purge(tmp_path):
    m,s=contracts();path=tmp_path/'知识库.sqlite'
    with s.KnowledgeStore(path) as store:
        first=store.add_source(m.KnowledgeEntry('蜗杆润滑','蜗杆使用油脂 A。',source_path=tmp_path/'参考.md'))
        duplicate=store.add_source(m.KnowledgeEntry('蜗杆润滑','蜗杆使用油脂 A。',source_path=tmp_path/'参考.md'))
        changed=store.add_source(m.KnowledgeEntry('蜗杆润滑','蜗杆使用油脂 B。',source_path=tmp_path/'参考.md'))
        assert first.status=='added' and duplicate.status=='duplicate'
        assert changed.source_id==first.source_id and changed.version==2
        assert store.get_source(first.source_id,1).entry.body=='蜗杆使用油脂 A。'
        store.set_active(first.source_id,False)
    with s.KnowledgeStore(path) as store:
        assert store.get_source(first.source_id).entry.body=='蜗杆使用油脂 B。'
        assert store.list_sources()==()
        assert len(store.list_sources(include_inactive=True))==1
        store.delete_source(first.source_id)
        assert store.list_sources(include_inactive=True)==()
        impact=store.preview_purge(first.source_id)
        assert impact.version_count==2
        store.purge_source(first.source_id)
        with pytest.raises(s.KnowledgeError):store.get_source(first.source_id,1)

def test_corrupt_and_unknown_database_preserved_and_readonly_no_writes(tmp_path):
    m,s=contracts();bad=tmp_path/'坏库.sqlite';bad.write_bytes(b'keep-original-not-a-database')
    old=bad.read_bytes()
    with pytest.raises(s.KnowledgeError):s.KnowledgeStore(bad)
    assert bad.read_bytes()==old
    known=tmp_path/'只读.sqlite'
    with s.KnowledgeStore(known) as store:result=store.add_case(m.HistoryCase('真实记录字段软件测试',equipment='EM1',observation='启动尖峰',actions='检查安装',outcome='尚无结论'))
    with s.KnowledgeStore(known,readonly=True) as store:
        assert store.get_source(result.source_id).kind=='case'
        with pytest.raises(s.KnowledgeError):store.add_source(m.KnowledgeEntry('禁止写入','数据'))
    future=tmp_path/'未来.sqlite'
    db=sqlite3.connect(future);db.execute('PRAGMA user_version=99');db.close()
    before=future.read_bytes()
    with pytest.raises(s.KnowledgeError):s.KnowledgeStore(future)
    assert future.read_bytes()==before

def test_permanent_delete_also_removes_sessions_with_body_snapshots(tmp_path):
    m,s=contracts()
    with s.KnowledgeStore(tmp_path/'store.sqlite') as store:
        saved=store.add_source(m.KnowledgeEntry('轴承测试材料','轴承安装方向必须记录。'))
        session=store.create_session('记录测试')
        snapshot={'evidence_id':'K1','source_id':saved.source_id,'snippet':'轴承安装方向必须记录。'}
        store.append_turn(session,'问','答',(snapshot,),scope_fingerprint='scope',status='ok',model='local-test',provider_name='protocol fixture')
        assert store.preview_purge(saved.source_id).session_ids==(session,)
        store.purge_source(saved.source_id)
        assert store.list_sessions()==()
def test_incomplete_v0_schema_is_not_migrated_or_overwritten(tmp_path):
    import sqlite3,hashlib,pytest
    from autoacoustics.knowledge.store import KnowledgeStore,KnowledgeError
    path=tmp_path/'incomplete.sqlite'
    with KnowledgeStore(path):pass
    with sqlite3.connect(path) as db:
        db.execute('DROP TABLE versions');db.execute('CREATE TABLE versions(unexpected TEXT)');db.execute('PRAGMA user_version=0')
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(KnowledgeError):KnowledgeStore(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before

def test_known_prerelease_v0_migration_preserves_identity_and_readonly_original(tmp_path):
    import sqlite3,hashlib,pytest
    from autoacoustics.knowledge.store import KnowledgeStore,KnowledgeError
    from autoacoustics.knowledge.models import KnowledgeEntry
    path=tmp_path/'known-v0.sqlite'
    with KnowledgeStore(path) as store:saved=store.add_source(KnowledgeEntry('旧版来源','需要保留的旧版正文。'))
    with sqlite3.connect(path) as db:db.execute('PRAGMA user_version=0')
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(KnowledgeError):KnowledgeStore(path,readonly=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    with KnowledgeStore(path) as store:
        record=store.get_source(saved.source_id)
        assert record.version==1 and record.entry.body=='需要保留的旧版正文。'
        assert store._db.execute('PRAGMA user_version').fetchone()[0]==1
