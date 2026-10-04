import importlib

def test_chinese_retrieval_versions_equipment_filters_and_no_basis(tmp_path):
    m=importlib.import_module('autoacoustics.knowledge.models')
    from autoacoustics.knowledge.store import KnowledgeStore
    from autoacoustics.knowledge.search import SearchEngine
    with KnowledgeStore(tmp_path/'store.sqlite') as store:
        source=store.add_source(m.KnowledgeEntry('蜗杆润滑记录','蜗杆应使用指定润滑脂；本记录不证明其他机构的故障。',tags=('润滑',)))
        store.add_case(m.HistoryCase('EM2 换向',equipment='EM2',conditions='loaded',observation='换向电流尖峰',actions='检查线束',outcome='等待复测'))
        engine=SearchEngine(store)
        evidence=engine.search_local('蜗杆为什么要润滑',m.SearchFilters())
        assert evidence and evidence[0].source_id==source.source_id
        assert store.get_source(source.source_id,1).text[evidence[0].start:evidence[0].end]==evidence[0].snippet
        assert not engine.search_local('换向电流',m.SearchFilters(kind='case',equipment='EM1'))
        assert not engine.search_local('座椅电机的股票价格',m.SearchFilters())
        updated=store.add_source(m.KnowledgeEntry('蜗杆润滑记录','蜗杆润滑脂已更换，原记录被更新。',tags=('润滑',),entry_id=source.source_id))
        assert all(e.version==updated.version for e in engine.search_local('蜗杆润滑',m.SearchFilters()))
        store.set_active(source.source_id,False)
        assert not engine.search_local('蜗杆润滑',m.SearchFilters())
