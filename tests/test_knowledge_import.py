import importlib,json

def test_readonly_text_and_mapped_case_import_preview_and_partial_errors(tmp_path):
    module=importlib.import_module('autoacoustics.knowledge.importers')
    from autoacoustics.knowledge.store import KnowledgeStore
    text=tmp_path/'中文知识.md';text.write_bytes('# 座椅电机\n\n测点距离需要记录。'.encode('gb18030'))
    before=text.read_bytes();preview=module.preview_import(text)
    assert preview.encoding=='gb18030' and preview.rows[0].entry.title=='座椅电机'
    csv=tmp_path/'案例.csv';csv.write_text('编号,标题,设备,观察,结果\n1,启动观察,EM1,启动异响,待复测\n2,,EM2,无题案例,\n',encoding='utf-8-sig')
    cases=module.preview_import(csv,kind='case',mapping={'source_record_id':'编号','title':'标题','equipment':'设备','observation':'观察','outcome':'结果'})
    assert cases.rows[0].status=='ready' and cases.rows[1].status=='failed'
    with KnowledgeStore(tmp_path/'store.sqlite') as store:
        assert module.import_preview(store,preview)[0].status=='added'
        assert module.import_preview(store,preview)[0].status=='duplicate'
        results=module.import_preview(store,cases)
        assert [row.status for row in results]==['added','failed']
    assert text.read_bytes()==before

def test_case_json_requires_declared_structure_and_no_fabricated_outcomes(tmp_path):
    from autoacoustics.knowledge.importers import preview_import
    path=tmp_path/'cases.json';path.write_text(json.dumps([{'title':'Case field test','equipment':'EM2','observation':'noise'}]),encoding='utf-8')
    preview=preview_import(path,kind='case')
    assert preview.rows[0].entry.outcome=='' and preview.rows[0].entry.actions==''
