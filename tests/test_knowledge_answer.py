import importlib,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from autoacoustics.model import AnalysisResult,AnalysisSummary,AnalysisSettings,AnalysisDetail,MeasurementContext,MetricResult,CalibrationProfile

def analysis(calibrated=True):
    return AnalysisResult('result-snapshot-id','file-hash',Path('真实录音.wav'),AnalysisSettings(channel=0,start=12000,end=24000),MeasurementContext(specimen='EM1'),
        CalibrationProfile('记录配置','声校准器证书记录',2,'FS') if calibrated else None,
        AnalysisSummary({'LAeq':MetricResult(67.4 if calibrated else None,'dB re 20 µPa','ok' if calibrated else 'uncalibrated'),
            'RMS':MetricResult(.03,'Pa' if calibrated else 'FS')}),
        AnalysisDetail({'waveform':np.zeros(100000),'wave_time':np.arange(100000)/48000}),
        {'sample_rate':48000,'source_unit':'FS','unit':'Pa' if calibrated else 'FS','channel_name':'mic1'})

def service(tmp_path):
    a=importlib.import_module('autoacoustics.knowledge.answer')
    from autoacoustics.knowledge.store import KnowledgeStore
    store=KnowledgeStore(tmp_path/'store.sqlite')
    return a,store,a.AnswerService(store)

def test_current_analysis_is_independent_evidence_and_never_sends_audio(tmp_path):
    a,store,worker=service(tmp_path)
    try:
        preview=worker.preview('本段的 LAeq 和校准来源是什么？',analysis=analysis())
        assert len(preview.evidence)==1 and preview.evidence[0].kind=='analysis'
        assert '67.4' in preview.facts_card and '声校准器证书记录' in preview.facts_card
        messages=worker.messages(preview)
        text=json.dumps(messages,ensure_ascii=False)
        assert 'waveform' not in text and 'wave_time' not in text
        assert '67.4' in text and '原样本' in preview.facts_card
        uncalibrated=worker.preview('LAeq 是多少？',analysis=analysis(False))
        assert '未校准' in uncalibrated.facts_card and '67.4' not in uncalibrated.facts_card
        stale=worker.preview('LAeq 是多少？',analysis=replace(analysis(),provenance={'source_state':'changed'}))
        assert not stale.evidence and any('过期' in note for note in stale.warnings)
    finally:store.close()

def test_grounded_answer_references_unknown_ids_and_no_evidence(tmp_path):
    a,store,worker=service(tmp_path)
    from autoacoustics.knowledge.models import KnowledgeEntry
    from autoacoustics.knowledge.provider import ProviderReply,ProviderConfig
    calls=[]
    class FixtureClient:
        def __init__(self,config):pass
        def complete(self,messages,cancel=None):
            calls.append(messages);payload=json.loads(messages[-1]['content']);identity=payload['evidence'][0]['evidence_id']
            return ProviderReply('ok',f'这是协议测试回答 [{identity}]','fixture')
    worker.client_factory=FixtureClient
    try:
        store.add_source(KnowledgeEntry('轴承安装记录','轴承安装方向需要按工装记录核对。'))
        preview=worker.preview('轴承安装方向')
        result=worker.ask(preview,ProviderConfig('Fixture','http://127.0.0.1:1/v1','fixture'))
        assert result.status=='ok' and result.cited_ids==(preview.evidence[0].evidence_id,)
        assert len(store.history(result.session_id))==1
        checked=a.validate_citations('没有引用的测试段落。\n\n另一段 [not_sent]',preview.evidence)
        assert not checked.verified and checked.unknown_ids==('not_sent',)
        empty=worker.preview('股票价格预算')
        no_answer=worker.ask(empty,ProviderConfig('Fixture','http://127.0.0.1:1/v1','fixture'))
        assert no_answer.status=='no_evidence' and not no_answer.text and len(calls)==1
    finally:store.close()

def test_updated_sources_cannot_use_old_preview_and_purged_body_not_restored(tmp_path):
    a,store,worker=service(tmp_path)
    from autoacoustics.knowledge.models import KnowledgeEntry
    from autoacoustics.knowledge.provider import ProviderConfig
    try:
        saved=store.add_source(KnowledgeEntry('润滑来源','蜗杆润滑记录。'))
        preview=worker.preview('蜗杆润滑')
        store.add_source(KnowledgeEntry('润滑来源','蜗杆润滑记录已经变化。',entry_id=saved.source_id))
        result=worker.ask(preview,ProviderConfig('Fixture','http://127.0.0.1:1/v1','fixture'))
        assert result.status=='stale_sources' and not result.text
    finally:store.close()

def test_history_scope_budget_case_draft_and_no_diagnosis(tmp_path):
    a,store,worker=service(tmp_path)
    from autoacoustics.knowledge.models import KnowledgeEntry,SearchFilters
    try:
        for index in range(10):store.add_source(KnowledgeEntry('轴承安装'+str(index),'轴承安装方向需核对。'+str(index)*1000))
        preview=worker.preview('轴承安装方向',char_limit=1500)
        assert len(preview.evidence)<=6 and sum(len(message['content']) for message in worker.messages(preview))<=1500
        session=store.create_session('History budget fixture')
        for index in range(10):store.append_turn(session,'轴承安装方向？','回答 '+str(index),(),scope_fingerprint=preview.scope_fingerprint,status='ok',model='fixture')
        next_preview=worker.preview('轴承安装方向',session_id=session,char_limit=1500)
        assert len(next_preview.history_messages)<=6
        switched=worker.preview('轴承安装方向',SearchFilters(equipment='other'),session_id=session)
        assert not switched.history_messages
        draft=a.case_draft_from_analysis(analysis())
        assert draft.observation==draft.actions==draft.outcome==''
        assert draft.analysis_ref=='result-snapshot-id' and draft.analysis_snapshot['metrics']['LAeq']['value']==67.4
    finally:store.close()

def test_unknown_source_origin_is_preserved_as_unknown_in_factcard(tmp_path):
    a,store,worker=service(tmp_path)
    try:
        result=replace(analysis(),provenance={'sample_rate':48000,'unit':'Pa','time_origin_seconds':None,'source_time_origin_known':False,'event_start_time_seconds':.25,'event_end_time_seconds':.5})
        facts=a.analysis_facts(result)
        assert facts['provenance']['source_time_origin_known'] is False
        preview=worker.preview('源时间起点是什么？',analysis=result)
        assert '源时间起点未知' in preview.facts_card and 'None s' not in preview.facts_card
        payload=str(worker.messages(preview));assert '0.0 s' not in payload
    finally:store.close()

def test_declared_pa_source_uses_recorded_source_without_fake_profile(tmp_path):
    a,store,worker=service(tmp_path)
    try:
        result=replace(analysis(),calibration=None,provenance={'sample_rate':48000,'source_unit':'Pa','unit':'Pa','calibration':{'source':'文件声明 Pa','coefficient':1.,'input_unit':'Pa','scaled_once':True}})
        preview=worker.preview('校准来源是什么？',analysis=result)
        assert '文件声明 Pa' in preview.facts_card and '未校准' not in preview.facts_card
        assert a.analysis_facts(result)['calibration']['mode']=='declared_pa'
    finally:store.close()
