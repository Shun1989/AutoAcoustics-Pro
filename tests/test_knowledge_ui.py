import os,importlib,json,threading,time
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QRect,QPoint
from autoacoustics.knowledge.models import KnowledgeEntry
from autoacoustics.knowledge.provider import ProviderConfig,save_provider
from autoacoustics.model import AnalysisResult,AnalysisSettings,AnalysisSummary,MeasurementContext,MetricResult,CalibrationProfile

@pytest.fixture
def dialogs():
    app=QApplication.instance() or QApplication([]);opened=[]
    def make(path,provider=None):
        module=importlib.import_module('autoacoustics.ui.knowledge_dialog')
        dialog=module.KnowledgeDialog(path,provider);opened.append(dialog);dialog.show();app.processEvents();return dialog
    yield make,app
    for dialog in opened:dialog.close()
    app.processEvents()

def wait(app,predicate,timeout=3):
    end=time.monotonic()+timeout
    while not predicate() and time.monotonic()<end:app.processEvents();time.sleep(.005)
    assert predicate();app.processEvents()

@contextmanager
def endpoint(delay=0):
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])));requests.append(request)
            payload=json.loads(request['messages'][-1]['content']);identity=payload['evidence'][0]['evidence_id']
            time.sleep(delay);body=json.dumps({'model':'ui-protocol-fixture','choices':[{'message':{'content':f'协议验收回答 [{identity}]'}}]}).encode()
            self.send_response(200);self.end_headers()
            try:self.wfile.write(body)
            except OSError:pass
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}/v1',requests
    finally:server.shutdown();server.server_close();thread.join(timeout=2)

def test_empty_local_management_import_edit_and_reopen(dialogs,tmp_path):
    make,app=dialogs;path=tmp_path/'local.sqlite';dialog=make(path)
    assert dialog.sources.count()==0 and not dialog.send_button.isEnabled()
    editor=dialog.edit_entry(kind='knowledge');editor.title.setText('轴承安装说明');editor.body.setPlainText('轴承安装方向应核对工装记录。');editor.save_button.click()
    assert dialog.sources.count()==1
    source=dialog.store.list_sources()[0];editor=dialog.edit_entry(source.source_id);editor.body.setPlainText('轴承安装方向按照工装版本二。');editor.save_button.click()
    assert dialog.store.get_source(source.source_id).version==2
    markdown=tmp_path/'资料.md';markdown.write_text('# 蜗杆润滑\n蜗杆润滑牌号以技术图纸为准。',encoding='utf-8')
    preview=dialog.import_file(markdown);assert 'UTF-8' in preview.summary.text().upper();preview.apply_button.click()
    assert dialog.sources.count()==2
    dialog.question.setPlainText('轴承安装方向');dialog.preview_button.click()
    assert dialog.evidence_list.count()>=1 and '本地' in dialog.status.text()
    dialog.close();reopened=make(path);assert reopened.sources.count()==2

def test_real_http_gui_stays_responsive_cancel_duplicate_and_no_evidence(dialogs,tmp_path):
    make,app=dialogs;dialog=make(tmp_path/'store.sqlite');dialog.store.add_source(KnowledgeEntry('轴承安装','轴承安装方向核对。'));dialog.refresh_sources()
    with endpoint(delay=1.5) as (url,requests):
        save_provider(dialog.store,ProviderConfig('UI 协议夹具',url,'fixture',timeout_s=3))
        dialog.question.setPlainText('轴承安装方向');dialog.preview_button.click();assert dialog.send_button.isEnabled()
        dialog.send_button.click();assert not dialog.send_button.isEnabled()
        wait(app,lambda:bool(requests));start=time.monotonic()
        dialog.cancel_button.click();wait(app,lambda:not dialog.busy,timeout=1)
        assert time.monotonic()-start<1 and len(requests)==1 and not dialog.answer.toPlainText()
        assert '取消' in dialog.status.text()
        dialog.question.setPlainText('股票预算');dialog.preview_button.click();assert not dialog.send_button.isEnabled()
        assert '未找到' in dialog.status.text()

def test_preview_freezes_analysis_and_history_sources_delete(dialogs,tmp_path):
    make,app=dialogs
    def result(value):return AnalysisResult('result-'+str(value),'hash',Path('原始.wav'),AnalysisSettings(channel=0,start=50,end=100),MeasurementContext(),CalibrationProfile('配置','证书',2,'FS'),AnalysisSummary({'LAeq':MetricResult(value,'dB re 20 µPa')}),provenance={'unit':'Pa','sample_rate':48000})
    current=[result(63.2)];dialog=make(tmp_path/'store.sqlite',lambda:current[0]);dialog.include_analysis.setChecked(True)
    with endpoint() as (url,requests):
        save_provider(dialog.store,ProviderConfig('UI 协议夹具',url,'fixture'))
        dialog.question.setPlainText('LAeq 是多少？');dialog.preview_button.click();current[0]=result(80.4)
        assert '63.2' in dialog.facts.toPlainText() and '80.4' not in dialog.facts.toPlainText()
        dialog.send_button.click();wait(app,lambda:not dialog.busy)
        serialized=json.dumps(requests[0],ensure_ascii=False);assert '63.2' in serialized and '80.4' not in serialized and 'waveform' not in serialized
        assert dialog.last_answer.status=='ok' and dialog.sessions.count()==1
        dialog.sessions.setCurrentRow(0);dialog.delete_history_button.click();assert dialog.sessions.count()==0
    saved=dialog.store.add_source(KnowledgeEntry('轴承安装','轴承安装方向。'));dialog.include_analysis.setChecked(False)
    dialog.question.setPlainText('轴承安装方向');dialog.preview_button.click()
    second=make(tmp_path/'store.sqlite');second.store.set_active(saved.source_id,False)
    dialog.send_button.click();wait(app,lambda:not dialog.busy)
    assert dialog.last_answer.status=='stale_sources' and not dialog.answer.toPlainText()

def test_knowledge_tabs_fit_small_screen_after_show_and_long_labels(dialogs,tmp_path,monkeypatch):
    from autoacoustics.ui import window_geometry
    from PyQt6.QtWidgets import QPushButton
    available=QRect(0,0,1280,680)
    monkeypatch.setattr(window_geometry,'available_geometry',lambda window:available)
    make,app=dialogs;dialog=make(tmp_path/'geometry.sqlite')
    for index in range(dialog.tabs.count()):
        dialog.tabs.setCurrentIndex(index);app.processEvents();assert available.contains(dialog.frameGeometry())
    dialog.target.setText('很长的中文服务提示 '*100);dialog.status.setText('很长的中文状态说明 '*100);app.processEvents()
    assert available.contains(dialog.frameGeometry()) and dialog.minimumSizeHint().height()<600
    close=dialog.findChild(QPushButton,'knowledgeClose');assert close.isVisible()
    assert available.contains(QRect(close.mapToGlobal(QPoint(0,0)),close.size()))
