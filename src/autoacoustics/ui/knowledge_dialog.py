"""Independent local knowledge workbench; explicit text preview before HTTP."""
from __future__ import annotations
from dataclasses import replace
import csv,html,io,json,threading,queue,hashlib
from pathlib import Path
from PyQt6.QtCore import Qt,QTimer,QUrl,QSettings
from PyQt6.QtGui import QDesktopServices,QPixmap
from PyQt6.QtWidgets import (QDialog,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QTabWidget,
    QListWidget,QListWidgetItem,QTextBrowser,QPlainTextEdit,QLineEdit,QPushButton,QLabel,
    QComboBox,QCheckBox,QSpinBox,QDoubleSpinBox,QSplitter,QFileDialog,QTableWidget,
    QTableWidgetItem,QDialogButtonBox,QScrollArea)
from .window_geometry import fit_to_available_screen,scroll_page,flexible_wrapped_label
from ..knowledge.models import KnowledgeEntry,HistoryCase,SearchFilters,KnowledgeError
from ..knowledge.store import KnowledgeStore
from ..knowledge.importers import preview_import,import_preview,_read
from ..knowledge.corpus import import_corpus
from ..knowledge.answer import AnswerService,case_draft_from_analysis,AnswerResult
from ..knowledge.provider import ProviderConfig,ProviderClient,save_provider,load_provider
from ..knowledge.credentials import CredentialVault,CredentialError

CASE_LABELS={'equipment':'设备','conditions':'工况','observation':'观察','actions':'已采取措施','outcome':'处理结果','tags':'标签','source_record_id':'文件行标识','analysis_ref':'分析引用'}
STYLE='''QDialog,QWidget{background:#122033;color:#e1eaf4;font-size:13px;}
QLineEdit,QPlainTextEdit,QTextBrowser,QListWidget,QTableWidget,QComboBox,QSpinBox,QDoubleSpinBox{background:#182b40;color:#e1eaf4;border:1px solid #365066;border-radius:4px;padding:4px;}
QPushButton{background:#25455c;border:1px solid #436378;border-radius:5px;padding:7px;}QPushButton:hover{background:#315b71;}QPushButton:disabled{color:#728391;background:#1b2e42;}
QTabBar::tab{background:#20364c;padding:9px;}QTabBar::tab:selected{background:#2c5268;}QHeaderView::section{background:#254358;color:#e1eaf4;padding:5px;}'''

def button(text,name,callback,layout):
    widget=QPushButton(text);widget.setObjectName(name);widget.clicked.connect(callback);layout.addWidget(widget);return widget

class EntryEditor(QDialog):
    def __init__(self,owner,kind='knowledge',record=None,draft=None):
        super().__init__(owner);self.owner=owner;self.kind=record.kind if record else kind;self.original=record.entry if record else draft
        self.setWindowTitle('编辑知识' if self.kind=='knowledge' else '编辑历史案例');self.resize(760,640)
        layout=QVBoxLayout(self);form=QFormLayout();layout.addLayout(form)
        self.title=QLineEdit();self.title.setObjectName('knowledgeEntryTitle');form.addRow('标题',self.title)
        self.tags=QLineEdit();self.tags.setPlaceholderText('用英文逗号分隔');form.addRow('标签',self.tags)
        self.equipment=QLineEdit();self.conditions=QLineEdit();form.addRow('设备',self.equipment);form.addRow('工况',self.conditions)
        self.body=QPlainTextEdit();self.body.setObjectName('knowledgeEntryBody')
        self.fields={}
        if self.kind=='knowledge':layout.addWidget(QLabel('正文'));layout.addWidget(self.body,1)
        else:
            for key in ('observation','actions','outcome'):
                edit=QPlainTextEdit();edit.setMaximumHeight(110);self.fields[key]=edit;layout.addWidget(QLabel(CASE_LABELS[key]));layout.addWidget(edit)
            layout.addWidget(QLabel('案例草稿只复制分析事实；观察、措施和结果需由用户填写。'))
        self.message=QLabel();self.message.setWordWrap(True);layout.addWidget(self.message)
        actions=QHBoxLayout();layout.addLayout(actions);self.save_button=button('保存本地版本','knowledgeEntrySave',self.save,actions);button('取消','knowledgeEntryCancel',self.reject,actions)
        if self.original:
            item=self.original;self.title.setText(item.title);self.tags.setText(', '.join(item.tags));self.equipment.setText(item.equipment);self.conditions.setText(item.conditions)
            if self.kind=='knowledge':self.body.setPlainText(item.body)
            else:
                for key,edit in self.fields.items():edit.setPlainText(getattr(item,key))
        fit_to_available_screen(self,(760,640))
    def save(self):
        try:
            values={'title':self.title.text().strip(),'tags':tuple(tag.strip() for tag in self.tags.text().split(',') if tag.strip()),'equipment':self.equipment.text().strip(),'conditions':self.conditions.text().strip()}
            if self.kind=='knowledge':values['body']=self.body.toPlainText()
            else:values.update({key:edit.toPlainText() for key,edit in self.fields.items()})
            cls=KnowledgeEntry if self.kind=='knowledge' else HistoryCase
            item=replace(self.original,**values) if self.original else cls(**values)
            if self.original and self.original.source_path:item=replace(item,source_version='本地修订；原文件版本 '+self.original.source_version)
            saved=self.owner.store.add_source(item) if self.kind=='knowledge' else self.owner.store.add_case(item)
            self.owner.refresh_sources();self.owner.invalidate_preview();self.owner.status.setText(f'本地已保存：版本 {saved.version}（{saved.status}）。');self.accept()
        except Exception as error:self.message.setText(str(error))

class ImportPreviewDialog(QDialog):
    def __init__(self,owner,path,mapping=None):
        super().__init__(owner);self.owner=owner;self.path=Path(path);self.preview=None;self.mapping={}
        self.setWindowTitle('资料导入预览');self.resize(800,600);layout=QVBoxLayout(self)
        self.summary=QLabel(str(self.path));self.summary.setWordWrap(True);layout.addWidget(self.summary)
        self.map_form=QFormLayout();layout.addLayout(self.map_form)
        if self.path.suffix.lower()=='.csv':
            _,text,_,_=_read(self.path);columns=tuple(csv.DictReader(io.StringIO(text)).fieldnames or ())
            for field,label in {'title':'标题（必选）',**CASE_LABELS}.items():
                combo=QComboBox();combo.addItem('不导入此字段',None)
                for column in columns:combo.addItem(column,column)
                if mapping and field in mapping:combo.setCurrentIndex(combo.findData(mapping[field]))
                self.mapping[field]=combo;self.map_form.addRow(label,combo)
            button('预览映射结果','knowledgeImportMap',self.refresh,layout)
        self.table=QTableWidget(0,4);self.table.setHorizontalHeaderLabels(['原行','标题','状态','错误 / 导入结果']);self.table.horizontalHeader().setStretchLastSection(True);layout.addWidget(self.table,1)
        actions=QHBoxLayout();layout.addLayout(actions);self.apply_button=button('导入以上可用记录','knowledgeImportApply',self.apply,actions);button('关闭','knowledgeImportClose',self.reject,actions)
        self.refresh()
        fit_to_available_screen(self,(800,600))
    def refresh(self):
        try:
            mapping={key:edit.currentData() for key,edit in self.mapping.items() if edit.currentData() is not None}
            self.preview=preview_import(self.path,mapping=mapping or None)
            self.summary.setText(f'{self.path}\n编码：{self.preview.encoding}；类型：{self.preview.kind}；共 {len(self.preview.rows)} 行。原文件保持只读。')
            self.table.setRowCount(len(self.preview.rows))
            for index,row in enumerate(self.preview.rows):
                for column,value in enumerate((row.row,row.entry.title if row.entry else '',row.status,row.error)):self.table.setItem(index,column,QTableWidgetItem(str(value)))
            self.apply_button.setEnabled(any(row.status=='ready' for row in self.preview.rows))
        except Exception as error:self.preview=None;self.summary.setText(str(error));self.apply_button.setEnabled(False)
    def apply(self):
        if self.preview is None:return
        outcomes=import_preview(self.owner.store,self.preview)
        for index,row in enumerate(outcomes):self.table.setItem(index,2,QTableWidgetItem(row.status));self.table.setItem(index,3,QTableWidgetItem(row.error or f'{row.source_id} / v{row.version}'))
        self.apply_button.setEnabled(False);self.owner.refresh_sources();self.owner.invalidate_preview();self.owner.status.setText(f'本地导入完成：{sum(row.status!="failed" for row in outcomes)} 条成功，{sum(row.status=="failed" for row in outcomes)} 条失败。')

class KnowledgeDialog(QDialog):
    """analysis_provider() -> AnalysisResult | None; snapshot read only on Preview."""
    def __init__(self,store_path,analysis_provider=None,parent=None):
        super().__init__(parent);self.store=KnowledgeStore(store_path);self.service=AnswerService(self.store);self.vault=CredentialVault()
        self.analysis_provider=analysis_provider;self.preview=None;self._preview_provider=None;self.session_id=None;self.last_answer=None;self.busy=False
        self._queue=queue.Queue();self._cancel=None;self._worker=None;self._closed=False;self._children=[];self._visible_evidence=();self._history_snapshots=()
        self._geometry_fitted=False
        self.corpus_busy=False;self._corpus_queue=queue.Queue();self._corpus_cancel=None;self._corpus_worker=None;self.last_corpus_result=None
        self.setWindowTitle('AutoAcoustics Pro · 本地知识与有来源问答');self.resize(1200,850);self.setStyleSheet(STYLE)
        layout=QVBoxLayout(self);self.tabs=QTabWidget();layout.addWidget(self.tabs,1)
        self._build_library();self._build_ask();self._build_settings()
        self.status=QLabel('本地知识库已打开；真实知识资料与真实 API 尚需用户配置和验收。');self.status.setObjectName('knowledgeStatus');flexible_wrapped_label(self.status)
        status_scroll=scroll_page(self.status,'knowledgeStatusScroll');status_scroll.setMinimumHeight(24);status_scroll.setMaximumHeight(72);layout.addWidget(status_scroll)
        button('关闭','knowledgeClose',self.close,layout)
        self.timer=QTimer(self);self.timer.setInterval(40);self.timer.timeout.connect(self.poll);self.timer.start()
        self.refresh_sources();self.refresh_history();self.load_settings()
        for label in self.findChildren(QLabel):
            if label.wordWrap():flexible_wrapped_label(label)
        fit_to_available_screen(self,(1200,850))
    def showEvent(self,event):
        super().showEvent(event)
        if not self._geometry_fitted:
            fit_to_available_screen(self,(1200,850));self._geometry_fitted=True
    def _build_library(self):
        page=QWidget();layout=QVBoxLayout(page);self.tabs.addTab(scroll_page(page,'knowledgeLibraryScroll'),'本地资料与案例')
        actions=QHBoxLayout();layout.addLayout(actions)
        button('新知识','knowledgeNewEntry',lambda:self.edit_entry(kind='knowledge'),actions);button('新案例','knowledgeNewCase',lambda:self.edit_entry(kind='case'),actions)
        button('当前分析 → 案例草稿','knowledgeCaseDraft',self.create_draft,actions);button('导入文件','knowledgeImportFile',self.choose_import,actions);button('刷新','knowledgeRefresh',self.refresh_sources,actions)
        corpus_actions=QHBoxLayout();layout.addLayout(corpus_actions)
        self.corpus_button=button('导入工程资料语料','knowledgeImportCorpus',self.choose_corpus,corpus_actions)
        self.local_corpus_button=button('本机工程资料','knowledgeLocalCorpus',self.import_local_corpus,corpus_actions)
        self.corpus_cancel_button=button('取消资料导入','knowledgeCancelCorpus',self.cancel_corpus,corpus_actions);self.corpus_cancel_button.setEnabled(False)
        self.corpus_note=QLabel('按原资料页号导入提取文字及原页图。提取 / 渲染不等于图表已理解；原文与工程解读条目分开保存。');flexible_wrapped_label(self.corpus_note);layout.addWidget(self.corpus_note)
        splitter=QSplitter();layout.addWidget(splitter,1);self.sources=QListWidget();self.sources.setObjectName('knowledgeSources');self.sources.currentItemChanged.connect(self.source_selected);splitter.addWidget(self.sources)
        self.source_text=QTextBrowser();self.source_text.setOpenExternalLinks(False);self.source_text.setObjectName('knowledgeSourceText');splitter.addWidget(self.source_text);splitter.setSizes([330,750])
        actions=QHBoxLayout();layout.addLayout(actions)
        button('编辑版本','knowledgeEditSource',self.edit_selected,actions);button('启用 / 停用','knowledgeToggleSource',self.toggle_selected,actions)
        button('打开原文件','knowledgeOpenSourceFile',self.open_selected_file,actions);button('查看原页图','knowledgeOpenPageImage',self.open_selected_image,actions);button('移出检索','knowledgeDeleteSource',self.delete_selected,actions);button('彻底删除…','knowledgePurgeSource',self.preview_purge_selected,actions)
    def _build_ask(self):
        page=QWidget();layout=QVBoxLayout(page);self.tabs.addTab(scroll_page(page,'knowledgeAskScroll'),'检索与问答')
        filters=QHBoxLayout();layout.addLayout(filters);self.kind=QComboBox()
        for label,kind in [('知识与案例','all'),('知识','knowledge'),('案例','case')]:self.kind.addItem(label,kind)
        filters.addWidget(self.kind);self.equipment=QLineEdit();self.equipment.setPlaceholderText('设备过滤（精确匹配）');filters.addWidget(self.equipment)
        self.conditions=QLineEdit();self.conditions.setPlaceholderText('工况过滤（精确匹配）');filters.addWidget(self.conditions);self.tags=QLineEdit();self.tags.setPlaceholderText('标签，用英文逗号分隔');filters.addWidget(self.tags)
        self.include_analysis=QCheckBox('附带当前分析的标量事实卡');self.include_analysis.setObjectName('knowledgeIncludeAnalysis');layout.addWidget(self.include_analysis)
        self.question=QPlainTextEdit();self.question.setObjectName('knowledgeQuestion');self.question.setPlaceholderText('输入问题。预览时重新检索当前范围；没有证据时不请求模型。');self.question.setMaximumHeight(100);layout.addWidget(self.question)
        actions=QHBoxLayout();layout.addLayout(actions);self.preview_button=button('本地检索 / 更新预览','knowledgePreview',self.prepare_preview,actions)
        self.send_button=button('发送以上预览给所选服务','knowledgeSend',self.send,actions);self.send_button.setEnabled(False)
        self.cancel_button=button('取消请求','knowledgeCancel',self.cancel,actions);self.cancel_button.setEnabled(False);button('新会话','knowledgeNewSession',self.new_session,actions)
        self.target=QLabel('没有发送目标。');self.target.setWordWrap(True);layout.addWidget(self.target)
        split=QSplitter();layout.addWidget(split,1);left=QWidget();ll=QVBoxLayout(left);split.addWidget(left)
        ll.addWidget(QLabel('将发送的证据（最多六段）'));self.evidence_list=QListWidget();self.evidence_list.setObjectName('knowledgeEvidence');self.evidence_list.itemDoubleClicked.connect(lambda item:self.open_evidence(item.data(Qt.ItemDataRole.UserRole)));ll.addWidget(self.evidence_list)
        self.facts=QPlainTextEdit();self.facts.setObjectName('knowledgeFacts');self.facts.setReadOnly(True);self.facts.setPlaceholderText('可选当前分析的事实卡');ll.addWidget(self.facts,1)
        self.payload=QPlainTextEdit();self.payload.setReadOnly(True);self.payload.setObjectName('knowledgeSendPayload');self.payload.setPlaceholderText('完整文本请求预览（不包含 API 密钥）');ll.addWidget(self.payload,1)
        right=QWidget();rl=QVBoxLayout(right);split.addWidget(right);rl.addWidget(QLabel('模型回答 / 本地历史（引用编号检查不保证语义正确）'))
        self.answer=QTextBrowser();self.answer.setObjectName('knowledgeAnswer');self.answer.setOpenLinks(False);self.answer.anchorClicked.connect(lambda url:self.open_evidence(url.toString().removeprefix('evidence:')));rl.addWidget(self.answer,2)
        self.sessions=QListWidget();self.sessions.setObjectName('knowledgeSessions');self.sessions.currentItemChanged.connect(self.history_selected);rl.addWidget(self.sessions,1)
        self.delete_history_button=button('删除所选会话及其正文快照','knowledgeDeleteHistory',self.delete_history,rl)
        split.setSizes([650,470])
        self.question.textChanged.connect(self.invalidate_preview);self.kind.currentIndexChanged.connect(self.invalidate_preview);self.include_analysis.toggled.connect(self.invalidate_preview)
        for widget in (self.equipment,self.conditions,self.tags):widget.textChanged.connect(self.invalidate_preview)
    def _build_settings(self):
        page=QWidget();scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(page);layout=QVBoxLayout(page);form=QFormLayout();layout.addLayout(form);self.tabs.addTab(scroll,'API 与凭据设置')
        note=QLabel('仅支持已配置的 OpenAI 风格文本聊天端点。远端使用 HTTPS，本机 loopback 可用 HTTP。点击发送会把预览中的问题、片段与可选事实卡传给所选服务；不发送原始音频。没有 API 仍可管理和检索本地资料。');note.setWordWrap(True);layout.addWidget(note)
        self.provider_name=QLineEdit();self.provider_url=QLineEdit();self.provider_model=QLineEdit()
        self.provider_url.setPlaceholderText('https://服务地址/v1 或 http://127.0.0.1:端口/v1')
        for label,widget in [('服务名称',self.provider_name),('base_url',self.provider_url),('模型',self.provider_model)]:form.addRow(label,widget)
        self.provider_timeout=QDoubleSpinBox();self.provider_timeout.setRange(.1,300);self.provider_timeout.setValue(30);form.addRow('超时（秒）',self.provider_timeout)
        self.provider_chars=QSpinBox();self.provider_chars.setRange(512,12000);self.provider_chars.setValue(12000);form.addRow('输入字符预算',self.provider_chars)
        self.provider_output=QSpinBox();self.provider_output.setRange(0,32768);self.provider_output.setSpecialValueText('使用服务默认值');form.addRow('输出 token 上限',self.provider_output)
        self.credential_ref=QLineEdit();self.credential_ref.setPlaceholderText('env:明确的环境变量名；或应用生成的 wincred: 引用');form.addRow('凭据引用（不含密钥）',self.credential_ref)
        self.new_secret=QLineEdit();self.new_secret.setEchoMode(QLineEdit.EchoMode.Password);self.new_secret.setObjectName('knowledgeNewSecret');self.new_secret.setPlaceholderText('留空保留引用；填写后保存到 Windows 凭据');form.addRow('新 / 替换 API 密钥',self.new_secret)
        self.credential_status=QLabel();form.addRow('密钥状态',self.credential_status)
        actions=QHBoxLayout();layout.addLayout(actions);button('保存配置','knowledgeSaveProvider',self.save_settings,actions);button('删除所选 Windows 凭据 / 清除引用','knowledgeDeleteCredential',self.delete_credential,actions)
        self.test_button=button('测试连接（发送最小文本请求）','knowledgeTestProvider',self.test_connection,actions);layout.addStretch()
    def selected_source(self):
        item=self.sources.currentItem();return item.data(Qt.ItemDataRole.UserRole) if item else None
    def refresh_sources(self):
        selected=self.selected_source();self.sources.clear()
        for record in self.store.list_sources(include_inactive=True):
            state='启用' if record.active else '停用';item=QListWidgetItem(f'{record.entry.title} · v{record.version} · {state} · {"知识" if record.kind=="knowledge" else "案例"}');item.setData(Qt.ItemDataRole.UserRole,record.source_id);self.sources.addItem(item)
            if record.source_id==selected:self.sources.setCurrentItem(item)
    def source_selected(self,item,*args):
        if not item:self.source_text.clear();return
        try:
            record=self.store.get_source(item.data(Qt.ItemDataRole.UserRole));entry=record.entry
            metadata=entry.metadata if record.kind=='knowledge' else {}
            page_info=f'\n原页：{metadata.get("page")} / {metadata.get("page_count")}；{metadata.get("coverage_status", "")}\n核验记录：{metadata.get("review_status", "未提供")}\n重复来源：{", ".join(metadata.get("aliases", ()))}' if metadata.get('page') else ''
            self.source_text.setPlainText(f'{entry.title}\n来源 ID：{record.source_id} / v{record.version}\n内容哈希：{record.content_hash}\n文件：{entry.source_path or "手工条目"}（{record.source_state}）\n更新时间：{entry.updated_at}{page_info}\n\n{record.text}')
        except Exception as error:self.source_text.setPlainText(str(error))
    def edit_entry(self,source_id=None,kind='knowledge',draft=None):
        try:record=self.store.get_source(source_id) if source_id else None;dialog=EntryEditor(self,kind,record,draft);self._children.append(dialog);dialog.show();return dialog
        except Exception as error:self.status.setText(str(error))
    def edit_selected(self):
        if self.selected_source():self.edit_entry(self.selected_source())
    def create_draft(self):
        result=self.analysis_provider() if self.analysis_provider else None
        if result is None:self.status.setText('尚无已完成且有效的当前分析，不能建立分析案例草稿。');return
        if result.provenance.get('source_state') in ('missing','changed','unverified','stale'):self.status.setText('分析来源已失效，请重新关联和分析后建草稿。');return
        self.edit_entry(kind='case',draft=case_draft_from_analysis(result))
    def choose_import(self):
        path,_=QFileDialog.getOpenFileName(self,'选择只读资料源','','知识/案例 (*.md *.markdown *.txt *.csv *.json)')
        if path:self.import_file(path)
    def import_file(self,path,mapping=None):
        try:dialog=ImportPreviewDialog(self,path,mapping);self._children.append(dialog);dialog.show();return dialog
        except Exception as error:self.status.setText(str(error))
    def toggle_selected(self):
        try:
            identity=self.selected_source()
            if identity:self.store.set_active(identity,not self.store.get_source(identity).active);self.refresh_sources();self.invalidate_preview()
        except Exception as error:self.status.setText(str(error))
    def _corpus_candidates(self):
        remembered=QSettings('AutoAcoustics','AutoAcoustics Pro').value('knowledge/corpus_manifest','')
        candidates=[Path(remembered)] if remembered else []
        candidates.append(Path.cwd()/'output'/'knowledge_corpus'/'source_manifest.json')
        return candidates
    def choose_corpus(self):
        initial=next((str(path) for path in self._corpus_candidates() if path.is_file()),'')
        path,_=QFileDialog.getOpenFileName(self,'选择工程资料提取清单',initial,'工程资料清单 (*.json)')
        if path:self.import_corpus_file(path)
    def import_local_corpus(self):
        path=next((path for path in self._corpus_candidates() if path.is_file()),None)
        if path:self.import_corpus_file(path)
        else:self.choose_corpus()
    def import_corpus_file(self,path):
        if self.corpus_busy or self._closed:return
        path=Path(path).resolve();self.corpus_busy=True;self._corpus_cancel=threading.Event();self.last_corpus_result=None
        self.corpus_button.setEnabled(False);self.local_corpus_button.setEnabled(False);self.corpus_cancel_button.setEnabled(True)
        self.invalidate_preview();self.status.setText('正在校验工程资料来源、连续页码和原文 / 原页图哈希…')
        database=self.store.path;cancel=self._corpus_cancel
        def progress(done,total,message):
            if cancel.is_set():raise KnowledgeError('工程资料导入已取消；本次写入已回滚。')
            self._corpus_queue.put(('progress',(done,total,message)))
        def run():
            try:
                # A worker owns its connection. It never uses the UI connection.
                with KnowledgeStore(database) as store:result=import_corpus(store,path,progress)
                self._corpus_queue.put(('complete',(path,result)))
            except Exception as error:self._corpus_queue.put(('failed',str(error)))
        self._corpus_worker=threading.Thread(target=run,daemon=True,name='AutoAcoustics-CorpusImport');self._corpus_worker.start()
    def cancel_corpus(self):
        if self._corpus_cancel:self._corpus_cancel.set();self.corpus_cancel_button.setEnabled(False);self.status.setText('正在取消资料导入，等待回滚…')
    def _poll_corpus(self):
        latest=None;terminal=None
        for _ in range(200):
            try:kind,value=self._corpus_queue.get_nowait()
            except queue.Empty:break
            if kind=='progress':latest=value
            else:terminal=(kind,value);break
        if latest:
            done,total,message=latest;self.status.setText(f'{done}/{total}：{message}')
        if terminal is None:return
        kind,value=terminal;self.corpus_busy=False;self._corpus_worker=None;self._corpus_cancel=None
        self.corpus_button.setEnabled(True);self.local_corpus_button.setEnabled(True);self.corpus_cancel_button.setEnabled(False)
        if kind=='failed':self.status.setText(value);return
        path,result=value;self.last_corpus_result=result;QSettings('AutoAcoustics','AutoAcoustics Pro').setValue('knowledge/corpus_manifest',str(path))
        self.refresh_sources();self.invalidate_preview()
        self.status.setText(f'工程资料导入：{result.documents} 份、{result.total_pages} 页；新增 {result.added}、更新 {result.updated}、重复 {result.duplicates}。原页图缺失 {result.render_missing_pages}；空文字 {result.image_only_pages}、正文缺失 / 可能不完整 {result.incomplete_text_pages}。独立记录已审阅文字 {result.text_reviewed_pages} 页、图像 {result.visual_reviewed_pages} 页；提取 / 渲染不表示已理解全部资料。')
    def delete_selected(self):
        if self.selected_source():self.store.delete_source(self.selected_source());self.refresh_sources();self.invalidate_preview();self.status.setText('来源已移出检索；历史版本仍可查看。')
    def preview_purge_selected(self):
        identity=self.selected_source()
        if not identity:return
        impact=self.store.preview_purge(identity);dialog=QDialog(self);dialog.setWindowTitle('彻底删除的具体影响');layout=QVBoxLayout(dialog)
        text=QLabel(f'来源 {identity}\n将删除 {impact.version_count} 个正文版本和 {len(impact.session_ids)} 个含此来源正文快照的会话。\n'+('\n'.join(impact.session_ids) or '没有相关会话。'));flexible_wrapped_label(text);layout.addWidget(scroll_page(text),1)
        actions=QHBoxLayout();layout.addLayout(actions)
        def apply():
            current=self.store.preview_purge(identity)
            if current!=impact:text.setText('其他窗口改变了删除范围，请关闭并重新预览。');return
            self.store.purge_source(identity);self.invalidate_preview();self.refresh_sources();self.refresh_history();self.answer.clear();self.facts.clear();self.status.setText('来源正文、历史版本及相关会话已彻底删除。');dialog.accept()
        button('执行以上彻底删除','knowledgeConfirmPurge',apply,actions);button('取消','knowledgeCancelPurge',dialog.reject,actions);self._children.append(dialog);fit_to_available_screen(dialog,(760,500));dialog.show();return dialog
    def open_selected_file(self):
        identity=self.selected_source()
        if identity:self.open_file(self.store.get_source(identity).entry.source_path)
    def open_file(self,path):
        if path and Path(path).is_file():QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))
        else:self.status.setText('源文件不可用；已保存的本地正文版本仍可查看。')
    def open_selected_image(self):
        identity=self.selected_source()
        if identity:
            record=self.store.get_source(identity)
            if record.kind=='knowledge':return self.show_page_image(record.entry.metadata)
        self.status.setText('此条目没有工程原页图。')
    def show_page_image(self,metadata):
        path=metadata.get('render_path','')
        if not path or not Path(path).is_file():self.status.setText('原页图尚未生成或已移走；不能视为已核验。');return
        try:
            digest=hashlib.sha256()
            with Path(path).open('rb') as stream:
                while block:=stream.read(1024*1024):digest.update(block)
            if digest.hexdigest()!=str(metadata.get('render_sha256','')).lower():self.status.setText('原页图哈希已变化；不能作为此固定版本的证据显示，请重新导入。');return
        except OSError:self.status.setText('原页图不可读取，请核对文件位置。');return
        pixmap=QPixmap(str(path))
        if pixmap.isNull():self.status.setText('原页图无法显示，请核对图像文件。');return
        dialog=QDialog(self);dialog.setWindowTitle(f'工程原页图 · 第{metadata.get("page", "?")}页');layout=QVBoxLayout(dialog)
        review_info=('\n审阅者：'+metadata.get('review_reviewer','')+'；日期：'+metadata.get('review_date','')+'\n'+metadata.get('review_notes','')) if metadata.get('review_reviewer') else ''
        note=QLabel(metadata.get('coverage_status','原页图未核验')+review_info);flexible_wrapped_label(note);layout.addWidget(note)
        label=QLabel();label.setPixmap(pixmap);label.setObjectName('knowledgePageImage');label.setAlignment(Qt.AlignmentFlag.AlignTop|Qt.AlignmentFlag.AlignLeft)
        scroll=QScrollArea();scroll.setWidget(label);scroll.setWidgetResizable(False);layout.addWidget(scroll,1)
        button('打开完整原页图','knowledgeOpenImageExternal',lambda:self.open_file(path),layout);button('关闭','knowledgeImageClose',dialog.accept,layout)
        self._children.append(dialog);fit_to_available_screen(dialog,(900,850));dialog.show();return dialog
    def invalidate_preview(self,*args):
        self.preview=None;self._preview_provider=None
        if hasattr(self,'send_button'):self.send_button.setEnabled(False)
        if hasattr(self,'payload'):self.payload.clear();self.facts.clear();self.evidence_list.clear();self.target.setText('输入或范围已变化，请重新检索生成预览。')
    def prepare_preview(self):
        if self.busy:return
        try:
            config=load_provider(self.store);analysis=self.analysis_provider() if self.include_analysis.isChecked() and self.analysis_provider else None
            filters=SearchFilters(self.kind.currentData(),self.equipment.text().strip(),self.conditions.text().strip(),tuple(tag.strip() for tag in self.tags.text().split(',') if tag.strip()))
            self.preview=self.service.preview(self.question.toPlainText(),filters,analysis,self.session_id,char_limit=config.context_char_limit if config else 12000)
            self._preview_provider=config;self._visible_evidence=self.preview.evidence;self.facts.setPlainText(self.preview.facts_card);self.evidence_list.clear()
            for evidence in self.preview.evidence:
                coverage=evidence.metadata.get('coverage_status','')
                state='\n'+coverage if coverage else ''
                item=QListWidgetItem(f'{evidence.title} / v{evidence.version}{state}\n[{evidence.evidence_id}]\n{evidence.snippet[:160]}');item.setData(Qt.ItemDataRole.UserRole,evidence.evidence_id);self.evidence_list.addItem(item)
            messages=self.service.messages(self.preview);self.payload.setPlainText(json.dumps(messages,ensure_ascii=False,indent=2));count=sum(len(item['content']) for item in messages)
            self.target.setText((f'发送目标：{config.name} · {config.base_url} · 模型 {config.model}\n' if config else '未配置 API；可以查看本地证据。\n')+f'预览已冻结：{len(self.preview.evidence)} 段证据，{len(self.preview.history_messages)} 条相关历史消息，共 {count}/{self.preview.char_limit} 内容字符。不包含原始音频或数组。')
            self.send_button.setEnabled(bool(config and self.preview.evidence))
            self.status.setText(('本地检索完成。' if self.preview.evidence else '未找到可用证据，未调用 API。')+' '.join(self.preview.warnings));self.answer.clear()
        except Exception as error:self.invalidate_preview();self.status.setText(str(error))
    def send(self):
        if self.busy or self.preview is None or self._preview_provider is None:return
        preview=self.preview;config=replace(self._preview_provider,remote_send_confirmed=True)
        # The explicit Send click authorizes exactly the visible snapshot/target.
        save_provider(self.store,config);self._start(lambda cancel:self.service.ask(preview,config,cancel=cancel),'answer')
    def _start(self,operation,kind):
        if self.busy:return
        self.busy=True;self._cancel=threading.Event();self.send_button.setEnabled(False);self.preview_button.setEnabled(False);self.test_button.setEnabled(False);self.cancel_button.setEnabled(True);self.answer.clear();self.status.setText('正在请求所选 API；本地分析窗口可继续操作。')
        for widget in (self.question,self.kind,self.equipment,self.conditions,self.tags,self.include_analysis):widget.setEnabled(False)
        cancel=self._cancel
        def run():
            try:result=operation(cancel)
            except Exception:result=AnswerResult('failed',message='请求处理失败；本地资料保留，请重新检查配置。')
            self._queue.put((kind,result))
            if self._closed:self.store.close()
        self._worker=threading.Thread(target=run,daemon=True,name='AutoAcoustics-Knowledge');self._worker.start()
    def cancel(self):
        if self._cancel:self._cancel.set();self.cancel_button.setEnabled(False);self.status.setText('正在取消请求…')
    def poll(self):
        self._poll_corpus()
        try:kind,result=self._queue.get_nowait()
        except queue.Empty:return
        self.busy=False;self._worker=None;self._cancel=None;self.preview_button.setEnabled(True);self.test_button.setEnabled(True);self.cancel_button.setEnabled(False)
        for widget in (self.question,self.kind,self.equipment,self.conditions,self.tags,self.include_analysis):widget.setEnabled(True)
        self.send_button.setEnabled(bool(self.preview and self.preview.evidence and self._preview_provider))
        if kind=='answer':
            self.last_answer=result;self._visible_evidence=result.evidence;self.render_answer(result.text,result.cited_ids);self.status.setText(f'{result.status}：{result.message}')
            if result.session_id:self.session_id=result.session_id
            self.refresh_history()
        else:self.status.setText(f'连接测试 {result.status}；实际模型：{result.model or "未返回"}。{result.message}')
    def render_answer(self,text,cited_ids=()):
        escaped=html.escape(text)
        for identity in cited_ids:
            token='['+html.escape(identity)+']';escaped=escaped.replace(token,f'<a href="evidence:{html.escape(identity,quote=True)}">{token}</a>')
        self.answer.setHtml('<div style="white-space:pre-wrap">'+escaped+'</div>')
    def open_evidence(self,identity):
        evidence=next((item for item in self._visible_evidence if item.evidence_id==identity),None)
        snapshot=next((item for item in self._history_snapshots if item.get('evidence_id')==identity),None)
        if evidence is None and snapshot is None:self.status.setText('该编号不是本次已发送的有效出处，不能跳转。');return
        source_id=evidence.source_id if evidence else snapshot['source_id'];version=evidence.version if evidence else snapshot['version'];kind=evidence.kind if evidence else snapshot['kind'];snippet=evidence.snippet if evidence else snapshot['snippet']
        dialog=QDialog(self);dialog.setWindowTitle('证据来源与固定版本');dialog.resize(780,570);layout=QVBoxLayout(dialog);browser=QTextBrowser();layout.addWidget(browser);path=None
        metadata=dict(evidence.metadata) if evidence else snapshot.get('metadata',{})
        content=f'[{identity}]\n类型：{kind}；来源 {source_id} / v{version}\n\n发送快照：\n{snippet}'
        if kind!='analysis':
            try:
                record=self.store.get_source(source_id,version);current=self.store.get_source(source_id);path=record.entry.source_path
                state='当前版本' if current.version==version and current.active and not current.deleted else '旧版 / 停用来源；此处仅显示原引用版本'
                if record.kind=='knowledge':metadata=dict(record.entry.metadata)
                page_info=f'\n原页：第{metadata.get("page")}页；{metadata.get("coverage_status", "原页图未核验")}\n核验记录：{metadata.get("review_status", "未提供")}' if metadata.get('page') else ''
                content+=f'\n\n本地记录：{state}；原文件 {record.source_state}\n位置：{evidence.start if evidence else snapshot.get("location",{}).get("start",0)}{page_info}\n\n{record.text}'
            except KnowledgeError:content+='\n\n本地来源已删除；只有会话中的发送快照可用。'
        browser.setPlainText(content);button('打开原文件','knowledgeEvidenceFile',lambda:self.open_file(path),layout)
        if metadata.get('page'):button('查看此出处的原页图','knowledgeEvidencePageImage',lambda:self.show_page_image(metadata),layout)
        button('关闭','knowledgeEvidenceClose',dialog.accept,layout);self._children.append(dialog);fit_to_available_screen(dialog,(780,570));dialog.show();return dialog
    def refresh_history(self):
        self.sessions.blockSignals(True);self.sessions.clear()
        for session in self.store.list_sessions():
            item=QListWidgetItem(session['title']+'\n'+session['created_at']);item.setData(Qt.ItemDataRole.UserRole,session['session_id']);self.sessions.addItem(item)
        self.sessions.blockSignals(False)
    def history_selected(self,item,*args):
        if item is None:return
        self.session_id=item.data(Qt.ItemDataRole.UserRole);turns=self.store.history(self.session_id);self._history_snapshots=tuple(snapshot for turn in turns for snapshot in turn['evidence']);self._visible_evidence=()
        text=[];valid=[]
        for turn in turns:
            text.append(f"问题：{turn['question']}\n模型：{turn['model']}；{turn['status']}；{turn['created_at']}\n{turn['answer']}")
            valid.extend(snapshot['evidence_id'] for snapshot in turn['evidence'])
        self.render_answer('\n\n'.join(text),tuple(valid));self.invalidate_preview();self.status.setText('显示本地会话快照；资料更新后原回答仍指向当时版本，语义需人工核对。')
    def new_session(self):self.session_id=None;self._history_snapshots=();self.answer.clear();self.invalidate_preview();self.status.setText('新会话；请检索并预览。')
    def delete_history(self):
        item=self.sessions.currentItem()
        if item:
            identity=item.data(Qt.ItemDataRole.UserRole);self.store.delete_session(identity)
            if self.session_id==identity:self.session_id=None
            self._history_snapshots=();self.answer.clear();self.invalidate_preview();self.refresh_history();self.status.setText('会话及其正文快照已从本地历史删除。')
    def load_settings(self):
        try:
            config=load_provider(self.store)
            if config:
                self.provider_name.setText(config.name);self.provider_url.setText(config.base_url);self.provider_model.setText(config.model);self.provider_timeout.setValue(config.timeout_s);self.provider_chars.setValue(config.context_char_limit);self.provider_output.setValue(config.output_token_limit or 0);self.credential_ref.setText(config.credential_ref or '')
            self.credential_status.setText({'available':'••••••••（引用可用）','missing':'引用不可读取 / 环境变量未设置','not_configured':'未配置密钥'}[self.vault.status(self.credential_ref.text().strip())])
        except Exception as error:self.status.setText(str(error))
    def settings_config(self):
        return ProviderConfig(self.provider_name.text().strip(),self.provider_url.text().strip(),self.provider_model.text().strip(),self.provider_timeout.value(),self.credential_ref.text().strip() or None,False,self.provider_chars.value(),self.provider_output.value() or None)
    def save_settings(self):
        new_reference=None;previous=self.credential_ref.text().strip()
        try:
            config=self.settings_config()
            if self.new_secret.text():
                name=previous.removeprefix('wincred:'+self.vault.namespace) if previous.startswith('wincred:'+self.vault.namespace) else None
                new_reference=self.vault.store(self.new_secret.text(),name=name);config=replace(config,credential_ref=new_reference)
            save_provider(self.store,config);self.credential_ref.setText(config.credential_ref or '');self.invalidate_preview();self.load_settings();self.status.setText('配置已保存；密钥仅保存在 Windows 凭据，数据库只含引用。')
        except Exception as error:
            if new_reference and new_reference!=previous:
                try:self.vault.delete(new_reference)
                except CredentialError:pass
            self.status.setText(str(error))
        finally:self.new_secret.clear()
    def delete_credential(self):
        try:
            reference=self.credential_ref.text().strip()
            if reference.startswith('wincred:'):self.vault.delete(reference)
            self.credential_ref.clear();self.new_secret.clear()
            config=load_provider(self.store)
            if config:save_provider(self.store,replace(config,credential_ref=None))
            self.invalidate_preview();self.load_settings();self.status.setText('应用凭据已删除 / 引用已清除；用户环境变量本身未修改。')
        except Exception as error:self.status.setText(str(error))
    def test_connection(self):
        if self.busy:return
        try:config=self.settings_config();config=replace(config,remote_send_confirmed=True);self._start(lambda cancel:ProviderClient(config).test_connection(cancel),'connection')
        except Exception as error:self.status.setText(str(error))
    def closeEvent(self,event):
        self._closed=True;self.timer.stop()
        if self._corpus_cancel:self._corpus_cancel.set()
        if self._cancel:self._cancel.set()
        if self._worker is None or not self._worker.is_alive():self.store.close()
        for child in self._children:child.close()
        event.accept()
