#!/usr/bin/env python3
"""Cross-check final WAV/HEAD batch reports with CSV and project snapshots.

Run from any directory with the project Python (standard library only):
    python -X utf8 tools/check_frozen_reports.py

No waveforms are loaded or recomputed. Exit 0 means all consistency checks pass;
exit 1 means a mismatch was recorded in docs/validation/final_frozen_report_check.json.
"""

import argparse, csv, hashlib, json, math, re, time, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifacts-dir',type=Path,default=ROOT/'output/frozen-bundle-validation')
parser.add_argument('--evidence',type=Path,default=ROOT/'docs/validation/final_frozen_report_check.json')
args=parser.parse_args(); OUT=args.artifacts_dir.resolve()
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
FILES=['batch.csv','batch.docx','真实音频项目.json','head_batch.csv','head_batch.docx','HEAD真实音频项目.json']
METRIC_NAMES={'LZeq':'平均 Z 声级 LZeq','LAeq':'平均 A 声级 LAeq','LAFmax':'最大 Fast A 声级 LAFmax','LASmax':'最大 Slow A 声级 LASmax','Nmean':'时变响度时间平均 Nmean','Nmax':'时变响度最大值 Nmax','Nstationary':'稳态响度 Nstationary','RMS':'均方根 RMS','RMS_dBFS':'数字 RMS dBFS','duration':'事件时长'}
STATUS={'ok':'有效','disabled':'未启用','silent':'静音','failed':'失败','partial':'部分指标失败','complete':'有效','estimated':'估计','unavailable':'不可用'}
CONTEXT={'specimen':'样件','version':'样件版本','test_level':'测量层级','mechanism':'机构','supply':'供电','supply_source':'供电来源','load':'负载','load_layout':'负载布局','fixture':'工装','fastening':'紧固方式','environment':'环境','temperature':'温度','mic_position':'麦克风测点','mic_distance':'麦克风距离','mic_direction':'麦克风方向','background':'背景','actual_movement':'实际运动方向','motor_rotation':'电机旋向','rotation_view':'旋向观察基准','stroke':'行程','cycle':'循环','repeat':'重复次数','notes':'备注'}
PROCESS={'fft_size':'FFT 请求点数','window':'窗函数','stft_size':'STFT 点数','stft_hop':'STFT 步长 点','spl_time_weighting':'SPL 时间计权','history_step':'SPL 历程步长 秒','field_type':'响度声场','remove_dc':'去直流','compute_loudness':'时变响度','stationary_loudness':'额外稳态响度'}
failures=[]; checks=0
def ck(value,message):
    global checks
    checks+=1
    if not value:failures.append(message)
def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()
def inventory():
    result={}
    for name in FILES:
        p=OUT/name;s=p.stat()
        result[name]={'bytes':s.st_size,'mtime_utc':datetime.fromtimestamp(s.st_mtime,timezone.utc).isoformat(),'mtime_ns':s.st_mtime_ns,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    return result
def ptext(node):
    return ''.join((part.text or '') if part.tag==W+'t' else '\n' if part.tag in (W+'br',W+'cr') else '' for part in node.iter())
def table(node):
    return [['\n'.join(ptext(p) for p in cell.findall(W+'p')) for cell in row.findall(W+'tc')] for row in node.findall(W+'tr')]
def pairs(nodes):
    result={}
    for node in nodes:
        if node.tag==W+'tbl':
            rows=table(node)
            if rows and rows[0]==['字段','记录','字段','记录']:
                for row in rows[1:]:
                    for i in (0,2):
                        if row[i]:result[row[i]]=row[i+1]
    return result
def paragraphs(nodes):return [ptext(n) for n in nodes if n.tag==W+'p']
def jsonparas(nodes):
    result={}
    for text in paragraphs(nodes):
        if '  ' in text:
            key,value=text.split('  ',1)
            try:result[key]=json.loads(value)
            except (ValueError,TypeError):result[key]=value
    return result
def shared(value):
    if isinstance(value,dict):return {k:shared(v) for k,v in value.items() if k not in ('event_start_sample','event_end_sample')}
    if isinstance(value,list):return [shared(v) for v in value]
    return value
def display(value,notes=False):
    if value is None or value=='' or value=='unknown':return '未填写' if notes else '未知'
    return str(value)
def source_summary(meta):
    data={k:meta[k] for k in ('format','subtype','bit_depth','source_dtype','normalization','lossy','source_quantity','calibration_applied_by_importer') if k in meta and not isinstance(meta[k],(dict,list))}
    head=meta.get('head',{});head=head if isinstance(head,dict) else {}
    data.update({'HEAD '+k:head[k] for k in ('version','release','byte_order','implementation_type','text_encoding') if k in head})
    sensor=meta.get('sensor',{});sensor=sensor if isinstance(sensor,dict) else {}
    if not sensor:
        raw=head.get('comment_metadata','')
        if isinstance(raw,str) and not re.search(r'<!\s*(DOCTYPE|ENTITY)\b',raw,re.I):
            opening=re.search(r'<SensorInfo\b[^>]*>',raw)
            if opening:
                tag=opening.group();tag=tag if tag.endswith('/>') else tag[:-1]+'/>'
                sensor=dict(ET.fromstring(tag).attrib)
    for key in ('Name','Manufacturer','SerialNumber','SensorType','Model','OutputUnit','Sensitivity','CalibrationFactor','CalibrationDate','Index','ChName'):
        if key in sensor:data['Sensor '+key+(' 原属性未用于重校准' if key in ('Sensitivity','CalibrationFactor') else '')]=sensor[key]
    return {k:display(v) for k,v in data.items()}
def sections(body):
    groups={};items={};current=None;run=None
    headings={'处理记录':'S','校准记录':'P','方法记录':'M','来源摘要':'H','质量记录':'Q'}
    for node in body:
        text=ptext(node) if node.tag==W+'p' else ''
        item=re.fullmatch(r'条目 (\d+) 测量记录',text)
        group=re.fullmatch(r'(处理记录|校准记录|方法记录|来源摘要|质量记录) ([SPMHQ]\d+)',text)
        if item:
            current=items.setdefault(int(item[1]),[])
        elif group:
            current=groups.setdefault(group[2],[])
        elif text=='工况记录 C1 共同字段':current=groups.setdefault('C1',[])
        elif text=='运行清单':current=None;run=[]
        elif text in ('批量指标摘要图','共同处理和测量记录'):current=None
        elif current is not None:current.append(node)
        elif run is not None and node.tag==W+'tbl' and not run:run=table(node)
    return groups,items,run
first=inventory();time.sleep(.25);second=inventory()
ck(first==second,'artifacts changed during initial stability check')
manifest_path=ROOT/'docs/validation/corpus_manifest.json'
corpus=json.loads(manifest_path.read_text(encoding='utf-8'))
member_by_name={Path(m['path']).name:m for m in corpus['members'] if m['kind'].lower() in ('wav','hdf')}
result={'schema_version':1,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Read-only final frozen batch DOCX XML/ZIP CRC vs CSV and saved immutable project snapshots; no analysis, rendering, tests or arrays loaded. Original member hashes/frames/fs cross-checked against existing corpus manifest.','artifact_stability_observed_seconds':.25,'artifacts':second,'corpus_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'runs':[],'passed':False}
for stem,project_name,expected_count,expected_metrics,notes in [('batch','真实音频项目.json',64,10,'样本名称来自文件标签；电机方向、观察基准及实际移动方向尚未确认。'),('head_batch','HEAD真实音频项目.json',66,9,'文件标签不是经确认的运动方向。')]:
    before_errors=len(failures)
    try:
        with (OUT/(stem+'.csv')).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        project=json.loads((OUT/project_name).read_text(encoding='utf-8'))
        snapshots={s['result']['result_id']:s['result'] for s in project['results']}
        ck(len(rows)==expected_count,stem+' row count')
        ck(len({r['item_id'] for r in rows})==expected_count,stem+' duplicate item IDs')
        ck(len({r['result_id'] for r in rows})==expected_count,stem+' duplicate result IDs')
        for snap in project['results']:
            ctx=snap['result']['context']
            ck(ctx.get('motor_rotation')=='unknown' and ctx.get('actual_movement')=='unknown',stem+' project contains claimed direction '+snap['result']['result_id'])
        with zipfile.ZipFile(OUT/(stem+'.docx')) as archive:
            crc_passed=archive.testzip() is None
            ck(crc_passed,stem+' DOCX CRC')
            xml_count=0
            for name in archive.namelist():
                if name.endswith(('.xml','.rels')):ET.fromstring(archive.read(name));xml_count+=1
            body=ET.fromstring(archive.read('word/document.xml')).find(W+'body')
        groups,items,run=sections(body)
        ck(len(items)==expected_count,stem+' DOCX item section count')
        ck(run is not None and len(run)==expected_count+1,stem+' run manifest table count')
        contexts=[json.loads(row['context_json']) for row in rows]
        common={k:v for k,v in contexts[0].items() if all(ctx.get(k)==v for ctx in contexts)}
        common_pairs=pairs(groups.get('C1',[]))
        for key,value in common.items():ck(common_pairs.get(CONTEXT.get(key,key))==display(value,key=='notes'),stem+' common context '+key)
        metric_count=0;required_valid=0;disabled_count=0
        for index,row in enumerate(rows,1):
            tag=stem+' item '+str(index)+': '
            ck(row['status']=='ok' and not row['error'],tag+'failed/partial row')
            saved=snapshots.get(row['result_id']);ck(saved is not None,tag+'result absent in project')
            if saved is None:continue
            prov=json.loads(row['provenance_json']);ctx=json.loads(row['context_json']);settings=json.loads(row['settings_json'])
            cal=json.loads(row['calibration_json']) if row['calibration_json'] else None
            ck(saved['input_hash']==row['input_hash'] and saved['path']==row['path'],tag+'input identity/path')
            for field,value in [('settings',settings),('context',ctx),('calibration',cal),('provenance',prov)]:
                ck(saved[field]==value,tag+'CSV/project '+field)
            ck(ctx['motor_rotation']=='unknown' and ctx['actual_movement']=='unknown' and ctx['notes']==notes,tag+'claimed direction or borrowed context')
            ck(ctx['specimen']==Path(row['path']).stem,tag+'specimen from file label')
            ck(saved['detail'] is None,tag+'batch unexpectedly retained detail')
            meta=prov['source_metadata'];member=member_by_name.get(Path(row['path']).name)
            ck(member is not None,tag+'source absent in archive manifest')
            if member:
                ck(member['sha256']==row['input_hash'],tag+'original source hash')
                ck(member['frames']==prov['source_frames'] and math.isclose(member['fs_hz'],prov['sample_rate'],rel_tol=1e-12),tag+'original frames/fs')
            ck(int(row['channel'])==settings['channel']==prov['channel'] and int(row['start_sample'])==settings['start']==prov['event_start_sample'],tag+'channel/start')
            ck(int(row['end_sample_exclusive'])==prov['event_end_sample']==prov['source_frames'],tag+'event end')
            nodes=items.get(index,[]);texts=paragraphs(nodes);field_pairs=pairs(nodes)
            ck('结果 ID '+row['result_id'] in texts and '输入 SHA256 '+row['input_hash'] in texts,tag+'DOCX identity')
            ck('source_metadata SHA256 '+digest(meta) in texts,tag+'DOCX full source metadata hash')
            ck('源文件 '+row['path'] in texts,tag+'DOCX source file')
            ck(any(text.startswith('分析条目 ID '+row['item_id']+'；状态 有效。') for text in texts),tag+'DOCX item identity/status')
            ck(field_pairs.get('原样本范围')=='['+row['start_sample']+', '+row['end_sample_exclusive']+')',tag+'DOCX sample interval')
            ck(field_pairs.get('原采样率')==str(prov['sample_rate'])+' Hz',tag+'DOCX sample rate')
            ck(field_pairs.get('输入／分析单位')==prov['source_unit']+' / '+prov['unit'],tag+'DOCX source/analysis units')
            if run and index<len(run):
                ck(run[index]==[str(index),row['path']+'\n'+row['item_id'],'有效','—'],tag+'DOCX run row')
            refs={p:field_pairs.get(label) for p,label in [('S','处理记录'),('P','校准记录'),('M','方法记录'),('H','来源摘要'),('Q','质量记录')]}
            for prefix,reference in refs.items():ck(reference in groups and str(reference).startswith(prefix),tag+'unresolved '+prefix+' reference')
            processing={k:v for k,v in settings.items() if k not in ('channel','start','end','event_label')}
            processing_pairs=pairs(groups.get(refs['S'],[]))
            for key,value in processing.items():
                text='自由场' if key=='field_type' and value=='free' else '扩散场' if key=='field_type' and value=='diffuse' else '是' if value is True else '否' if value is False else display(value)
                ck(processing_pairs.get(PROCESS.get(key,key))==text,tag+'S setting '+key)
            expected_cal=cal if cal is not None else prov.get('calibration')
            if isinstance(expected_cal,dict) and isinstance(expected_cal.get('applicable_hashes'),list) and len(expected_cal['applicable_hashes'])>8:
                hashes=expected_cal['applicable_hashes'];ck(row['input_hash'] in hashes,tag+'profile source applicability')
                expected_cal={**expected_cal,'applicable_hashes':{'记录':str(len(hashes))+' 项','完整清单 SHA256':digest(hashes),'说明':'完整清单保存在原结果及 CSV 校准记录，本条输入 SHA256 见测量配置'}}
            ck(jsonparas(groups.get(refs['P'],[])).get('calibration')==expected_cal,tag+'P calibration')
            actual_methods=jsonparas(groups.get(refs['M'],[]))
            for key in ('versions','method_revision','methods'):
                if key in prov:ck(actual_methods.get(key)==shared(prov[key]),tag+'M '+key)
            method_tables=[table(n) for n in groups.get(refs['M'],[]) if n.tag==W+'tbl']
            metric_method_map={r[0]:r[1] for t in method_tables if t and t[0]==['指标','指标方法'] for r in t[1:]}
            ck(pairs(groups.get(refs['H'],[]))==source_summary(meta),tag+'H source/sensor summary')
            quality_texts=paragraphs(groups.get(refs['Q'],[]))
            expected_flags='来源质量标记 '+('；'.join(map(str,prov.get('quality',[]))) or '无已记录标记，不代表全部前端状态已知')
            ck(expected_flags in quality_texts,tag+'Q flags')
            expected_warnings=['质量或适用性说明 '+str(v) for v in saved['summary']['warnings']]
            actual_warnings=[v for v in quality_texts if v.startswith('质量或适用性说明 ')]
            ck(actual_warnings==expected_warnings,tag+'Q warnings')
            for key,value in ctx.items():
                if key not in common:ck(field_pairs.get(CONTEXT.get(key,key))==display(value,key=='notes'),tag+'item context '+key)
            metric_tables=[table(n) for n in nodes if n.tag==W+'tbl']
            tables=[t for t in metric_tables if t and t[0]==['指标','数值','单位','状态','方法及说明']]
            ck(len(tables)==1,tag+'metric table count')
            doc_metrics={r[0]:r[1:] for t in tables for r in t[1:]}
            metrics=saved['summary']['metrics']
            ck(len(metrics)==expected_metrics and len(doc_metrics)==expected_metrics,tag+'metric count')
            ck({k[:-6] for k in row if k.endswith('.value')}==set(metrics),tag+'CSV metric set')
            for name,metric in metrics.items():
                metric_count+=1;value=metric['value']
                ck(metric['status'] in ('ok','disabled','silent'),tag+'failed/partial/unavailable metric '+name)
                if name in ('LZeq','LAeq','LAFmax','LASmax','Nmean','Nmax'):
                    ck(metric['status']=='ok' and isinstance(value,(int,float)) and math.isfinite(value) and (not name.startswith('N') or value>=0),tag+'required metric '+name)
                    required_valid+=int(metric['status']=='ok' and value is not None and math.isfinite(value))
                if value is None:
                    ck(row[name+'.value']=='',tag+'disabled value not empty '+name)
                    expected_display='—'
                    if metric['status']=='disabled':disabled_count+=1
                else:
                    ck(float(row[name+'.value'])==value,tag+'CSV numeric '+name)
                    expected_display='−∞' if value==-math.inf else format(value,'.10g')
                for field in ('unit','status','method','message'):ck(row[name+'.'+field]==metric[field],tag+'CSV '+name+' '+field)
                label=METRIC_NAMES[name]
                explanation='\n'.join(p for p in ((refs['M']+' / '+name) if metric['method'] else '',metric['message']) if p) or '方法未记录'
                ck(doc_metrics.get(label)==[expected_display,metric['unit'],STATUS.get(metric['status'],metric['status']),explanation],tag+'DOCX metric '+name)
                ck(metric_method_map.get(label)==(metric['method'] or '方法未记录'),tag+'resolved metric method '+name)
        result['runs'].append({'name':stem,'csv_rows':len(rows),'docx_item_sections':len(items),'project_snapshot_count':len(project['results']),'metric_records_checked':metric_count,'required_valid_metric_records':required_valid,'disabled_empty_metric_records':disabled_count,'resolved_shared_records':sorted(groups),'docx_xml_parts_parsed':xml_count,'archive_crc_passed':crc_passed,'new_inconsistencies':len(failures)-before_errors,'passed':len(failures)==before_errors,'direction_scope':'motor_rotation and actual_movement unknown; specimen from file label and explicit unconfirmed-label notes'})
    except Exception as error:
        failures.append(stem+' parse/check error: '+type(error).__name__+': '+str(error))
        result['runs'].append({'name':stem,'passed':False,'error':str(error)})
third=inventory();ck(second==third,'artifacts changed while checking')
result['assertions_checked']=checks
result['total_metric_records']=sum(run.get('metric_records_checked',0) for run in result['runs'])
result['total_required_valid_metric_records']=sum(run.get('required_valid_metric_records',0) for run in result['runs'])
result['total_disabled_empty_metric_records']=sum(run.get('disabled_empty_metric_records',0) for run in result['runs'])
result['passed']=not failures;result['inconsistency_count']=len(failures);result['inconsistencies']=failures[:100]
result['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
target=args.evidence.resolve()
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'passed':result['passed'],'assertions':checks,'metrics':result['total_metric_records'],'required_valid':result['total_required_valid_metric_records'],'disabled_empty':result['total_disabled_empty_metric_records'],'runs':result['runs'],'inconsistency_count':len(failures),'first_inconsistencies':failures[:15],'evidence_file':str(target)},ensure_ascii=True))

raise SystemExit(0 if result['passed'] else 1)

