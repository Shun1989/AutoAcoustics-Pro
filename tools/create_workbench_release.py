"""Package the UI revision using evidence from that same frozen executable."""
import argparse, hashlib, json, re, shutil, zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,default=ROOT/'dist/AutoAcousticsProWorkbench')
    parser.add_argument('--evidence',type=Path,default=ROOT/'output/workbench-frozen')
    args=parser.parse_args();bundle=args.bundle.resolve();evidence=args.evidence.resolve()
    executable=bundle/(bundle.name+'.exe')
    snapshot=read(ROOT/'output/workbench_build_source_snapshot.json')
    assert digest(executable)==snapshot['exe_sha256']
    assert snapshot['source_unchanged_during_build'] is True
    for name,expected in snapshot['product_source_sha256'].items():
        assert digest(ROOT/name)==expected, 'Product source changed after build: '+name
    workflow=read(evidence/'bundle_validation.json')
    invocation=read(evidence/'invocation.json')
    consistency=read(evidence/'report_consistency.json')
    assert workflow['frozen'] and workflow['passed']
    assert invocation['exit_code']==0 and invocation['exe_sha256']==digest(executable)
    assert workflow['workbench_experience']['four_quadrants_simultaneously_visible']
    assert len(workflow['formats'])==18 and workflow['batch']['count']==64 and workflow['head']['count']==66
    assert consistency['passed']
    log=(ROOT/'output/workbench_pytest.log').read_text(encoding='utf-8')
    match=re.search(r'(\d+) passed, (\d+) skipped, (\d+) warnings, (\d+) subtests passed in ([\d.]+)s',log)
    assert match, 'Missing completed full test result'
    original={'spec':digest(ROOT/'REWRITE_PRODUCT_SPEC.md'),'audio_archive':digest(ROOT/'压缩.zip')}
    assert original['spec']=='3d1e135cdb1b4a967c8aab334086f1d3297ef79ef409cf0b9eb51e04de7d1f64'
    assert original['audio_archive']=='6b0194790b61f8dbc3098a0ac99e52b63f97e0bb3b49f30e208bbb9144ccc45a'
    for source,name in ((evidence/'bundle_validation.json','workbench_validation.json'),
                        (evidence/'report_consistency.json','report_consistency.json'),
                        (evidence/'invocation.json','verification_invocation.json'),
                        (ROOT/'output/workbench_build_source_snapshot.json','build_source_snapshot.json'),
                        (ROOT/'output/workbench_pytest.log','source_tests.txt')):
        shutil.copyfile(source,bundle/name)
    screenshots=bundle/'workbench_screenshots';screenshots.mkdir(exist_ok=True)
    for name in ('desktop_four_quadrants.png','desktop_stft_popout.png','desktop_batch_workspace.png'):
        shutil.copyfile(evidence/name,screenshots/name)
    manifest={'schema_version':1,'version':'0.2.0','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Four-quadrant desktop experience repair; same frozen EXE real-audio workflow and report consistency',
        'exe_sha256':digest(executable),'original_inputs_sha256':original,
        'source_snapshot':snapshot,'source_tests':dict(zip(('passed','skipped','warnings','subtests','seconds'),
            [int(v) for v in match.groups()[:4]]+[float(match.group(5))])),
        'frozen_workflow_passed':True,'human_acceptance':False,'external_clean_Windows_tested':False,
        'real_NI_hardware_tested':False,'real_HEAD_control_tested':False,'independent_absolute_calibration_tested':False,
        'previous_release_performance_retested':False,
        'excluded_scope':['EoL','ODS','PHM','enterprise functions','automatic OK/NG or fault diagnosis'],
        'files':[]}
    for path in sorted(bundle.rglob('*')):
        if path.is_file() and path.name!='release_manifest.json':
            manifest['files'].append({'path':path.relative_to(bundle).as_posix(),'bytes':path.stat().st_size,'sha256':digest(path)})
    (bundle/'release_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    package=ROOT/'AutoAcousticsPro-0.2.0-Windows.zip'
    files=sorted(p for p in bundle.rglob('*') if p.is_file())
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files: archive.write(path,path.relative_to(bundle.parent).as_posix())
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None and len(archive.infolist())==len(files)
        for path in files:
            assert hashlib.sha256(archive.read(path.relative_to(bundle.parent).as_posix())).hexdigest()==digest(path)
    record={'zip':str(package),'sha256':digest(package),'bytes':package.stat().st_size,
        'files':len(files),'all_bytes_verified':True,'exe_sha256':digest(executable)}
    (ROOT/'output/workbench_release.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))

if __name__=='__main__': main()
