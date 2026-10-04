"""Record the verified local delivery, without promoting pending external gates."""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(1024*1024):value.update(chunk)
    return value.hexdigest()

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,default=ROOT/'dist/AutoAcousticsPro')
    parser.add_argument('--suite-log',type=Path,default=ROOT/'.superpowers/sdd/2026-10-03-autoacoustics-desktop/full_suite_release.log')
    args=parser.parse_args();bundle=args.bundle.resolve()
    workflow=read(ROOT/'docs/validation/frozen_bundle_validation.json')
    resources=read(ROOT/'docs/validation/frozen_pipeline_resource_600s.json')
    self_check=read(ROOT/'docs/validation/final_self_check.json')
    reports=read(ROOT/'docs/validation/final_frozen_report_check.json')
    knowledge_create=read(ROOT/'docs/validation/frozen_knowledge_create.json')
    knowledge_reopen=read(ROOT/'docs/validation/frozen_knowledge_reopen.json')
    relocation=read(ROOT/'docs/validation/portable_relocation_validation.json')
    word_open=read(ROOT/'docs/validation/final_word_open.json')
    word_visual=read(ROOT/'docs/validation/final_word_visual.json')
    supplemental=read(ROOT/'docs/validation/loudness_supplemental_validation.json')
    exe_sha=digest(bundle/'AutoAcousticsPro.exe')
    if (not supplemental.get('passed') or supplemental.get('exe_sha256') != exe_sha
        or not supplemental.get('product_source_unchanged_since_build')
        or supplemental.get('new_audio_reference_cases') != 21
        or supplemental.get('product_audio_reference_cases_total') != 24
        or supplemental.get('new_cases_frozen_EXE_executed') is not False):
        raise RuntimeError('Supplemental source-reference evidence is incomplete or misstates its execution scope.')
    for relative,expected_sha in supplemental['product_source_sha256'].items():
        if digest(ROOT/relative) != expected_sha:
            raise RuntimeError(f'Product source changed after supplemental validation: {relative}')
    for relative,expected_sha in supplemental['evidence_sha256'].items():
        if digest(ROOT/relative) != expected_sha:
            raise RuntimeError(f'Supplemental reference evidence changed: {relative}')
    for name,proof in (('workflow',workflow),('resources',resources),('self-check',self_check),
                       ('knowledge create',knowledge_create),('knowledge reopen',knowledge_reopen),
                       ('relocation',relocation),('Word open',word_open),('Word visual',word_visual)):
        if proof.get('exe_sha256') != exe_sha:
            raise RuntimeError(f'{name} evidence does not belong to the current executable.')
    if not workflow.get('passed') or not workflow.get('frozen') or not resources.get('passed'):
        raise RuntimeError('Final frozen workflow and resource gates must pass before recording a delivery.')
    ni_contract=workflow.get('NI_software_contract',{})
    condition_filter=workflow.get('recorded_condition_filter',{})
    if (not ni_contract.get('passed') or ni_contract.get('is_simulated') is not True
        or ni_contract.get('real_hardware_tested') is not False
        or not condition_filter.get('project_reopen')
        or not condition_filter.get('original_selection_identity_preserved')
        or condition_filter.get('independent_repeated_trials_confirmed') is not False):
        raise RuntimeError('Current NI software and recorded-condition checks are incomplete or overstate their scope.')
    build_snapshot=ROOT/'output/build_source_snapshot.json'
    if (supplemental.get('build_source_snapshot_sha256') != digest(build_snapshot)
        or read(build_snapshot).get('exe_sha256') != exe_sha
        or not read(build_snapshot).get('source_unchanged_during_build')):
        raise RuntimeError('Current executable is not bound to the verified build source snapshot.')
    if (not self_check.get('frozen') or not self_check.get('loudness_finite')
        or not self_check.get('optimized_backend',{}).get('compiled_nonlinear_recurrence')
        or self_check.get('optimized_backend_total_max_error') != 0.0
        or self_check.get('optimized_backend_specific_max_error') != 0.0
        or not reports.get('passed') or reports.get('inconsistency_count') != 0):
        raise RuntimeError('Final compiled backend and exported-report checks must pass.')
    for proof in (knowledge_create,knowledge_reopen):
        if (not proof.get('passed') or not proof.get('frozen')
            or not proof.get('checks') or not all(proof['checks'].values())):
            raise RuntimeError('Frozen knowledge create and cross-process reopen must pass.')
    if (knowledge_create.get('pid') == knowledge_reopen.get('pid')
        or not knowledge_create.get('windows_credentials',{}).get('passed')
        or not relocation.get('passed') or not relocation.get('frozen')
        or not relocation.get('HEAD_C2_software',{}).get('batch_csv_docx')
        or not word_open.get('passed') or not word_visual.get('passed')):
        raise RuntimeError('Credential, relocated C2, and final Word checks must pass.')
    for entry in word_open['documents']:
        if digest(ROOT/'output/frozen-bundle-validation'/entry['filename']) != entry['sha256']:
            raise RuntimeError('Word evidence refers to an earlier report export.')
    open_hashes={entry['filename']:entry['sha256'] for entry in word_open['documents']}
    visual_hashes={entry['filename']:entry['docx_sha256'] for entry in word_visual['documents']
                   if entry.get('all_pages_visually_reviewed')}
    if len(open_hashes) != 5 or open_hashes != visual_hashes:
        raise RuntimeError('Visual review must cover all five current Word reports.')
    opened={entry['filename']:entry for entry in word_open['documents']}
    for entry in word_visual['documents']:
        native=opened[entry['filename']]
        if (entry['pages'] != native['pages']
            or entry.get('covered_page_range') != [1,native['pages']]
            or entry.get('pdf_sha256') != digest(Path(native['internal_qa_pdf']))):
            raise RuntimeError('Visual PDF hash or reviewed page coverage is stale or incomplete.')
    for filename,entry in reports['artifacts'].items():
        if digest(ROOT/'output/frozen-bundle-validation'/filename) != entry['sha256']:
            raise RuntimeError('Report consistency evidence refers to an earlier export.')
    for path in (ROOT/'docs').rglob('*'):
        if path.is_file() and path.name not in ('release_manifest.json','release_zip.json'):
            bundled=bundle/'_internal'/path.relative_to(ROOT)
            if not bundled.is_file() or digest(bundled) != digest(path):
                raise RuntimeError(f'Bundled documentation or evidence is stale: {path.relative_to(ROOT)}')
    suite=args.suite_log.read_text(encoding='utf-8-sig',errors='replace')
    match=re.search(r'(\d+) passed, (\d+) skipped, (\d+) warnings, (\d+) subtests passed in ([\d.]+)s',suite)
    if not match or re.search(r'\d+ failed',suite):raise RuntimeError('The final full-suite result is not a passing result.')
    inventory=read(bundle/'licenses/distribution_manifest.json')
    manifest={'schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'name':'AutoAcoustics Pro','version':'0.1.0','delivery':'Windows GUI onedir; copy the complete directory',
        'exe_sha256':exe_sha,
        'inputs':{'original_spec_sha256':digest(ROOT/'REWRITE_PRODUCT_SPEC.md'),
                  'original_archive_sha256':digest(ROOT/'压缩.zip')},
        'suite':dict(zip(('passed','skipped','warnings','subtests','seconds'),
                        [int(v) for v in match.groups()[:4]]+[float(match.group(5))])),
        'workflow':workflow,'single_channel_resource':resources,
        'compiled_backend_self_check':self_check,'exported_report_consistency':reports,
        'supplemental_source_loudness_references':supplemental,
        'build_source_snapshot':read(build_snapshot),
        'knowledge_create':knowledge_create,'knowledge_cross_process_reopen':knowledge_reopen,
        'portable_relocation':relocation,'native_Word_open':word_open,'native_Word_visual':word_visual,
        'environment':'This Windows 11 host with PATH limited to System32/Windows and no PYTHONHOME/PYTHONPATH',
        'external_clean_Windows_tested':False,'external_first_time_user_tested':False,
        'real_NI_hardware_tested':False,'real_HEAD_control_tested':False,
        'independent_absolute_calibration_tested':False,'real_LLM_API_tested':False,
        'real_user_knowledge_and_case_tested':False,
        'public_redistribution_ready':inventory['public_redistribution_ready'],
        'remaining_inputs':['Exact NI/HEAD/PCB/GRAS models and acquisition/calibration chain',
            'NI-DAQmx driver and actual equipment; HEAD official SDK/Recorder license and version',
            'Independent calibration recording and pre/post calibration records',
            'User-confirmed knowledge and historical cases; actual API configuration',
            'Second clean Windows machine and first-time-user workflow'],
        'excluded_scope':['EoL','ODS','PHM','enterprise functions','automatic OK/NG or fault diagnosis'],
        'files':[]}
    for path in sorted(bundle.rglob('*')):
        if path.is_file() and path.name!='release_manifest.json':
            manifest['files'].append({'path':path.relative_to(bundle).as_posix(),
                'bytes':path.stat().st_size,'sha256':digest(path)})
    manifest['package_bytes']=sum(item['bytes'] for item in manifest['files'])
    target=bundle/'release_manifest.json'
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'docs/validation/release_manifest.json').write_text(target.read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps({'manifest':str(target),'exe_sha256':manifest['exe_sha256'],
        'files':len(manifest['files']),'package_bytes':manifest['package_bytes']},ensure_ascii=False))

if __name__=='__main__':main()
