"""Gate the live desktop release on tests and the exact frozen EXE."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

from autoacoustics import __version__

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=ROOT/'dist/AutoAcousticsProLive')
    parser.add_argument('--live', type=Path, default=ROOT/'output/live-nvh-fixed-frozen')
    parser.add_argument('--regression', type=Path, default=ROOT/'output/live-regression-fixed-frozen')
    parser.add_argument('--snapshot', type=Path, default=ROOT/'output/live_build_source_snapshot.json')
    parser.add_argument('--test-log', type=Path, default=ROOT/'output/live_pytest_fixed_final.log')
    parser.add_argument('--repeat-prefix', type=Path, default=ROOT/'output/live-repeat-exit-')
    parser.add_argument('--version', default=__version__)
    options = parser.parse_args()
    require(options.version==__version__, 'Release version does not match current product source')
    bundle = options.bundle.resolve()
    executable = bundle/(bundle.name+'.exe')
    exe_hash = digest(executable)
    snapshot = read(options.snapshot)
    require(snapshot['source_unchanged_during_build'] and snapshot['exe_sha256']==exe_hash,
            'Build snapshot does not match executable')
    for name, expected in snapshot['product_source_sha256'].items():
        require(digest(ROOT/name)==expected, 'Source changed after build: '+name)
    live = read(options.live/'live_nvh_validation.json')
    regression = read(options.regression/'bundle_validation.json')
    for directory, workflow in ((options.live,live),(options.regression,regression)):
        invocation = read(directory/'invocation.json')
        require(workflow['frozen'] and workflow['passed'], 'Frozen workflow failed: '+str(directory))
        require(invocation['exit_code']==0 and invocation['exe_unchanged']
                and invocation['exe_sha256']==exe_hash, 'Frozen process failed or EXE changed')
    require(live.get('capture',{}).get('frames_written',0)>0, 'Real microphone was not recorded')
    require(not live['capture']['flags'] and not live['capture']['errors'], 'Microphone integrity flags')
    require(len(regression['formats'])==18 and regression['batch']['count']==64
            and regression['head']['count']==66, 'Real audio coverage incomplete')
    consistency = read(options.regression/'report_consistency.json')
    require(consistency['passed'], 'Reports differ from computed results')
    repetitions = []
    for index in (1,2,3):
        directory = Path(str(options.repeat_prefix)+str(index))
        invocation = read(directory/'invocation.json')
        workflow = read(directory/'live_nvh_validation.json')
        require(invocation['exit_code']==0 and invocation['exe_sha256']==exe_hash
                and invocation['exe_unchanged'] and workflow['passed'],
                'Repeated process exit failed')
        repetitions.append(invocation)
    log_path = options.test_log
    log = log_path.read_text(encoding='utf-8')
    match = re.search(r'(\d+) passed, (\d+) skipped, (\d+) warnings, (\d+) subtests passed in ([\d.]+)s',log)
    require(match is not None and 'Fatal Python error' not in log
            and 'fatal exception' not in log
            and not re.search(r'\b[1-9]\d* (?:failed|errors?)\b',log),
            'Full source suite did not complete successfully')
    original = {'spec':digest(ROOT/'REWRITE_PRODUCT_SPEC.md'),
                'audio_archive':digest(ROOT/'压缩.zip')}
    require(original['spec']=='3d1e135cdb1b4a967c8aab334086f1d3297ef79ef409cf0b9eb51e04de7d1f64', 'Original spec changed')
    require(original['audio_archive']=='6b0194790b61f8dbc3098a0ac99e52b63f97e0bb3b49f30e208bbb9144ccc45a', 'Original audio changed')
    docs = bundle/'_internal/docs'
    shutil.copytree(ROOT/'docs',docs,dirs_exist_ok=True)
    for name in ('release_manifest.json','release_zip.json'):
        (docs/'validation'/name).unlink(missing_ok=True)
    readme = (ROOT/'README.md').read_text(encoding='utf-8')
    (bundle/'README.md').write_text(readme.replace('(docs/','(_internal/docs/'),encoding='utf-8')
    copies = [(options.live/'live_nvh_validation.json','live_nvh_validation.json'),
              (options.live/'invocation.json','live_invocation.json'),
              (options.regression/'bundle_validation.json','frozen_bundle_validation.json'),
              (options.regression/'invocation.json','regression_invocation.json'),
              (options.regression/'report_consistency.json','report_consistency.json'),
              (options.snapshot,'build_source_snapshot.json'),
              (log_path,'source_tests.txt')]
    for source,name in copies:
        shutil.copyfile(source,bundle/name)
    (bundle/'exit_repetitions.json').write_text(json.dumps(repetitions,ensure_ascii=False,indent=2),encoding='utf-8')
    screenshots = bundle/'screenshots'
    screenshots.mkdir(exist_ok=True)
    for source in options.live.glob('desktop_*.png'):
        shutil.copyfile(source,screenshots/source.name)
    for source in options.regression.glob('desktop_*.png'):
        shutil.copyfile(source,screenshots/('regression_'+source.name))
    manifest = {'schema_version':1,'version':options.version,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'exe_sha256':exe_hash,
        'scope':'Live soundcard recording, rotational NVH evidence, local engineering knowledge',
        'source_snapshot':snapshot,'original_inputs_sha256':original,
        'source_tests':dict(zip(('passed','skipped','warnings','subtests','seconds'),
            [int(v) for v in match.groups()[:4]]+[float(match.group(5))])),
        'frozen_workflow_passed':True,'real_microphone_tested':True,
        'human_acceptance':False,'external_clean_Windows_tested':False,
        'real_NI_hardware_tested':False,'real_HEAD_control_tested':False,
        'independent_absolute_calibration_tested':False,
        'previous_release_performance_retested':False,
        'excluded_scope':['EoL','ODS','PHM','enterprise functions','automatic OK/NG or fault diagnosis'],
        'files':[]}
    for path in sorted(bundle.rglob('*')):
        if path.is_file() and path.name!='release_manifest.json':
            manifest['files'].append({'path':path.relative_to(bundle).as_posix(),
                                      'bytes':path.stat().st_size,'sha256':digest(path)})
    (bundle/'release_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'exe_sha256':exe_hash,'files':len(manifest['files']),
                      'tests':manifest['source_tests']},ensure_ascii=False))


if __name__=='__main__':
    main()
