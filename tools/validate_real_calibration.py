"""Fit eight product imports and verify the remaining 56 known HEAD pairs."""
from pathlib import Path
import hashlib
import json
import tempfile
import zipfile
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests/reference'))
from head_pa_reader import read_head_pa_bytes
from autoacoustics.importers import import_signal
from autoacoustics.model import CalibrationProfile
from autoacoustics.calibration import apply_calibration,save_profile,load_profile
from autoacoustics.project import write_json_atomic

def validate(destination=ROOT/'docs/validation/real_calibration_validation.json'):
    manifest=json.loads((ROOT/'docs/validation/corpus_manifest.json').read_text(encoding='utf-8'))
    train=set(manifest['reference_regression']['training_ids']) if 'reference_regression' in manifest else set(next(v['training_ids'] for v in manifest.values() if isinstance(v,dict) and 'training_ids' in v))
    archive_path=ROOT/'压缩.zip'
    archive_hash=hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert archive_hash==manifest['archive']['sha256']
    with tempfile.TemporaryDirectory(prefix='autoacoustics-calibration-') as work,zipfile.ZipFile(archive_path) as archive:
        paths={};refs={};hashes=[]
        for member in archive.namelist():
            if member.lower().endswith('.wav'):
                path=Path(work)/Path(member).name
                data=archive.read(member);path.write_bytes(data)
                paths[path.stem]=path;hashes.append(hashlib.sha256(data).hexdigest())
            elif member.lower().endswith('.hdf'):
                refs[Path(member).stem]=member
        numerator=0.;denominator=0.
        for stem in sorted(train):
            signal=import_signal(paths[stem])
            reference=read_head_pa_bytes(archive.read(refs[stem]))
            x=signal.samples[0];y=reference.samples[0]
            numerator+=float(np.dot(x,y));denominator+=float(np.dot(x,x))
        coefficient=numerator/denominator
        profile=CalibrationProfile('本 ZIP 的 HEAD Pa 参考换算','HEAD Pa reference（本ZIP专用，同源导出）',
            coefficient,'FS',profile_id='head-pa-reference-v1',channel=0,
            applicable_hashes=tuple(sorted(hashes)),date='2026-10-03',mode='reference',details={
                'archive_sha256':archive_hash,'training_ids':sorted(train),
                'formula':'sum(raw_FS * HEAD_Pa) / sum(raw_FS²), all aligned original frames',
                'dc_removed':False,'independent_absolute_calibration':False,'blind_holdout':False,
                'limitation':'仅证明当前 ZIP 的同源 HEAD Pa 导出一致性，不能作为其他文件默认灵敏度。'})
        profile_path=ROOT/'profiles/head-pa-reference.json'
        save_profile(profile,profile_path);profile=load_profile(profile_path)
        rows=[]
        for stem,path in sorted(paths.items()):
            if stem in train: continue
            physical=apply_calibration(import_signal(path),profile)
            reference=read_head_pa_bytes(archive.read(refs[stem]))
            x=physical.samples[0];y=reference.samples[0]
            rms_x=np.sqrt(np.mean(x*x));rms_y=np.sqrt(np.mean(y*y))
            row={'id':stem,'frames':len(x),'rms_difference_db':float(20*np.log10(rms_x/rms_y)),
                'normalized_rmse_percent':float(100*np.sqrt(np.mean((x-y)**2))/rms_y),
                'correlation':float(np.corrcoef(x,y)[0,1])}
            assert abs(row['rms_difference_db'])<=.05 and row['normalized_rmse_percent']<=.2 and row['correlation']>=.99999
            rows.append(row)
        output={'scope':'formal product WAV/calibration vs existing HEAD Pa regression; not absolute or blind calibration',
            'archive_sha256':archive_hash,'training_count':len(train),'holdout_count':len(rows),
            'coefficient':coefficient,'profile':str(profile_path.relative_to(ROOT)),
            'maximum_absolute_rms_difference_db':max(abs(r['rms_difference_db']) for r in rows),
            'median_absolute_rms_difference_db':float(np.median([abs(r['rms_difference_db']) for r in rows])),
            'maximum_normalized_rmse_percent':max(r['normalized_rmse_percent'] for r in rows),
            'median_normalized_rmse_percent':float(np.median([r['normalized_rmse_percent'] for r in rows])),
            'minimum_correlation':min(r['correlation'] for r in rows),'rows':rows,'passed':True}
        write_json_atomic(output,destination)
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest()==archive_hash
    return output

if __name__=='__main__':
    output=validate()
    print(json.dumps({k:v for k,v in output.items() if k!='rows'},ensure_ascii=False,indent=2))
