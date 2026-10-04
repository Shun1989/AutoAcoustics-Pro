"""Chunked synthetic 10-minute two-channel float WAV, resource tests only."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'output/resource-fixtures/600s-dual.wav'
target.parent.mkdir(parents=True,exist_ok=True)
fs=48000
axis=np.arange(fs)/fs
block=np.stack((.02*np.sin(2*np.pi*1000*axis),.015*np.sin(2*np.pi*500*axis)),axis=1)
with sf.SoundFile(target,'w',samplerate=fs,channels=2,subtype='FLOAT') as stream:
    for _ in range(600):stream.write(block)
request=json.loads((ROOT/'output/frozen_bundle_request.json').read_text(encoding='utf-8'))
request.update(run_head=True,long_fixture=str(target))
(ROOT/'output/frozen_bundle_request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2),encoding='utf-8')
print(target)
