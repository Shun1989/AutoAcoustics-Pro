"""Qualitative historical-plot comparison; PNG has no numerical curve truth."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from autoacoustics.importers import import_signal
from autoacoustics.analysis.pipeline import analyze
from autoacoustics.calibration import load_profile
from autoacoustics.model import AnalysisSettings

def main():
    directory=ROOT/'output/png-comparison';directory.mkdir(parents=True,exist_ok=True)
    profile=load_profile(ROOT/'profiles/head-pa-reference.json');rows=[]
    for name,png_n,png_t,png_la,png_la_t in [('01-01CW',7.22,2.45,52.46,2.86),('01-32CCW',8.83,4.87,54.96,4.71)]:
        result=analyze(import_signal(ROOT/f'data/corpus/WAV/{name}.wav'),profile,AnalysisSettings(channel=0))
        arrays=result.detail.arrays
        index=int(np.argmax(arrays['loudness']))
        row={'id':name,'png_peak_sone':png_n,'png_peak_time_s':png_t,'png_A_peak_db':png_la,
             'png_A_peak_time_s':png_la_t,'product_peak_sone':result.summary.metrics['Nmax'].value,
             'product_peak_time_s':float(arrays['loudness_time'][index]),
             'product_LAeq_db':result.summary.metrics['LAeq'].value,
             'product_LAFmax_db':result.summary.metrics['LAFmax'].value,
             'product_LASmax_db':result.summary.metrics['LASmax'].value,
             'method':'PNG labels manually read; free-field ISO 532-1; complete recording, no DC removal',
             'scope':'qualitative curve/annotation comparison, not full HEAD numerical equivalence'}
        rows.append(row)
        figure,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
        axes[0].plot(arrays['spl_time'],arrays['laf'],label='LAF');axes[0].plot(arrays['spl_time'],arrays['las'],label='LAS')
        axes[0].set(ylabel='dB re 20 uPa',title=f'{name} - full original recording');axes[0].legend()
        axes[1].plot(arrays['loudness_time'],arrays['loudness'],label='ISO 532-1, free')
        axes[1].axhline(png_n,color='orange',linestyle='--',label='Historical PNG peak annotation')
        axes[1].set(xlabel='Original time / s',ylabel='Loudness / sone');axes[1].legend()
        for axis in axes:axis.grid(alpha=.3)
        figure.tight_layout();figure.savefig(directory/f'{name}.png',dpi=140);plt.close(figure)
    (ROOT/'docs/validation/png_comparison.json').write_text(json.dumps({'evidence_scope':'PNG qualitative only','cases':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
