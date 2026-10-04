import sys
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from autoacoustics.resource_probe import run_probe
parser=argparse.ArgumentParser()
parser.add_argument('--duration',type=int,default=600)
parser.add_argument('--channels',type=int,default=1)
parser.add_argument('--summary-only',action='store_true')
args=parser.parse_args()
print(run_probe(ROOT/f'docs/validation/pipeline_resource_{args.duration}s_{args.channels}ch.json',
                args.duration,args.channels,not args.summary_only))
