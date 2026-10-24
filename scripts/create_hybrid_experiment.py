#!/usr/bin/env python3
"""Create a fresh isolated experiment config from the hybrid template."""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import yaml

repo = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('experiment_id',help='E.g. hybrid_002 (must not exist)')
parser.add_argument('--template',default='config/experiments/hybrid_001.yaml')
args=parser.parse_args()
if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{2,63}',args.experiment_id):
    raise SystemExit('Invalid experiment_id')
dest=repo/'config/experiments'/f'{args.experiment_id}.yaml'
if dest.exists():raise SystemExit(f'Already exists: {dest}')
config=yaml.safe_load((repo/args.template).read_text(encoding='utf8'))
config['experiment_id']=args.experiment_id
dest.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf8')
print('Created:',dest)
print('Run directly with python scripts/run_hybrid_experiment.py --config',dest.relative_to(repo))
