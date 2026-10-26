"""Download only the retained FS1/profile data from the official UCI archive.

The 73 MB ZIP is read in memory; unrelated sensor files are never extracted.
The compact cache preserves all 600 original FS1 readings per cycle, profile
labels, and 100 disjoint six-reading averages used by the experiment.
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
CACHE = ROOT/'data'/'real_datasets'/'hydraulic_fs1'
URL = 'https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip'
PROFILE_NAMES = ['cooler_pct','valve_pct','pump_leakage','accumulator_bar','unstable_flag']


def load_data():
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE/'fs1.npz'
    if not path.exists():
        print('Downloading official UCI hydraulic archive (only FS1 and profile retained)...',flush=True)
        response = requests.get(URL, timeout=(20,180))
        response.raise_for_status()
        hashes = {}
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            selected = {}
            for name in ('FS1.txt','profile.txt'):
                matches = [p for p in archive.namelist() if Path(p).name == name]
                if len(matches) != 1:
                    raise ValueError(f'Expected one archive member for {name}: {matches}')
                raw = archive.read(matches[0])
                hashes[name] = hashlib.sha256(raw).hexdigest()
                selected[name] = np.loadtxt(io.BytesIO(raw))
            notes = [p for p in archive.namelist() if Path(p).name.lower() in ('readme.txt','readme')]
            for name in notes:
                (CACHE/'source_readme.txt').write_bytes(archive.read(name))
        raw, profile = selected['FS1.txt'], selected['profile.txt'].astype(int)
        if raw.shape != (2205,600) or profile.shape != (2205,5):
            raise ValueError(f'Unexpected source shapes: {raw.shape}, {profile.shape}')
        assert np.isfinite(raw).all()
        aggregated = raw.reshape(2205,100,6).mean(axis=2)
        time_seconds = np.arange(600).reshape(100,6).mean(axis=1)/10.
        np.savez_compressed(path, raw_fs1=raw, y=aggregated, profile=profile,
                            time_seconds=time_seconds, cycle_ids=np.arange(1,2206))
        meta = dict(source_url=URL, source_page='https://archive.ics.uci.edu/dataset/447',
                    archive_sha256=hashlib.sha256(response.content).hexdigest(),
                    member_sha256=hashes, sensor='FS1', unit='litres/minute',
                    original_sampling_hz=10, cycle_seconds=60, original_points=600,
                    aggregation='100 disjoint groups of six consecutive readings; arithmetic mean',
                    time_origin='first raw sample indexed at 0 seconds; aggregated input is bin sample-time mean',
                    profile_columns=PROFILE_NAMES, stable_flag_value=0,
                    citation='Helwig, Pignanelli and Schuetze (2015), UCI, DOI 10.24432/C5CW21',
                    license='CC BY 4.0; see source UCI page',
                    archive_retained=False, unrelated_sensor_files_extracted=False)
        (CACHE/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    with np.load(path) as saved:
        data = {key:saved[key].copy() for key in saved.files}
    frame = pd.DataFrame(data['profile'],columns=PROFILE_NAMES)
    frame.insert(0,'cycle_id',data['cycle_ids'])
    frame.to_csv(CACHE/'profiles.csv',index=False)
    counts = frame[frame.unstable_flag==0].groupby(PROFILE_NAMES[:4],sort=True).agg(
        stable_cycles=('cycle_id','size'), first_cycle=('cycle_id','min'), last_cycle=('cycle_id','max'))
    counts = counts.reset_index().sort_values('stable_cycles',ascending=False,kind='stable')
    counts.to_csv(CACHE/'stable_condition_counts.csv',index=False)
    print(counts.head(15).to_string(index=False),flush=True)
    print(f'Cached {len(frame)} cycles, {int((frame.unstable_flag==0).sum())} stable. '
          f'Cache size {path.stat().st_size/2**20:.2f} MiB',flush=True)
    return data, counts


if __name__=='__main__':
    load_data()
