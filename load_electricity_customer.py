"""Retain only the first customer's series from the official UCI archive.

The 249 MB ZIP stays in memory; the 678 MB text and other customers are not
saved to disk. Customer MT_001 is chosen before model fitting or evaluation.
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'data'/'real_datasets'/'electricity_mt001'
URL='https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip'
CUSTOMER='MT_001'


def load_data():
    CACHE.mkdir(parents=True,exist_ok=True)
    path=CACHE/'series.npz'
    if not path.exists():
        archive_buffer=io.BytesIO();hasher=hashlib.sha256();total=0;last=0
        print('Downloading official UCI electricity archive into memory...',flush=True)
        with requests.get(URL,stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(2**20):
                archive_buffer.write(chunk);hasher.update(chunk);total+=len(chunk)
                if total-last>=32*2**20:
                    print(f'Received {total/2**20:.0f} MiB',flush=True);last=total
        archive_buffer.seek(0)
        print('Extracting only timestamps and MT_001 into compact cache...',flush=True)
        with zipfile.ZipFile(archive_buffer) as archive:
            names=[name for name in archive.namelist() if Path(name).name=='LD2011_2014.txt']
            if len(names)!=1:raise ValueError(names)
            with archive.open(names[0]) as stream:
                frame=pd.read_csv(stream,sep=';',decimal=',',usecols=[0,1])
        if frame.columns[1]!=CUSTOMER:raise ValueError(f'Unexpected first customer: {frame.columns[1]}')
        stamps=pd.to_datetime(frame.iloc[:,0]);values=frame.iloc[:,1].to_numpy(float)
        if not np.isfinite(values).all() or np.any(values<0):raise ValueError('Invalid source readings')
        if stamps.duplicated().any() or not stamps.is_monotonic_increasing:raise ValueError('Timestamp ordering')
        np.savez_compressed(path,timestamps=stamps.to_numpy(dtype='datetime64[ns]'),y=values)
        metadata=dict(source_url=URL,source_page='https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014',
            citation='Trindade (2015), ElectricityLoadDiagrams20112014, UCI, DOI 10.24432/C58C86',
            license='CC BY 4.0',customer=CUSTOMER,customer_selection='First source customer, chosen before evaluation.',
            archive_sha256=hasher.hexdigest(),archive_bytes=total,archive_retained=False,other_customers_retained=False,
            unit='kW',timestamps='Recorded Portuguese local-time labels; no timezone conversion.',
            source_warning='Pre-activation zeros and DST adjustments; see UCI documentation.',
            n_observations=len(values),first_timestamp=str(stamps.iloc[0]),last_timestamp=str(stamps.iloc[-1]))
        (CACHE/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    with np.load(path) as saved:data={key:saved[key] for key in saved.files}
    print(f'{CUSTOMER}: {len(data["y"])} readings; cache {path.stat().st_size/2**20:.2f} MiB',flush=True)
    return data


def daily_curves(data):
    """Calendar-day groups using source labels, with quality/exclusion flags."""
    index=pd.DatetimeIndex(data['timestamps']);series=pd.Series(data['y'],index=index)
    curves=[];records=[]
    for date,day in series.groupby(index.normalize()):
        hours=day.index.hour+day.index.minute/60
        complete=len(day)==96 and np.array_equal(hours,np.arange(96)/4)
        # Portuguese clock changes: last Sunday of March and October.
        next_week=date+pd.Timedelta(days=7)
        dst=date.month in (3,10) and date.dayofweek==6 and next_week.month!=date.month
        record=dict(date=str(date.date()),n_readings=len(day),complete=bool(complete),dst_adjusted=bool(dst),
                    all_zero=bool((day==0).all()),weekday=int(date.dayofweek),daily_mean=float(day.mean()),
                    daily_min=float(day.min()),daily_max=float(day.max()),zero_fraction=float((day==0).mean()))
        records.append(record)
        if complete:curves.append((np.datetime64(date.date()),day.to_numpy()))
    return np.array([date for date,_ in curves]),np.stack([y for _,y in curves]),pd.DataFrame(records)


if __name__=='__main__':
    data=load_data();dates,y,quality=daily_curves(data)
    quality.to_csv(CACHE/'daily_quality.csv',index=False)
    print(quality[quality.date.between('2014-01-01','2014-03-31')].to_string(index=False),flush=True)
