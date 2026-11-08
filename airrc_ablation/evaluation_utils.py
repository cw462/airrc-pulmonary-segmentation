import csv
import math
import numpy as np


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    return None if isinstance(value,float) and not math.isfinite(value) else value


def stats(values):
    values=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {'mean':float(np.mean(values)) if values else None,
            'sample_std':float(np.std(values,ddof=1)) if len(values)>1 else None,
            'median':float(np.median(values)) if values else None,'n_valid':len(values)}


def csv_output(path,rows,fields):
    temp=path.with_suffix('.csv.part')
    with temp.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in rows:writer.writerow({name:('nan' if row.get(name) is None else row.get(name)) for name in fields})
    temp.replace(path)
