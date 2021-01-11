import pandas as pd
import numpy as np

def show_status(config_fname):
    status = pd.read_json(config_fname)
    status.insert(0, "Algorithm", status.index)
    pandas_df_to_markdown_table(status)

def pandas_df_to_markdown_table(df):
    from IPython.display import Markdown, display
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    display(Markdown(df_formatted.to_csv(sep="|", index=False)))

def load_batch(log_records, idx_offset=0):
    dat = pd.DataFrame()
    for path, label in log_records:
        d = pd.read_csv(path)
        d = d.set_index('iteration').rename(columns={'exploitability':label})
        dat = dat.join(d, how='outer')
    dat.index += idx_offset
    return dat

def log_interp(df, method="index"):
    df.index = df.index.map(np.log)
    df = df.apply(np.log)
    
    df = df.interpolate(method=method)
    
    df.index = df.index.map(np.exp)
    df = df.apply(np.exp)
    return df