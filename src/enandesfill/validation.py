import pandas as pd

def verify_and_regularize(df, start_date, end_date, dayfirst=True):
    """Verifica columnas y genera secuencia completa de fechas.
    df: columnas 'date' y 'value'. 'date' puede ser dd/mm/yyyy o ISO.
    start_date/end_date: 'dd/mm/yyyy'.
    Devuelve DataFrame con fechas completas y value (NaN donde falte).
    """
    if not {'date','value'}.issubset(df.columns):
        raise ValueError("Se requieren columnas 'date' y 'value'.")
    f = pd.to_datetime(df['date'], dayfirst=dayfirst, errors='coerce')
    if f.isna().any():
        f = pd.to_datetime(df['date'], dayfirst=False, errors='coerce')
    if f.isna().any():
        raise ValueError("Fechas inválidas, use dd/mm/yyyy o ISO (YYYY-MM-DD).")
    df2 = pd.DataFrame({'date': f, 'value': pd.to_numeric(df['value'], errors='coerce')})
    s = pd.date_range(pd.to_datetime(start_date, dayfirst=True),
                      pd.to_datetime(end_date, dayfirst=True), freq='D')
    out = pd.DataFrame({'date': s}).merge(df2, how='left', on='date')
    out['date'] = out['date'].dt.strftime('%d/%m/%Y')
    return out
