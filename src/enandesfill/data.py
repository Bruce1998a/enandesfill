import pandas as pd
from importlib import resources

def load_example():
    """Carga dataset de ejemplo con NA (2020).
    Devuelve DataFrame con columnas: date (dd/mm/yyyy), value (float).
    """
    with resources.files(__package__).joinpath('ejemplo_fill_py.csv').open('rb') as f:
        df = pd.read_csv(f)
    return df
