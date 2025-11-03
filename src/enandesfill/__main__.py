import argparse
import pandas as pd
from .validation import verify_and_regularize
from .imputers import fill_series

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='CSV con columnas date,value')
    p.add_argument('--output', required=True, help='CSV de salida')
    p.add_argument('--variable', type=int, default=1, help='1=pp, 2=tmedia, 3=tmax, 4=tmin')
    p.add_argument('--method', type=str, default='stl_kalman',
                   choices=['kalman','stl_kalman','knn','mice','rf','missforest','xgboost','lightgbm','autoencoder'])
    p.add_argument('--start', required=False, help='dd/mm/yyyy (opcional)')
    p.add_argument('--end', required=False, help='dd/mm/yyyy (opcional)')
    p.add_argument('--seasonal', type=int, default=365, help='periodo estacional (serie diaria=365)')
    args = p.parse_args()

    df = pd.read_csv(args.input)
    if args.start and args.end:
        df = verify_and_regularize(df, args.start, args.end)
    res = fill_series(df, variable=args.variable, method=args.method, seasonal_period=args.seasonal)
    res.to_csv(args.output, index=False)
    print('Guardado en', args.output)

if __name__ == '__main__':
    main()
