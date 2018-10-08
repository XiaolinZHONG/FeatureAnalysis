import pandas as pd
import numpy as np


class PandasTools():
    def __init__(self, data_path, sep=",", nan_values=None, time_column=True):
        self.data_path = data_path
        self.nan_values = nan_values
        self.time_column = time_column
        self.sep = sep

    def read_data(self, nrows=None):
        na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A',
                     'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null', '\\n', '\\N', "-900", "-1"]
        na_values = na_values.append(self.nan_values)
        reader = pd.read_csv(self.data_path, sep=self.sep, iterator=True, parse_dates=self.time_column,
                             na_values=na_values, nrows=nrows, infer_datetime_format=True,
                             error_bad_lines=False, engine='c')
        chunks = []
        loop = True
        i = 1
        while loop:
            try:
                chunk = reader.get_chunk(300000000)
                chunks.append(chunk)
                print("Reading Data Chunks ", i)
                i += 1
            except StopIteration:
                loop = False
                print('Iteration is stopped.')
        data = pd.concat(chunks, ignore_index=True)
        print("Read Data Done")

        df0 = data.select_dtypes(include=["int"]).apply(pd.to_numeric, downcast="unsigned")
        df1 = data.select_dtypes(include=["float"]).apply(pd.to_numeric, downcast="float")
        df2 = data.select_dtypes(exclude=["int", "float"])
        data = pd.concat([df2, df1, df0], axis=1)
        print(data.info())
        return data
