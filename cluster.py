import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from decouple import config

class Cluster:
    def __init__():
        self.data_root = config('DATA_ROOT')
        self.soper = None
        self.data_file_path = None
        self.drop_columns = [
            'Unnamed: 0', 'SOPER', 'SEQ_NO', 'FACTORY', 'SHOP', 'LINE', 'INIT_LINE', 'RES_ID', 'SHAPE', 'PROD_TYPE', 'MAT_ID', 'W_MAT_ID',
            'DO_NO', 'WO_NO', 'WO_TYPE', 'WORKER', 'CUST_ID', 'RECIPE', 'STEP_COND', 'PARAM_NAME_KO', 'PARAM_NAME_CN', 'LSL', 'CENTER', 'USL', 
            'PART_MONTH', 'CREATE_USER_ID', 'CREATE_TIME', 'UPDATE_USER_ID', 'UPDATE_TIME', 'RCLAMP', 'SHIFT'
        ]

    def preprocess_input_data( prod_date, line='PK03', step='P30' ):
        self.prod_date = prod_date
        self.line = line
        self.step = step

        if self.line == 'PK03':
            if self.step == 'P30':
                self.soper = '9020'

        if self.soper is not None:
            self.data_file_path = Path( self.data_root ) / self.prod_date / f"{self.line}-{self.soper}-{self.prod_date}.csv"
            df = pd.read_csv( self.data_file_path )
            df.drop(drop_cols, axis=1, inplace=True)

            self.df = df[df.STEP_NAME == int(self.step[1:])]

            params = P30_df.PARAM_NAME_EN.value_counts()
            params = [param for param in params.index if param.startswith('Bolt_Tightening')]


        