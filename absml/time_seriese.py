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
        self.k = 2

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

            params_angle = [int(param[-2:]) for param in params if param.startswith('Bolt_Tightening_Angle')]
            params_torque = [int(param[-2:]) for param in params if param.startswith('Bolt_Tightening_Torque')]
            params_count = [int(param[-2:]) for param in params if param.startswith('Bolt_Tightening_Count')]
            params_rd = [int(param[-2:]) for param in params if param.startswith('Bolt_Tightening_RD_Angle')]

            self.num_angle = max(params_angle) if len(params_angle) > 0 else 0
            self.num_torque = max(params_angle) if len(params_torque) > 0 else 0
            self.num_count = max(params_count) if len(params_count) > 0 else 0
            self.num_rd = max(params_rd) if len(params_rd) > 0 else 0

            cols = []

            self.angle_cols = [f"Bolt_Tightening_Angle_{i+1:02d}" for i in range(self.num_angle)]
            self.torque_cols = [f"Bolt_Tightening_Torque_{i+1:02d}" for i in range(self.num_torque)]
            self.count_cols = [f"Bolt_Tightening_Count_{i+1:02d}" for i in range(self.num_count)]
            self.rd_cols = [f"Bolt_Tightening_RD_Angle_{i+1:02d}" for i in range(self.num_rd)]
            cols = self.angle_cols + self.torque_cols + self.count_cols + self.rd_cols

            pivot_df = df[df['PARAM_NAME_EN'].isin(cols)]
            self.pivot_df = pd.pivot_table(pivot_df, index='LOT_ID', columns='PARAM_NAME_EN', values='VALUE', aggfunc="max")
            
            self.pivot_df[angle_cols] = pivot_df[angle_cols].astype(np.float16)
            self.pivot_df[torque_cols] = pivot_df[torque_cols].astype(np.float32)
            self.pivot_df[rd_cols] = pivot_df[rd_cols].astype(np.float16)
            self.pivot_df[count_cols] = pivot_df[count_cols].astype(np.int32)

    def _plot_kmean_scatters( X_trains, param_pair ):
        kmeans = KMeans(n_clusters=self.k, n_init=20, random_state=42)
        num_charts = len(X_trains)
        row_num, col_num = num_charts // 2 + num_charts % 2, 2
        fig, ax = plt.subplots( ncols=col_num, nrows=row_num, sharex=False, figsize=(14, 3 * row_num ) )

        anomali_indice = []
        for i in range( row_num ):
            ax0 = ax[i, 0] if row_num > 1 else ax[0]
            y0_pred = kmeans.fit_predict(X_trains[i*2])
            anomali_indice.append(y0_pred == 1)
            ax0.scatter( X_trains[i*2][:,0], X_trains[i*2][:,1], c=y0_pred )

            if i * 2 + 2 <= num_charts:
                ax1 = ax[i, 1] if row_num > 1 else ax[1]
                y1_pred = kmeans.fit_predict(X_trains[i*2 + 1])
                ax1.scatter( X_trains[i*2 + 1][:,0], X_trains[i*2 + 1][:,1], c=y1_pred )

            if upper(param_pair) == 'AR' or upper(param_pair) == 'RA':
                ax0.set_xlabel("Angle")
                ax1.set_xlabel("Angle")
                ax0.set_ylabel("Rundown Angle", rotation=0, labelpad=10)
                ax1.set_ylabel("Rundown Angle", rotation=0, labelpad=10)
                ax0.set_title(f"Rundown Angle and Angle at Bolt #{i*2+1}")
                ax1.set_title(f"Rundown Angle and Angle at Bolt #{i*2+2}")
            elif upper(param_pair) == 'RT' or upper(param_pair) == 'TR':
                ax0.set_xlabel("Rundown Angle")
                ax1.set_xlabel("Rundown Angle")
                ax0.set_ylabel("Torque", rotation=0, labelpad=10)
                ax1.set_ylabel("Torque", rotation=0, labelpad=10)
                ax0.set_title(f"Rundown Angle and Torque at Bolt #{i*2+1}")
                ax1.set_title(f"Rundown Angle and Torque at Bolt #{i*2+2}")
            else:
                ax0.set_xlabel("Angle")
                ax1.set_xlabel("Angle")
                ax0.set_ylabel("Torque", rotation=0, labelpad=10)
                ax1.set_ylabel("Torque", rotation=0, labelpad=10)
                ax0.set_title(f"Torque and Angle at Bolt #{i*2+1}")
                ax1.set_title(f"Torque and Angle at Bolt #{i*2+2}")
                
        plt.show()
        return anomali_indice

    def _plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
        core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
        core_mask[dbscan.core_sample_indices_] = True
        anomalies_mask = dbscan.labels_ == -1
        non_core_mask = ~(core_mask | anomalies_mask)

        cores = dbscan.components_
        anomalies = X[anomalies_mask]
        non_cores = X[non_core_mask]
        
        plt.scatter(cores[:, 0], cores[:, 1], c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
        plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
        plt.scatter(anomalies[:, 0], anomalies[:, 1], c="r", marker="x", s=100)
        plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
        if show_xlabels:
            plt.xlabel("Angles")
        else:
            plt.tick_params(labelbottom=False)
        if show_ylabels:
            plt.ylabel("$x_2$", rotation=0)
        else:
            plt.tick_params(labelleft=False)
        plt.title(f"eps={dbscan.eps:.2f}, min_samples={dbscan.min_samples}")
        plt.grid()
        plt.gca().set_axisbelow(True)


    def fit_models( prod_date, param_pair='TA' ):
        X_trains = []
        if upper(param_pair) == 'AR' or upper(param_pair) == 'RA':  #Angles and Rundown
            df = self.pivot_df[self.angle_cols + self.rd_cols]
            for i in range( min([self.num_angle, self.num_rd]) ):
                X_trains.append( df[[f'Bolt_Tightening_Angle_{i+1:02d}', f'Bolt_Tightening_RD_Angle_{i+1:02d}']].values )
        elif upper(param_pair) == 'RT' or upper(param_pair) == 'TR': #Torques and Rundown
            df = self.pivot_df[self.rd_cols + self.torque_cols]
            for i in range( min([self.num_rd, self.num_torque]) ):
                X_trains.append( df[[f'Bolt_Tightening_RD_Angle_{i+1:02d}', f'Bolt_Tightening_Torque_{i+1:02d}']].values )
        else: #Torques and Angles
            df = self.pivot_df[self.angle_cols + self.torque_cols]
            for i in range( min([self.num_rd, self.num_torque]) ):
                X_trains.append( df[[f'Bolt_Tightening_Angle_{i+1:02d}', f'Bolt_Tightening_Torque_{i+1:02d}']].values )

        kmean_anomali_idx = _plot_kmean_scatters( X_trains, param_pair )
        





        