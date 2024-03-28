
import numpy as np
import pandas as pd
import datetime as dt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DataManipulator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self._replace_datetime()
        self._drop_columns()
        self._preprocess_data()
        self._rename_columns()
    
    def _replace_datetime(self):
        self.data['created'] = self.data['created'].str.replace(r'^0014', '2014', regex=True)
        self.data['ended'] = self.data['ended'].str.replace(r'^0014', '2014', regex=True)
        self.data['created'] = self.data['created'].str.replace(r'^0015', '2015', regex=True)
        self.data['ended'] = self.data['ended'].str.replace(r'^0015', '2015', regex=True)

    def _drop_columns(self):
        col_to_remove=['distance', 'managerVehicle', 'Mon','Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun', 'reportedZip']
        self.data.drop(col_to_remove, axis=1, inplace=True)

    def _preprocess_data(self):
        self.data['created'] = pd.to_datetime(self.data['created'])
        self.data['ended'] = pd.to_datetime(self.data['ended'])
        self.data['month'] = self.data['created'].dt.month

    def _rename_columns(self):
        cols_to_rename={
            'created': 'sessionStartDatetime',
            'ended': 'sessionEndDatetime',
            'dollars': 'chargingCost',
            'platform': 'appUsed'
            }
        self.data.rename(cols_to_rename, axis=1, inplace=True)

    @staticmethod
    def _append_columns_dtype(data1):
        # convert facilityType to Catagorical data
        data1['facilityType_cat'] = data1['facilityType'].replace([1,2,3,4],['Manufacturing','Office','R&D','Others'])
        # convert Catgorical to Ordinal data
        data1['appUsed_ord'] = data1['appUsed'].map({'ios':0,'android':1, 'web':2})


    @staticmethod
    def _remove_outliers(data1, col_name='chargeTimeHrs'):

        Q1= data1[col_name].quantile(0.25)
        Q3= data1[col_name].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound= Q1 - 1.5* IQR
        upper_bound= Q3 + 1.5* IQR

        data2 = data1[(data1[col_name] > lower_bound) & (data1[col_name] < upper_bound)]

        return data2

class DataVisualizer:
    def __init__(self, df):
        self.df=df
   
    def plot_sessions_per_hour(self, time_column, count_column, title, xlabel, ylabel):
        # Aggregate data: Group by time_column and count occurrences in count_column
        aggregated_data = self.df.groupby(self.df[time_column].dt.hour)[count_column].count()
        
        # Ensure x data matches aggregated y data
        x = aggregated_data.index
        y = aggregated_data.values

        # Plotting
        plt.figure()
        plt.plot(x, y, linestyle='solid', marker='o')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(x)  
        plt.grid(True)
        plt.show()

    def plot_energy_consumption(self, x_col='startTime', y_col='kwhTotal'):

        avg_energy_by_hour = self.df.groupby(x_col)[y_col].mean().reset_index()

        # Plotting
        plt.figure(figsize=(12, 6))
        sns.barplot(data=avg_energy_by_hour, x=x_col, y=y_col, palette='coolwarm')
        plt.title('Average Total Energy Consumption by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Total Energy Consumption (kWh)')
        plt.xticks(rotation=45)  
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()


    @staticmethod
    def plot_daily_energy_consumption(df0, y='kwhTotal', x='weekday', hue='facilityType_cat' ):

        plt.figure(figsize=(10,6))
        sns.swarmplot(data=df0, y=y, x=x, hue=hue)
        plt.title('Daily Energy Consumption')
        plt.xlabel('Days')
        plt.ylabel('Energy (kWh)')
        plt.show() 

    @staticmethod
    def plot_charging_cost_charging_duration(df1, title, y='chargeTimeHrs', x='chargingCost', hue='weekday'):
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=df1, y=y, x=x, hue=hue)
        plt.title(title)
        plt.xlabel('Charging Cost ($)')
        plt.ylabel('Charging Duration (hrs)')
        plt.show()

    @staticmethod
    def plot_chargetime_per_day_app(df1):
        plt.figure(figsize=(10,6))
        sns.swarmplot(data=df1, y='chargeTimeHrs', x='weekday', hue='appUsed')
        plt.title('Charging Duration per day w.r.t App used')
        plt.xlabel('Day of the week')
        plt.ylabel('Charging Duration (hrs)')
        plt.show()

    @staticmethod
    def plot_corrmat(df1):
        numeric_cols = df1.select_dtypes(include=[np.number]).columns.tolist()
        df1= df1[numeric_cols].copy()

        corrmat = df1.corr()

        plt.figure(figsize=(12,12))  
        sns.heatmap(corrmat, annot = True, cmap = 'mako', center = 0)
        plt.show()

    @staticmethod
    def plot_evcs_facility(df1):
        plt.figure(figsize=(10,6))
        df1['facilityType_cat'].value_counts().plot(kind='pie', autopct='%1.1f%%', shadow=False, pctdistance=0.85,
                                                    startangle=90)
        # draw circle
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        
        # Adding Circle in Pie chart
        fig.gca().add_artist(centre_circle)

        plt.title('Charging Stations installed at different facilities')
        plt.ylabel('')
        plt.show()

    @staticmethod
    def plot_multiple_features_facility(df1, vars, hue='facilityType_cat', palette='Set1'):
        sns.pairplot(df1, vars=vars, hue=hue, palette=palette)
        plt.show()

class DataHandler:
    def __init__(self):
        self.data=None

    # Function to categorize each session
    def categorize_duration(self, row):
        if row['chargeTimeHrs'] < 2:
            return 'Short'
        elif row['chargeTimeHrs'] <= 4:
            return 'Medium'
        else:
            return 'Long'
    
    def apply_cat_duration(self, data):
        if self.data is None:
            self.data = data.copy()  
        else:
            self.data.update(data)   
        self.data['Duration_Category'] = self.data.apply(self.categorize_duration, axis=1)
    
    def check_distribution_duration_cat(self):
        # Check the distribution of the new categories
        print(self.data['Duration_Category'].value_counts())
        
    def plot_distribution_duration_cat(self):

        plt.figure(figsize=(8, 6))
        sns.countplot(data=self.data, x='Duration_Category', order=['Short', 'Medium', 'Long'])
        plt.title('Distribution of Charging Session Durations')
        plt.xlabel('Duration Category')
        plt.ylabel('Number of Sessions')
        plt.show()


class DimensionReducer:
    def __init__(self, data, req_cols):
        self.data = data[req_cols]
        self.scaled_data = None
        self.pca = None
        self.principal_components = None
        self.component_matrix = None
        self.principal_component_df = None 

    def scale_features(self):
        scaler = StandardScaler()
        self.scaled_data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)

    def apply_pca(self, n_components=3):
        self.pca = PCA(n_components=n_components)
        self.principal_components = self.pca.fit_transform(self.scaled_data)
        self.component_matrix = self.pca.components_.T

        # Store the principal components in a DataFrame
        pc_columns = ['PC' + str(i+1) for i in range(n_components)]
        self.principal_component_df = pd.DataFrame(self.principal_components, columns=pc_columns)

    def print_component_matrix(self):
        component_df = pd.DataFrame(self.component_matrix, index=self.data.columns, columns=['W' + str(i+1) for i in range(self.component_matrix.shape[1])])
        print(component_df)

    def visualize_variance(self):
        sns.barplot(x=list(range(1, self.pca.n_components_ + 1)), y=self.pca.explained_variance_, palette='GnBu_r')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.title('PCA Variance Explained')
        plt.show()

    def print_explained_variance(self):
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()
        print("Explained Variance Ratio:", explained_variance)
        print("Cumulative Explained Variance:", cumulative_variance)

    def visualize_principal_components(self, dim=3):
        if dim == 3:
            fig = plt.figure(figsize=(13, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.principal_components[:, 0], self.principal_components[:, 1], self.principal_components[:, 2], c='darkred', marker='o')
            ax.set_title('3D Projection of Data In the Reduced Dimension')
            plt.show()
        elif dim == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.principal_components[:, 0], self.principal_components[:, 1], alpha=0.5)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('PCA with 2 Components of EV Charging Data')
            plt.grid(True)
            plt.show()
        else:
            print("Visualization for the given dimension is not supported.")


class KMeansElbowMethod:
    def __init__(self, data, k_range):
        self.data = data
        self.k_range = k_range
        self.inertias = []

    def find_optimal_clusters(self):
        for k in self.k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.data)
            self.inertias.append(kmeans.inertia_)

    def plot_elbow_curve(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.k_range, self.inertias, 'o-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.grid(True)
        plt.show()


class ClusterAnalysis:
    def __init__(self, data):
        self.data = data
    
    def apply_kmeans(self, df, n_clusters=4):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km_clusters = kmeans.fit_predict(self.data)
        self.data['KM_clusters'] = km_clusters
        df.loc[:, 'KM_clusters'] = km_clusters
        self.visualize_clusters('KM_clusters', 'K-Means Clustering')
        return df
    
    def apply_agglomerative(self, df, n_clusters=4):
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        ac_clusters = agglomerative.fit_predict(self.data)
        self.data['AC_clusters'] = ac_clusters
        df.loc[:, 'AC_clusters'] = ac_clusters
        self.visualize_clusters('AC_clusters', 'Agglomerative Clustering')
        return df
    
    def apply_dbscan(self, eps=0.2, min_samples=4):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.data['DBSCAN_clusters'] = dbscan.fit_predict(self.data)
        self.visualize_clusters('DBSCAN_clusters', 'DBSCAN Clustering')
    
    def apply_gmm(self, n_components=4):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        self.data['GMM_clusters'] = gmm.fit_predict(self.data)
        self.visualize_clusters('GMM_clusters', 'Gaussian Mixture Model')
    
    def visualize_clusters(self, cluster_col, title):
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x='PC1', y='PC2', hue=cluster_col, data=self.data, palette='viridis', alpha=0.5)
        plt.title(title)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def compute_metrics(self, cluster_col):
        silhouette = metrics.silhouette_score(self.data[['PC1', 'PC2']], self.data[cluster_col])
        calinski_harabasz = metrics.calinski_harabasz_score(self.data[['PC1', 'PC2']], self.data[cluster_col])
        davies_bouldin = metrics.davies_bouldin_score(self.data[['PC1', 'PC2']], self.data[cluster_col])
        print(f"{cluster_col} Metrics:")
        print(f"Silhouette Score: {silhouette:.2f}")
        print(f"Calinski-Harabasz Index: {calinski_harabasz:.2f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.2f}")


class SegementationPlotter:
    def __init__(self, df):
        self.df=df
        self.pal =['blue', 'green', 'red', 'purple']

    def plot_km_cluster_distribution(self, cluster_col='KM_clusters'):
        plt.figure(figsize=(13, 8))
        sns.countplot(x=cluster_col, data=self.df, palette=self.pal)
        plt.title('Distribution Of KM Clusters')
        plt.show()

    def plot_cluster_scatter(self, x_col, y_col, title, cluster_col='KM_clusters'):
        plt.figure(figsize=(13, 8))
        sns.scatterplot(x=x_col, y=y_col, hue=cluster_col, data=self.df, palette=self.pal, style=cluster_col, legend='full')
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        plt.show()

    def plot_cluster_strip_boxen(self, y_col, title, cluster_col='KM_clusters'):
        plt.figure(figsize=(13, 8))
        sns.stripplot(data=self.df, x=cluster_col, y=y_col, size=3, palette=self.pal)  
        sns.boxenplot(x=cluster_col, y=y_col, data=self.df, palette=self.pal)
        plt.title(title)
        plt.show()


class RFMAnalysis:
    def __init__(self, data):
        self.data = data.copy()
        self.rfm_df = None
        self.cluster_summary = None

    def calculate_rfm(self):
        # Set the reference date as one day after the last charging session in your data
        reference_date = self.data['sessionStartDatetime'].max() + pd.Timedelta(days=1)

        # Calculate Recency, Frequency, and Monetary values
        self.data['Recency'] = (reference_date - self.data.groupby('userId')['sessionStartDatetime'].transform('max')).dt.days
        self.data['Frequency'] = self.data.groupby('userId')['sessionId'].transform('count')
        self.data['Monetary'] = self.data.groupby('userId')['chargingCost'].transform('sum')

        # Drop duplicates to get a DataFrame with one row per user
        self.rfm_df = self.data[['userId', 'Recency', 'Frequency', 'Monetary']].drop_duplicates()

    def assign_rfm_quartiles(self):
        # Assign quartile labels to Recency, Frequency, and Monetary
        bins = [self.rfm_df['Recency'].min(), 30, 60, 90, self.rfm_df['Recency'].max()]

        self.rfm_df['R_Quartile'] = pd.cut(self.rfm_df['Recency'], bins=bins, labels=['4', '3', '2', '1'], include_lowest=True)
        self.rfm_df['F_Quartile'] = pd.qcut(self.rfm_df['Frequency'], 4, labels=['1', '2', '3', '4'], duplicates='drop')
        self.rfm_df['M_Quartile'] = pd.cut(self.rfm_df['Monetary'], bins=bins, labels=['1', '2', '3', '4'], include_lowest=True)

        # Combine the quartile values to create an RFM Score
        self.rfm_df['RFM_Score'] = self.rfm_df['R_Quartile'].astype(str) + self.rfm_df['F_Quartile'].astype(str) + self.rfm_df['M_Quartile'].astype(str)

    def perform_clustering(self, n_clusters=4):
        # Scale the RFM values
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(self.rfm_df[['Recency', 'Frequency', 'Monetary']])

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    def summarize_clusters(self):
        # Aggregate the RFM values by Cluster to understand the average characteristics
        self.cluster_summary = self.rfm_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'count']
        }).reset_index()
        self.cluster_summary.columns = ['Cluster', 'Average Recency', 'Average Frequency', 'Average Monetary', 'Count']

    def visualize_clusters(self):
        # 3D visualization of the clusters
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        colors = ['blue', 'green', 'red', 'purple']

        for i in range(len(self.cluster_summary)):
            ax.scatter(self.rfm_df[self.rfm_df['Cluster'] == i]['Recency'],
                       self.rfm_df[self.rfm_df['Cluster'] == i]['Frequency'],
                       self.rfm_df[self.rfm_df['Cluster'] == i]['Monetary'],
                       s=50, c=colors[i], label=f'Cluster {i}')

        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        plt.title('3D view of K-Means Clusters')
        plt.legend()
        plt.show()