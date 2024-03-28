
from utils import DataManipulator, DataVisualizer, DataHandler, DimensionReducer, KMeansElbowMethod, ClusterAnalysis, SegementationPlotter, RFMAnalysis

# Load and preprocess data
manipulator = DataManipulator('data/evcs_data.csv')
manipulator.load_data()
df_evcs = manipulator.data

df_evcs.info()

df=df_evcs.copy()

print(df.head())

print(df[df.isnull().any(axis=1)])

print(df.isnull().sum())
print(df.duplicated().sum())

columns=['kwhTotal','chargingCost','startTime','endTime','chargeTimeHrs']

print(df[columns].describe())


visualizer = DataVisualizer(df)


# visualize Charging sessions per Hour

visualizer.plot_sessions_per_hour(
    'sessionStartDatetime',  
    'sessionId',
    'Charging sessions per Hour',
    'Time of Day',
    'Sessions',
)
        
# visualize energy consumption
visualizer.plot_energy_consumption()

dfj=df.copy()

print(dfj['appUsed'].unique())
print(dfj['facilityType'].unique())

print(dfj['userId'].nunique())
print(dfj['stationId'].nunique())
print(dfj['locationId'].nunique())


manipulator._append_columns_dtype(dfj)

dfj.info()

print(dfj.head())

# visualize daily energy consumption
visualizer.plot_daily_energy_consumption(dfj) 

# remove outliers in charging duration
dfk = manipulator._remove_outliers(dfj)

# Charging Cost with respect to the charging duration

# visualize charging cost w.r.t charging duration 
visualizer.plot_charging_cost_charging_duration(dfk, "Charging cost w.r.t charging duration")


# Average charging duration per day 

# visualize mean chargeTimeHr per day w.r.t to platform used
visualizer.plot_chargetime_per_day_app(dfk)

count_percent= dfk['appUsed'].value_counts() /len(dfk) * 100
print(f'App used count % \n {count_percent}')


visualizer.plot_corrmat(dfk)  


# Installed Charging Stations Share at the Facility
# pie chart of facilityType
visualizer.plot_evcs_facility(dfk)


vars=['chargeTimeHrs','kwhTotal', 'chargingCost']
visualizer.plot_multiple_features_facility(dfk, vars)

handler=DataHandler()
handler.apply_cat_duration(dfk)
handler.check_distribution_duration_cat()
handler.plot_distribution_duration_cat()

# Usage
req_cols = ['userId', 'kwhTotal', 'chargeTimeHrs', 'chargingCost', 'stationId', 'facilityType', 'sessionId', 'appUsed_ord', 'startTime', 'endTime']
dimension_reducer = DimensionReducer(dfk, req_cols)

dimension_reducer.scale_features()

dimension_reducer.apply_pca(n_components=3)
dimension_reducer.print_component_matrix()
dimension_reducer.visualize_variance()
dimension_reducer.print_explained_variance()
dimension_reducer.visualize_principal_components(dim=3)

# To apply and visualize PCA with 2 components
dimension_reducer.apply_pca(n_components=2)
dimension_reducer.print_component_matrix()
dimension_reducer.visualize_variance()
dimension_reducer.print_explained_variance()
dimension_reducer.visualize_principal_components(dim=2)

df_PCA2 = dimension_reducer.principal_component_df
print(df_PCA2.head())  
print(df_PCA2.describe().T)

elbow_method = KMeansElbowMethod(df_PCA2, range(1, 11))
elbow_method.find_optimal_clusters()
elbow_method.plot_elbow_curve()


# Clustering Methods

cluster_analyzer = ClusterAnalysis(df_PCA2)  

# Apply clustering algorithms
cluster_analyzer.apply_kmeans(dfk)
cluster_analyzer.apply_agglomerative(dfk)
cluster_analyzer.apply_dbscan()
cluster_analyzer.apply_gmm()

print(dfk.head())

# Compute metrics for K-means clustering
cluster_analyzer.compute_metrics('KM_clusters')

# Compute metrics for Agglomerative Clustering
cluster_analyzer.compute_metrics('AC_clusters')


splotter=SegementationPlotter(dfk)
splotter.plot_km_cluster_distribution(cluster_col='KM_clusters')
splotter.plot_cluster_strip_boxen('kwhTotal', 'KMeans Clusters of EV Charging Behavior by total energy consumption')
splotter.plot_cluster_scatter('chargeTimeHrs', 'kwhTotal', 'KMeans Clusters of EV Charging Behavior by charging duration')
splotter.plot_cluster_scatter('startTime', 'kwhTotal', 'KMeans Clusters of EV Charging Behavior by Hour of Day')



# RFM (Recency, Frequency, Monetary) analysis for EV charging

rfm_analyzer = RFMAnalysis(dfj)

rfm_analyzer.calculate_rfm()
rfm_analyzer.assign_rfm_quartiles()
rfm_analyzer.perform_clustering(n_clusters=4)
rfm_analyzer.summarize_clusters()

print(rfm_analyzer.cluster_summary)

rfm_analyzer.visualize_clusters()




