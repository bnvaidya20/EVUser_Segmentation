# Electric Vehicle (EV) Charging Sessions Analysis and EV User Segmentation

This project delves into the analysis of electric vehicle (EV) charging sessions, utilizing high-resolution data to understand EV user behavior and segment users based on their charging patterns. By employing various data analysis and machine learning techniques, this study aims to provide insights into the diverse needs and preferences of EV users, particularly within a workplace charging context.

## Dataset

The dataset is comprised of 3395 detailed records of EV charging sessions in .CSV format. These records encompass various attributes including charging duration, energy consumption, charging cost, station ID, user ID, and the platform used for charging, among others. The data encapsulates the charging activities of 85 EV drivers across 105 stations situated in 25 different workplace sites. These sites span an array of facility types such as R&D centers, manufacturing sites, testing facilities, and office headquarters, offering a comprehensive view of workplace EV charging dynamics.

## Analysis Methodology

The analysis methodology encompasses a thorough examination of the EV charging data, focusing on identifying patterns and trends in EV charging behavior. Key aspects such as charging duration, charging cost, total energy consumption, and start time are scrutinized to draw meaningful conclusions about user preferences and charging habits.

## Analysis Steps

1. **Data Preprocessing**: Cleaning and preparing the data for analysis, including handling  outliers.
2. **Exploratory Data Analysis (EDA)**: Conducting an initial exploration to understand the dataset's characteristics and identify significant variables.
3. **Principal Component Analysis (PCA)**: Applying PCA to reduce the dimensionality of the data, capturing the most important information in fewer dimensions.
4. **Use of Clustering Techniques**: Strategic application of various clustering techniques including K-means clustering, Agglomerative clustering, DBSCAN clustering, and Gaussian Mixture Model clustering to segment the user base according to their charging habits and preferences.
5. **User Segmentation**: Applying clustering techniques like K-means and RFM analysis to segment EV users based on various attributes.

## Results and Analysis of EV User Segmentation

### K-means Clustering 

#### Total Energy Consumption

 [Total Energy Consumption](img/Fig_tec.png)

- **Cluster 0**: Users in this cluster tend to have minimal energy consumption during their charging sessions. These could be users with short commutes or those who predominantly rely on charging at home or at destinations with longer dwell times, thus rarely needing substantial charges from public or workplace charging stations.
- **Cluster 1**: This cluster signifies users with high energy consumption per charging session. Users in this category might be those who rely heavily on public or workplace charging as their primary charging source, possibly due to lack of home charging options.
- **Cluster 2**: Users in this cluster exhibit moderate energy consumption during their charging sessions. Their charging behavior might be characterized by a mix of regular commutes and occasional long trips, leading to a moderate need for energy per session.
- **Cluster 3**: This cluster is marked by very low energy consumption. This segment might include users who only need minimal energy per session, possibly due to having multiple charging options available to them (home, work, public) or using EVs for very short distances.

#### Total Energy vs. Charging Duration

[Total Energy vs. Charging Duration](img/Fig_tec_cd.png)

- **Cluster 0**: Characterized by medium-duration charging sessions with low to medium energy consumption. This group likely represents regular users utilizing workplace charging for daily commutes.
- **Cluster 1**: Identified by long-duration charging sessions and a wide range of energy consumption, suggesting the use of fast chargers or charging from a significantly depleted state.
- **Cluster 2 & 3**: These clusters include users engaging in short to medium-duration charging sessions with low to medium energy consumption, indicative of occasional users or those with access to alternative charging solutions.

#### Total Energy vs. Start Time

[Total Energy vs. Start Time](img/Fig_tec_st.png)

- **Cluster 0 & 3**: These clusters primarily involve daytime and late-afternoon to evening charging sessions, respectively, with moderate energy consumption, reflecting workplace or commuter charging patterns.
- **Cluster 1**: Encompasses late morning to night charging sessions with higher energy consumption, possibly representing less frequent but more prolonged charging sessions.
- **Cluster 2**: Exhibits a broad range of start times with varying energy consumption, suggesting a mix of overnight and daytime charging behaviors.

### RFM (Recency, Frequency, Monetary) Analysis

The RFM analysis segments users into four distinct clusters based on their recency of use, frequency of charging sessions, and monetary spending on charging.

[RFM (Recency, Frequency, Monetary)](img/Fig_rfm.png)

- **Cluster 0 (Frequent/Low Spend Users)**: High frequency but low monetary spending per session.
- **Cluster 1 (New/Low Engagement Users)**: Recent engagement with low frequency and spending.
- **Cluster 2 (Loyal/High-Value Users)**: High frequency and monetary spending, indicating a core user base.
- **Cluster 3 (Lapsed/Low Engagement Users)**: Low engagement and spending, suggesting lapsed or infrequent users.

### Strategic Implications

The segmentation offers actionable insights for tailoring services and engagement strategies to cater to the distinct needs of each user group. Enhancing user experience, optimizing charging infrastructure, and implementing targeted marketing strategies could significantly improve service adoption and user satisfaction.

## Conclusion

This analysis underscores the diversity within the EV user base and highlights the importance of understanding user behavior for effective management and expansion of EV charging services. By segmenting users based on their charging patterns and preferences, service providers can develop more personalized and efficient solutions, fostering the broader adoption of electric vehicles.
