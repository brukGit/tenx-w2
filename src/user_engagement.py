import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class UserEngagement:
    """
    A class to analyze and aggregate user engagement metrics based on MSISDN (customer ID).
    
    Metrics:
        - Sessions frequency
        - Duration of the session
        - Total traffic (download and upload in bytes)

    Attributes:
        df (DataFrame): DataFrame containing the telecom dataset.
    """

    def __init__(self, df):
        """
        Initializes the UserEngagement class with the provided DataFrame.
        
        :param df: DataFrame containing columns like 'MSISDN/Number', 'Dur. (ms)', 'Total DL (Bytes)', and 'Total UL (Bytes)'.
        """
        self.df = df

    def aggregate_user_metrics(self):
        """
        Aggregates user engagement metrics for each customer (MSISDN/Number).
        
        Metrics:
            - Sessions frequency: Count of sessions per user.
            - Total session duration: Sum of 'Dur. (ms)' per user.
            - Total data traffic: Sum of 'Total DL (Bytes)' + 'Total UL (Bytes)' per user.
        
        :return: DataFrame containing aggregated metrics for each user.
        """
        # Aggregate metrics by MSISDN/Number
        user_metrics = self.df.groupby('MSISDN/Number').agg({
            'MSISDN/Number': 'count',  # Sessions frequency
            'Dur. (ms)': 'sum',  # Total session duration
            'Total DL (Bytes)': 'sum',  # Total download traffic
            'Total UL (Bytes)': 'sum'  # Total upload traffic
        }).rename(columns={
            'MSISDN/Number': 'Sessions Frequency',
            'Dur. (ms)': 'Total Session Duration (ms)',
            'Total DL (Bytes)': 'Total Download (Bytes)',
            'Total UL (Bytes)': 'Total Upload (Bytes)'
        })
        
        # Calculate total traffic (Download + Upload)
        user_metrics['Total Traffic (Bytes)'] = user_metrics['Total Download (Bytes)'] + user_metrics['Total Upload (Bytes)']

        return user_metrics

    def top_customers_by_engagement(self, n=10):
        """
        Reports the top N customers per engagement metric:
            - Sessions Frequency
            - Total Session Duration (ms)
            - Total Traffic (Bytes)
        
        :param n: Number of top customers to return, default is 10.
        :return: Dictionary containing DataFrames for top N customers per metric.
        """
        # Get the aggregated user metrics
        user_metrics = self.aggregate_user_metrics()

        # Sort and get top N customers for each metric
        top_sessions = user_metrics.sort_values('Sessions Frequency', ascending=False).head(n)
        top_duration = user_metrics.sort_values('Total Session Duration (ms)', ascending=False).head(n)
        top_traffic = user_metrics.sort_values('Total Traffic (Bytes)', ascending=False).head(n)

        # Return the top N customers for each metric
        return {
            'Top Sessions Frequency': top_sessions,
            'Top Total Session Duration': top_duration,
            'Top Total Traffic': top_traffic
        }
    def classify_customers_by_engagement(self):
        """
        Normalizes the engagement metrics and classifies customers into 
        three groups using K-Means clustering.
        
        :return: DataFrame with customer IDs and their assigned cluster.
        """
        # Define metrics to be used for clustering
        metrics = ['Sessions Frequency', 'Session Duration', 'Total Traffic (Bytes)']

        # Aggregate per customer the three metrics
        user_metrics = self.df.groupby('MSISDN/Number').agg({
            'Dur. (ms)': 'sum',  # Total session duration
            'Total DL (Bytes)': 'sum',  # Total download data
            'Total UL (Bytes)': 'sum',  # Total upload data
            'MSISDN/Number': 'count'  # Session frequency
        })

        # Calculate total session traffic (DL + UL)
        user_metrics['Total Traffic (Bytes)'] = user_metrics['Total DL (Bytes)'] + user_metrics['Total UL (Bytes)']
        user_metrics.rename(columns={'MSISDN/Number': 'Sessions Frequency', 'Dur. (ms)': 'Session Duration'}, inplace=True)

        # Normalize the data
        scaler = StandardScaler()
        normalized_metrics = scaler.fit_transform(user_metrics[metrics])

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        user_metrics['Cluster'] = kmeans.fit_predict(normalized_metrics)

        # Assign clusters to customers
        user_clusters = user_metrics[['Sessions Frequency', 'Session Duration', 'Total Traffic (Bytes)', 'Cluster']]

        return user_clusters
    
    def analyze_clusters(self, clustered_data):
        """
        Compute the minimum, maximum, average, and total non-normalized engagement metrics for each cluster.
        Visualize and interpret the results.

        :param clustered_data: DataFrame containing engagement metrics with an additional column 'Cluster'.
        :return: Dictionary with descriptive stats for each cluster.
        """
        # Metrics to evaluate
        metrics = ['Sessions Frequency', 'Session Duration', 'Total Traffic (Bytes)']

        # Initialize dictionary to store cluster analysis
        cluster_stats = {}

        # Iterate over each cluster and compute the descriptive statistics
        for cluster in clustered_data['Cluster'].unique():
            cluster_data = clustered_data[clustered_data['Cluster'] == cluster]

            stats = {}
            for metric in metrics:
                stats[metric] = {
                    'min': cluster_data[metric].min(),
                    'max': cluster_data[metric].max(),
                    'mean': cluster_data[metric].mean(),
                    'total': cluster_data[metric].sum()
                }
            cluster_stats[cluster] = stats

        # Create visualizations for each metric
        self._plot_cluster_stats(cluster_stats)

        return cluster_stats

    def _plot_cluster_stats(self, cluster_stats):
        """
        Plot the minimum, maximum, average, and total for each metric and each cluster.

        :param cluster_stats: Dictionary with descriptive stats for each cluster.
        :param metrics: List of engagement metrics.
        
        """
        
        # Engagement metrics
        metrics = ['Sessions Frequency', 'Session Duration', 'Total Traffic (Bytes)']
        for metric in metrics:
            min_vals, max_vals, mean_vals, total_vals = [], [], [], []
            clusters = []

            for cluster, stats in cluster_stats.items():
                clusters.append(cluster)
                min_vals.append(stats[metric]['min'])
                max_vals.append(stats[metric]['max'])
                mean_vals.append(stats[metric]['mean'])
                total_vals.append(stats[metric]['total'])

            # Plot the results
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{metric} Statistics by Cluster', fontsize=16)

            axes[0, 0].bar(clusters, min_vals, color='blue')
            axes[0, 0].set_title(f'Min {metric}')
            axes[0, 0].set_ylabel(metric)

            axes[0, 1].bar(clusters, max_vals, color='orange')
            axes[0, 1].set_title(f'Max {metric}')
            axes[0, 1].set_ylabel(metric)

            axes[1, 0].bar(clusters, mean_vals, color='green')
            axes[1, 0].set_title(f'Mean {metric}')
            axes[1, 0].set_ylabel(metric)

            axes[1, 1].bar(clusters, total_vals, color='red')
            axes[1, 1].set_title(f'Total {metric}')
            axes[1, 1].set_ylabel(metric)

            plt.tight_layout()
            plt.show()

    def top_users_per_application(self):
        """
        Aggregates total traffic per application for each user and returns the top 10 most engaged users per application.
        
        :param df: DataFrame containing user engagement data.
        :return: Dictionary containing the top 10 users per application based on total traffic (DL + UL).
        """
        # Applications and their respective DL and UL columns
        application_columns = {
            'Social Media': ['Social Media DL (Bytes)', 'Social Media UL (Bytes)'],
            'Google': ['Google DL (Bytes)', 'Google UL (Bytes)'],
            'Email': ['Email DL (Bytes)', 'Email UL (Bytes)'],
            'Youtube': ['Youtube DL (Bytes)', 'Youtube UL (Bytes)'],
            'Netflix': ['Netflix DL (Bytes)', 'Netflix UL (Bytes)'],
            'Gaming': ['Gaming DL (Bytes)', 'Gaming UL (Bytes)'],
            'Other': ['Other DL (Bytes)', 'Other UL (Bytes)']
        }

        # Initialize a dictionary to store top 10 users for each application
        top_users_per_app = {}

        for app, cols in application_columns.items():
            # Sum DL + UL traffic for each user (MSISDN/Number) per application
            self.df[app + ' Total (Bytes)'] = self.df[cols[0]] + self.df[cols[1]]

            # Aggregate total traffic per user and sort by the top 10 most engaged users
            top_users = self.df.groupby('MSISDN/Number')[app + ' Total (Bytes)'].sum().sort_values(ascending=False).head(10)

            # Store the result in the dictionary
            top_users_per_app[app] = top_users

        return top_users_per_app
    

    def plot_top_applications(self):
        """
        Plots the top 3 most used applications based on total data traffic.

        :param df: DataFrame containing user engagement data.
        """
        # Applications and their respective DL and UL columns
        application_columns = {
            'Social Media': ['Social Media DL (Bytes)', 'Social Media UL (Bytes)'],
            'Google': ['Google DL (Bytes)', 'Google UL (Bytes)'],
            'Email': ['Email DL (Bytes)', 'Email UL (Bytes)'],
            'Youtube': ['Youtube DL (Bytes)', 'Youtube UL (Bytes)'],
            'Netflix': ['Netflix DL (Bytes)', 'Netflix UL (Bytes)'],
            'Gaming': ['Gaming DL (Bytes)', 'Gaming UL (Bytes)'],
            'Other': ['Other DL (Bytes)', 'Other UL (Bytes)']
        }

        # Compute total data usage for each application
        app_usage = {}
        for app, cols in application_columns.items():
            total_traffic = self.df[cols[0]].sum() + self.df[cols[1]].sum()
            app_usage[app] = total_traffic

        # Convert to DataFrame for plotting
        app_usage_df = pd.DataFrame(list(app_usage.items()), columns=['Application', 'Total Traffic (Bytes)'])
        
        # Find the top 3 applications by total traffic
        top_applications = app_usage_df.sort_values(by='Total Traffic (Bytes)', ascending=False).head(3)

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_applications, x='Application', y='Total Traffic (Bytes)', palette='viridis')

        plt.title('Top 3 Most Used Applications')
        plt.xlabel('Application')
        plt.ylabel('Total Traffic (Bytes)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def k_means_optimal(self):
        """
        Determines the optimal number of clusters for K-Means clustering using the elbow method.
        
        :param metrics: List of column names representing engagement metrics for clustering.
        :return: None
        """
        # Aggregate per customer the required metrics
        user_metrics = self.df.groupby('MSISDN/Number').agg({
            'Dur. (ms)': 'sum',  # Total session duration
            'Total DL (Bytes)': 'sum',  # Total download data
            'Total UL (Bytes)': 'sum',  # Total upload data
            'MSISDN/Number': 'count'  # Session frequency
        })
        
        # Calculate total session traffic (DL + UL)
        user_metrics['Total Traffic (Bytes)'] = user_metrics['Total DL (Bytes)'] + user_metrics['Total UL (Bytes)']
        user_metrics.rename(columns={'MSISDN/Number': 'Sessions Frequency', 'Dur. (ms)': 'Session Duration'}, inplace=True)

        metrics = ['Sessions Frequency', 'Session Duration', 'Total Traffic (Bytes)']

        # Normalize the data
        scaler = StandardScaler()
        normalized_metrics = scaler.fit_transform(user_metrics[metrics])

        # Calculate inertia for a range of k values
        inertias = []
        k_range = range(1, 11)  # Testing k from 1 to 10
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(normalized_metrics)
            inertias.append(kmeans.inertia_)

        # Plot the elbow curve
        plt.figure(figsize=(8, 6))
        plt.plot(k_range, inertias, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.xticks(k_range)
        plt.grid(True)
        plt.show()

