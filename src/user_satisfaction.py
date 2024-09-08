import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine

class UserSatisfaction:
    """
    A class to analyze user satisfaction based on engagement and experience metrics.
    
    This class combines the functionality of UserEngagement and UserExperience
    to calculate satisfaction scores, perform clustering, build a regression model,
    and export results to a PostgreSQL database.
    
    Attributes:
        df (pd.DataFrame): The input DataFrame containing user data.
        engagement_model (UserEngagement): An instance of the UserEngagement class.
        experience_model (UserExperience): An instance of the UserExperience class.
    """

    def __init__(self, df, engagement_model, experience_model):
        """
        Initialize the UserSatisfaction class.

        Args:
            df (pd.DataFrame): The input DataFrame containing user data.
            engagement_model (UserEngagement): An instance of the UserEngagement class.
            experience_model (UserExperience): An instance of the UserExperience class.
        """
        self.df = df
        self.engagement_model = engagement_model
        self.experience_model = experience_model
        self.satisfaction_scores = None

    def calculate_euclidean_distance(self, point, centroid):
        """
        Calculate the Euclidean distance between a point and a centroid.

        Args:
            point (np.array): A data point.
            centroid (np.array): A cluster centroid.

        Returns:
            float: The Euclidean distance between the point and the centroid.
        """
        print(f"calculate_euclidean_distance called with point: {point}, centroid: {centroid}")
        distance = np.linalg.norm(point - centroid)
        print(f"Calculated distance: {distance}")
        return distance

    def assign_engagement_score(self):
        """
        Assign engagement scores to each user based on their distance from the least engaged cluster.

        Returns:
            pd.DataFrame: DataFrame with user IDs and their engagement scores.
        """
        clustered_data = self.engagement_model.classify_customers_by_engagement()
        
        # Debug: Print column names and data types
        print("Columns in clustered_data:", clustered_data.columns)
        print("Data types:", clustered_data.dtypes)
        
        least_engaged_cluster = clustered_data['Cluster'].min()
        least_engaged_centroid = clustered_data[clustered_data['Cluster'] == least_engaged_cluster].mean()
        
        engagement_features = ['Sessions Frequency', 'Session Duration', 'Total Traffic (Bytes)']
        
        # Vectorized distance calculation
        points = clustered_data[engagement_features].values
        centroid = least_engaged_centroid[engagement_features].values
        
        # Calculate distances using numpy operations
        differences = points - centroid
        engagement_scores = np.linalg.norm(differences, axis=1)
        
        # Ensure MSISDN/Number is present and used correctly
        if 'MSISDN/Number' not in clustered_data.columns:
            print("MSISDN/Number not found in columns. Using index as MSISDN/Number.")
            msisdn_numbers = clustered_data.index
        else:
            msisdn_numbers = clustered_data['MSISDN/Number']
        
        result = pd.DataFrame({
            'MSISDN/Number': msisdn_numbers,
            'Engagement Score': engagement_scores
        })
        
        # Debug: Print first few rows of result and data types
        print("First few rows of result DataFrame:")
        print(result.head())
        print("Result data types:", result.dtypes)
        
        return result

    def assign_experience_score(self):
        """
        Assign experience scores to each user based on their distance from the worst experience cluster.

        Returns:
            pd.DataFrame: DataFrame with user IDs and their experience scores.
        """
        clustered_data = self.experience_model.perform_kmeans_clustering()
        worst_experience_cluster = clustered_data['Cluster'].max()
        # Select only numeric columns for centroid calculation
        numeric_columns = ['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
                       'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
                       'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']
    
        worst_experience_centroid = clustered_data[clustered_data['Cluster'] == worst_experience_cluster][numeric_columns].mean()
        
        experience_scores = clustered_data.apply(
        lambda row: self.calculate_euclidean_distance(row[numeric_columns], 
                                                      worst_experience_centroid),
        axis=1
    )
        
        result = pd.DataFrame({'MSISDN/Number': clustered_data.index, 'Experience Score': experience_scores})
        return result.reset_index(drop=True)

    def calculate_satisfaction_scores(self):
        """
        Calculate satisfaction scores as the average of engagement and experience scores.

        Returns:
            pd.DataFrame: DataFrame with user IDs, engagement scores, experience scores, and satisfaction scores.
        """
        engagement_scores = self.assign_engagement_score()
        experience_scores = self.assign_experience_score()
        
        print("Engagement scores shape:", engagement_scores.shape)
        print("Experience scores shape:", experience_scores.shape)
        
        # Merge the DataFrames
        satisfaction_scores = pd.merge(engagement_scores, experience_scores, on='MSISDN/Number', how='outer')
        
        print("Merged satisfaction scores shape:", satisfaction_scores.shape)
        print("Merged satisfaction scores columns:", satisfaction_scores.columns)
        print("First few rows of merged satisfaction scores:")
        print(satisfaction_scores.head())
        print("Data types of merged satisfaction scores:")
        print(satisfaction_scores.dtypes)
        
        satisfaction_scores['Satisfaction Score'] = (satisfaction_scores['Engagement Score'] + satisfaction_scores['Experience Score']) / 2
        
        self.satisfaction_scores = satisfaction_scores
        return satisfaction_scores

    def get_top_satisfied_customers(self, n=10):
        """
        Get the top n satisfied customers based on their satisfaction scores.

        Args:
            n (int): Number of top customers to return. Defaults to 10.

        Returns:
            pd.DataFrame: DataFrame with the top n satisfied customers and their scores.
        """
        if self.satisfaction_scores is None:
            self.calculate_satisfaction_scores()
        
        return self.satisfaction_scores.nlargest(n, 'Satisfaction Score')

    def plot_satisfaction_scores(self):
        """
        Plot the distribution of satisfaction scores and highlight the top 10 satisfied customers.
        """
        if self.satisfaction_scores is None:
            self.calculate_satisfaction_scores()
        
        top_10 = self.get_top_satisfied_customers()
        
        plt.figure(figsize=(12, 6))
        sns.histplot(self.satisfaction_scores['Satisfaction Score'], kde=True)
        plt.title('Distribution of Satisfaction Scores')
        plt.xlabel('Satisfaction Score')
        plt.ylabel('Frequency')
        
        for score in top_10['Satisfaction Score']:
            plt.axvline(x=score, color='r', linestyle='--', alpha=0.5)
        
        plt.text(0.95, 0.95, 'Top 10 Customers', transform=plt.gca().transAxes, 
                 verticalalignment='top', horizontalalignment='right',
                 color='r', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

    def build_regression_model(self):
        """
        Build a Random Forest regression model to predict satisfaction scores.

        Returns:
            tuple: Trained model, mean squared error, and R-squared score.
        """
        if self.satisfaction_scores is None:
            self.calculate_satisfaction_scores()
        
        X = self.satisfaction_scores[['Engagement Score', 'Experience Score']]
        y = self.satisfaction_scores['Satisfaction Score']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return model, mse, r2

    def perform_kmeans_clustering(self):
        """
        Perform k-means clustering (k=2) on engagement and experience scores.

        Returns:
            pd.DataFrame: DataFrame with user IDs, scores, and assigned clusters.
        """
        if self.satisfaction_scores is None:
            self.calculate_satisfaction_scores()
        
        X = self.satisfaction_scores[['Engagement Score', 'Experience Score']]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        clustered_data = self.satisfaction_scores.copy()
        clustered_data['Cluster'] = clusters
        
        return clustered_data

    def plot_kmeans_results(self):
        """
        Plot the results of k-means clustering on engagement and experience scores.
        """
        clustered_data = self.perform_kmeans_clustering()
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=clustered_data, x='Engagement Score', y='Experience Score', hue='Cluster', palette='viridis')
        plt.title('K-means Clustering of Users based on Engagement and Experience Scores')
        plt.xlabel('Engagement Score')
        plt.ylabel('Experience Score')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.show()

    def aggregate_scores_by_cluster(self):
        """
        Aggregate average satisfaction and experience scores per cluster.

        Returns:
            pd.DataFrame: DataFrame with average scores for each cluster.
        """
        clustered_data = self.perform_kmeans_clustering()
        
        aggregated_scores = clustered_data.groupby('Cluster').agg({
            'Satisfaction Score': 'mean',
            'Experience Score': 'mean'
        }).reset_index()
        
        return aggregated_scores

    def plot_aggregated_scores(self):
        """
        Plot the aggregated satisfaction and experience scores per cluster.
        """
        aggregated_scores = self.aggregate_scores_by_cluster()
        
        plt.figure(figsize=(10, 6))
        x = range(len(aggregated_scores))
        width = 0.35
        
        plt.bar(x, aggregated_scores['Satisfaction Score'], width, label='Satisfaction Score')
        plt.bar([i + width for i in x], aggregated_scores['Experience Score'], width, label='Experience Score')
        
        plt.xlabel('Cluster')
        plt.ylabel('Average Score')
        plt.title('Average Satisfaction and Experience Scores per Cluster')
        plt.xticks([i + width/2 for i in x], aggregated_scores['Cluster'])
        plt.legend()
        plt.tight_layout()
        plt.show()

    def export_to_postgres(self, engine, table_name='user_satisfaction_scores'):
        """
        Export user satisfaction data to a PostgreSQL database.

        Args:
            engine (sqlalchemy.engine.base.Engine): SQLAlchemy engine for database connection.
            table_name (str): Name of the table to create in the database.

        Returns:
            str: SQL query to select all data from the exported table.
        """
        if self.satisfaction_scores is None:
            self.calculate_satisfaction_scores()
        
        self.satisfaction_scores.to_sql(table_name, engine, schema='public', if_exists='replace', index=False)
        
        select_query = f"SELECT * FROM public.{table_name} LIMIT 10;"
        
        return select_query