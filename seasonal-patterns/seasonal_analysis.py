import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

class SeasonalGeographicAnalysis:
    def __init__(self):
        self.color_palette = sns.color_palette("YlOrRd", as_cmap=True)  # Colorblind friendly
        plt.style.use('seaborn')

    def load_and_process_data(self, orders_path, customers_path):
        # Load datasets
        orders = pd.read_csv(orders_path)
        customers = pd.read_csv(customers_path)
        
        # Process timestamps and merge data
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders['season'] = orders['order_purchase_timestamp'].dt.quarter  # Simple season mapping
        orders['month'] = orders['order_purchase_timestamp'].dt.month
        
        # Merge with customer location data
        self.data = orders.merge(customers, on='customer_id')
        
        # Create Brazilian season mapping (Southern Hemisphere)
        self.season_map = {
            1: 'Summer',  # Jan-Mar
            2: 'Autumn',  # Apr-Jun
            3: 'Winter',  # Jul-Sep
            4: 'Spring'   # Oct-Dec
        }
        
        return self.data

    def create_geographic_seasonal_heatmap(self, product_category):
        """Create heatmap of sales by state and season for specific product category"""
        # Filter for product category
        category_data = self.data[self.data['product_category_name'] == product_category]
        
        # Create state x season matrix
        heatmap_data = pd.crosstab(
            category_data['customer_state'],
            category_data['season'].map(self.season_map)
        )
        
        # Normalize by state population/total orders to show relative demand
        heatmap_data_normalized = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            heatmap_data_normalized,
            cmap=self.color_palette,
            annot=True,
            fmt='.2%',
            cbar_kws={'label': 'Relative Demand'}
        )
        
        plt.title(f'Seasonal Demand Patterns by State: {product_category}')
        plt.xlabel('Season')
        plt.ylabel('State')
        
        return plt

    def identify_seasonal_trends(self):
        """Identify strong seasonal patterns in product categories"""
        seasonal_variation = {}
        
        for category in self.data['product_category_name'].unique():
            category_data = self.data[self.data['product_category_name'] == category]
            seasonal_counts = category_data.groupby('season').size()
            
            # Calculate coefficient of variation
            cv = seasonal_counts.std() / seasonal_counts.mean()
            seasonal_variation[category] = cv
            
        # Return products with highest seasonal variation
        return pd.Series(seasonal_variation).sort_values(ascending=False)

    def visualize_state_category_correlation(self, top_n_categories=10):
        """Visualize which states have unusual demand for specific categories"""
        # Calculate state-category preference scores
        state_category_counts = pd.crosstab(
            self.data['customer_state'], 
            self.data['product_category_name']
        )
        
        # Normalize by state
        preferences = state_category_counts.div(state_category_counts.sum(axis=1), axis=0)
        
        # Find most variable categories
        category_variance = preferences.var()
        top_categories = category_variance.nlargest(top_n_categories).index
        
        # Create heatmap for top categories
        plt.figure(figsize=(15, 10))
        sns.heatmap(
            preferences[top_categories],
            cmap=self.color_palette,
            annot=True,
            fmt='.2%'
        )
        
        plt.title('Regional Product Category Preferences')
        plt.xlabel('Product Category')
        plt.ylabel('State')
        plt.xticks(rotation=45, ha='right')
        
        return plt

if __name__ == "__main__":
    analysis = SeasonalGeographicAnalysis()
    # Add example usage here once we have the data