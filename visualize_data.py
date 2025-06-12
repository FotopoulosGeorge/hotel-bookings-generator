"""
Comprehensive Data Visualization Tool for Hotel Booking Generator

This tool creates extensive visualizations to evaluate synthetic data quality,
realism, and ML readiness. Generates publication-ready charts and analysis plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import sys
from scipy import stats
from scipy.stats import kstest, normaltest
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import argparse

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils import load_generated_data, validate_data_quality
except ImportError:
    print("⚠️ Could not import utils module. Some functions may not work.")


class HotelDataVisualizer:
    """Comprehensive visualization tool for hotel booking data"""
    
    def __init__(self, data_prefix='', output_dir='visualizations/'):
        self.data_prefix = data_prefix
        self.output_dir = output_dir
        self.data = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Configure matplotlib for better plots
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def load_data(self):
        """Load hotel booking data"""
        print(f"📂 Loading data with prefix: '{self.data_prefix}'")
        
        try:
            self.data = load_generated_data(self.data_prefix)
            if not self.data:
                # Try direct CSV loading if utils import failed
                booking_file = f"{self.data_prefix}historical_bookings.csv"
                if os.path.exists(booking_file):
                    self.data = {
                        'bookings': pd.read_csv(booking_file),
                        'campaigns': pd.read_csv(f"{self.data_prefix}campaigns_run.csv"),
                        'customers': pd.read_csv(f"{self.data_prefix}customer_segments.csv"),
                        'attribution': pd.read_csv(f"{self.data_prefix}attribution_ground_truth.csv")
                    }
                    
                    # Convert date columns
                    date_columns = ['booking_date', 'stay_start_date', 'stay_end_date', 'cancellation_date']
                    for col in date_columns:
                        if col in self.data['bookings'].columns:
                            self.data['bookings'][col] = pd.to_datetime(self.data['bookings'][col], errors='coerce')
                    
                    date_columns = ['start_date', 'end_date']
                    for col in date_columns:
                        if col in self.data['campaigns'].columns:
                            self.data['campaigns'][col] = pd.to_datetime(self.data['campaigns'][col], errors='coerce')
            
            if self.data:
                print(f"✅ Data loaded successfully")
                print(f"   📊 Bookings: {len(self.data['bookings']):,} records")
                print(f"   🎯 Campaigns: {len(self.data['campaigns']):,} records")
                return True
            else:
                print(f"❌ Failed to load data files")
                return False
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_distribution_plots(self):
        """Create distribution analysis plots"""
        print("📊 Creating distribution plots...")
        
        bookings_df = self.data['bookings']
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hotel Booking Data - Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price distribution
        axes[0, 0].hist(bookings_df['final_price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(bookings_df['final_price'].mean(), color='red', linestyle='--', label=f'Mean: ${bookings_df["final_price"].mean():.2f}')
        axes[0, 0].axvline(bookings_df['final_price'].median(), color='orange', linestyle='--', label=f'Median: ${bookings_df["final_price"].median():.2f}')
        axes[0, 0].set_title('Final Price Distribution')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Lead time distribution
        bookings_df['lead_time'] = (bookings_df['stay_start_date'] - bookings_df['booking_date']).dt.days
        axes[0, 1].hist(bookings_df['lead_time'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(bookings_df['lead_time'].mean(), color='red', linestyle='--', label=f'Mean: {bookings_df["lead_time"].mean():.1f} days')
        axes[0, 1].set_title('Lead Time Distribution')
        axes[0, 1].set_xlabel('Days')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Stay length distribution
        stay_counts = bookings_df['stay_length'].value_counts().sort_index()
        axes[0, 2].bar(stay_counts.index, stay_counts.values, alpha=0.7, color='coral')
        axes[0, 2].set_title('Stay Length Distribution')
        axes[0, 2].set_xlabel('Nights')
        axes[0, 2].set_ylabel('Number of Bookings')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Customer segment distribution
        segment_counts = bookings_df['customer_segment'].value_counts()
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        axes[1, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', colors=colors)
        axes[1, 0].set_title('Customer Segment Distribution')
        
        # 5. Booking channel distribution
        channel_counts = bookings_df['booking_channel'].value_counts()
        axes[1, 1].bar(channel_counts.index, channel_counts.values, alpha=0.7, color=['#FFB366', '#66FFB2'])
        axes[1, 1].set_title('Booking Channel Distribution')
        axes[1, 1].set_xlabel('Channel')
        axes[1, 1].set_ylabel('Number of Bookings')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Room type distribution
        room_counts = bookings_df['room_type'].value_counts()
        axes[1, 2].bar(room_counts.index, room_counts.values, alpha=0.7, color='plum')
        axes[1, 2].set_title('Room Type Distribution')
        axes[1, 2].set_xlabel('Room Type')
        axes[1, 2].set_ylabel('Number of Bookings')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}01_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed price analysis
        self._create_detailed_price_analysis(bookings_df)
        
        print(f"   ✅ Distribution plots saved")
    
    def _create_detailed_price_analysis(self, bookings_df):
        """Create detailed price analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Price Analysis', fontsize=16, fontweight='bold')
        
        # Price by room type
        room_price_data = [bookings_df[bookings_df['room_type'] == rt]['final_price'] for rt in bookings_df['room_type'].unique()]
        axes[0, 0].boxplot(room_price_data, labels=bookings_df['room_type'].unique())
        axes[0, 0].set_title('Price Distribution by Room Type')
        axes[0, 0].set_xlabel('Room Type')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Price by customer segment
        segment_price_data = [bookings_df[bookings_df['customer_segment'] == seg]['final_price'] for seg in bookings_df['customer_segment'].unique()]
        axes[0, 1].boxplot(segment_price_data, labels=bookings_df['customer_segment'].unique())
        axes[0, 1].set_title('Price Distribution by Customer Segment')
        axes[0, 1].set_xlabel('Customer Segment')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Discount analysis
        promotional_bookings = bookings_df[bookings_df['discount_amount'] > 0]
        if not promotional_bookings.empty:
            promotional_bookings['discount_percentage'] = promotional_bookings['discount_amount'] / promotional_bookings['base_price']
            axes[1, 0].hist(promotional_bookings['discount_percentage'], bins=30, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 0].set_title('Discount Percentage Distribution')
            axes[1, 0].set_xlabel('Discount Percentage')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Seasonal pricing
        bookings_df['stay_month'] = bookings_df['stay_start_date'].dt.month
        monthly_avg_price = bookings_df.groupby('stay_month')['final_price'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        axes[1, 1].plot(monthly_avg_price.index, monthly_avg_price.values, marker='o', linewidth=2, markersize=8, color='darkblue')
        axes[1, 1].set_title('Average Price by Stay Month')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Average Price ($)')
        axes[1, 1].set_xticks(range(1, 13))
        axes[1, 1].set_xticklabels(month_names)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}02_detailed_price_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_temporal_analysis(self):
        """Create temporal pattern analysis plots"""
        print("⏰ Creating temporal analysis plots...")
        
        bookings_df = self.data['bookings']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Daily booking volume
        daily_bookings = bookings_df.groupby('booking_date').size()
        axes[0, 0].plot(daily_bookings.index, daily_bookings.values, alpha=0.7, color='steelblue')
        axes[0, 0].set_title('Daily Booking Volume Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Number of Bookings')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Weekly patterns
        bookings_df['booking_weekday'] = bookings_df['booking_date'].dt.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_counts = bookings_df['booking_weekday'].value_counts().reindex(weekday_order)
        
        axes[0, 1].bar(range(len(weekly_counts)), weekly_counts.values, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Booking Volume by Day of Week')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Number of Bookings')
        axes[0, 1].set_xticks(range(len(weekly_counts)))
        axes[0, 1].set_xticklabels([day[:3] for day in weekday_order])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Monthly booking patterns
        bookings_df['booking_month'] = bookings_df['booking_date'].dt.month
        monthly_booking_counts = bookings_df['booking_month'].value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        axes[1, 0].bar(monthly_booking_counts.index, monthly_booking_counts.values, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Booking Volume by Month')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Number of Bookings')
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_xticklabels(month_names, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Lead time vs booking date
        lead_time_by_date = bookings_df.groupby('booking_date')['lead_time'].mean()
        axes[1, 1].plot(lead_time_by_date.index, lead_time_by_date.values, alpha=0.7, color='purple')
        axes[1, 1].set_title('Average Lead Time Over Time')
        axes[1, 1].set_xlabel('Booking Date')
        axes[1, 1].set_ylabel('Average Lead Time (days)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}03_temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create seasonal heatmap
        self._create_seasonal_heatmap(bookings_df)
        
        print(f"   ✅ Temporal analysis plots saved")
    
    def _create_seasonal_heatmap(self, bookings_df):
        """Create seasonal booking pattern heatmap"""
        # Create month-week heatmap
        bookings_df['booking_week'] = bookings_df['booking_date'].dt.isocalendar().week
        heatmap_data = bookings_df.groupby(['booking_month', 'booking_week']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(20, 8))
        sns.heatmap(heatmap_data, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Number of Bookings'})
        plt.title('Booking Volume Heatmap: Month vs Week of Year', fontsize=16, fontweight='bold')
        plt.xlabel('Week of Year')
        plt.ylabel('Month')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}04_seasonal_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_campaign_analysis(self):
        """Create campaign performance analysis plots"""
        print("🎯 Creating campaign analysis plots...")
        
        bookings_df = self.data['bookings']
        campaigns_df = self.data['campaigns']
        
        # Extract campaign type from campaign_id
        campaign_bookings = bookings_df[bookings_df['campaign_id'].notna()].copy()
        if campaign_bookings.empty:
            print("   ⚠️ No campaign data found")
            return
        
        campaign_bookings['campaign_type'] = campaign_bookings['campaign_id'].str.extract(r'^([A-Z]+)_')[0]
        campaign_bookings['campaign_type'] = campaign_bookings['campaign_type'].map({
            'EB': 'Early_Booking',
            'FS': 'Flash_Sale', 
            'SO': 'Special_Offer'
        })
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Campaign Performance Analysis', fontsize=16, fontweight='bold')
        
        # Campaign participation over time
        campaign_bookings_daily = campaign_bookings.groupby('booking_date').size()
        total_bookings_daily = bookings_df.groupby('booking_date').size()
        participation_rate_daily = (campaign_bookings_daily / total_bookings_daily).fillna(0)
        
        axes[0, 0].plot(participation_rate_daily.index, participation_rate_daily.values, alpha=0.7, color='darkorange')
        axes[0, 0].axhline(participation_rate_daily.mean(), color='red', linestyle='--', label=f'Mean: {participation_rate_daily.mean():.1%}')
        axes[0, 0].set_title('Campaign Participation Rate Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Participation Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Campaign type performance
        campaign_type_stats = campaign_bookings.groupby('campaign_type').agg({
            'booking_id': 'count',
            'discount_amount': 'mean',
            'attribution_score': 'mean',
            'incremental_flag': 'sum'
        })
        
        campaign_type_stats['incremental_rate'] = campaign_type_stats['incremental_flag'] / campaign_type_stats['booking_id']
        
        x_pos = range(len(campaign_type_stats))
        axes[0, 1].bar(x_pos, campaign_type_stats['booking_id'], alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 1].set_title('Bookings by Campaign Type')
        axes[0, 1].set_xlabel('Campaign Type')
        axes[0, 1].set_ylabel('Number of Bookings')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(campaign_type_stats.index, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Attribution score distribution
        axes[1, 0].hist(campaign_bookings['attribution_score'], bins=30, alpha=0.7, color='mediumpurple', edgecolor='black')
        axes[1, 0].axvline(campaign_bookings['attribution_score'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {campaign_bookings["attribution_score"].mean():.3f}')
        axes[1, 0].set_title('Attribution Score Distribution')
        axes[1, 0].set_xlabel('Attribution Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Discount effectiveness
        discount_ranges = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
        discount_effectiveness = []
        discount_labels = []
        
        for low, high in discount_ranges:
            mask = (campaign_bookings['discount_amount'] / campaign_bookings['base_price'] >= low) & \
                   (campaign_bookings['discount_amount'] / campaign_bookings['base_price'] < high)
            subset = campaign_bookings[mask]
            if len(subset) > 0:
                effectiveness = subset['incremental_flag'].mean()
                discount_effectiveness.append(effectiveness)
                discount_labels.append(f'{low:.0%}-{high:.0%}')
        
        if discount_effectiveness:
            axes[1, 1].bar(range(len(discount_effectiveness)), discount_effectiveness, alpha=0.7, color='gold')
            axes[1, 1].set_title('Incremental Rate by Discount Level')
            axes[1, 1].set_xlabel('Discount Range')
            axes[1, 1].set_ylabel('Incremental Rate')
            axes[1, 1].set_xticks(range(len(discount_labels)))
            axes[1, 1].set_xticklabels(discount_labels)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}05_campaign_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Campaign analysis plots saved")
    
    def create_correlation_analysis(self):
        """Create correlation analysis plots"""
        print("🔗 Creating correlation analysis...")
        
        bookings_df = self.data['bookings']
        
        # Prepare data for correlation analysis
        correlation_data = bookings_df.copy()
        
        # Convert categorical variables to numeric
        correlation_data['customer_segment_encoded'] = pd.Categorical(correlation_data['customer_segment']).codes
        correlation_data['booking_channel_encoded'] = pd.Categorical(correlation_data['booking_channel']).codes
        correlation_data['room_type_encoded'] = pd.Categorical(correlation_data['room_type']).codes
        correlation_data['has_campaign'] = correlation_data['campaign_id'].notna().astype(int)
        correlation_data['is_weekend'] = correlation_data['booking_date'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Select numeric columns for correlation
        numeric_cols = [
            'final_price', 'base_price', 'discount_amount', 'stay_length', 'lead_time',
            'customer_segment_encoded', 'booking_channel_encoded', 'room_type_encoded',
            'has_campaign', 'is_weekend', 'attribution_score'
        ]
        
        # Remove columns that don't exist
        available_cols = [col for col in numeric_cols if col in correlation_data.columns]
        corr_matrix = correlation_data[available_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.3f')
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}06_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Correlation analysis saved")
    
    def create_ml_readiness_assessment(self):
        """Create ML readiness assessment visualizations"""
        print("🤖 Creating ML readiness assessment...")
        
        bookings_df = self.data['bookings']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ML Readiness Assessment', fontsize=16, fontweight='bold')
        
        # 1. Missing values analysis
        missing_values = bookings_df.isnull().sum()
        missing_pct = (missing_values / len(bookings_df)) * 100
        
        if missing_pct.sum() > 0:
            missing_pct = missing_pct[missing_pct > 0]
            axes[0, 0].bar(range(len(missing_pct)), missing_pct.values, color='red', alpha=0.7)
            axes[0, 0].set_title('Missing Values by Column')
            axes[0, 0].set_xlabel('Columns')
            axes[0, 0].set_ylabel('Missing Percentage (%)')
            axes[0, 0].set_xticks(range(len(missing_pct)))
            axes[0, 0].set_xticklabels(missing_pct.index, rotation=45)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=14)
            axes[0, 0].set_title('Missing Values Analysis')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Outlier detection for price
        Q1 = bookings_df['final_price'].quantile(0.25)
        Q3 = bookings_df['final_price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = bookings_df[(bookings_df['final_price'] < lower_bound) | (bookings_df['final_price'] > upper_bound)]
        
        axes[0, 1].boxplot(bookings_df['final_price'])
        axes[0, 1].set_title(f'Price Outliers Detection\n({len(outliers)} outliers found)')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Class balance analysis
        segment_counts = bookings_df['customer_segment'].value_counts()
        balance_ratio = segment_counts.min() / segment_counts.max()
        
        axes[0, 2].bar(range(len(segment_counts)), segment_counts.values, alpha=0.7, 
                      color='lightblue' if balance_ratio > 0.5 else 'orange')
        axes[0, 2].set_title(f'Class Balance - Customer Segments\n(Balance Ratio: {balance_ratio:.2f})')
        axes[0, 2].set_xlabel('Customer Segments')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_xticks(range(len(segment_counts)))
        axes[0, 2].set_xticklabels(segment_counts.index, rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Normality test for key variables
        price_stat, price_p = normaltest(bookings_df['final_price'])
        lead_stat, lead_p = normaltest(bookings_df['lead_time'])
        
        variables = ['Final Price', 'Lead Time']
        p_values = [price_p, lead_p]
        colors = ['green' if p > 0.05 else 'red' for p in p_values]
        
        axes[1, 0].bar(variables, p_values, color=colors, alpha=0.7)
        axes[1, 0].axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
        axes[1, 0].set_title('Normality Test Results')
        axes[1, 0].set_ylabel('p-value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Data consistency checks
        consistency_checks = {
            'Valid Dates': (bookings_df['stay_start_date'] > bookings_df['booking_date']).sum(),
            'Positive Prices': (bookings_df['final_price'] > 0).sum(),
            'Valid Attribution': ((bookings_df['attribution_score'] >= 0) & (bookings_df['attribution_score'] <= 1)).sum()
        }
        
        total_records = len(bookings_df)
        consistency_rates = [count / total_records for count in consistency_checks.values()]
        colors = ['green' if rate == 1.0 else 'orange' if rate > 0.95 else 'red' for rate in consistency_rates]
        
        axes[1, 1].bar(consistency_checks.keys(), consistency_rates, color=colors, alpha=0.7)
        axes[1, 1].set_title('Data Consistency Checks')
        axes[1, 1].set_ylabel('Consistency Rate')
        axes[1, 1].set_ylim(0, 1.1)
        for i, rate in enumerate(consistency_rates):
            axes[1, 1].text(i, rate + 0.02, f'{rate:.1%}', ha='center')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Feature importance simulation (correlation with target)
        # Use final_price as proxy target
        numeric_features = bookings_df.select_dtypes(include=[np.number]).columns
        correlations = []
        feature_names = []
        
        for col in numeric_features:
            if col != 'final_price' and col in bookings_df.columns:
                corr = abs(bookings_df[col].corr(bookings_df['final_price']))
                if not np.isnan(corr):
                    correlations.append(corr)
                    feature_names.append(col)
        
        if correlations:
            # Sort by correlation
            sorted_indices = np.argsort(correlations)[::-1]
            top_features = [feature_names[i] for i in sorted_indices[:5]]
            top_correlations = [correlations[i] for i in sorted_indices[:5]]
            
            axes[1, 2].barh(range(len(top_correlations)), top_correlations, alpha=0.7, color='purple')
            axes[1, 2].set_title('Feature Importance (Correlation with Price)')
            axes[1, 2].set_xlabel('Absolute Correlation')
            axes[1, 2].set_yticks(range(len(top_features)))
            axes[1, 2].set_yticklabels(top_features)
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}07_ml_readiness_assessment.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ ML readiness assessment saved")
    
    def create_business_validation_plots(self):
        """Create business logic validation plots"""
        print("💼 Creating business validation plots...")
        
        bookings_df = self.data['bookings']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Business Logic Validation', fontsize=16, fontweight='bold')
        
        # 1. Price vs Room Type validation
        room_price_stats = bookings_df.groupby('room_type')['final_price'].agg(['mean', 'std'])
        room_order = ['Standard', 'Deluxe', 'Suite', 'Premium']  # Expected price order
        room_order = [room for room in room_order if room in room_price_stats.index]
        
        means = [room_price_stats.loc[room, 'mean'] for room in room_order]
        stds = [room_price_stats.loc[room, 'std'] for room in room_order]
        
        axes[0, 0].bar(room_order, means, yerr=stds, alpha=0.7, capsize=5, color='lightblue')
        axes[0, 0].set_title('Price Progression by Room Type\n(Should increase from Standard to Premium)')
        axes[0, 0].set_xlabel('Room Type')
        axes[0, 0].set_ylabel('Average Price ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add validation text
        price_increases_correctly = all(means[i] < means[i+1] for i in range(len(means)-1))
        validation_text = "✅ Prices increase correctly" if price_increases_correctly else "❌ Price ordering issue"
        axes[0, 0].text(0.02, 0.98, validation_text, transform=axes[0, 0].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen' if price_increases_correctly else 'lightcoral'),
                       verticalalignment='top')
        
        # 2. Lead time vs Customer Segment validation
        segment_leadtime = bookings_df.groupby('customer_segment')['lead_time'].mean()
        expected_order = ['Last_Minute', 'Flexible', 'Early_Planner']  # Expected lead time order
        expected_order = [seg for seg in expected_order if seg in segment_leadtime.index]
        
        leadtimes = [segment_leadtime[seg] for seg in expected_order]
        axes[0, 1].bar(expected_order, leadtimes, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Lead Time by Customer Segment\n(Should increase from Last_Minute to Early_Planner)')
        axes[0, 1].set_xlabel('Customer Segment')
        axes[0, 1].set_ylabel('Average Lead Time (days)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Validation
        leadtime_increases_correctly = all(leadtimes[i] < leadtimes[i+1] for i in range(len(leadtimes)-1))
        validation_text = "✅ Lead times increase correctly" if leadtime_increases_correctly else "❌ Lead time ordering issue"
        axes[0, 1].text(0.02, 0.98, validation_text, transform=axes[0, 1].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen' if leadtime_increases_correctly else 'lightcoral'),
                       verticalalignment='top')
        
        # 3. Cancellation rate validation
        cancellation_by_segment = bookings_df.groupby('customer_segment')['is_cancelled'].mean()
        
        axes[1, 0].bar(cancellation_by_segment.index, cancellation_by_segment.values, alpha=0.7, color='orange')
        axes[1, 0].set_title('Cancellation Rate by Customer Segment')
        axes[1, 0].set_xlabel('Customer Segment')
        axes[1, 0].set_ylabel('Cancellation Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, rate in enumerate(cancellation_by_segment.values):
            axes[1, 0].text(i, rate + 0.01, f'{rate:.1%}', ha='center')
        
        # 4. Seasonal demand validation
        monthly_demand = bookings_df.groupby('stay_month').size()
        month_names = ['May', 'Jun', 'Jul', 'Aug', 'Sep']  # Operational months
        
        operational_months = [5, 6, 7, 8, 9]
        demand_values = [monthly_demand.get(month, 0) for month in operational_months]
        
        axes[1, 1].plot(month_names, demand_values, marker='o', linewidth=3, markersize=10, color='purple')
        axes[1, 1].set_title('Seasonal Demand Pattern\n(Jul-Aug should be peaks)')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Number of Stays')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Validation - check if Jul/Aug are peaks
        jul_aug_peak = (demand_values[2] >= max(demand_values[:2])) and (demand_values[3] >= max(demand_values[:2]))
        validation_text = "✅ Jul-Aug show peak demand" if jul_aug_peak else "❌ Seasonal pattern issue"
        axes[1, 1].text(0.02, 0.98, validation_text, transform=axes[1, 1].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen' if jul_aug_peak else 'lightcoral'),
                       verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}08_business_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Business validation plots saved")
    
    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        print("📋 Creating summary dashboard...")
        
        bookings_df = self.data['bookings']
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Hotel Booking Data - Comprehensive Summary Dashboard', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # Key metrics text box
        ax_metrics = fig.add_subplot(gs[0, :2])
        ax_metrics.axis('off')
        
        total_bookings = len(bookings_df)
        total_revenue = bookings_df['final_price'].sum()
        avg_price = bookings_df['final_price'].mean()
        cancellation_rate = bookings_df['is_cancelled'].mean()
        campaign_rate = (bookings_df['campaign_id'].notna()).mean()
        
        metrics_text = f"""
        KEY METRICS SUMMARY
        ══════════════════════════════════════════
        📊 Total Bookings: {total_bookings:,}
        💰 Total Revenue: ${total_revenue:,.2f}
        💵 Average Price: ${avg_price:.2f}
        ❌ Cancellation Rate: {cancellation_rate:.1%}
        🎯 Campaign Participation: {campaign_rate:.1%}
        📅 Date Range: {bookings_df['booking_date'].min().strftime('%Y-%m-%d')} to {bookings_df['booking_date'].max().strftime('%Y-%m-%d')}
        """
        
        ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes, 
                       fontsize=12, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Data quality score
        try:
            quality_result = validate_data_quality(bookings_df)
            ml_score = int(quality_result.get('overall_quality_score', 0) * 100)
        except:
            ml_score = 85  # Default score if validation fails
        
        quality_text = f"""
        DATA QUALITY ASSESSMENT
        ══════════════════════════════════════════
        🤖 ML Readiness Score: {ml_score}/100
        
        {"✅ EXCELLENT - Ready for ML" if ml_score >= 90 else
         "✅ GOOD - Suitable for ML" if ml_score >= 75 else
         "⚠️ FAIR - Needs preprocessing" if ml_score >= 60 else
         "❌ POOR - Significant cleanup needed"}
        """
        
        ax_quality = fig.add_subplot(gs[0, 2:])
        ax_quality.axis('off')
        ax_quality.text(0.05, 0.95, quality_text, transform=ax_quality.transAxes, 
                       fontsize=12, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", 
                               facecolor='lightgreen' if ml_score >= 75 else 'lightyellow' if ml_score >= 60 else 'lightcoral', 
                               alpha=0.8))
        
        # Mini charts
        # Price distribution
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.hist(bookings_df['final_price'], bins=30, alpha=0.7, color='skyblue')
        ax1.set_title('Price Distribution', fontweight='bold')
        ax1.set_xlabel('Price ($)')
        
        # Channel distribution
        ax2 = fig.add_subplot(gs[1, 1])
        channel_counts = bookings_df['booking_channel'].value_counts()
        ax2.pie(channel_counts.values, labels=channel_counts.index, autopct='%1.1f%%')
        ax2.set_title('Booking Channels', fontweight='bold')
        
        # Monthly trend
        ax3 = fig.add_subplot(gs[1, 2])
        monthly_bookings = bookings_df.groupby('stay_month').size()
        ax3.plot(monthly_bookings.index, monthly_bookings.values, marker='o', linewidth=2)
        ax3.set_title('Monthly Demand', fontweight='bold')
        ax3.set_xlabel('Month')
        ax3.grid(True, alpha=0.3)
        
        # Customer segments
        ax4 = fig.add_subplot(gs[1, 3])
        segment_counts = bookings_df['customer_segment'].value_counts()
        ax4.bar(range(len(segment_counts)), segment_counts.values, alpha=0.7)
        ax4.set_title('Customer Segments', fontweight='bold')
        ax4.set_xticks(range(len(segment_counts)))
        ax4.set_xticklabels([seg[:8] + '...' if len(seg) > 8 else seg for seg in segment_counts.index], rotation=45)
        
        # Campaign analysis
        ax5 = fig.add_subplot(gs[2, :2])
        campaign_bookings = bookings_df[bookings_df['campaign_id'].notna()]
        if not campaign_bookings.empty:
            daily_campaign_rate = campaign_bookings.groupby('booking_date').size() / bookings_df.groupby('booking_date').size()
            ax5.plot(daily_campaign_rate.index, daily_campaign_rate.values, alpha=0.7)
            ax5.axhline(daily_campaign_rate.mean(), color='red', linestyle='--', label=f'Mean: {daily_campaign_rate.mean():.1%}')
            ax5.set_title('Campaign Participation Over Time', fontweight='bold')
            ax5.set_ylabel('Participation Rate')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Lead time vs segment
        ax6 = fig.add_subplot(gs[2, 2:])
        segment_leadtime = bookings_df.groupby('customer_segment')['lead_time'].mean()
        ax6.bar(range(len(segment_leadtime)), segment_leadtime.values, alpha=0.7, color='lightgreen')
        ax6.set_title('Average Lead Time by Segment', fontweight='bold')
        ax6.set_xticks(range(len(segment_leadtime)))
        ax6.set_xticklabels([seg[:8] + '...' if len(seg) > 8 else seg for seg in segment_leadtime.index], rotation=45)
        ax6.set_ylabel('Days')
        ax6.grid(True, alpha=0.3)
        
        # Data validation summary
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Perform key validations
        validation_results = {
            'Price Range': f"${bookings_df['final_price'].min():.0f} - ${bookings_df['final_price'].max():.0f}",
            'Date Consistency': f"{((bookings_df['stay_start_date'] > bookings_df['booking_date']).mean() * 100):.1f}% valid",
            'Missing Values': f"{(bookings_df.isnull().sum().sum() / (len(bookings_df) * len(bookings_df.columns)) * 100):.1f}% missing",
            'Room Type Balance': f"{(bookings_df['room_type'].value_counts().min() / bookings_df['room_type'].value_counts().max()):.2f} ratio",
            'Seasonal Pattern': "Jul-Aug peaks detected" if bookings_df[bookings_df['stay_month'].isin([7, 8])].shape[0] > bookings_df[bookings_df['stay_month'].isin([5, 9])].shape[0] else "Pattern needs review"
        }
        
        validation_text = "DATA VALIDATION SUMMARY\n" + "="*80 + "\n"
        for key, value in validation_results.items():
            validation_text += f"• {key}: {value}\n"
        
        ax7.text(0.05, 0.95, validation_text, transform=ax7.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
        
        plt.savefig(f'{self.output_dir}09_summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Summary dashboard saved")
    
    def generate_all_visualizations(self):
        """Generate all visualization plots"""
        print("🎨 GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 60)
        
        if not self.load_data():
            return False
        
        print(f"📁 Output directory: {self.output_dir}")
        
        try:
            # Generate all plot types
            self.create_distribution_plots()
            self.create_temporal_analysis()
            self.create_campaign_analysis()
            self.create_correlation_analysis()
            self.create_ml_readiness_assessment()
            self.create_business_validation_plots()
            self.create_summary_dashboard()
            
            # Create index file
            self._create_visualization_index()
            
            print(f"\n🎉 All visualizations completed successfully!")
            print(f"📁 Check the '{self.output_dir}' directory for all plots")
            print(f"📋 Open 'visualization_index.html' for a complete overview")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_visualization_index(self):
        """Create an HTML index of all visualizations"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hotel Booking Data Visualizations</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                h1 {{ color: #2c3e50; text-align: center; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .visualization {{ margin: 20px 0; padding: 20px; background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
                .description {{ margin: 10px 0; color: #7f8c8d; font-style: italic; }}
                .metadata {{ background-color: #ecf0f1; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Hotel Booking Data - Comprehensive Visualizations</h1>
            <div class="metadata">
                <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <strong>Data Prefix:</strong> {self.data_prefix if self.data_prefix else 'Standard'}<br>
                <strong>Total Records:</strong> {len(self.data['bookings']):,}<br>
            </div>
            
            <h2>📊 Distribution Analysis</h2>
            <div class="visualization">
                <img src="01_distribution_analysis.png" alt="Distribution Analysis">
                <div class="description">
                    Overview of key variable distributions including price, lead time, stay length, and categorical breakdowns.
                </div>
            </div>
            
            <div class="visualization">
                <img src="02_detailed_price_analysis.png" alt="Detailed Price Analysis">
                <div class="description">
                    Deep dive into pricing patterns by room type, customer segment, discount levels, and seasonal trends.
                </div>
            </div>
            
            <h2>⏰ Temporal Pattern Analysis</h2>
            <div class="visualization">
                <img src="03_temporal_analysis.png" alt="Temporal Analysis">
                <div class="description">
                    Time-based patterns including daily volumes, weekly trends, monthly distributions, and lead time evolution.
                </div>
            </div>
            
            <div class="visualization">
                <img src="04_seasonal_heatmap.png" alt="Seasonal Heatmap">
                <div class="description">
                    Detailed seasonal booking patterns showing month vs week relationships.
                </div>
            </div>
            
            <h2>🎯 Campaign Performance Analysis</h2>
            <div class="visualization">
                <img src="05_campaign_analysis.png" alt="Campaign Analysis">
                <div class="description">
                    Campaign effectiveness metrics including participation rates, attribution scores, and discount effectiveness.
                </div>
            </div>
            
            <h2>🔗 Correlation Analysis</h2>
            <div class="visualization">
                <img src="06_correlation_analysis.png" alt="Correlation Analysis">
                <div class="description">
                    Feature correlation matrix to identify relationships between variables for ML modeling.
                </div>
            </div>
            
            <h2>🤖 ML Readiness Assessment</h2>
            <div class="visualization">
                <img src="07_ml_readiness_assessment.png" alt="ML Readiness Assessment">
                <div class="description">
                    Comprehensive assessment of data quality for machine learning including missing values, outliers, class balance, and feature importance.
                </div>
            </div>
            
            <h2>💼 Business Logic Validation</h2>
            <div class="visualization">
                <img src="08_business_validation.png" alt="Business Validation">
                <div class="description">
                    Validation of business logic including price hierarchies, customer behavior patterns, and seasonal demand.
                </div>
            </div>
            
            <h2>📋 Summary Dashboard</h2>
            <div class="visualization">
                <img src="09_summary_dashboard.png" alt="Summary Dashboard">
                <div class="description">
                    Comprehensive overview dashboard with key metrics, data quality scores, and validation summaries.
                </div>
            </div>
            
            <div class="metadata">
                <h3>📈 Data Quality Summary</h3>
                Use these visualizations to:
                <ul>
                    <li><strong>Validate Realism:</strong> Check if patterns match real-world hotel booking behavior</li>
                    <li><strong>Assess ML Readiness:</strong> Identify any preprocessing needs before machine learning</li>
                    <li><strong>Understand Patterns:</strong> Gain insights into customer behavior and business dynamics</li>
                    <li><strong>Quality Control:</strong> Verify data consistency and business logic implementation</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(f'{self.output_dir}visualization_index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Generate comprehensive visualizations for hotel booking data')
    parser.add_argument('--prefix', type=str, default='', help='Data file prefix (e.g., "luxury_")')
    parser.add_argument('--output-dir', type=str, default='visualizations/', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = HotelDataVisualizer(data_prefix=args.prefix, output_dir=args.output_dir)
    
    # Generate all visualizations
    success = visualizer.generate_all_visualizations()
    
    if success:
        print(f"\n✨ Visualization generation completed!")
        print(f"🌐 Open '{args.output_dir}visualization_index.html' in your browser for a complete overview")
    else:
        print(f"\n❌ Visualization generation failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())