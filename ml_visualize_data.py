"""
ML-Focused Data Visualization Tool for Hotel Booking Generator

This tool creates comprehensive visualizations specifically designed for 
validating synthetic data quality for machine learning applications.
Clear separation between booking and stay patterns, with focus on data quality metrics.
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import sys
from scipy import stats
from scipy.stats import normaltest, jarque_bera
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import argparse

warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100


class MLDataVisualizer:
    """Comprehensive ML-focused visualization tool for hotel booking data"""
    
    def __init__(self, data_prefix='', output_dir='ml_visualizations/'):
        self.data_prefix = data_prefix
        self.output_dir = output_dir
        self.bookings_df = None
        self.campaigns_df = None
        self.customers_df = None
        self.attribution_df = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load hotel booking data from CSV files"""
        print(f"Loading data with prefix: '{self.data_prefix}'")
        
        try:
            # Load CSV files
            self.bookings_df = pd.read_csv(f"output/{self.data_prefix}historical_bookings.csv")
            self.campaigns_df = pd.read_csv(f"output/{self.data_prefix}campaigns_run.csv")
            self.customers_df = pd.read_csv(f"output/{self.data_prefix}customer_segments.csv")
            self.attribution_df = pd.read_csv(f"output/{self.data_prefix}attribution_ground_truth.csv")
            
            # Convert date columns
            date_columns = ['booking_date', 'stay_start_date', 'stay_end_date', 'cancellation_date']
            for col in date_columns:
                if col in self.bookings_df.columns:
                    self.bookings_df[col] = pd.to_datetime(self.bookings_df[col], errors='coerce')
            
            # Add derived features
            self.bookings_df['lead_time'] = (self.bookings_df['stay_start_date'] - self.bookings_df['booking_date']).dt.days
            self.bookings_df['booking_month'] = self.bookings_df['booking_date'].dt.month
            self.bookings_df['stay_month'] = self.bookings_df['stay_start_date'].dt.month
            self.bookings_df['booking_weekday'] = self.bookings_df['booking_date'].dt.dayofweek
            self.bookings_df['stay_weekday'] = self.bookings_df['stay_start_date'].dt.dayofweek
            
            print(f"Data loaded successfully")
            print(f"   Bookings: {len(self.bookings_df):,} records")
            print(f"   Campaigns: {len(self.campaigns_df):,} records")
            print(f"   Customers: {len(self.customers_df):,} records")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_critical_ml_metrics(self):
        """Create visualization focusing on critical ML data quality metrics"""
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 3, hspace=0.3, wspace=0.25)
        
        fig.suptitle('ML Data Quality Dashboard - Critical Metrics', fontsize=18, fontweight='bold')
        
        # 1. STAY Month Distribution (THE KEY ISSUE)
        ax1 = fig.add_subplot(gs[0, :])
        stay_counts = self.bookings_df['stay_month'].value_counts().sort_index()
        stay_pct = (stay_counts / stay_counts.sum() * 100)
        
        bars = ax1.bar(stay_counts.index, stay_pct.values, color='lightcoral', alpha=0.7, edgecolor='black')
        
        # Add target lines for seasonal hotels
        if len(stay_counts) <= 5:  # Likely seasonal
            targets = {5: 18, 6: 22, 7: 26, 8: 24, 9: 10}
            for month, target in targets.items():
                if month in stay_counts.index:
                    ax1.axhline(y=target, color='green', linestyle='--', alpha=0.5)
                    ax1.text(month + 0.3, target + 0.5, f'Target: {target}%', fontsize=9, color='green')
        
        # Highlight problematic months
        for i, (month, pct) in enumerate(stay_pct.items()):
            ax1.text(month, pct + 0.5, f'{pct:.1f}%', ha='center', fontweight='bold')
            if len(stay_counts) <= 5 and month in targets and abs(pct - targets[month]) > 5:
                bars[i].set_color('red')
                bars[i].set_alpha(0.8)
        
        ax1.set_title('STAY Month Distribution - Critical for Seasonal Hotels', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Percentage of Stays (%)')
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax1.grid(True, alpha=0.3)
        
        # 2. Booking vs Stay Month Comparison
        ax2 = fig.add_subplot(gs[1, 0])
        booking_counts = self.bookings_df['booking_month'].value_counts().sort_index()
        
        x = np.arange(1, 13)
        width = 0.35
        
        # Normalize to percentages
        booking_pct = booking_counts.reindex(range(1, 13), fill_value=0) / booking_counts.sum() * 100
        stay_pct_full = stay_counts.reindex(range(1, 13), fill_value=0) / stay_counts.sum() * 100
        
        ax2.bar(x - width/2, booking_pct.values, width, label='Booking Month', alpha=0.7, color='skyblue')
        ax2.bar(x + width/2, stay_pct_full.values, width, label='Stay Month', alpha=0.7, color='lightcoral')
        
        ax2.set_title('Booking vs Stay Month Distribution')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Lead Time Distribution by Month
        ax3 = fig.add_subplot(gs[1, 1])
        lead_time_by_month = self.bookings_df.groupby('stay_month')['lead_time'].mean()
        
        ax3.plot(lead_time_by_month.index, lead_time_by_month.values, marker='o', linewidth=2, markersize=8)
        ax3.set_title('Average Lead Time by Stay Month')
        ax3.set_xlabel('Stay Month')
        ax3.set_ylabel('Average Lead Time (days)')
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax3.grid(True, alpha=0.3)
        
        # 4. Price Distribution Check (ML Critical)
        ax4 = fig.add_subplot(gs[1, 2])
        price_stats = self.bookings_df.groupby('room_type')['final_price'].agg(['mean', 'std'])
        room_types = ['Standard', 'Deluxe', 'Suite', 'Premium']
        existing_rooms = [rt for rt in room_types if rt in price_stats.index]
        
        means = [price_stats.loc[rt, 'mean'] for rt in existing_rooms]
        stds = [price_stats.loc[rt, 'std'] for rt in existing_rooms]
        
        bars = ax4.bar(existing_rooms, means, yerr=stds, capsize=5, alpha=0.7)
        
        # Check if prices increase monotonically
        prices_correct = all(means[i] < means[i+1] for i in range(len(means)-1))
        bar_color = 'green' if prices_correct else 'red'
        for bar in bars:
            bar.set_color(bar_color)
        
        ax4.set_title(f'Price Hierarchy {"OK" if prices_correct else "ERROR"}')
        ax4.set_xlabel('Room Type')
        ax4.set_ylabel('Average Price ($)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Missing Values Heatmap
        ax5 = fig.add_subplot(gs[2, 0])
        missing_data = self.bookings_df.isnull().sum()
        missing_pct = (missing_data / len(self.bookings_df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        
        if len(missing_pct) > 0:
            colors = ['red' if pct > 5 else 'orange' if pct > 1 else 'yellow' for pct in missing_pct.values]
            bars = ax5.barh(range(len(missing_pct)), missing_pct.values, color=colors, alpha=0.7)
            ax5.set_yticks(range(len(missing_pct)))
            ax5.set_yticklabels(missing_pct.index)
            ax5.set_xlabel('Missing %')
            
            for i, pct in enumerate(missing_pct.values):
                ax5.text(pct + 0.1, i, f'{pct:.2f}%', va='center')
        else:
            ax5.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=16, color='green')
        
        ax5.set_title('Missing Data Analysis')
        ax5.grid(True, alpha=0.3)
        
        # 6. Feature Distributions for ML
        ax6 = fig.add_subplot(gs[2, 1])
        
        # Check normality of key numeric features
        numeric_features = ['final_price', 'lead_time', 'stay_length', 'discount_amount']
        normality_results = {}
        
        for feature in numeric_features:
            if feature in self.bookings_df.columns:
                data = self.bookings_df[feature].dropna()
                if len(data) > 0:
                    stat, p_value = normaltest(data)
                    normality_results[feature] = p_value
        
        features = list(normality_results.keys())
        p_values = list(normality_results.values())
        colors = ['green' if p > 0.05 else 'red' for p in p_values]
        
        bars = ax6.bar(range(len(features)), p_values, color=colors, alpha=0.7)
        ax6.axhline(y=0.05, color='black', linestyle='--', label='alpha = 0.05')
        ax6.set_xticks(range(len(features)))
        ax6.set_xticklabels(features, rotation=45)
        ax6.set_ylabel('Normality Test p-value')
        ax6.set_title('Feature Normality Tests (for ML)')
        ax6.set_yscale('log')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Class Balance Analysis
        ax7 = fig.add_subplot(gs[2, 2])
        
        # Analyze customer segment balance
        segment_counts = self.bookings_df['customer_segment'].value_counts()
        imbalance_ratio = segment_counts.min() / segment_counts.max()
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(segment_counts)))
        wedges, texts, autotexts = ax7.pie(segment_counts.values, labels=segment_counts.index, 
                                           autopct='%1.1f%%', colors=colors)
        
        balance_status = "Balanced" if imbalance_ratio > 0.5 else "Imbalanced"
        balance_color = "green" if imbalance_ratio > 0.5 else "red"
        ax7.set_title(f'Customer Segment Balance\n(Ratio: {imbalance_ratio:.2f} - {balance_status})', 
                     color=balance_color)
        
        # 8. Correlation Matrix for Key Features
        ax8 = fig.add_subplot(gs[3, :2])
        
        # Select key features for correlation
        correlation_features = ['final_price', 'lead_time', 'stay_length', 'discount_amount']
        
        # Add encoded categorical features
        self.bookings_df['room_type_encoded'] = pd.Categorical(self.bookings_df['room_type']).codes
        self.bookings_df['segment_encoded'] = pd.Categorical(self.bookings_df['customer_segment']).codes
        self.bookings_df['channel_encoded'] = pd.Categorical(self.bookings_df['booking_channel']).codes
        
        correlation_features.extend(['room_type_encoded', 'segment_encoded', 'channel_encoded'])
        
        # Calculate correlation matrix
        corr_data = self.bookings_df[correlation_features].dropna()
        corr_matrix = corr_data.corr()
        
        # Plot heatmap
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax8, cbar_kws={"shrink": .8})
        ax8.set_title('Feature Correlation Matrix')
        
        # 9. Data Quality Score
        ax9 = fig.add_subplot(gs[3, 2])
        ax9.axis('off')
        
        # Calculate ML readiness score
        ml_score = 100
        issues = []
        
        # Check for issues
        if len(missing_pct) > 0 and missing_pct.max() > 5:
            ml_score -= 20
            issues.append("High missing values")
        
        if not prices_correct:
            ml_score -= 15
            issues.append("Price hierarchy violation")
        
        if imbalance_ratio < 0.3:
            ml_score -= 10
            issues.append("Severe class imbalance")
        
        may_spike_pct = stay_pct.get(5, 0)
        if may_spike_pct > 25:  # May has more than 25% of stays
            ml_score -= 15
            issues.append(f"May spike: {may_spike_pct:.1f}%")
        
        # Display score
        score_color = 'green' if ml_score >= 80 else 'orange' if ml_score >= 60 else 'red'
        ax9.text(0.5, 0.7, f'ML Readiness Score', ha='center', fontsize=16, fontweight='bold')
        ax9.text(0.5, 0.5, f'{ml_score}/100', ha='center', fontsize=48, fontweight='bold', color=score_color)
        
        if issues:
            ax9.text(0.5, 0.2, 'Issues:', ha='center', fontsize=12, fontweight='bold')
            issues_text = '\n'.join([f'- {issue}' for issue in issues])
            ax9.text(0.5, 0.1, issues_text, ha='center', fontsize=10, va='top')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}01_ml_critical_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return ml_score, issues
    
    def create_temporal_analysis(self):
        """Create detailed temporal analysis with clear labeling"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('Temporal Analysis - Booking vs Stay Patterns', fontsize=16, fontweight='bold')
        
        # 1. Daily BOOKINGS over time
        ax1 = axes[0, 0]
        daily_bookings = self.bookings_df.groupby('booking_date').size()
        ax1.plot(daily_bookings.index, daily_bookings.values, alpha=0.7, color='blue')
        ax1.set_title('Daily BOOKING Volume (When Reservations Are Made)')
        ax1.set_xlabel('Booking Date')
        ax1.set_ylabel('Number of Bookings')
        ax1.grid(True, alpha=0.3)
        
        # 2. Daily STAYS over time
        ax2 = axes[0, 1]
        daily_stays = self.bookings_df.groupby('stay_start_date').size()
        ax2.plot(daily_stays.index, daily_stays.values, alpha=0.7, color='green')
        ax2.set_title('Daily STAY Volume (When Guests Actually Stay)')
        ax2.set_xlabel('Stay Start Date')
        ax2.set_ylabel('Number of Stays')
        ax2.grid(True, alpha=0.3)
        
        # 3. Lead time evolution
        ax3 = axes[1, 0]
        self.bookings_df['booking_month_year'] = self.bookings_df['booking_date'].dt.to_period('M')
        monthly_lead_time = self.bookings_df.groupby('booking_month_year')['lead_time'].mean()
        
        ax3.plot(monthly_lead_time.index.to_timestamp(), monthly_lead_time.values, marker='o')
        ax3.set_title('Average Lead Time Evolution')
        ax3.set_xlabel('Booking Month')
        ax3.set_ylabel('Average Lead Time (days)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Booking-to-Stay Flow
        ax4 = axes[1, 1]
        
        # Create flow matrix
        flow_matrix = pd.crosstab(self.bookings_df['booking_month'], self.bookings_df['stay_month'])
        
        # Plot heatmap
        sns.heatmap(flow_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Count'})
        ax4.set_title('Booking Month to Stay Month Flow')
        ax4.set_xlabel('Stay Month')
        ax4.set_ylabel('Booking Month')
        
        # 5. Weekly patterns - Bookings
        ax5 = axes[2, 0]
        weekday_bookings = self.bookings_df['booking_weekday'].value_counts().sort_index()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        ax5.bar(weekday_bookings.index, weekday_bookings.values, alpha=0.7, color='blue')
        ax5.set_title('Booking Patterns by Day of Week')
        ax5.set_xlabel('Day of Week')
        ax5.set_ylabel('Number of Bookings')
        ax5.set_xticks(range(7))
        ax5.set_xticklabels(days)
        ax5.grid(True, alpha=0.3)
        
        # 6. Weekly patterns - Stays
        ax6 = axes[2, 1]
        weekday_stays = self.bookings_df['stay_weekday'].value_counts().sort_index()
        
        ax6.bar(weekday_stays.index, weekday_stays.values, alpha=0.7, color='green')
        ax6.set_title('Stay Patterns by Day of Week')
        ax6.set_xlabel('Day of Week')
        ax6.set_ylabel('Number of Stays')
        ax6.set_xticks(range(7))
        ax6.set_xticklabels(days)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}02_temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_feature_engineering_insights(self):
        """Visualizations to guide feature engineering for ML"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('Feature Engineering Insights for ML', fontsize=16, fontweight='bold')
        
        # 1. Price variations by features
        ax1 = axes[0, 0]
        price_by_segment_channel = self.bookings_df.groupby(['customer_segment', 'booking_channel'])['final_price'].mean().unstack()
        price_by_segment_channel.plot(kind='bar', ax=ax1)
        ax1.set_title('Price by Segment and Channel')
        ax1.set_xlabel('Customer Segment')
        ax1.set_ylabel('Average Price ($)')
        ax1.legend(title='Channel')
        ax1.grid(True, alpha=0.3)
        
        # 2. Discount effectiveness
        ax2 = axes[0, 1]
        
        # Create discount bins
        self.bookings_df['has_discount'] = self.bookings_df['discount_amount'] > 0
        discount_impact = self.bookings_df.groupby('has_discount').agg({
            'is_cancelled': 'mean',
            'stay_length': 'mean',
            'lead_time': 'mean'
        })
        
        x = np.arange(len(discount_impact.columns))
        width = 0.35
        
        no_disc = discount_impact.loc[False].values
        with_disc = discount_impact.loc[True].values
        
        ax2.bar(x - width/2, no_disc, width, label='No Discount', alpha=0.7)
        ax2.bar(x + width/2, with_disc, width, label='With Discount', alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Cancel Rate', 'Avg Stay', 'Avg Lead'])
        ax2.set_title('Impact of Discounts')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cancellation predictors
        ax3 = axes[0, 2]
        
        # Cancellation rate by lead time bins
        self.bookings_df['lead_time_bin'] = pd.cut(self.bookings_df['lead_time'], 
                                                   bins=[0, 7, 30, 60, 120, 365],
                                                   labels=['0-7d', '8-30d', '31-60d', '61-120d', '120+d'])
        
        cancel_by_lead = self.bookings_df.groupby('lead_time_bin')['is_cancelled'].mean()
        
        ax3.bar(range(len(cancel_by_lead)), cancel_by_lead.values, alpha=0.7, color='red')
        ax3.set_xticks(range(len(cancel_by_lead)))
        ax3.set_xticklabels(cancel_by_lead.index)
        ax3.set_title('Cancellation Rate by Lead Time')
        ax3.set_ylabel('Cancellation Rate')
        ax3.grid(True, alpha=0.3)
        
        # 4. Stay length patterns
        ax4 = axes[1, 0]
        stay_length_by_segment = self.bookings_df.groupby('customer_segment')['stay_length'].value_counts().unstack(fill_value=0)
        stay_length_by_segment.T.plot(kind='bar', stacked=True, ax=ax4)
        ax4.set_title('Stay Length Distribution by Segment')
        ax4.set_xlabel('Stay Length (nights)')
        ax4.set_ylabel('Count')
        ax4.legend(title='Segment', bbox_to_anchor=(1.05, 1))
        ax4.grid(True, alpha=0.3)
        
        # 5. Campaign effectiveness
        ax5 = axes[1, 1]
        
        campaign_bookings = self.bookings_df[self.bookings_df['campaign_id'].notna()].copy()
        if len(campaign_bookings) > 0:
            campaign_bookings['campaign_type'] = campaign_bookings['campaign_id'].str.extract(r'^([A-Z]+)_')[0]
            campaign_effectiveness = campaign_bookings.groupby('campaign_type').agg({
                'incremental_flag': 'mean',
                'discount_amount': 'mean'
            })
            
            ax5_twin = ax5.twinx()
            
            x = range(len(campaign_effectiveness))
            ax5.bar(x, campaign_effectiveness['incremental_flag'], alpha=0.7, color='green', label='Incremental Rate')
            ax5_twin.plot(x, campaign_effectiveness['discount_amount'], 'ro-', label='Avg Discount')
            
            ax5.set_xticks(x)
            ax5.set_xticklabels(campaign_effectiveness.index)
            ax5.set_ylabel('Incremental Rate', color='green')
            ax5_twin.set_ylabel('Average Discount ($)', color='red')
            ax5.set_title('Campaign Type Effectiveness')
            ax5.grid(True, alpha=0.3)
        
        # 6. Room type preferences
        ax6 = axes[1, 2]
        room_segment_dist = pd.crosstab(self.bookings_df['customer_segment'], self.bookings_df['room_type'])
        room_segment_dist.plot(kind='bar', stacked=True, ax=ax6)
        ax6.set_title('Room Type Preferences by Segment')
        ax6.set_xlabel('Customer Segment')
        ax6.set_ylabel('Bookings')
        ax6.legend(title='Room Type')
        ax6.grid(True, alpha=0.3)
        
        # 7. Seasonality impact on price
        ax7 = axes[2, 0]
        monthly_metrics = self.bookings_df.groupby('stay_month').agg({
            'final_price': 'mean',
            'booking_id': 'count'
        })
        
        ax7_twin = ax7.twinx()
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        x = monthly_metrics.index
        
        ax7.bar(x, monthly_metrics['booking_id'], alpha=0.5, color='blue', label='Volume')
        ax7_twin.plot(x, monthly_metrics['final_price'], 'ro-', linewidth=2, markersize=8, label='Avg Price')
        
        ax7.set_xticks(x)
        ax7.set_xticklabels([months[m-1] for m in x])
        ax7.set_ylabel('Booking Volume', color='blue')
        ax7_twin.set_ylabel('Average Price ($)', color='red')
        ax7.set_title('Seasonality: Volume vs Price')
        ax7.grid(True, alpha=0.3)
        
        # 8. Feature importance hints
        ax8 = axes[2, 1]
        
        # Simple correlation with price as target
        numeric_cols = ['lead_time', 'stay_length', 'room_type_encoded', 'segment_encoded', 'channel_encoded']
        correlations = {}
        
        for col in numeric_cols:
            if col in self.bookings_df.columns:
                corr = abs(self.bookings_df[col].corr(self.bookings_df['final_price']))
                correlations[col] = corr
        
        sorted_corr = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
        
        ax8.barh(range(len(sorted_corr)), list(sorted_corr.values()), alpha=0.7)
        ax8.set_yticks(range(len(sorted_corr)))
        ax8.set_yticklabels(list(sorted_corr.keys()))
        ax8.set_xlabel('Absolute Correlation with Price')
        ax8.set_title('Feature Importance Indicators')
        ax8.grid(True, alpha=0.3)
        
        # 9. Data splits recommendation
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        # Calculate recommended splits
        total_records = len(self.bookings_df)
        train_size = int(0.7 * total_records)
        val_size = int(0.15 * total_records)
        test_size = total_records - train_size - val_size
        
        # Time-based split recommendation
        date_range = self.bookings_df['booking_date'].max() - self.bookings_df['booking_date'].min()
        
        recommendation_text = f"""
        ML Data Split Recommendations:
        
        Total Records: {total_records:,}
        
        Random Split:
        - Train: {train_size:,} (70%)
        - Validation: {val_size:,} (15%)
        - Test: {test_size:,} (15%)
        
        Time-Based Split:
        - Use last 20% of bookings as test
        - Maintains temporal integrity
        - Better for production scenarios
        
        Feature Engineering Tips:
        - Create day_of_week features
        - Add is_weekend flags
        - Calculate price_per_night
        - Include rolling averages
        - Add holiday indicators
        """
        
        ax9.text(0.1, 0.9, recommendation_text, transform=ax9.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}03_feature_engineering_insights.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_data_quality_report(self):
        """Create comprehensive data quality report for ML"""
        fig = plt.figure(figsize=(16, 20))
        gs = gridspec.GridSpec(5, 2, hspace=0.3, wspace=0.25)
        
        fig.suptitle('Comprehensive Data Quality Report for ML', fontsize=18, fontweight='bold')
        
        # 1. Overall statistics summary
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        stats_text = f"""
        Dataset Overview:
        - Total bookings: {len(self.bookings_df):,}
        - Date range: {self.bookings_df['booking_date'].min().strftime('%Y-%m-%d')} to {self.bookings_df['booking_date'].max().strftime('%Y-%m-%d')}
        - Unique customers: {self.bookings_df['customer_id'].nunique():,}
        - Total revenue: ${self.bookings_df['final_price'].sum():,.2f}
        - Average booking value: ${self.bookings_df['final_price'].mean():.2f}
        - Cancellation rate: {self.bookings_df['is_cancelled'].mean():.1%}
        """
        
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # 2. Numeric feature distributions
        ax2 = fig.add_subplot(gs[1:3, 0])
        
        numeric_features = ['final_price', 'lead_time', 'stay_length', 'discount_amount']
        fig_dist, axes_dist = plt.subplots(2, 2, figsize=(8, 6))
        axes_dist = axes_dist.ravel()
        
        for i, feature in enumerate(numeric_features):
            if feature in self.bookings_df.columns:
                data = self.bookings_df[feature].dropna()
                axes_dist[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                axes_dist[i].axvline(data.mean(), color='red', linestyle='--', label=f'mean={data.mean():.1f}')
                axes_dist[i].axvline(data.median(), color='green', linestyle='--', label=f'median={data.median():.1f}')
                axes_dist[i].set_title(feature)
                axes_dist[i].legend(fontsize=8)
                axes_dist[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_dist.savefig(f'{self.output_dir}temp_distributions.png', dpi=150)
        plt.close(fig_dist)
        
        from PIL import Image
        img = Image.open(f'{self.output_dir}temp_distributions.png')
        ax2.imshow(img)
        ax2.axis('off')
        ax2.set_title('Numeric Feature Distributions')
        os.remove(f'{self.output_dir}temp_distributions.png')
        
        # 3. Categorical feature balance
        ax3 = fig.add_subplot(gs[1:3, 1])
        
        categorical_features = ['customer_segment', 'room_type', 'booking_channel']
        imbalance_scores = {}
        
        y_position = 0.9
        for feature in categorical_features:
            if feature in self.bookings_df.columns:
                counts = self.bookings_df[feature].value_counts()
                imbalance_ratio = counts.min() / counts.max()
                imbalance_scores[feature] = imbalance_ratio
                
                # Display distribution
                dist_text = f"{feature}: "
                for val, count in counts.items():
                    pct = count / counts.sum() * 100
                    dist_text += f"{val}={pct:.1f}% "
                
                color = 'green' if imbalance_ratio > 0.5 else 'orange' if imbalance_ratio > 0.3 else 'red'
                ax3.text(0.05, y_position, dist_text, transform=ax3.transAxes, fontsize=10, color=color)
                ax3.text(0.95, y_position, f"Balance: {imbalance_ratio:.2f}", transform=ax3.transAxes, 
                        fontsize=10, color=color, ha='right')
                y_position -= 0.15
        
        ax3.set_title('Categorical Feature Balance')
        ax3.axis('off')
        
        # 4. Data consistency checks
        ax4 = fig.add_subplot(gs[3, :])
        
        consistency_checks = {
            'Valid stay dates': (self.bookings_df['stay_start_date'] > self.bookings_df['booking_date']).mean(),
            'Positive prices': (self.bookings_df['final_price'] > 0).mean(),
            'Valid discounts': (self.bookings_df['discount_amount'] <= self.bookings_df['base_price']).mean(),
            'Attribution in range': ((self.bookings_df['attribution_score'] >= 0) & 
                                   (self.bookings_df['attribution_score'] <= 1)).mean(),
            'Valid cancellations': (self.bookings_df[self.bookings_df['is_cancelled']]['cancellation_date'].notna()).mean()
        }
        
        checks = list(consistency_checks.keys())
        rates = list(consistency_checks.values())
        colors = ['green' if r == 1.0 else 'orange' if r > 0.95 else 'red' for r in rates]
        
        bars = ax4.barh(range(len(checks)), rates, color=colors, alpha=0.7)
        ax4.set_yticks(range(len(checks)))
        ax4.set_yticklabels(checks)
        ax4.set_xlabel('Pass Rate')
        ax4.set_xlim(0, 1.1)
        
        for i, rate in enumerate(rates):
            ax4.text(rate + 0.01, i, f'{rate:.1%}', va='center')
        
        ax4.set_title('Data Consistency Checks')
        ax4.grid(True, alpha=0.3)
        
        # 5. ML-specific warnings
        ax5 = fig.add_subplot(gs[4, :])
        ax5.axis('off')
        
        warnings = []
        
        # Check for May spike
        may_stays_pct = (self.bookings_df['stay_month'] == 5).mean() * 100
        if may_stays_pct > 25:
            warnings.append(f"WARNING: May stay concentration: {may_stays_pct:.1f}% (should be ~18%)")
        
        # Check for data leakage risks
        if 'attribution_score' in self.bookings_df.columns:
            warnings.append("WARNING: Remove 'attribution_score' from features - it's derived from the target")
        
        # Check for high cardinality
        if self.bookings_df['customer_id'].nunique() > len(self.bookings_df) * 0.8:
            warnings.append("WARNING: High cardinality in customer_id - consider aggregated features instead")
        
        # Check temporal integrity
        warnings.append("TIP: Use time-based train/test split to avoid data leakage")
        
        # Display warnings
        warning_text = "ML-Specific Warnings and Recommendations:\n\n"
        for warning in warnings:
            warning_text += f"{warning}\n"
        
        ax5.text(0.05, 0.95, warning_text, transform=ax5.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}04_data_quality_report.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_executive_summary(self, ml_score, issues):
        """Create a one-page executive summary"""
        fig = plt.figure(figsize=(11, 8.5))  # Letter size
        gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.3, 
                              left=0.1, right=0.95, top=0.92, bottom=0.08)
        
        # Title
        fig.suptitle('Hotel Booking Data - Executive Summary', fontsize=20, fontweight='bold')
        
        # 1. Key metrics box
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.axis('off')
        
        key_metrics = f"""
        KEY METRICS
        ├─ Total Bookings: {len(self.bookings_df):,}
        ├─ Revenue: ${self.bookings_df['final_price'].sum():,.0f}
        ├─ Avg Price: ${self.bookings_df['final_price'].mean():.0f}
        ├─ Cancellation Rate: {self.bookings_df['is_cancelled'].mean():.1%}
        └─ Campaign Participation: {(self.bookings_df['campaign_id'].notna()).mean():.1%}
        """
        
        ax1.text(0.05, 0.95, key_metrics, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # 2. ML Score
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        score_color = 'green' if ml_score >= 80 else 'orange' if ml_score >= 60 else 'red'
        ax2.text(0.5, 0.5, f'{ml_score}', ha='center', va='center', fontsize=72,
                fontweight='bold', color=score_color)
        ax2.text(0.5, 0.15, 'ML Readiness', ha='center', fontsize=12)
        
        # 3. Monthly distribution
        ax3 = fig.add_subplot(gs[1, :])
        stay_dist = self.bookings_df['stay_month'].value_counts().sort_index()
        stay_pct = stay_dist / stay_dist.sum() * 100
        
        colors = ['red' if month == 5 and pct > 25 else 'lightblue' for month, pct in stay_pct.items()]
        ax3.bar(stay_dist.index, stay_pct.values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add target line for May
        if 5 in stay_dist.index:
            ax3.axhline(y=18, color='green', linestyle='--', alpha=0.5)
            ax3.text(5.3, 18.5, 'Target: 18%', fontsize=9, color='green')
        
        ax3.set_title('Stay Distribution by Month (Key Issue: May Spike)', fontweight='bold')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Percentage of Stays (%)')
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax3.grid(True, alpha=0.3)
        
        # 4. Issues summary
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        issues_text = "DATA QUALITY ISSUES:\n"
        if issues:
            for issue in issues:
                issues_text += f"- {issue}\n"
        else:
            issues_text += "- No critical issues found\n"
        
        issues_text += "\nRECOMMENDATIONS:\n"
        issues_text += "- Fix May spike by improving distribution logic\n"
        issues_text += "- Use time-based train/test splits\n"
        issues_text += "- Consider feature engineering for seasonality\n"
        
        ax4.text(0.05, 0.95, issues_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
        
        plt.savefig(f'{self.output_dir}00_executive_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all ML-focused visualizations"""
        print("GENERATING ML-FOCUSED VISUALIZATIONS")
        print("=" * 60)
        
        if not self.load_data():
            return False
        
        print(f"Output directory: {self.output_dir}")
        
        try:
            # Generate all visualizations
            ml_score, issues = self.create_critical_ml_metrics()
            print("Created critical ML metrics dashboard")
            
            self.create_temporal_analysis()
            print("Created temporal analysis")
            
            self.create_feature_engineering_insights()
            print("Created feature engineering insights")
            
            self.create_data_quality_report()
            print("Created data quality report")
            
            self.create_executive_summary(ml_score, issues)
            print("Created executive summary")
            
            # Create index HTML
            self.create_visualization_index(ml_score)
            
            print(f"\nAll visualizations completed successfully!")
            print(f"Check the '{self.output_dir}' directory for all visualizations")
            print(f"Open '{self.output_dir}index.html' for overview")
            
            return True
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_visualization_index(self, ml_score):
        """Create HTML index for all visualizations"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML-Focused Hotel Data Visualizations</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                h1 {{ color: #2c3e50; text-align: center; }}
                .summary {{ background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                .warning {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .success {{ background-color: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .viz-container {{ margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Hotel Booking Data - ML Analysis</h1>
            
            <div class="summary">
                <h2>Quick Summary</h2>
                <div class="metric"><strong>ML Readiness Score:</strong> {ml_score}/100</div>
                <div class="metric"><strong>Total Records:</strong> {len(self.bookings_df):,}</div>
                <div class="metric"><strong>Date Range:</strong> {self.bookings_df['booking_date'].min().strftime('%Y-%m-%d')} to {self.bookings_df['booking_date'].max().strftime('%Y-%m-%d')}</div>
            </div>
            
            <div class="warning">
                <h3>Critical Finding: May Stay Spike</h3>
                <p>Analysis shows {(self.bookings_df['stay_month'] == 5).mean() * 100:.1f}% of stays are in May (target: 18%). 
                This indicates the seasonal distribution logic needs adjustment.</p>
            </div>
            
            <div class="viz-container">
                <h2>Visualizations</h2>
                <img src="00_executive_summary.png" alt="Executive Summary">
                <img src="01_ml_critical_metrics.png" alt="ML Critical Metrics">
                <img src="02_temporal_analysis.png" alt="Temporal Analysis">
                <img src="03_feature_engineering_insights.png" alt="Feature Engineering">
                <img src="04_data_quality_report.png" alt="Data Quality Report">
            </div>
        </body>
        </html>
        """
        
        with open(f'{self.output_dir}index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate ML-focused visualizations for hotel booking data')
    parser.add_argument('--prefix', type=str, default='', help='Data file prefix')
    parser.add_argument('--output-dir', type=str, default='ml_visualizations/', help='Output directory')
    
    args = parser.parse_args()
    
    visualizer = MLDataVisualizer(data_prefix=args.prefix, output_dir=args.output_dir)
    success = visualizer.generate_all_visualizations()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())