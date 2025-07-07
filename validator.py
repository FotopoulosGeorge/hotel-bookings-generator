"""
Comprehensive Data Validation Suite
Creates multiple visualizations for thorough data quality validation
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg', force=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.stats import normaltest
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# Set consistent plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11


class ComprehensiveDataValidator:
    """Comprehensive validation suite with multiple visualization types"""
    
    def __init__(self, data_prefix='', output_dir='validation_suite/'):
        self.data_prefix = data_prefix
        self.output_dir = output_dir
        self.bookings_df = None
        self.campaigns_df = None
        self.customers_df = None
        self.validation_results = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load and prepare data"""
        print(f"🔍 Loading data for comprehensive validation...")
        
        try:
            base_path = f"output/{self.data_prefix}"
            
            self.bookings_df = pd.read_csv(f"{base_path}historical_bookings.csv")
            self.campaigns_df = pd.read_csv(f"{base_path}campaigns_run.csv")
            self.customers_df = pd.read_csv(f"{base_path}customer_segments.csv")
            
            # Date conversions
            date_cols = ['booking_date', 'stay_start_date', 'stay_end_date', 'cancellation_date']
            for col in date_cols:
                if col in self.bookings_df.columns:
                    self.bookings_df[col] = pd.to_datetime(self.bookings_df[col], errors='coerce')
            
            # Derived features
            self.bookings_df['lead_time'] = (self.bookings_df['stay_start_date'] - self.bookings_df['booking_date']).dt.days
            self.bookings_df['booking_month'] = self.bookings_df['booking_date'].dt.month
            self.bookings_df['stay_month'] = self.bookings_df['stay_start_date'].dt.month
            self.bookings_df['booking_weekday'] = self.bookings_df['booking_date'].dt.dayofweek
            
            print(f"   ✅ Loaded {len(self.bookings_df):,} bookings")
            print(f"   📅 Date range: {self.bookings_df['booking_date'].min().date()} to {self.bookings_df['booking_date'].max().date()}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False
    
    def validate_temporal_patterns(self):
        """Validation 1: Temporal patterns and consistency"""
        print("📅 Validating temporal patterns...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Temporal Pattern Validation', fontsize=16, fontweight='bold')
        
        # 1. Booking vs Stay Month Comparison
        ax1 = axes[0, 0]
        
        booking_monthly = self.bookings_df['booking_month'].value_counts().sort_index()
        stay_monthly = self.bookings_df['stay_month'].value_counts().sort_index()
        
        months = range(1, 13)
        month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        
        booking_pct = booking_monthly.reindex(months, fill_value=0) / booking_monthly.sum() * 100
        stay_pct = stay_monthly.reindex(months, fill_value=0) / stay_monthly.sum() * 100
        
        x = np.arange(len(months))
        width = 0.35
        
        ax1.bar(x - width/2, booking_pct, width, label='Booking Month', alpha=0.7, color='skyblue')
        ax1.bar(x + width/2, stay_pct, width, label='Stay Month', alpha=0.7, color='lightcoral')
        
        ax1.set_title('Booking vs Stay Month Distribution')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(month_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Check for realistic patterns
        stay_peak_months = stay_pct.nlargest(3).index.tolist()
        if set([7, 8]).intersection(set(stay_peak_months)):
            ax1.text(0.02, 0.98, '✅ Summer peaks detected', transform=ax1.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'), verticalalignment='top')
        else:
            ax1.text(0.02, 0.98, '⚠️ Unusual seasonal pattern', transform=ax1.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'), verticalalignment='top')
        
        # 2. Lead Time Distribution by Month
        ax2 = axes[0, 1]
        
        monthly_lead_time = self.bookings_df.groupby('booking_month')['lead_time'].mean()
        ax2.plot(monthly_lead_time.index, monthly_lead_time.values, marker='o', linewidth=2, markersize=8)
        ax2.set_title('Average Lead Time by Booking Month')
        ax2.set_xlabel('Booking Month')
        ax2.set_ylabel('Average Lead Time (days)')
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(month_names)
        ax2.grid(True, alpha=0.3)
        
        # Validate lead time patterns
        lead_time_range = monthly_lead_time.max() - monthly_lead_time.min()
        if lead_time_range > 30:
            ax2.text(0.02, 0.98, f'✅ Seasonal lead time variation: {lead_time_range:.1f} days', 
                    transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'),
                    verticalalignment='top')
        else:
            ax2.text(0.02, 0.98, f'⚠️ Low lead time variation: {lead_time_range:.1f} days', 
                    transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'),
                    verticalalignment='top')
        
        # 3. Weekly Booking Patterns
        ax3 = axes[0, 2]
        
        weekly_bookings = self.bookings_df['booking_weekday'].value_counts().sort_index()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        bars = ax3.bar(range(7), weekly_bookings.values, alpha=0.7, color='lightgreen')
        ax3.set_title('Booking Volume by Day of Week')
        ax3.set_xlabel('Day of Week')
        ax3.set_ylabel('Number of Bookings')
        ax3.set_xticks(range(7))
        ax3.set_xticklabels(day_names)
        ax3.grid(True, alpha=0.3)
        
        # Check for realistic weekly patterns
        weekend_bookings = weekly_bookings.iloc[5:].sum()  # Sat, Sun
        weekday_bookings = weekly_bookings.iloc[:5].sum()   # Mon-Fri
        weekend_ratio = weekend_bookings / weekday_bookings
        
        if 0.3 <= weekend_ratio <= 0.8:
            ax3.text(0.02, 0.98, f'✅ Realistic weekend ratio: {weekend_ratio:.2f}', 
                    transform=ax3.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'),
                    verticalalignment='top')
        else:
            ax3.text(0.02, 0.98, f'⚠️ Unusual weekend ratio: {weekend_ratio:.2f}', 
                    transform=ax3.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'),
                    verticalalignment='top')
        
        # 4. Daily Booking Volume Timeline
        ax4 = axes[1, 0]
        
        daily_bookings = self.bookings_df.groupby('booking_date').size()
        ax4.plot(daily_bookings.index, daily_bookings.values, alpha=0.7, color='purple')
        ax4.set_title('Daily Booking Volume Over Time')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Bookings per Day')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Check for consistency
        daily_std = daily_bookings.std()
        daily_mean = daily_bookings.mean()
        cv = daily_std / daily_mean  # Coefficient of variation
        
        if cv < 0.5:
            ax4.text(0.02, 0.98, f'✅ Stable daily volume (CV: {cv:.2f})', 
                    transform=ax4.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'),
                    verticalalignment='top')
        else:
            ax4.text(0.02, 0.98, f'⚠️ High daily volatility (CV: {cv:.2f})', 
                    transform=ax4.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'),
                    verticalalignment='top')
        
        # 5. Lead Time vs Booking Date
        ax5 = axes[1, 1]
        
        # Sample data for scatter plot (use every 10th point to avoid overcrowding)
        sample_data = self.bookings_df.iloc[::10]
        ax5.scatter(sample_data['booking_date'], sample_data['lead_time'], alpha=0.5, s=10)
        ax5.set_title('Lead Time vs Booking Date')
        ax5.set_xlabel('Booking Date')
        ax5.set_ylabel('Lead Time (days)')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Check for realistic lead time range
        lead_time_min = self.bookings_df['lead_time'].min()
        lead_time_max = self.bookings_df['lead_time'].max()
        
        if lead_time_min >= 0 and lead_time_max <= 365:
            ax5.text(0.02, 0.98, f'✅ Valid lead time range: {lead_time_min}-{lead_time_max} days', 
                    transform=ax5.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'),
                    verticalalignment='top')
        else:
            ax5.text(0.02, 0.98, f'❌ Invalid lead times: {lead_time_min}-{lead_time_max} days', 
                    transform=ax5.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'),
                    verticalalignment='top')
        
        # 6. Stay Length Distribution
        ax6 = axes[1, 2]
        
        stay_length_counts = self.bookings_df['stay_length'].value_counts().sort_index()
        ax6.bar(stay_length_counts.index[:15], stay_length_counts.values[:15], alpha=0.7, color='orange')
        ax6.set_title('Stay Length Distribution (1-15 nights)')
        ax6.set_xlabel('Stay Length (nights)')
        ax6.set_ylabel('Number of Bookings')
        ax6.grid(True, alpha=0.3)
        
        # Check for realistic stay length patterns
        avg_stay = self.bookings_df['stay_length'].mean()
        if 3 <= avg_stay <= 10:
            ax6.text(0.02, 0.98, f'✅ Realistic avg stay: {avg_stay:.1f} nights', 
                    transform=ax6.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'),
                    verticalalignment='top')
        else:
            ax6.text(0.02, 0.98, f'⚠️ Unusual avg stay: {avg_stay:.1f} nights', 
                    transform=ax6.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'),
                    verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}01_temporal_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store validation results
        self.validation_results['temporal'] = {
            'weekend_ratio': weekend_ratio,
            'lead_time_range': lead_time_range,
            'daily_volatility': cv,
            'avg_stay_length': avg_stay,
            'valid_lead_times': lead_time_min >= 0 and lead_time_max <= 365
        }
        
        print("   ✅ Temporal validation completed")
    
    def validate_business_logic(self):
        """Validation 2: Business logic and pricing rules"""
        print("💼 Validating business logic...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Business Logic Validation', fontsize=16, fontweight='bold')
        
        # 1. Price Hierarchy by Room Type
        ax1 = axes[0, 0]
        
        room_price_stats = self.bookings_df.groupby('room_type')['final_price'].agg(['mean', 'std', 'count'])
        expected_order = ['Standard', 'Deluxe', 'Suite', 'Premium']
        existing_rooms = [rt for rt in expected_order if rt in room_price_stats.index]
        
        if len(existing_rooms) >= 2:
            means = [room_price_stats.loc[rt, 'mean'] for rt in existing_rooms]
            stds = [room_price_stats.loc[rt, 'std'] for rt in existing_rooms]
            
            bars = ax1.bar(existing_rooms, means, yerr=stds, capsize=5, alpha=0.7)
            
            # Check price ordering
            prices_correct = all(means[i] <= means[i+1] * 1.05 for i in range(len(means)-1))
            
            bar_color = 'green' if prices_correct else 'red'
            for bar in bars:
                bar.set_color(bar_color)
            
            ax1.set_title(f'Price Hierarchy {"✅ OK" if prices_correct else "❌ ERROR"}')
            
            # Add price values as text
            for i, (room, mean) in enumerate(zip(existing_rooms, means)):
                ax1.text(i, mean + stds[i] + 10, f'${mean:.0f}', ha='center', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'Insufficient room types', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Price Hierarchy - Insufficient Data')
            prices_correct = True  # Can't validate
        
        ax1.set_xlabel('Room Type')
        ax1.set_ylabel('Average Price ($)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Lead Time by Customer Segment
        ax2 = axes[0, 1]
        
        segment_leadtime = self.bookings_df.groupby('customer_segment')['lead_time'].agg(['mean', 'std'])
        expected_segments = ['Last_Minute', 'Flexible', 'Early_Planner']
        existing_segments = [seg for seg in expected_segments if seg in segment_leadtime.index]
        
        if len(existing_segments) >= 2:
            means = [segment_leadtime.loc[seg, 'mean'] for seg in existing_segments]
            stds = [segment_leadtime.loc[seg, 'std'] for seg in existing_segments]
            
            bars = ax2.bar(existing_segments, means, yerr=stds, capsize=5, alpha=0.7, color='lightblue')
            
            # Check ordering (should increase)
            leadtime_correct = all(means[i] <= means[i+1] * 1.2 for i in range(len(means)-1))
            
            if leadtime_correct:
                ax2.text(0.02, 0.98, '✅ Lead times increase correctly', transform=ax2.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'), verticalalignment='top')
            else:
                ax2.text(0.02, 0.98, '⚠️ Lead time ordering issue', transform=ax2.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'), verticalalignment='top')
            
            ax2.set_title('Lead Time by Customer Segment')
            
            # Add values as text
            for i, (seg, mean) in enumerate(zip(existing_segments, means)):
                ax2.text(i, mean + stds[i] + 5, f'{mean:.0f}d', ha='center', fontweight='bold')
        else:
            leadtime_correct = True  # Can't validate
            ax2.text(0.5, 0.5, 'Insufficient segments', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Lead Time - Insufficient Data')
        
        ax2.set_xlabel('Customer Segment')
        ax2.set_ylabel('Average Lead Time (days)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Cancellation Rate by Segment
        ax3 = axes[0, 2]
        
        segment_cancellation = self.bookings_df.groupby('customer_segment')['is_cancelled'].agg(['mean', 'count'])
        
        cancellation_rates = segment_cancellation['mean'] * 100
        booking_counts = segment_cancellation['count']
        
        bars = ax3.bar(cancellation_rates.index, cancellation_rates.values, alpha=0.7, color='orange')
        ax3.set_title('Cancellation Rate by Customer Segment')
        ax3.set_xlabel('Customer Segment')
        ax3.set_ylabel('Cancellation Rate (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, (segment, rate) in enumerate(cancellation_rates.items()):
            ax3.text(i, rate + 0.5, f'{rate:.1f}%', ha='center', fontweight='bold')
        
        # Check for reasonable cancellation rates
        overall_cancellation = self.bookings_df['is_cancelled'].mean() * 100
        if 5 <= overall_cancellation <= 25:
            ax3.text(0.02, 0.98, f'✅ Realistic cancellation: {overall_cancellation:.1f}%', 
                    transform=ax3.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'),
                    verticalalignment='top')
        else:
            ax3.text(0.02, 0.98, f'⚠️ Unusual cancellation: {overall_cancellation:.1f}%', 
                    transform=ax3.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'),
                    verticalalignment='top')
        
        # 4. Price vs Discount Relationship
        ax4 = axes[1, 0]
        
        promotional_bookings = self.bookings_df[self.bookings_df['discount_amount'] > 0]
        if len(promotional_bookings) > 0:
            promotional_bookings = promotional_bookings.copy()
            promotional_bookings['discount_pct'] = promotional_bookings['discount_amount'] / promotional_bookings['base_price'] * 100
            
            # Sample for scatter plot
            sample_promo = promotional_bookings.sample(min(1000, len(promotional_bookings)))
            ax4.scatter(sample_promo['base_price'], sample_promo['discount_pct'], alpha=0.5, s=10)
            ax4.set_title('Discount Percentage vs Base Price')
            ax4.set_xlabel('Base Price ($)')
            ax4.set_ylabel('Discount Percentage (%)')
            ax4.grid(True, alpha=0.3)
            
            # Check discount reasonableness
            max_discount = promotional_bookings['discount_pct'].max()
            avg_discount = promotional_bookings['discount_pct'].mean()
            
            if max_discount <= 50 and avg_discount <= 30:
                ax4.text(0.02, 0.98, f'✅ Reasonable discounts (max: {max_discount:.1f}%)', 
                        transform=ax4.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'),
                        verticalalignment='top')
            else:
                ax4.text(0.02, 0.98, f'⚠️ High discounts (max: {max_discount:.1f}%)', 
                        transform=ax4.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'),
                        verticalalignment='top')
        else:
            ax4.text(0.5, 0.5, 'No promotional bookings found', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Discount Analysis - No Data')
            max_discount = 0
        
        # 5. Channel Distribution with Pricing
        ax5 = axes[1, 1]
        
        channel_stats = self.bookings_df.groupby('booking_channel').agg({
            'final_price': 'mean',
            'booking_id': 'count'
        })
        
        # Create dual axis
        ax5_twin = ax5.twinx()
        
        x_pos = range(len(channel_stats))
        bars1 = ax5.bar([x - 0.2 for x in x_pos], channel_stats['booking_id'], 
                       width=0.4, alpha=0.7, color='skyblue', label='Booking Count')
        bars2 = ax5_twin.bar([x + 0.2 for x in x_pos], channel_stats['final_price'], 
                            width=0.4, alpha=0.7, color='lightcoral', label='Avg Price')
        
        ax5.set_title('Channel Performance: Volume vs Price')
        ax5.set_xlabel('Booking Channel')
        ax5.set_ylabel('Booking Count', color='blue')
        ax5_twin.set_ylabel('Average Price ($)', color='red')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(channel_stats.index, rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Check channel balance
        channel_counts = channel_stats['booking_id']
        channel_balance = channel_counts.min() / channel_counts.max()
        
        if channel_balance > 0.3:
            ax5.text(0.02, 0.98, f'✅ Balanced channels (ratio: {channel_balance:.2f})', 
                    transform=ax5.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'),
                    verticalalignment='top')
        else:
            ax5.text(0.02, 0.98, f'⚠️ Imbalanced channels (ratio: {channel_balance:.2f})', 
                    transform=ax5.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'),
                    verticalalignment='top')
        
        # 6. Data Consistency Summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Perform consistency checks
        consistency_checks = {
            'Valid Prices': (self.bookings_df['final_price'] > 0).mean(),
            'Valid Dates': (self.bookings_df['stay_start_date'] > self.bookings_df['booking_date']).mean(),
            'Valid Lead Times': (self.bookings_df['lead_time'] >= 0).mean(),
            'Valid Stay Lengths': (self.bookings_df['stay_length'] > 0).mean(),
            'Price ≤ Base Price': (self.bookings_df['final_price'] <= self.bookings_df['base_price']).mean()
        }
        
        y_pos = 0.9
        ax6.text(0.1, 0.95, 'Data Consistency Checks:', transform=ax6.transAxes, 
                fontsize=14, fontweight='bold')
        
        for check_name, pass_rate in consistency_checks.items():
            color = 'green' if pass_rate >= 0.99 else 'orange' if pass_rate >= 0.95 else 'red'
            status = '✅' if pass_rate >= 0.99 else '⚠️' if pass_rate >= 0.95 else '❌'
            
            ax6.text(0.1, y_pos, f'{status} {check_name}: {pass_rate:.1%}', 
                    transform=ax6.transAxes, fontsize=11, color=color)
            y_pos -= 0.12
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}02_business_logic_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store validation results
        self.validation_results['business_logic'] = {
            'price_hierarchy_correct': prices_correct,
            'leadtime_ordering_correct': leadtime_correct,
            'overall_cancellation_rate': overall_cancellation,
            'max_discount_percentage': max_discount,
            'channel_balance': channel_balance,
            'consistency_checks': consistency_checks
        }
        
        print("   ✅ Business logic validation completed")
    
    def validate_campaign_effectiveness(self):
        """Validation 3: Campaign analysis and attribution"""
        print("🎯 Validating campaign effectiveness...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Campaign Effectiveness Validation', fontsize=16, fontweight='bold')
        
        campaign_bookings = self.bookings_df[self.bookings_df['campaign_id'].notna()]
        
        if len(campaign_bookings) == 0:
            # Handle case with no campaign bookings
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No campaign bookings found', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Campaign Analysis - No Data')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}03_campaign_validation.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.validation_results['campaigns'] = {'no_campaigns': True}
            print("   ⚠️ No campaign data found")
            return
        
        # Extract campaign types
        campaign_bookings = campaign_bookings.copy()
        campaign_bookings['campaign_type'] = campaign_bookings['campaign_id'].str.extract(r'^([A-Z]+)_')[0]
        campaign_bookings['campaign_type'] = campaign_bookings['campaign_type'].map({
            'EB': 'Early_Booking',
            'FS': 'Flash_Sale', 
            'SO': 'Special_Offer',
            'ADV': 'Advance_Campaign',
            'LM': 'LastMinute_Sale'
        })
        
        # 1. Campaign Participation Over Time
        ax1 = axes[0, 0]
        
        # Daily campaign participation rate
        daily_total = self.bookings_df.groupby('booking_date').size()
        daily_campaign = campaign_bookings.groupby('booking_date').size()
        daily_participation = (daily_campaign / daily_total).fillna(0)
        
        ax1.plot(daily_participation.index, daily_participation.values * 100, alpha=0.7, color='purple')
        ax1.axhline(daily_participation.mean() * 100, color='red', linestyle='--', 
                   label=f'Mean: {daily_participation.mean():.1%}')
        ax1.set_title('Campaign Participation Rate Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Participation Rate (%)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Campaign Type Performance
        ax2 = axes[0, 1]
        
        campaign_type_stats = campaign_bookings.groupby('campaign_type').agg({
            'booking_id': 'count',
            'discount_amount': 'mean',
            'attribution_score': 'mean',
            'incremental_flag': 'mean'
        })
        
        if len(campaign_type_stats) > 0:
            bars = ax2.bar(range(len(campaign_type_stats)), 
                          campaign_type_stats['booking_id'], alpha=0.7)
            ax2.set_title('Bookings by Campaign Type')
            ax2.set_xlabel('Campaign Type')
            ax2.set_ylabel('Number of Bookings')
            ax2.set_xticks(range(len(campaign_type_stats)))
            ax2.set_xticklabels(campaign_type_stats.index, rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add booking counts as labels
            for i, count in enumerate(campaign_type_stats['booking_id']):
                ax2.text(i, count + 10, str(count), ha='center', fontweight='bold')
        
        # 3. Attribution Score Distribution
        ax3 = axes[0, 2]
        
        valid_attribution = campaign_bookings['attribution_score'].dropna()
        if len(valid_attribution) > 0:
            ax3.hist(valid_attribution, bins=30, alpha=0.7, color='gold', edgecolor='black')
            ax3.axvline(valid_attribution.mean(), color='red', linestyle='--', 
                       label=f'Mean: {valid_attribution.mean():.3f}')
            ax3.set_title('Attribution Score Distribution')
            ax3.set_xlabel('Attribution Score')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Check attribution score validity
            invalid_scores = ((valid_attribution < 0) | (valid_attribution > 1)).sum()
            if invalid_scores == 0:
                ax3.text(0.02, 0.98, '✅ All scores in [0,1]', transform=ax3.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'), verticalalignment='top')
            else:
                ax3.text(0.02, 0.98, f'❌ {invalid_scores} invalid scores', transform=ax3.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'), verticalalignment='top')
        
        # 4. Incremental Rate by Campaign Type
        ax4 = axes[1, 0]
        
        if len(campaign_type_stats) > 0 and 'incremental_flag' in campaign_type_stats.columns:
            incremental_rates = campaign_type_stats['incremental_flag'] * 100
            bars = ax4.bar(range(len(incremental_rates)), incremental_rates.values, 
                          alpha=0.7, color='lightgreen')
            ax4.set_title('Incremental Rate by Campaign Type')
            ax4.set_xlabel('Campaign Type')
            ax4.set_ylabel('Incremental Rate (%)')
            ax4.set_xticks(range(len(incremental_rates)))
            ax4.set_xticklabels(incremental_rates.index, rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Add percentage labels
            for i, rate in enumerate(incremental_rates.values):
                ax4.text(i, rate + 1, f'{rate:.1f}%', ha='center', fontweight='bold')
        
        # 5. Discount vs Attribution Relationship
        ax5 = axes[1, 1]
        
        if len(campaign_bookings) > 100:  # Sample if too many points
            sample_campaign = campaign_bookings.sample(min(1000, len(campaign_bookings)))
        else:
            sample_campaign = campaign_bookings
        
        discount_pct = sample_campaign['discount_amount'] / sample_campaign['base_price'] * 100
        attribution = sample_campaign['attribution_score']
        
        valid_data = ~(discount_pct.isna() | attribution.isna())
        if valid_data.sum() > 0:
            ax5.scatter(discount_pct[valid_data], attribution[valid_data], alpha=0.6, s=15)
            ax5.set_title('Discount % vs Attribution Score')
            ax5.set_xlabel('Discount Percentage (%)')
            ax5.set_ylabel('Attribution Score')
            ax5.grid(True, alpha=0.3)
            
            # Calculate correlation
            correlation = discount_pct[valid_data].corr(attribution[valid_data])
            ax5.text(0.02, 0.98, f'Correlation: {correlation:.3f}', transform=ax5.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'), verticalalignment='top')
        
        # 6. Campaign Summary Statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        total_bookings = len(self.bookings_df)
        campaign_bookings_count = len(campaign_bookings)
        participation_rate = campaign_bookings_count / total_bookings
        
        summary_text = f"""Campaign Summary Statistics:

Total Bookings: {total_bookings:,}
Campaign Bookings: {campaign_bookings_count:,}
Participation Rate: {participation_rate:.1%}

Average Discount: ${campaign_bookings['discount_amount'].mean():.2f}
Average Attribution: {campaign_bookings['attribution_score'].mean():.3f}

Incremental Bookings: {campaign_bookings['incremental_flag'].sum():,}
Incremental Rate: {campaign_bookings['incremental_flag'].mean():.1%}
"""
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}03_campaign_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store validation results
        self.validation_results['campaigns'] = {
            'participation_rate': participation_rate,
            'avg_attribution_score': campaign_bookings['attribution_score'].mean(),
            'incremental_rate': campaign_bookings['incremental_flag'].mean(),
            'campaign_types_count': len(campaign_type_stats),
            'valid_attribution_scores': invalid_scores == 0 if len(valid_attribution) > 0 else True
        }
        
        print("   ✅ Campaign validation completed")
    
    def validate_statistical_properties(self):
        """Validation 4: Statistical properties and distributions"""
        print("📊 Validating statistical properties...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Statistical Properties Validation', fontsize=16, fontweight='bold')
        
        # 1. Price Distribution with Normality Test
        ax1 = axes[0, 0]
        
        prices = self.bookings_df['final_price']
        ax1.hist(prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        
        # Add normal curve for comparison
        mu, sigma = prices.mean(), prices.std()
        x = np.linspace(prices.min(), prices.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label='Normal curve')
        
        # Normality test
        stat, p_value = normaltest(prices)
        ax1.set_title(f'Price Distribution (normality p={p_value:.4f})')
        ax1.set_xlabel('Final Price ($)')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if p_value > 0.05:
            ax1.text(0.02, 0.98, '✅ Approximately normal', transform=ax1.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'), verticalalignment='top')
        else:
            ax1.text(0.02, 0.98, '⚠️ Non-normal distribution', transform=ax1.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'), verticalalignment='top')
        
        # 2. Lead Time Distribution
        ax2 = axes[0, 1]
        
        lead_times = self.bookings_df['lead_time']
        ax2.hist(lead_times, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_title('Lead Time Distribution')
        ax2.set_xlabel('Lead Time (days)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Check for outliers
        Q1 = lead_times.quantile(0.25)
        Q3 = lead_times.quantile(0.75)
        IQR = Q3 - Q1
        outliers = lead_times[(lead_times < Q1 - 1.5*IQR) | (lead_times > Q3 + 1.5*IQR)]
        outlier_rate = len(outliers) / len(lead_times)
        
        ax2.text(0.02, 0.98, f'Outliers: {outlier_rate:.1%}', transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor='lightgreen' if outlier_rate < 0.05 else 'lightyellow'),
                verticalalignment='top')
        
        # 3. Correlation Matrix
        ax3 = axes[0, 2]
        
        # Select numeric columns for correlation
        numeric_cols = ['final_price', 'base_price', 'discount_amount', 'lead_time', 'stay_length']
        existing_cols = [col for col in numeric_cols if col in self.bookings_df.columns]
        
        if len(existing_cols) >= 3:
            corr_matrix = self.bookings_df[existing_cols].corr()
            
            # Create heatmap
            im = ax3.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax3.set_xticks(range(len(existing_cols)))
            ax3.set_yticks(range(len(existing_cols)))
            ax3.set_xticklabels(existing_cols, rotation=45)
            ax3.set_yticklabels(existing_cols)
            ax3.set_title('Feature Correlation Matrix')
            
            # Add correlation values
            for i in range(len(existing_cols)):
                for j in range(len(existing_cols)):
                    text = ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)
            
            # Add colorbar
            plt.colorbar(im, ax=ax3, shrink=0.8)
        else:
            ax3.text(0.5, 0.5, 'Insufficient numeric columns', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Correlation Matrix - Insufficient Data')
        
        # 4. Box Plot by Room Type
        ax4 = axes[1, 0]
        
        room_types = self.bookings_df['room_type'].unique()
        price_data = [self.bookings_df[self.bookings_df['room_type'] == rt]['final_price'] for rt in room_types]
        
        bp = ax4.boxplot(price_data, labels=room_types, patch_artist=True)
        
        # Color boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        
        ax4.set_title('Price Distribution by Room Type')
        ax4.set_xlabel('Room Type')
        ax4.set_ylabel('Final Price ($)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 5. Price vs Lead Time Scatter
        ax5 = axes[1, 1]
        
        # Sample data for performance
        sample_data = self.bookings_df.sample(min(2000, len(self.bookings_df)))
        ax5.scatter(sample_data['lead_time'], sample_data['final_price'], alpha=0.5, s=10)
        ax5.set_title('Price vs Lead Time')
        ax5.set_xlabel('Lead Time (days)')
        ax5.set_ylabel('Final Price ($)')
        ax5.grid(True, alpha=0.3)
        
        # Calculate correlation
        price_leadtime_corr = self.bookings_df['final_price'].corr(self.bookings_df['lead_time'])
        ax5.text(0.02, 0.98, f'Correlation: {price_leadtime_corr:.3f}', transform=ax5.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'), verticalalignment='top')
        
        # 6. Summary Statistics Table
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate key statistics
        stats_summary = f"""Key Statistical Properties:

PRICE STATISTICS:
Mean: ${prices.mean():.2f}
Median: ${prices.median():.2f}
Std Dev: ${prices.std():.2f}
Skewness: {prices.skew():.3f}

LEAD TIME STATISTICS:
Mean: {lead_times.mean():.1f} days
Median: {lead_times.median():.1f} days
Std Dev: {lead_times.std():.1f} days

DATA QUALITY:
Outlier Rate: {outlier_rate:.1%}
Price-Lead Correlation: {price_leadtime_corr:.3f}
"""
        
        ax6.text(0.1, 0.9, stats_summary, transform=ax6.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}04_statistical_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store validation results
        self.validation_results['statistical'] = {
            'price_normality_p': p_value,
            'outlier_rate': outlier_rate,
            'price_leadtime_correlation': price_leadtime_corr,
            'price_skewness': prices.skew(),
            'price_mean': prices.mean(),
            'price_std': prices.std()
        }
        
        print("   ✅ Statistical validation completed")
    
    def create_validation_summary(self):
        """Create comprehensive validation summary"""
        print("📋 Creating validation summary...")
        
        # Calculate overall validation score
        validation_score = 100
        critical_issues = []
        warnings = []
        
        # Temporal validation
        if 'temporal' in self.validation_results:
            temporal = self.validation_results['temporal']
            if not temporal.get('valid_lead_times', True):
                validation_score -= 20
                critical_issues.append("Invalid lead times detected")
            if temporal.get('daily_volatility', 0) > 1.0:
                validation_score -= 10
                warnings.append(f"High daily booking volatility: {temporal['daily_volatility']:.2f}")
        
        # Business logic validation
        if 'business_logic' in self.validation_results:
            business = self.validation_results['business_logic']
            if not business.get('price_hierarchy_correct', True):
                validation_score -= 15
                critical_issues.append("Price hierarchy violation")
            
            consistency = business.get('consistency_checks', {})
            for check_name, pass_rate in consistency.items():
                if pass_rate < 0.95:
                    validation_score -= 10
                    critical_issues.append(f"Low consistency in {check_name}: {pass_rate:.1%}")
        
        # Campaign validation
        if 'campaigns' in self.validation_results:
            campaigns = self.validation_results['campaigns']
            if not campaigns.get('no_campaigns', False):
                if not campaigns.get('valid_attribution_scores', True):
                    validation_score -= 15
                    critical_issues.append("Invalid attribution scores")
                if campaigns.get('participation_rate', 0) < 0.1:
                    validation_score -= 5
                    warnings.append(f"Low campaign participation: {campaigns['participation_rate']:.1%}")
        
        # Statistical validation
        if 'statistical' in self.validation_results:
            statistical = self.validation_results['statistical']
            if statistical.get('outlier_rate', 0) > 0.1:
                validation_score -= 5
                warnings.append(f"High outlier rate: {statistical['outlier_rate']:.1%}")
        
        # Create summary HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive Data Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .header {{ text-align: center; background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; }}
                .score {{ font-size: 48px; font-weight: bold; text-align: center; margin: 20px; 
                         color: {'green' if validation_score >= 90 else 'orange' if validation_score >= 75 else 'red'}; }}
                .section {{ margin: 20px 0; padding: 20px; background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .critical {{ background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 10px; margin: 10px 0; }}
                .warning {{ background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 10px; margin: 10px 0; }}
                .success {{ background-color: #d4edda; border-left: 5px solid #28a745; padding: 10px; margin: 10px 0; }}
                .visualization {{ margin: 10px 0; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Comprehensive Data Validation Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Dataset: {len(self.bookings_df):,} bookings analyzed</p>
            </div>
            
            <div class="section">
                <h2>Overall Validation Score</h2>
                <div class="score">{validation_score}/100</div>
                <p style="text-align: center; font-size: 18px;">
                    Status: <strong>{'EXCELLENT' if validation_score >= 90 else 'GOOD' if validation_score >= 75 else 'FAIR' if validation_score >= 60 else 'POOR'}</strong>
                </p>
            </div>
            
            <div class="section">
                <h2>Issues Summary</h2>
                {"".join([f'<div class="critical">❌ <strong>Critical:</strong> {issue}</div>' for issue in critical_issues]) if critical_issues else '<div class="success">✅ No critical issues found!</div>'}
                {"".join([f'<div class="warning">⚠️ <strong>Warning:</strong> {warning}</div>' for warning in warnings]) if warnings else ''}
            </div>
            
            <div class="section">
                <h2>Validation Categories</h2>
                <table>
                    <tr><th>Category</th><th>Status</th><th>Key Findings</th></tr>
                    <tr>
                        <td>Temporal Patterns</td>
                        <td>{'✅ PASS' if self.validation_results.get('temporal', {}).get('valid_lead_times', True) else '❌ FAIL'}</td>
                        <td>Lead time range, daily volatility, seasonal patterns</td>
                    </tr>
                    <tr>
                        <td>Business Logic</td>
                        <td>{'✅ PASS' if self.validation_results.get('business_logic', {}).get('price_hierarchy_correct', True) else '❌ FAIL'}</td>
                        <td>Price hierarchies, customer behavior, data consistency</td>
                    </tr>
                    <tr>
                        <td>Campaign Effectiveness</td>
                        <td>{'✅ PASS' if self.validation_results.get('campaigns', {}).get('valid_attribution_scores', True) else '❌ FAIL'}</td>
                        <td>Attribution scores, participation rates, incremental impact</td>
                    </tr>
                    <tr>
                        <td>Statistical Properties</td>
                        <td>{'✅ PASS' if self.validation_results.get('statistical', {}).get('outlier_rate', 0) < 0.1 else '⚠️ WARNING'}</td>
                        <td>Distributions, correlations, outliers</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Detailed Visualizations</h2>
                <div class="visualization">
                    <h3>1. Temporal Pattern Validation</h3>
                    <img src="01_temporal_validation.png" alt="Temporal Validation">
                    <p>Validates booking patterns, lead times, seasonal trends, and temporal consistency.</p>
                </div>
                
                <div class="visualization">
                    <h3>2. Business Logic Validation</h3>
                    <img src="02_business_logic_validation.png" alt="Business Logic Validation">
                    <p>Validates price hierarchies, customer behavior patterns, and business rule compliance.</p>
                </div>
                
                <div class="visualization">
                    <h3>3. Campaign Effectiveness Validation</h3>
                    <img src="03_campaign_validation.png" alt="Campaign Validation">
                    <p>Validates campaign attribution, participation rates, and promotional effectiveness.</p>
                </div>
                
                <div class="visualization">
                    <h3>4. Statistical Properties Validation</h3>
                    <img src="04_statistical_validation.png" alt="Statistical Validation">
                    <p>Validates distributions, correlations, outliers, and statistical assumptions.</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {'<div class="success">✅ Your data appears to be of high quality and ready for ML applications!</div>' if validation_score >= 90 else ''}
                {'<div class="warning">⚠️ Consider addressing the warnings above before proceeding with ML applications.</div>' if 75 <= validation_score < 90 else ''}
                {'<div class="critical">❌ Critical issues found. Please address these before using the data for ML.</div>' if validation_score < 75 else ''}
                
                <h3>Next Steps:</h3>
                <ul>
                    <li>Review all visualizations for any anomalies</li>
                    <li>Address any critical issues identified</li>
                    <li>Consider feature engineering based on correlation analysis</li>
                    <li>Use time-based train/test splits for ML models</li>
                    <li>Monitor data quality in production using these same validation checks</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(f'{self.output_dir}validation_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"   ✅ Validation summary created")
        print(f"   📊 Overall Score: {validation_score}/100")
        return validation_score
    
    def run_comprehensive_validation(self):
        """Run all validation checks"""
        print("🔍 COMPREHENSIVE DATA VALIDATION SUITE")
        print("=" * 60)
        
        if not self.load_data():
            return False
        
        print(f"Output directory: {self.output_dir}")
        
        try:
            self.validate_temporal_patterns()
            self.validate_business_logic()
            self.validate_campaign_effectiveness()
            self.validate_statistical_properties()
            score = self.create_validation_summary()
            
            print(f"\n🎉 Comprehensive validation completed!")
            print(f"📁 Check '{self.output_dir}validation_report.html' for full report")
            print(f"📊 Final Validation Score: {score}/100")
            
            if score >= 90:
                print("✅ EXCELLENT - Your data is ready for ML applications!")
            elif score >= 75:
                print("✅ GOOD - Minor issues detected, data is suitable for ML")
            elif score >= 60:
                print("⚠️ FAIR - Some issues found, consider preprocessing")
            else:
                print("❌ POOR - Significant issues found, review and fix before ML use")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during validation: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function"""
    print("🔍 Comprehensive Data Validation Suite")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser(description='Run comprehensive data validation')
    parser.add_argument('--prefix', type=str, default='', help='Data file prefix')
    parser.add_argument('--output-dir', type=str, default='validation_suite/', help='Output directory')
    
    args = parser.parse_args()
    
    validator = ComprehensiveDataValidator(data_prefix=args.prefix, output_dir=args.output_dir)
    success = validator.run_comprehensive_validation()
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())