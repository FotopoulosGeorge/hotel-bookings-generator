"""
Utility functions for hotel booking data generator

Helper functions for data analysis, visualization, and configuration management.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
from typing import Dict, List, Tuple, Optional


def load_generated_data(prefix: str = '') -> Dict:
    """
    Load previously generated data files
    
    Args:
        prefix: File prefix (e.g., 'luxury_' for luxury scenario files)
    
    Returns:
        Dictionary containing loaded dataframes and objects
    """
    data = {}
    
    try:
        # Load CSV files
        data['bookings'] = pd.read_csv(f'output/{prefix}historical_bookings.csv')
        data['campaigns'] = pd.read_csv(f'output/{prefix}campaigns_run.csv')
        data['customers'] = pd.read_csv(f'output/{prefix}customer_segments.csv')
        data['attribution'] = pd.read_csv(f'output/{prefix}attribution_ground_truth.csv')
        
        # Convert date columns
        date_columns = {
            'bookings': ['booking_date', 'stay_start_date', 'stay_end_date', 'cancellation_date'],
            'campaigns': ['start_date', 'end_date']
        }
        
        for df_name, columns in date_columns.items():
            for col in columns:
                if col in data[df_name].columns:
                    data[df_name][col] = pd.to_datetime(data[df_name][col], errors='coerce')
        
        # Load pickle file
        with open(f'output/{prefix}baseline_demand_model.pkl', 'rb') as f:
            data['baseline_demand'] = pickle.load(f)
        
        print(f"✅ Successfully loaded data files with prefix '{prefix}'")
        return data
        
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        print(f"Make sure files with prefix '{prefix}' exist in output/ directory")
        return {}


def analyze_booking_patterns(bookings_df: pd.DataFrame) -> Dict:
    """
    Analyze booking patterns in the generated data
    
    Args:
        bookings_df: Bookings dataframe
    
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Basic statistics
    analysis['total_bookings'] = len(bookings_df)
    analysis['date_range'] = (bookings_df['booking_date'].min(), bookings_df['booking_date'].max())
    analysis['avg_booking_value'] = bookings_df['final_price'].mean()
    analysis['total_revenue'] = bookings_df['final_price'].sum()
    
    # Channel analysis
    analysis['channel_distribution'] = bookings_df['booking_channel'].value_counts(normalize=True)
    analysis['channel_avg_price'] = bookings_df.groupby('booking_channel')['final_price'].mean()
    
    # Segment analysis
    analysis['segment_distribution'] = bookings_df['customer_segment'].value_counts(normalize=True)
    analysis['segment_avg_price'] = bookings_df.groupby('customer_segment')['final_price'].mean()
    
    # Room type analysis
    analysis['room_type_distribution'] = bookings_df['room_type'].value_counts(normalize=True)
    analysis['room_type_avg_price'] = bookings_df.groupby('room_type')['final_price'].mean()
    
    # Campaign analysis
    campaign_bookings = bookings_df[bookings_df['campaign_id'].notna()]
    analysis['campaign_participation_rate'] = len(campaign_bookings) / len(bookings_df)
    analysis['campaign_avg_discount'] = campaign_bookings['discount_amount'].mean()
    
    # Cancellation analysis
    analysis['cancellation_rate'] = bookings_df['is_cancelled'].mean()
    analysis['cancellation_by_segment'] = bookings_df.groupby('customer_segment')['is_cancelled'].mean()
    
    # Overbooking analysis
    analysis['overbooking_rate'] = bookings_df['is_overbooked'].mean()
    
    # Seasonal patterns
    bookings_df['booking_month'] = bookings_df['booking_date'].dt.month
    bookings_df['stay_month'] = bookings_df['stay_start_date'].dt.month
    analysis['booking_seasonality'] = bookings_df['booking_month'].value_counts().sort_index()
    analysis['stay_seasonality'] = bookings_df['stay_month'].value_counts().sort_index()
    
    return analysis


def create_summary_report(data: Dict, output_file: Optional[str] = None) -> str:
    """
    Create a comprehensive summary report of the generated data
    
    Args:
        data: Dictionary from load_generated_data()
        output_file: Optional filename to save report
    
    Returns:
        Report string
    """
    bookings = data['bookings']
    campaigns = data['campaigns']
    customers = data['customers']
    analysis = analyze_booking_patterns(bookings)
    
    report = []
    report.append("🏨 HOTEL BOOKING DATA GENERATION REPORT")
    report.append("=" * 60)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Data period: {analysis['date_range'][0].strftime('%Y-%m-%d')} to {analysis['date_range'][1].strftime('%Y-%m-%d')}")
    report.append("")
    
    # Overview
    report.append("📊 OVERVIEW")
    report.append("-" * 30)
    report.append(f"Total bookings: {analysis['total_bookings']:,}")
    report.append(f"Total campaigns: {len(campaigns):,}")
    report.append(f"Total customers: {len(customers):,}")
    report.append(f"Total revenue: ${analysis['total_revenue']:,.2f}")
    report.append(f"Average booking value: ${analysis['avg_booking_value']:.2f}")
    report.append("")
    
    # Channel performance
    report.append("🏢 CHANNEL PERFORMANCE")
    report.append("-" * 30)
    for channel, share in analysis['channel_distribution'].items():
        avg_price = analysis['channel_avg_price'][channel]
        report.append(f"{channel}: {share:.1%} of bookings, ${avg_price:.2f} avg price")
    report.append("")
    
    # Customer segments
    report.append("👥 CUSTOMER SEGMENTS")
    report.append("-" * 30)
    for segment, share in analysis['segment_distribution'].items():
        avg_price = analysis['segment_avg_price'][segment]
        cancellation = analysis['cancellation_by_segment'][segment]
        report.append(f"{segment}: {share:.1%} of bookings, ${avg_price:.2f} avg price, {cancellation:.1%} cancellation")
    report.append("")
    
    # Campaign performance
    report.append("🎯 CAMPAIGN PERFORMANCE")
    report.append("-" * 30)
    report.append(f"Campaign participation rate: {analysis['campaign_participation_rate']:.1%}")
    report.append(f"Average campaign discount: ${analysis['campaign_avg_discount']:.2f}")
    
    # Campaign details
    campaign_summary = campaigns.groupby('campaign_type').agg({
        'campaign_id': 'count',
        'actual_bookings': 'sum',
        'incremental_bookings': 'sum',
        'discount_percentage': 'mean'
    }).round(3)
    
    for campaign_type, row in campaign_summary.iterrows():
        incremental_rate = row['incremental_bookings'] / row['actual_bookings'] if row['actual_bookings'] > 0 else 0
        report.append(f"{campaign_type}: {row['campaign_id']} campaigns, {row['actual_bookings']} bookings, {incremental_rate:.1%} incremental")
    report.append("")
    
    # Operations
    report.append("🔄 OPERATIONS")
    report.append("-" * 30)
    report.append(f"Cancellation rate: {analysis['cancellation_rate']:.1%}")
    report.append(f"Overbooking rate: {analysis['overbooking_rate']:.1%}")
    report.append("")
    
    # Seasonality
    report.append("📅 SEASONALITY")
    report.append("-" * 30)
    report.append("Stay month distribution:")
    for month, count in analysis['stay_seasonality'].items():
        month_name = datetime(2024, month, 1).strftime('%B')
        share = count / analysis['stay_seasonality'].sum()
        report.append(f"  {month_name}: {count:,} stays ({share:.1%})")
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"📄 Report saved to {output_file}")
    
    return report_text


def create_visualizations(data: Dict, output_dir: str = 'plots/') -> None:
    """
    Create visualization plots for the generated data
    
    Args:
        data: Dictionary from load_generated_data()
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    bookings = data['bookings']
    campaigns = data['campaigns']
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Booking volume over time
    plt.figure(figsize=(12, 6))
    daily_bookings = bookings.groupby('booking_date').size()
    daily_bookings.plot(kind='line', alpha=0.7)
    plt.title('Daily Booking Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Bookings')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}booking_volume_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Channel and segment distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Channel distribution
    bookings['booking_channel'].value_counts().plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%')
    axes[0,0].set_title('Booking Channel Distribution')
    axes[0,0].set_ylabel('')
    
    # Segment distribution
    bookings['customer_segment'].value_counts().plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')
    axes[0,1].set_title('Customer Segment Distribution')
    axes[0,1].set_ylabel('')
    
    # Room type distribution
    bookings['room_type'].value_counts().plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Room Type Distribution')
    axes[1,0].set_xlabel('Room Type')
    axes[1,0].set_ylabel('Number of Bookings')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Price distribution
    axes[1,1].hist(bookings['final_price'], bins=30, alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Final Price Distribution')
    axes[1,1].set_xlabel('Final Price ($)')
    axes[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}distribution_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Campaign performance
    campaign_performance = campaigns[campaigns['actual_bookings'] > 0].copy()
    campaign_performance['incremental_rate'] = (
        campaign_performance['incremental_bookings'] / campaign_performance['actual_bookings']
    )
    
    if len(campaign_performance) > 0:
        plt.figure(figsize=(12, 8))
        
        # Create subplot for campaign types
        campaign_type_perf = campaign_performance.groupby('campaign_type').agg({
            'actual_bookings': 'sum',
            'incremental_bookings': 'sum',
            'discount_percentage': 'mean'
        })
        campaign_type_perf['incremental_rate'] = (
            campaign_type_perf['incremental_bookings'] / campaign_type_perf['actual_bookings']
        )
        
        x = range(len(campaign_type_perf))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], campaign_type_perf['actual_bookings'], width, 
                label='Total Bookings', alpha=0.8)
        plt.bar([i + width/2 for i in x], campaign_type_perf['incremental_bookings'], width,
                label='Incremental Bookings', alpha=0.8)
        
        plt.xlabel('Campaign Type')
        plt.ylabel('Number of Bookings')
        plt.title('Campaign Performance by Type')
        plt.xticks(x, campaign_type_perf.index, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}campaign_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Seasonal patterns
    bookings['stay_month'] = bookings['stay_start_date'].dt.month
    seasonal_data = bookings.groupby(['stay_month', 'customer_segment']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(12, 6))
    seasonal_data.plot(kind='bar', stacked=True)
    plt.title('Seasonal Booking Patterns by Customer Segment')
    plt.xlabel('Stay Month')
    plt.ylabel('Number of Bookings')
    plt.legend(title='Customer Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}seasonal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to {output_dir}")


def compare_scenarios(scenario_prefixes: List[str], metrics: List[str] = None) -> pd.DataFrame:
    """
    Compare key metrics across different scenario runs
    
    Args:
        scenario_prefixes: List of file prefixes for different scenarios
        metrics: List of metrics to compare (if None, uses default set)
    
    Returns:
        DataFrame with comparison results
    """
    if metrics is None:
        metrics = [
            'total_bookings', 'total_revenue', 'avg_booking_value', 
            'campaign_participation_rate', 'cancellation_rate', 'overbooking_rate'
        ]
    
    comparison_data = []
    
    for prefix in scenario_prefixes:
        try:
            data = load_generated_data(prefix)
            if data:
                analysis = analyze_booking_patterns(data['bookings'])
                
                row = {'scenario': prefix.rstrip('_') if prefix.endswith('_') else prefix}
                for metric in metrics:
                    if metric in analysis:
                        row[metric] = analysis[metric]
                    else:
                        row[metric] = None
                
                comparison_data.append(row)
                
        except Exception as e:
            print(f"⚠️ Error processing scenario {prefix}: {e}")
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df.set_index('scenario', inplace=True)
        return df
    else:
        return pd.DataFrame()


def validate_data_quality(bookings_df: pd.DataFrame) -> Dict:
    """
    Perform data quality checks on generated booking data
    
    Args:
        bookings_df: Bookings dataframe
    
    Returns:
        Dictionary with quality check results
    """
    quality_checks = {}
    
    # Check for missing values
    quality_checks['missing_values'] = bookings_df.isnull().sum().to_dict()
    
    # Check for invalid dates
    invalid_dates = bookings_df[bookings_df['stay_start_date'] <= bookings_df['booking_date']]
    quality_checks['invalid_stay_dates'] = len(invalid_dates)
    
    # Check for negative prices
    negative_prices = bookings_df[bookings_df['final_price'] < 0]
    quality_checks['negative_prices'] = len(negative_prices)
    
    # Check for excessive discounts
    excessive_discounts = bookings_df[bookings_df['discount_amount'] > bookings_df['base_price']]
    quality_checks['excessive_discounts'] = len(excessive_discounts)
    
    # Check attribution scores
    invalid_attribution = bookings_df[
        (bookings_df['attribution_score'] < 0) | (bookings_df['attribution_score'] > 1)
    ]
    quality_checks['invalid_attribution_scores'] = len(invalid_attribution)
    
    # Check for cancelled bookings with future cancellation dates
    future_cancellations = bookings_df[
        (bookings_df['is_cancelled'] == True) & 
        (bookings_df['cancellation_date'] > bookings_df['stay_start_date'])
    ]
    quality_checks['future_cancellations'] = len(future_cancellations)
    
    # Overall quality score
    total_issues = sum([
        quality_checks['invalid_stay_dates'],
        quality_checks['negative_prices'], 
        quality_checks['excessive_discounts'],
        quality_checks['invalid_attribution_scores'],
        quality_checks['future_cancellations']
    ])
    
    quality_checks['overall_quality_score'] = 1 - (total_issues / len(bookings_df))
    quality_checks['total_issues'] = total_issues
    
    return quality_checks


if __name__ == "__main__":
    # Example usage
    print("🔧 Hotel Booking Data Generator Utilities")
    print("This module provides helper functions for analyzing generated data.")
    print("\nExample usage:")
    print("  from utils import load_generated_data, analyze_booking_patterns")
    print("  data = load_generated_data('luxury_')")
    print("  analysis = analyze_booking_patterns(data['bookings'])")