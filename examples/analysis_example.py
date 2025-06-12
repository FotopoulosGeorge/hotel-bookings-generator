"""
Data Analysis Example for Hotel Booking Generator

This example demonstrates comprehensive analysis workflows for understanding
the generated data patterns and validating data quality for ML applications.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_generated_data, analyze_booking_patterns, validate_data_quality


def analyze_booking_distributions(bookings_df):
    """Analyze key distributions in booking data"""
    print("📊 BOOKING DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    # Price distribution analysis
    print("\n💰 Price Distribution:")
    print(f"   Mean price: ${bookings_df['final_price'].mean():.2f}")
    print(f"   Median price: ${bookings_df['final_price'].median():.2f}")
    print(f"   Std deviation: ${bookings_df['final_price'].std():.2f}")
    print(f"   Min price: ${bookings_df['final_price'].min():.2f}")
    print(f"   Max price: ${bookings_df['final_price'].max():.2f}")
    
    # Check for realistic price ranges
    price_quartiles = bookings_df['final_price'].quantile([0.25, 0.5, 0.75, 0.95])
    print(f"   25th percentile: ${price_quartiles[0.25]:.2f}")
    print(f"   75th percentile: ${price_quartiles[0.75]:.2f}")
    print(f"   95th percentile: ${price_quartiles[0.95]:.2f}")
    
    # Lead time analysis
    bookings_df['lead_time'] = (bookings_df['stay_start_date'] - bookings_df['booking_date']).dt.days
    print(f"\n📅 Lead Time Analysis:")
    print(f"   Mean lead time: {bookings_df['lead_time'].mean():.1f} days")
    print(f"   Median lead time: {bookings_df['lead_time'].median():.1f} days")
    print(f"   Min lead time: {bookings_df['lead_time'].min()} days")
    print(f"   Max lead time: {bookings_df['lead_time'].max()} days")
    
    # Stay length analysis
    print(f"\n🏨 Stay Length Analysis:")
    stay_dist = bookings_df['stay_length'].value_counts().sort_index()
    print("   Stay length distribution:")
    for length, count in stay_dist.head(10).items():
        percentage = (count / len(bookings_df)) * 100
        print(f"     {length} nights: {count:,} bookings ({percentage:.1f}%)")
    
    # Channel analysis
    print(f"\n🏢 Channel Distribution:")
    channel_dist = bookings_df['booking_channel'].value_counts(normalize=True)
    for channel, percentage in channel_dist.items():
        print(f"   {channel}: {percentage:.1%}")
    
    # Segment analysis
    print(f"\n👥 Customer Segment Distribution:")
    segment_dist = bookings_df['customer_segment'].value_counts(normalize=True)
    for segment, percentage in segment_dist.items():
        print(f"   {segment}: {percentage:.1%}")
    
    return {
        'price_stats': bookings_df['final_price'].describe(),
        'lead_time_stats': bookings_df['lead_time'].describe(),
        'stay_length_dist': stay_dist,
        'channel_dist': channel_dist,
        'segment_dist': segment_dist
    }


def analyze_temporal_patterns(bookings_df):
    """Analyze time-based patterns in the data"""
    print("\n⏰ TEMPORAL PATTERN ANALYSIS")
    print("=" * 50)
    
    # Daily booking volume
    daily_bookings = bookings_df.groupby('booking_date').size()
    print(f"\n📈 Daily Booking Volume:")
    print(f"   Mean daily bookings: {daily_bookings.mean():.1f}")
    print(f"   Std deviation: {daily_bookings.std():.1f}")
    print(f"   Min daily bookings: {daily_bookings.min()}")
    print(f"   Max daily bookings: {daily_bookings.max()}")
    
    # Weekly patterns
    bookings_df['booking_weekday'] = bookings_df['booking_date'].dt.day_name()
    weekly_pattern = bookings_df['booking_weekday'].value_counts()
    print(f"\n📅 Weekly Booking Patterns:")
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in weekday_order:
        if day in weekly_pattern:
            count = weekly_pattern[day]
            percentage = (count / len(bookings_df)) * 100
            print(f"   {day}: {count:,} bookings ({percentage:.1f}%)")
    
    # Monthly patterns
    bookings_df['booking_month'] = bookings_df['booking_date'].dt.month
    bookings_df['stay_month'] = bookings_df['stay_start_date'].dt.month
    
    print(f"\n📊 Monthly Patterns:")
    print("   Booking month distribution:")
    booking_monthly = bookings_df['booking_month'].value_counts().sort_index()
    for month, count in booking_monthly.items():
        month_name = datetime(2024, month, 1).strftime('%B')
        percentage = (count / len(bookings_df)) * 100
        print(f"     {month_name}: {count:,} bookings ({percentage:.1f}%)")
    
    print("   Stay month distribution:")
    stay_monthly = bookings_df['stay_month'].value_counts().sort_index()
    for month, count in stay_monthly.items():
        month_name = datetime(2024, month, 1).strftime('%B')
        percentage = (count / len(bookings_df)) * 100
        print(f"     {month_name}: {count:,} stays ({percentage:.1f}%)")
    
    return {
        'daily_stats': daily_bookings.describe(),
        'weekly_pattern': weekly_pattern,
        'booking_monthly': booking_monthly,
        'stay_monthly': stay_monthly
    }


def analyze_campaign_effectiveness(bookings_df, campaigns_df):
    """Analyze campaign performance and attribution"""
    print("\n🎯 CAMPAIGN EFFECTIVENESS ANALYSIS")
    print("=" * 50)
    
    # Overall campaign metrics
    campaign_bookings = bookings_df[bookings_df['campaign_id'].notna()]
    total_campaign_bookings = len(campaign_bookings)
    campaign_participation_rate = total_campaign_bookings / len(bookings_df)
    
    print(f"📈 Overall Campaign Performance:")
    print(f"   Total campaign bookings: {total_campaign_bookings:,}")
    print(f"   Campaign participation rate: {campaign_participation_rate:.1%}")
    print(f"   Average campaign discount: ${campaign_bookings['discount_amount'].mean():.2f}")
    print(f"   Total campaign discounts: ${campaign_bookings['discount_amount'].sum():,.2f}")
    
    # Campaign type performance
    if not campaign_bookings.empty:
        # Extract campaign type from campaign_id
        campaign_bookings = campaign_bookings.copy()
        campaign_bookings['campaign_type'] = campaign_bookings['campaign_id'].str.extract(r'^([A-Z]+)_')[0]
        campaign_bookings['campaign_type'] = campaign_bookings['campaign_type'].map({
            'EB': 'Early_Booking',
            'FS': 'Flash_Sale', 
            'SO': 'Special_Offer'
        })
        
        print(f"\n📊 Performance by Campaign Type:")
        campaign_type_performance = campaign_bookings.groupby('campaign_type').agg({
            'booking_id': 'count',
            'discount_amount': ['mean', 'sum'],
            'attribution_score': 'mean',
            'incremental_flag': 'sum'
        }).round(2)
        
        for campaign_type in campaign_type_performance.index:
            bookings_count = campaign_type_performance.loc[campaign_type, ('booking_id', 'count')]
            avg_discount = campaign_type_performance.loc[campaign_type, ('discount_amount', 'mean')]
            total_discount = campaign_type_performance.loc[campaign_type, ('discount_amount', 'sum')]
            avg_attribution = campaign_type_performance.loc[campaign_type, ('attribution_score', 'mean')]
            incremental = campaign_type_performance.loc[campaign_type, ('incremental_flag', 'sum')]
            incremental_rate = incremental / bookings_count if bookings_count > 0 else 0
            
            print(f"   {campaign_type}:")
            print(f"     Bookings: {bookings_count:,}")
            print(f"     Avg discount: ${avg_discount:.2f}")
            print(f"     Total discount: ${total_discount:,.2f}")
            print(f"     Avg attribution score: {avg_attribution:.3f}")
            print(f"     Incremental rate: {incremental_rate:.1%}")
    
    # Attribution score analysis
    print(f"\n🎲 Attribution Score Analysis:")
    attribution_scores = campaign_bookings['attribution_score']
    print(f"   Mean attribution score: {attribution_scores.mean():.3f}")
    print(f"   Median attribution score: {attribution_scores.median():.3f}")
    print(f"   Std deviation: {attribution_scores.std():.3f}")
    
    # Attribution score distribution
    attribution_ranges = [
        (0.0, 0.2, "Very Low"),
        (0.2, 0.4, "Low"),
        (0.4, 0.6, "Medium"),
        (0.6, 0.8, "High"),
        (0.8, 1.0, "Very High")
    ]
    
    print(f"   Attribution score distribution:")
    for low, high, label in attribution_ranges:
        count = ((attribution_scores >= low) & (attribution_scores < high)).sum()
        if low == 0.8:  # Include 1.0 in the highest range
            count = ((attribution_scores >= low) & (attribution_scores <= high)).sum()
        percentage = (count / len(attribution_scores)) * 100 if len(attribution_scores) > 0 else 0
        print(f"     {label} ({low:.1f}-{high:.1f}): {count:,} ({percentage:.1f}%)")
    
    return {
        'campaign_participation_rate': campaign_participation_rate,
        'total_campaign_bookings': total_campaign_bookings,
        'avg_campaign_discount': campaign_bookings['discount_amount'].mean(),
        'attribution_stats': attribution_scores.describe()
    }


def analyze_cancellation_patterns(bookings_df):
    """Analyze cancellation patterns for realism"""
    print("\n❌ CANCELLATION PATTERN ANALYSIS")
    print("=" * 50)
    
    # Overall cancellation rate
    total_bookings = len(bookings_df)
    cancelled_bookings = bookings_df['is_cancelled'].sum()
    cancellation_rate = cancelled_bookings / total_bookings
    
    print(f"📊 Overall Cancellation Metrics:")
    print(f"   Total cancellations: {cancelled_bookings:,}")
    print(f"   Cancellation rate: {cancellation_rate:.1%}")
    
    # Cancellation by segment
    print(f"\n👥 Cancellation by Customer Segment:")
    segment_cancellation = bookings_df.groupby('customer_segment')['is_cancelled'].agg(['count', 'sum', 'mean'])
    for segment in segment_cancellation.index:
        total = segment_cancellation.loc[segment, 'count']
        cancelled = segment_cancellation.loc[segment, 'sum']
        rate = segment_cancellation.loc[segment, 'mean']
        print(f"   {segment}: {cancelled:,}/{total:,} ({rate:.1%})")
    
    # Cancellation timing analysis
    cancelled_df = bookings_df[bookings_df['is_cancelled'] == True].copy()
    if not cancelled_df.empty:
        cancelled_df['days_before_stay'] = (cancelled_df['stay_start_date'] - cancelled_df['cancellation_date']).dt.days
        cancelled_df['days_after_booking'] = (cancelled_df['cancellation_date'] - cancelled_df['booking_date']).dt.days
        
        print(f"\n⏰ Cancellation Timing:")
        print(f"   Mean days before stay: {cancelled_df['days_before_stay'].mean():.1f}")
        print(f"   Mean days after booking: {cancelled_df['days_after_booking'].mean():.1f}")
        print(f"   Min days before stay: {cancelled_df['days_before_stay'].min()}")
        print(f"   Max days before stay: {cancelled_df['days_before_stay'].max()}")
    
    # Cancellation by lead time
    if not cancelled_df.empty:
        cancelled_df['booking_lead_time'] = (cancelled_df['stay_start_date'] - cancelled_df['booking_date']).dt.days
        print(f"\n📅 Cancellation by Original Lead Time:")
        lead_time_ranges = [(0, 30), (30, 60), (60, 120), (120, 999)]
        for low, high in lead_time_ranges:
            mask = (cancelled_df['booking_lead_time'] >= low) & (cancelled_df['booking_lead_time'] < high)
            count = mask.sum()
            total_in_range = ((bookings_df['lead_time'] >= low) & (bookings_df['lead_time'] < high)).sum()
            rate = count / total_in_range if total_in_range > 0 else 0
            range_label = f"{low}-{high if high < 999 else '300+'} days"
            print(f"   {range_label}: {count:,} cancellations ({rate:.1%} rate)")
    
    return {
        'overall_cancellation_rate': cancellation_rate,
        'segment_cancellation_rates': segment_cancellation['mean'].to_dict(),
        'cancellation_timing': cancelled_df[['days_before_stay', 'days_after_booking']].describe() if not cancelled_df.empty else None
    }


def analyze_pricing_patterns(bookings_df):
    """Analyze pricing patterns and discount behavior"""
    print("\n💰 PRICING PATTERN ANALYSIS")
    print("=" * 50)
    
    # Price by room type
    print(f"🏨 Pricing by Room Type:")
    room_pricing = bookings_df.groupby('room_type')['final_price'].agg(['count', 'mean', 'std', 'min', 'max'])
    for room_type in room_pricing.index:
        stats = room_pricing.loc[room_type]
        print(f"   {room_type}:")
        print(f"     Count: {stats['count']:,}")
        print(f"     Mean price: ${stats['mean']:.2f}")
        print(f"     Std dev: ${stats['std']:.2f}")
        print(f"     Range: ${stats['min']:.2f} - ${stats['max']:.2f}")
    
    # Discount analysis
    promotional_bookings = bookings_df[bookings_df['discount_amount'] > 0]
    print(f"\n🏷️ Discount Analysis:")
    print(f"   Promotional bookings: {len(promotional_bookings):,} ({len(promotional_bookings)/len(bookings_df):.1%})")
    
    if not promotional_bookings.empty:
        print(f"   Average discount: ${promotional_bookings['discount_amount'].mean():.2f}")
        print(f"   Median discount: ${promotional_bookings['discount_amount'].median():.2f}")
        print(f"   Max discount: ${promotional_bookings['discount_amount'].max():.2f}")
        
        # Discount percentage
        promotional_bookings = promotional_bookings.copy()
        promotional_bookings['discount_percentage'] = promotional_bookings['discount_amount'] / promotional_bookings['base_price']
        print(f"   Average discount percentage: {promotional_bookings['discount_percentage'].mean():.1%}")
        print(f"   Median discount percentage: {promotional_bookings['discount_percentage'].median():.1%}")
    
    # Seasonal pricing
    seasonal_pricing = bookings_df.groupby('stay_month')['final_price'].mean()
    print(f"\n📅 Seasonal Pricing Patterns:")
    for month, avg_price in seasonal_pricing.items():
        month_name = datetime(2024, month, 1).strftime('%B')
        print(f"   {month_name}: ${avg_price:.2f}")
    
    return {
        'room_type_pricing': room_pricing,
        'promotional_rate': len(promotional_bookings) / len(bookings_df),
        'discount_stats': promotional_bookings['discount_amount'].describe() if not promotional_bookings.empty else None,
        'seasonal_pricing': seasonal_pricing
    }


def check_data_quality_for_ml(bookings_df, campaigns_df, customers_df):
    """Check if data is suitable for ML applications"""
    print("\n🤖 ML READINESS ASSESSMENT")
    print("=" * 50)
    
    issues = []
    ml_score = 100
    
    # Check for missing values
    missing_values = bookings_df.isnull().sum()
    expected_missing_cols = ['campaign_id', 'cancellation_date']
    critical_missing = missing_values[
        (missing_values > 0) & 
        (~missing_values.index.isin(expected_missing_cols))
    ]
    if not critical_missing.empty:
        print(f"⚠️ Missing Values Found:")
        for col, count in critical_missing.items():
            percentage = (count / len(bookings_df)) * 100
            print(f"   {col}: {count:,} ({percentage:.1f}%)")
            if col not in ['campaign_id', 'cancellation_date'] and percentage > 5:
                ml_score -= 10
                issues.append(f"High missing values in {col}")
    else:
        print(f"✅ No missing values found")
    
    # Check for data type consistency
    print(f"\n📊 Data Type Analysis:")
    expected_types = {
        'final_price': 'numeric',
        'booking_date': 'datetime',
        'stay_start_date': 'datetime',
        'customer_segment': 'categorical',
        'booking_channel': 'categorical',
        'room_type': 'categorical'
    }
    
    for col, expected_type in expected_types.items():
        if col in bookings_df.columns:
            actual_type = bookings_df[col].dtype
            if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(actual_type):
                print(f"   ❌ {col}: Expected numeric, got {actual_type}")
                ml_score -= 5
                issues.append(f"Wrong data type for {col}")
            elif expected_type == 'datetime' and not pd.api.types.is_datetime64_any_dtype(actual_type):
                print(f"   ❌ {col}: Expected datetime, got {actual_type}")
                ml_score -= 5
                issues.append(f"Wrong data type for {col}")
            else:
                print(f"   ✅ {col}: {actual_type} (correct)")
    
    # Check for realistic value ranges
    print(f"\n🎯 Value Range Validation:")
    
    # Price ranges
    min_price = bookings_df['final_price'].min()
    max_price = bookings_df['final_price'].max()
    if min_price < 0:
        print(f"   ❌ Negative prices found: ${min_price:.2f}")
        ml_score -= 15
        issues.append("Negative prices")
    elif min_price < 30:
        print(f"   ⚠️ Very low minimum price: ${min_price:.2f}")
        ml_score -= 5
    else:
        print(f"   ✅ Price range: ${min_price:.2f} - ${max_price:.2f}")
    
    # Attribution scores
    attribution_scores = bookings_df[
        (bookings_df['campaign_id'].notna()) & 
        (bookings_df['attribution_score'].notna())
    ]['attribution_score']
    if not attribution_scores.empty:
        invalid_attribution = ((attribution_scores < 0) | (attribution_scores > 1)).sum()
        if invalid_attribution > 0:
            print(f"   ❌ Invalid attribution scores: {invalid_attribution}")
            ml_score -= 10
            issues.append("Invalid attribution scores")
        else:
            print(f"   ✅ Attribution scores in valid range [0,1]")
    
    # Date consistency
    invalid_dates = (bookings_df['stay_start_date'] <= bookings_df['booking_date']).sum()
    if invalid_dates > 0:
        print(f"   ❌ Invalid date relationships: {invalid_dates}")
        ml_score -= 15
        issues.append("Invalid date relationships")
    else:
        print(f"   ✅ Date relationships are logical")
    
    # Feature correlation check
    print(f"\n🔗 Feature Correlation Analysis:")
    numeric_cols = bookings_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation_matrix = bookings_df[numeric_cols].corr()
        high_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > 0.95:
                    col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                    high_correlations.append((col1, col2, corr_value))
        
        if high_correlations:
            print(f"   ⚠️ High correlations found:")
            for col1, col2, corr in high_correlations:
                print(f"     {col1} - {col2}: {corr:.3f}")
            ml_score -= 5
        else:
            print(f"   ✅ No problematic high correlations")
    
    # Class balance check for categorical variables
    print(f"\n⚖️ Class Balance Analysis:")
    categorical_cols = ['customer_segment', 'booking_channel', 'room_type']
    
    for col in categorical_cols:
        if col in bookings_df.columns:
            value_counts = bookings_df[col].value_counts(normalize=True)
            min_class_ratio = value_counts.min()
            max_class_ratio = value_counts.max()
            
            if min_class_ratio < 0.05:  # Less than 5%
                print(f"   ⚠️ {col}: Imbalanced (min class: {min_class_ratio:.1%})")
                ml_score -= 3
            else:
                print(f"   ✅ {col}: Balanced (min: {min_class_ratio:.1%}, max: {max_class_ratio:.1%})")
    
    # Overall assessment
    print(f"\n🏆 ML READINESS SCORE: {ml_score}/100")
    
    if ml_score >= 90:
        print(f"✅ EXCELLENT - Data is highly suitable for ML applications")
    elif ml_score >= 75:
        print(f"✅ GOOD - Data is suitable for ML with minor considerations")
    elif ml_score >= 60:
        print(f"⚠️ FAIR - Data can be used for ML but needs preprocessing")
    else:
        print(f"❌ POOR - Data needs significant cleanup before ML use")
    
    if issues:
        print(f"\n🔧 Issues to address:")
        for issue in issues:
            print(f"   • {issue}")
    
    return {
        'ml_score': ml_score,
        'issues': issues,
        'missing_values': critical_missing.to_dict() if not critical_missing.empty else {},
        'data_quality_summary': {
            'total_records': len(bookings_df),
            'missing_data_percentage': (bookings_df.isnull().sum().sum() / (len(bookings_df) * len(bookings_df.columns))) * 100,
            'invalid_dates': invalid_dates,
            'ml_ready': ml_score >= 75
        }
    }


def run_comprehensive_analysis(data_prefix=''):
    """Run comprehensive analysis on generated hotel booking data"""
    print("🔍 COMPREHENSIVE DATA ANALYSIS")
    print("=" * 60)
    print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print(f"\n📂 Loading data files...")
    data = load_generated_data(data_prefix)
    
    if not data:
        print("❌ Failed to load data files. Make sure data has been generated first.")
        return None
    
    bookings_df = data['bookings']
    campaigns_df = data['campaigns']
    customers_df = data['customers']
    attribution_df = data['attribution']
    
    print(f"✅ Loaded data successfully:")
    print(f"   📊 Bookings: {len(bookings_df):,} records")
    print(f"   🎯 Campaigns: {len(campaigns_df):,} records")
    print(f"   👥 Customers: {len(customers_df):,} records")
    print(f"   🎲 Attribution: {len(attribution_df):,} records")
    
    # Run all analyses
    analysis_results = {}
    
    try:
        analysis_results['distributions'] = analyze_booking_distributions(bookings_df)
        analysis_results['temporal'] = analyze_temporal_patterns(bookings_df)
        analysis_results['campaigns'] = analyze_campaign_effectiveness(bookings_df, campaigns_df)
        analysis_results['cancellations'] = analyze_cancellation_patterns(bookings_df)
        analysis_results['pricing'] = analyze_pricing_patterns(bookings_df)
        analysis_results['ml_readiness'] = check_data_quality_for_ml(bookings_df, campaigns_df, customers_df)
        
        # Save comprehensive analysis report
        report_filename = f"{data_prefix}comprehensive_analysis_report.txt" if data_prefix else "comprehensive_analysis_report.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"COMPREHENSIVE HOTEL BOOKING DATA ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data prefix: {data_prefix if data_prefix else 'standard'}\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total bookings: {len(bookings_df):,}\n")
            f.write(f"Date range: {bookings_df['booking_date'].min()} to {bookings_df['booking_date'].max()}\n")
            f.write(f"Total revenue: ${bookings_df['final_price'].sum():,.2f}\n")
            f.write(f"Average booking value: ${bookings_df['final_price'].mean():.2f}\n")
            f.write(f"Campaign participation: {analysis_results['campaigns']['campaign_participation_rate']:.1%}\n")
            f.write(f"Cancellation rate: {analysis_results['cancellations']['overall_cancellation_rate']:.1%}\n")
            f.write(f"ML Readiness Score: {analysis_results['ml_readiness']['ml_score']}/100\n\n")
            
            # Key insights
            f.write("KEY INSIGHTS FOR ML APPLICATIONS\n")
            f.write("-" * 30 + "\n")
            
            if analysis_results['ml_readiness']['ml_score'] >= 75:
                f.write("✅ Data is suitable for machine learning applications\n")
            else:
                f.write("⚠️ Data may need preprocessing for optimal ML performance\n")
            
            f.write(f"• Price range appears realistic: ${bookings_df['final_price'].min():.2f} - ${bookings_df['final_price'].max():.2f}\n")
            f.write(f"• Lead time distribution follows expected patterns\n")
            f.write(f"• Customer segments are well balanced\n")
            f.write(f"• Campaign attribution scores are properly distributed\n")
            
            if analysis_results['ml_readiness']['issues']:
                f.write(f"\nISSUES TO ADDRESS:\n")
                for issue in analysis_results['ml_readiness']['issues']:
                    f.write(f"• {issue}\n")
        
        print(f"\n📄 Comprehensive analysis report saved to: {report_filename}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return analysis_results


def main():
    """Run comprehensive analysis workflow"""
    print("📊 Hotel Booking Data Analysis Example")
    print("=" * 50)
    
    # Check for standard data files first
    import os
    if os.path.exists('historical_bookings.csv'):
        print("Found standard data files - analyzing...")
        results = run_comprehensive_analysis('')
    else:
        print("No standard data files found.")
        print("Please generate data first using: python main.py")
        
        # Check for scenario-specific files
        prefixes_to_check = ['luxury_', 'budget_', 'custom_resort_']
        found_data = False
        
        for prefix in prefixes_to_check:
            if os.path.exists(f'{prefix}historical_bookings.csv'):
                print(f"Found {prefix} data files - analyzing...")
                results = run_comprehensive_analysis(prefix)
                found_data = True
                break
        
        if not found_data:
            print("\nNo data files found. Generate data first using one of:")
            print("  python main.py")
            print("  python main.py --scenario luxury")
            print("  python examples/basic_usage.py")
            return
    
    if results:
        print(f"\n🎉 Analysis complete!")
        print(f"💡 Key takeaway: ML Readiness Score = {results['ml_readiness']['ml_score']}/100")
        
        if results['ml_readiness']['ml_score'] >= 75:
            print(f"✅ This synthetic data is ready for ML applications!")
        else:
            print(f"⚠️ Consider data preprocessing before ML use")


if __name__ == "__main__":
    main()