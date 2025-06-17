"""
Data Processing Module

Handles post-processing operations including cancellation logic,
data validation, and file I/O operations.
"""

import pandas as pd
import random
import pickle
from datetime import timedelta


class DataProcessor:
    """Handles data post-processing, validation, and I/O operations"""
    
    def __init__(self, config):
        self.config = config
    
    def apply_cancellation_logic(self, bookings):
        """Apply cancellation modeling to existing bookings"""
        print("📋 Applying cancellation logic...")
        
        cancelled_count = 0
        
        for booking in bookings:
            customer_segment = booking['customer_segment']
            cancellation_config = self.config.CANCELLATION_CONFIG[customer_segment]
            
            # Calculate cancellation probability based on lead time
            advance_days = (booking['stay_start_date'] - booking['booking_date']).days
            
            # Base cancellation rate adjusted by lead time
            lead_time_factor = min(2.0, advance_days / 30.0)
            adjusted_rate = (cancellation_config['base_cancellation_rate'] * 
                           (1 + (lead_time_factor - 1) * cancellation_config['lead_time_multiplier']))
            
            # Cap the cancellation rate at reasonable levels
            final_cancellation_rate = min(0.40, max(0.02, adjusted_rate))
            
            # Determine if booking gets cancelled
            is_cancelled = random.random() < final_cancellation_rate
            
            if is_cancelled:
                cancellation_date = self._generate_cancellation_date(booking, cancellation_config)
                if cancellation_date:
                    cancelled_count += 1
                else:
                    is_cancelled = False
                    cancellation_date = None
            else:
                cancellation_date = None
            
            # Update booking with cancellation status
            booking['is_cancelled'] = is_cancelled
            booking['cancellation_date'] = cancellation_date
            
            # Adjust attribution score for cancelled bookings
            if is_cancelled and booking['campaign_id']:
                booking['attribution_score'] *= 0.3
                booking['incremental_flag'] = False
        
        print(f"✅ Applied cancellations: {cancelled_count:,} bookings cancelled ({cancelled_count/len(bookings):.1%})")
        return bookings
    
    def _generate_cancellation_date(self, booking, cancellation_config):
        """Generate a valid cancellation date for a booking"""
        min_cancel_days, max_cancel_days = cancellation_config['cancellation_window_days']
        min_days_before_stay = cancellation_config['min_days_before_stay']
        
        latest_cancellation_date = booking['stay_start_date'] - timedelta(days=min_days_before_stay)
        earliest_cancellation_date = booking['booking_date'] + timedelta(days=min_cancel_days)
        
        # If the window is valid, generate cancellation date
        if earliest_cancellation_date <= latest_cancellation_date:
            cancellation_window_days = (latest_cancellation_date - earliest_cancellation_date).days
            if cancellation_window_days > 0:
                cancellation_date = earliest_cancellation_date + timedelta(
                    days=random.randint(0, cancellation_window_days)
                )
            else:
                cancellation_date = earliest_cancellation_date
            
            return cancellation_date
        
        return None
    
    def validate_data(self, bookings, campaigns):
        """Validate generated data against configuration targets"""
        df_bookings = pd.DataFrame(bookings)
        
        print("\n" + "="*80)
        print("📊 CONFIGURATION VALIDATION REPORT")
        print("="*80)
        
        # Channel distribution validation
        channel_dist = df_bookings['booking_channel'].value_counts(normalize=True)
        connected_actual = channel_dist.get('Connected_Agent', 0)
        online_actual = channel_dist.get('Online_Direct', 0)
        
        print(f"🏢 CHANNEL DISTRIBUTION:")
        print(f"   Connected Agent: {connected_actual:.1%} (target: {self.config.TARGET_CONNECTED_AGENT_SHARE:.0%})")
        print(f"   Online Direct: {online_actual:.1%} (target: {self.config.TARGET_ONLINE_DIRECT_SHARE:.0%})")
        
        # Promotional rates validation
        connected_bookings = df_bookings[df_bookings['booking_channel'] == 'Connected_Agent']
        online_bookings = df_bookings[df_bookings['booking_channel'] == 'Online_Direct']
        
        connected_promo_actual = (connected_bookings['discount_amount'] > 0).mean() if len(connected_bookings) > 0 else 0
        online_promo_actual = (online_bookings['discount_amount'] > 0).mean() if len(online_bookings) > 0 else 0
        
        print(f"🏷️  PROMOTIONAL RATES:")
        print(f"   Connected Agent: {connected_promo_actual:.1%} (target: {self.config.CONNECTED_AGENT_PROMO_RATE:.0%})")
        print(f"   Online Direct: {online_promo_actual:.1%} (target: {self.config.ONLINE_DIRECT_PROMO_RATE:.0%})")
        
        # Segment distribution validation
        segment_dist = df_bookings['customer_segment'].value_counts(normalize=True)
        print(f"👥 CUSTOMER SEGMENTS:")
        for segment, target_share in [(s, data['market_share']) for s, data in self.config.CUSTOMER_SEGMENTS.items()]:
            actual_share = segment_dist.get(segment, 0)
            print(f"   {segment}: {actual_share:.1%} (target: {target_share:.0%})")
        
        # Campaign performance
        campaign_bookings = df_bookings[df_bookings['campaign_id'].notna()]
        campaign_rate = len(campaign_bookings) / len(df_bookings)
        print(f"🎯 CAMPAIGN PERFORMANCE:")
        print(f"   Campaign participation rate: {campaign_rate:.1%}")
        
        total_promo_rate = (df_bookings['discount_amount'] > 0).mean()
        print(f"   Total promotional rate: {total_promo_rate:.1%}")
        
        # Date validation
        invalid_dates = df_bookings[df_bookings['stay_start_date'] <= df_bookings['booking_date']]
        
        # Operational season validation
        df_bookings['stay_month'] = df_bookings['stay_start_date'].dt.month
        non_operational_stays = df_bookings[~df_bookings['stay_month'].isin(self.config.OPERATIONAL_MONTHS)]
        operational_stays = df_bookings[df_bookings['stay_month'].isin(self.config.OPERATIONAL_MONTHS)]
        
        print(f"📅 DATE VALIDATION:")
        print(f"   Invalid stay dates: {len(invalid_dates)} ({len(invalid_dates)/len(df_bookings):.1%})")
        print(f"   Operational season stays: {len(operational_stays)} ({len(operational_stays)/len(df_bookings):.1%})")
        print(f"   Non-operational season stays: {len(non_operational_stays)} ({len(non_operational_stays)/len(df_bookings):.1%})")
        
        if len(non_operational_stays) > 0:
            print(f"   ⚠️  CRITICAL: {len(non_operational_stays)} stays found outside operational months {self.config.OPERATIONAL_MONTHS}")
            month_breakdown = non_operational_stays['stay_month'].value_counts().sort_index()
            print(f"   Month breakdown: {dict(month_breakdown)}")
            
            # Show sample of problematic bookings
            sample_problems = non_operational_stays[['booking_date', 'stay_start_date', 'stay_month', 'campaign_id', 'customer_segment']].head(5)
            print(f"   Sample problematic bookings:")
            for _, row in sample_problems.iterrows():
                print(f"     Booking: {row['booking_date'].strftime('%Y-%m-%d')} → Stay: {row['stay_start_date'].strftime('%Y-%m-%d')} (Month {row['stay_month']}) Campaign: {row['campaign_id']} Segment: {row['customer_segment']}")
        else:
            print(f"   ✅ All stays are within operational season!")
        
        print("="*80)
        
        return {
            'channel_distribution': channel_dist,
            'promotional_rates': {
                'connected_agent': connected_promo_actual,
                'online_direct': online_promo_actual
            },
            'segment_distribution': segment_dist,
            'campaign_participation_rate': campaign_rate,
            'invalid_dates_count': len(invalid_dates),
            'non_operational_stays_count': len(non_operational_stays)
        }
    
    def save_data(self, bookings, campaigns, customers, attribution_data, baseline_demand):
        """Save all generated data"""
        # Convert to DataFrames

        import os
        os.makedirs('output', exist_ok=True)  # Create output folder

        df_bookings = pd.DataFrame(bookings)
        df_bookings.to_csv('output/historical_bookings.csv', index=False)
        
        df_campaigns = pd.DataFrame(campaigns)
        df_campaigns['target_segments'] = df_campaigns['target_segments'].apply(lambda x: ';'.join(x))
        df_campaigns['room_types_eligible'] = df_campaigns['room_types_eligible'].apply(lambda x: ';'.join(x))
        df_campaigns.to_csv('output/campaigns_run.csv', index=False)
        
        df_customers = pd.DataFrame(customers)
        df_customers['booking_history'] = df_customers['booking_history'].apply(lambda x: ';'.join(x))
        # Handle campaign exposures
        df_customers['campaign_exposures'] = df_customers['campaign_exposures'].apply(
            lambda x: ';'.join([f"{exp['campaign_id']}:{exp['exposure_date'].strftime('%Y-%m-%d')}" for exp in x])
        )
        df_customers.to_csv('output/customer_segments.csv', index=False)
        
        df_attribution = pd.DataFrame(attribution_data)
        df_attribution.to_csv('output/attribution_ground_truth.csv', index=False)
        
        with open('output/baseline_demand_model.pkl', 'wb') as f:
            pickle.dump(baseline_demand, f)
        
        print(f"\n✅ Saved all data files:")
        print(f"   📄 output/historical_bookings.csv ({len(bookings):,} records)")
        print(f"   📄 output/campaigns_run.csv ({len(campaigns)} records)")
        print(f"   📄 output/customer_segments.csv ({len(customers):,} records)")
        print(f"   📄 output/attribution_ground_truth.csv ({len(attribution_data):,} records)")
        print(f"   📄 output/baseline_demand_model.pkl")
        
        return {
            'bookings_file': 'output/historical_bookings.csv',
            'campaigns_file': 'output/campaigns_run.csv',
            'customers_file': 'output/customer_segments.csv',
            'attribution_file': 'output/attribution_ground_truth.csv',
            'demand_model_file': 'output/baseline_demand_model.pkl'
        }
    
    def perform_data_quality_checks(self, bookings):
        """Perform comprehensive data quality checks"""
        df_bookings = pd.DataFrame(bookings)
        
        quality_report = {
            'total_records': len(df_bookings),
            'missing_values': {},
            'data_type_issues': [],
            'business_logic_violations': [],
            'statistical_anomalies': []
        }
        
        # Check for missing values
        missing_values = df_bookings.isnull().sum()
        quality_report['missing_values'] = missing_values[missing_values > 0].to_dict()
        
        # Check business logic violations
        # 1. Stay dates before booking dates
        invalid_stay_dates = df_bookings[df_bookings['stay_start_date'] <= df_bookings['booking_date']]
        if len(invalid_stay_dates) > 0:
            quality_report['business_logic_violations'].append({
                'issue': 'Invalid stay dates (stay_start_date <= booking_date)',
                'count': len(invalid_stay_dates)
            })
        
        # 2. Negative prices
        negative_prices = df_bookings[df_bookings['final_price'] < 0]
        if len(negative_prices) > 0:
            quality_report['business_logic_violations'].append({
                'issue': 'Negative final prices',
                'count': len(negative_prices)
            })
        
        # 3. Discounts greater than base price
        excessive_discounts = df_bookings[df_bookings['discount_amount'] > df_bookings['base_price']]
        if len(excessive_discounts) > 0:
            quality_report['business_logic_violations'].append({
                'issue': 'Discounts greater than base price',
                'count': len(excessive_discounts)
            })
        
        # 4. Attribution scores outside 0-1 range
        invalid_attribution = df_bookings[
            (df_bookings['attribution_score'] < 0) | (df_bookings['attribution_score'] > 1)
        ]
        if len(invalid_attribution) > 0:
            quality_report['business_logic_violations'].append({
                'issue': 'Attribution scores outside [0,1] range',
                'count': len(invalid_attribution)
            })
        
        # Statistical anomalies
        # 1. Price outliers (more than 3 standard deviations from mean)
        price_mean = df_bookings['final_price'].mean()
        price_std = df_bookings['final_price'].std()
        price_outliers = df_bookings[
            abs(df_bookings['final_price'] - price_mean) > 3 * price_std
        ]
        if len(price_outliers) > 0:
            quality_report['statistical_anomalies'].append({
                'issue': 'Price outliers (>3 std dev)',
                'count': len(price_outliers),
                'examples': price_outliers['final_price'].tolist()[:5]
            })
        
        # Calculate overall quality score
        total_issues = sum([
            len(quality_report['missing_values']),
            len(quality_report['business_logic_violations']),
            len(quality_report['statistical_anomalies'])
        ])
        
        quality_report['overall_quality_score'] = max(0, 1 - (total_issues / 20))  # Normalize to 0-1
        
        return quality_report