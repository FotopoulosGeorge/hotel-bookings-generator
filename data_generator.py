"""
Hotel Booking Data Generator

Main data generation logic for creating realistic hotel booking datasets
with campaigns, customer behavior, cancellations, and overbooking.
"""

import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta
import pickle
import random
import warnings
from config import HotelBusinessConfig

warnings.filterwarnings('ignore')


class ConfigurableHotelBookingGenerator:
    def __init__(self, config=None):
        """Initialize with configuration object"""
        self.config = config or HotelBusinessConfig()
        
        # Initialize counters
        self.campaign_counter = 1000
        self.booking_counter = 10000  
        self.customer_counter = 1000
        
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        print(f"🏨 Initialized Hotel Booking Generator")
        print(f"   📅 Simulation period: {min(self.config.SIMULATION_YEARS)}-{max(self.config.SIMULATION_YEARS)}")
        print(f"   🏖️  Operational months: {self.config.OPERATIONAL_MONTHS}")
        print(f"   👥 Target customers: {self.config.DATA_CONFIG['total_customers']:,}")
        print(f"   📊 Target channel split: {self.config.TARGET_CONNECTED_AGENT_SHARE:.0%}/{self.config.TARGET_ONLINE_DIRECT_SHARE:.0%}")
    
    def generate_campaigns(self):
        """Generate promotional campaigns based on configuration"""
        campaigns = []
        
        for year in self.config.SIMULATION_YEARS:
            # Early booking campaigns
            for month in self.config.CAMPAIGN_TYPES['Early_Booking']['campaign_months']:
                campaign_year = year - 1 if month > 9 else year
                if campaign_year < 2021:
                    continue
                    
                for _ in range(self.config.CAMPAIGN_TYPES['Early_Booking']['campaigns_per_month']):
                    start_date = datetime(campaign_year, month, random.randint(1, 15))
                    duration = random.randint(*self.config.CAMPAIGN_TYPES['Early_Booking']['duration_range'])
                    end_date = start_date + timedelta(days=duration)
                    
                    campaign = {
                        'campaign_id': f'EB_{self.campaign_counter}',
                        'campaign_type': 'Early_Booking',
                        'start_date': start_date,
                        'end_date': end_date,
                        'discount_percentage': random.uniform(*self.config.CAMPAIGN_TYPES['Early_Booking']['discount_range']),
                        'target_segments': self.config.CAMPAIGN_TYPES['Early_Booking']['target_segments'],
                        'channel': self.config.CAMPAIGN_TYPES['Early_Booking']['preferred_channel'],
                        'room_types_eligible': self.config.ROOM_TYPES,
                        'advance_booking_requirements': self.config.CAMPAIGN_TYPES['Early_Booking']['advance_booking_requirement'],
                        'capacity_limit': random.randint(*self.config.CAMPAIGN_TYPES['Early_Booking']['capacity_range']),
                        'actual_bookings': 0,
                        'incremental_bookings': 0
                    }
                    campaigns.append(campaign)
                    self.campaign_counter += 1
            
            # Flash sales (during operational season)
            for month in self.config.OPERATIONAL_MONTHS:
                num_flash_sales = random.randint(*self.config.CAMPAIGN_TYPES['Flash_Sale']['campaigns_per_month'])
                for _ in range(num_flash_sales):
                    start_date = datetime(year, month, random.randint(1, 25))
                    duration = random.randint(*self.config.CAMPAIGN_TYPES['Flash_Sale']['duration_range'])
                    end_date = start_date + timedelta(days=duration)
                    
                    campaign = {
                        'campaign_id': f'FS_{self.campaign_counter}',
                        'campaign_type': 'Flash_Sale',
                        'start_date': start_date,
                        'end_date': end_date,
                        'discount_percentage': random.uniform(*self.config.CAMPAIGN_TYPES['Flash_Sale']['discount_range']),
                        'target_segments': self.config.CAMPAIGN_TYPES['Flash_Sale']['target_segments'],
                        'channel': random.choice(['Online_Direct', 'Connected_Agent']),
                        'room_types_eligible': random.sample(self.config.ROOM_TYPES, random.randint(2, 4)),
                        'advance_booking_requirements': self.config.CAMPAIGN_TYPES['Flash_Sale']['advance_booking_requirement'],
                        'capacity_limit': random.randint(*self.config.CAMPAIGN_TYPES['Flash_Sale']['capacity_range']),
                        'actual_bookings': 0,
                        'incremental_bookings': 0
                    }
                    campaigns.append(campaign)
                    self.campaign_counter += 1
            
            # Special offers (shoulder season)
            for month in self.config.CAMPAIGN_TYPES['Special_Offer']['target_months']:
                for _ in range(self.config.CAMPAIGN_TYPES['Special_Offer']['campaigns_per_month']):
                    start_date = datetime(year, month, random.randint(5, 20))
                    duration = random.randint(*self.config.CAMPAIGN_TYPES['Special_Offer']['duration_range'])
                    end_date = start_date + timedelta(days=duration)
                    
                    campaign = {
                        'campaign_id': f'SO_{self.campaign_counter}',
                        'campaign_type': 'Special_Offer',
                        'start_date': start_date,
                        'end_date': end_date,
                        'discount_percentage': random.uniform(*self.config.CAMPAIGN_TYPES['Special_Offer']['discount_range']),
                        'target_segments': self.config.CAMPAIGN_TYPES['Special_Offer']['target_segments'],
                        'channel': 'Mixed',
                        'room_types_eligible': self.config.ROOM_TYPES,
                        'advance_booking_requirements': self.config.CAMPAIGN_TYPES['Special_Offer']['advance_booking_requirement'],
                        'capacity_limit': random.randint(*self.config.CAMPAIGN_TYPES['Special_Offer']['capacity_range']),
                        'actual_bookings': 0,
                        'incremental_bookings': 0
                    }
                    campaigns.append(campaign)
                    self.campaign_counter += 1
        
        print(f"✅ Generated {len(campaigns)} campaigns")
        return campaigns
    
    def generate_baseline_demand(self):
        """Generate baseline demand using configured parameters"""
        baseline_demand = {}
        
        for year in self.config.SIMULATION_YEARS:
            for month in range(1, 13):
                try:
                    days_in_month = (datetime(year, month + 1, 1) - datetime(year, month, 1)).days if month < 12 else 31
                except ValueError:
                    days_in_month = 30
                    
                for day in range(1, days_in_month + 1):
                    try:
                        date = datetime(year, month, day)
                        weekday = date.weekday()
                        
                        # Apply configured seasonal and weekly patterns
                        seasonal_factor = self.config.SEASONAL_DEMAND_MULTIPLIERS[month]
                        weekly_factor = self.config.WEEKLY_DEMAND_MULTIPLIERS[weekday]
                        base_demand = self.config.DATA_CONFIG['base_daily_demand'] * seasonal_factor * weekly_factor
                        
                        # Add configured noise
                        noise = np.random.normal(0, self.config.DATA_CONFIG['demand_noise_std'] * base_demand)
                        final_demand = max(self.config.DATA_CONFIG['min_daily_bookings'], base_demand + noise)
                        
                        baseline_demand[date] = final_demand
                        
                    except ValueError:
                        continue
        
        return baseline_demand
    
    def generate_customers(self):
        """Generate customer profiles using configuration"""
        customers = []
        total_customers = self.config.DATA_CONFIG['total_customers']
        
        for i in range(total_customers):
            # Select segment based on configured market shares
            segments = list(self.config.CUSTOMER_SEGMENTS.keys())
            market_shares = [seg_data['market_share'] for seg_data in self.config.CUSTOMER_SEGMENTS.values()]
            
            # Normalize to ensure probabilities sum to exactly 1.0
            market_shares = np.array(market_shares)
            market_shares = market_shares / market_shares.sum()
            
            segment_choice = np.random.choice(segments, p=market_shares)
            
            segment_config = self.config.CUSTOMER_SEGMENTS[segment_choice]
            
            customer = {
                'customer_id': f'CUST_{self.customer_counter}',
                'segment': segment_choice,
                'price_sensitivity': segment_config['price_sensitivity'] + np.random.normal(0, 0.1),
                'planning_horizon': random.randint(*segment_config['advance_booking_days']),
                'channel_preference': np.random.choice(
                    list(segment_config['channel_preference_weights'].keys()),
                    p=np.array(list(segment_config['channel_preference_weights'].values())) / 
                      sum(segment_config['channel_preference_weights'].values())
                ),
                'loyalty_status': np.random.choice(
                    list(segment_config['loyalty_distribution'].keys()),
                    p=np.array(list(segment_config['loyalty_distribution'].values())) /
                      sum(segment_config['loyalty_distribution'].values())
                ),
                'campaign_fatigue': 0,
                'booking_history': []
            }
            customers.append(customer)
            self.customer_counter += 1
        
        return customers
    
    def get_base_price_for_date(self, date, room_type):
        """Get base price for a specific date and room type using periodic pricing"""
        if not self.config.PERIODIC_BASE_PRICING['enabled']:
            return self.config.BASE_PRICES[room_type]
        
        # Convert date to datetime if it's a string
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        elif hasattr(date, 'date'):
            date = date.date()
        
        # Get pricing periods for this room type
        pricing_periods = self.config.PERIODIC_BASE_PRICING['pricing_periods'].get(room_type, [])
        
        # Find the applicable pricing period
        for period in pricing_periods:
            start_date = datetime.strptime(period['start_date'], '%Y-%m-%d').date()
            end_date = datetime.strptime(period['end_date'], '%Y-%m-%d').date()
            
            if start_date <= date <= end_date:
                return period['base_price']
        
        # Fallback to static pricing if no period found
        print(f"⚠️ No pricing period found for {room_type} on {date}, using static pricing")
        return self.config.BASE_PRICES[room_type]
    
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
                # Generate cancellation date
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
    
    def generate_overbooked_bookings(self, original_bookings):
        """Generate additional overbooked bookings based on configuration"""
        if not self.config.OVERBOOKING_CONFIG['enable_overbooking']:
            print("📋 Overbooking disabled - skipping")
            return []
        
        print("📋 Generating overbooked bookings...")
        
        overbooked_bookings = []
        
        # Group bookings by stay date to apply overbooking logic
        bookings_by_date = {}
        for booking in original_bookings:
            stay_date = booking['stay_start_date']
            if stay_date not in bookings_by_date:
                bookings_by_date[stay_date] = []
            bookings_by_date[stay_date].append(booking)
        
        # Apply overbooking for each stay date
        for stay_date, date_bookings in bookings_by_date.items():
            if len(date_bookings) == 0:
                continue
                
            month = stay_date.month
            if month not in [5, 6, 7, 8, 9]:
                continue
            
            # Calculate overbooking rate for this date
            base_rate = self.config.OVERBOOKING_CONFIG['base_overbooking_rate']
            seasonal_multiplier = self.config.OVERBOOKING_CONFIG['seasonal_overbooking_multipliers'].get(month, 1.0)
            
            # Calculate target number of overbooked rooms
            current_bookings = len([b for b in date_bookings if not b.get('is_cancelled', False)])
            target_overbooked = int(current_bookings * base_rate * seasonal_multiplier)
            
            if target_overbooked == 0:
                continue
            
            # Generate overbooked bookings by duplicating patterns from existing bookings
            for _ in range(target_overbooked):
                template_booking = random.choice(date_bookings)
                
                overbooked_booking = template_booking.copy()
                overbooked_booking['booking_id'] = f'BK_{self.booking_counter}'
                overbooked_booking['customer_id'] = f'CUST_{self.customer_counter}'
                overbooked_booking['is_overbooked'] = True
                overbooked_booking['is_cancelled'] = False
                overbooked_booking['cancellation_date'] = None
                
                # Adjust booking date to be closer to stay date
                days_before_stay = random.randint(1, 14)
                overbooked_booking['booking_date'] = stay_date - timedelta(days=days_before_stay)
                
                # Apply channel-specific adjustments
                channel = overbooked_booking['booking_channel']
                channel_multiplier = self.config.OVERBOOKING_CONFIG['channel_overbooking_rates'].get(channel, 1.0)
                
                # Apply campaign-specific adjustments if applicable
                if overbooked_booking['campaign_id']:
                    campaign_type = overbooked_booking['campaign_id'].split('_')[0]
                    campaign_type_map = {'EB': 'Early_Booking', 'FS': 'Flash_Sale', 'SO': 'Special_Offer'}
                    campaign_type_full = campaign_type_map.get(campaign_type, 'Special_Offer')
                    campaign_multiplier = self.config.OVERBOOKING_CONFIG['campaign_overbooking_adjustment'].get(campaign_type_full, 1.0)
                    
                    overbooked_booking['attribution_score'] *= campaign_multiplier
                
                overbooked_bookings.append(overbooked_booking)
                self.booking_counter += 1
                self.customer_counter += 1
        
        print(f"✅ Generated {len(overbooked_bookings):,} overbooked bookings")
        return overbooked_bookings
    
    def calculate_attribution_score(self, booking, campaigns, customer):
        """Calculate attribution score using configured parameters"""
        if booking['campaign_id'] is None:
            return 0.0, None
        
        campaign = next((c for c in campaigns if c['campaign_id'] == booking['campaign_id']), None)
        if not campaign:
            return 0.0, None
        
        attr_config = self.config.ATTRIBUTION_CONFIG
        base_score = attr_config['base_attribution_score']
        
        # Campaign type specific logic
        if campaign['campaign_type'] == 'Early_Booking':
            days_since_start = (booking['booking_date'] - campaign['start_date']).days
            for threshold, factor in attr_config['temporal_decay'].items():
                if days_since_start <= threshold:
                    base_score *= factor
                    break
        
        elif campaign['campaign_type'] == 'Flash_Sale':
            days_since_start = (booking['booking_date'] - campaign['start_date']).days
            urgency_config = self.config.CAMPAIGN_TYPES['Flash_Sale']['urgency_decay']
            urgency_factor = urgency_config.get(days_since_start, 0.3)
            base_score *= urgency_factor
        
        # Segment matching
        if customer['segment'] in campaign.get('target_segments', []):
            base_score *= attr_config['segment_match_boost']
        else:
            base_score *= attr_config['segment_mismatch_penalty']
        
        # Channel alignment
        if (campaign['channel'] == customer['channel_preference'] or campaign['channel'] == 'Mixed'):
            base_score *= attr_config['channel_alignment_boost']
        
        # Customer fatigue
        fatigue_penalty = customer['campaign_fatigue'] * attr_config['fatigue_penalty_per_exposure']
        fatigue_factor = max(attr_config['min_fatigue_factor'], 1 - fatigue_penalty)
        base_score *= fatigue_factor
        
        # Add realistic model uncertainty
        model_uncertainty = np.random.normal(0, attr_config['model_uncertainty_std'])
        noisy_score = base_score + model_uncertainty
        
        # Occasional significant errors
        if random.random() < attr_config['high_error_probability']:
            error_factor = random.uniform(*attr_config['high_error_range'])
            noisy_score *= error_factor
        
        attribution_score = max(0.0, min(1.0, noisy_score))
        
        # Determine incrementality
        cannibalization_threshold = random.uniform(*attr_config['cannibalization_threshold_range'])
        is_incremental = attribution_score > cannibalization_threshold
        
        return attribution_score, is_incremental
    
    def generate_stay_dates(self, booking_date, customer, selected_campaign):
        """Generate stay dates using configuration"""
        stay_config = self.config.DATA_CONFIG['stay_length_distribution']
        stay_weights = np.array(list(stay_config.values()))
        stay_weights_normalized = stay_weights / stay_weights.sum()
        
        stay_length = np.random.choice(
            list(stay_config.keys()),
            p=stay_weights_normalized
        )
        
        if selected_campaign and selected_campaign['campaign_type'] == 'Early_Booking':
            # Use configured seasonal weights for early booking stays
            eb_config = self.config.CAMPAIGN_TYPES['Early_Booking']
            
            if booking_date.month <= 4:
                season_start = datetime(booking_date.year, 5, 1)
            else:
                season_start = datetime(booking_date.year + 1, 5, 1)
            
            # Select month based on configured weights
            seasonal_weights = eb_config['seasonal_stay_weights']
            weights_array = np.array(list(seasonal_weights.values()))
            weights_normalized = weights_array / weights_array.sum()
            
            selected_month = np.random.choice(
                list(seasonal_weights.keys()),
                p=weights_normalized
            )
            
            month_start = datetime(season_start.year, selected_month, 1)
            try:
                if selected_month == 9:
                    month_end = datetime(season_start.year, selected_month, 30)
                else:
                    month_end = datetime(season_start.year, selected_month + 1, 1) - timedelta(days=1)
                
                max_start_date = month_end - timedelta(days=int(stay_length))
                days_available = max(0, (max_start_date - month_start).days)
                
                if days_available > 0:
                    stay_start_date = month_start + timedelta(days=random.randint(0, days_available))
                else:
                    stay_start_date = month_start
                    
            except ValueError:
                stay_start_date = month_start
        else:
            # Regular booking logic
            max_advance = min(customer['planning_horizon'], 300)
            base_stay_date = booking_date + timedelta(days=random.randint(1, max_advance))
            
            # Apply seasonal bias to reduce September clustering
            if base_stay_date.month == 9 and random.random() < 0.4:
                target_month = random.choice([7, 8])
                try:
                    stay_start_date = base_stay_date.replace(month=target_month)
                except ValueError:
                    stay_start_date = datetime(base_stay_date.year, target_month, 
                                             min(base_stay_date.day, 31))
            else:
                stay_start_date = base_stay_date

        # Ensure stay is within operational months
        if stay_start_date.month not in self.config.OPERATIONAL_MONTHS:
            # Adjust to nearest operational month
            if stay_start_date.month < 5:
                stay_start_date = stay_start_date.replace(month=random.randint(5,9), day=random.randint(1,30))
            elif stay_start_date.month > 9:
                stay_start_date = stay_start_date.replace(month=random.randint(5,9), day=random.randint(1,30))
        
        if stay_start_date <= booking_date:
        # Push stay date to at least 1 day after booking
            stay_start_date = booking_date + timedelta(days=random.randint(1, 30))
        
        # Re-validate operational months    
        if stay_start_date.month not in self.config.OPERATIONAL_MONTHS:
            # Find next operational month
            target_month = 5 if stay_start_date.month < 5 else 9
            stay_start_date = stay_start_date.replace(month=target_month, day=1)
        stay_end_date = stay_start_date + timedelta(days=int(stay_length))
        return stay_start_date, stay_end_date, stay_length
    
    def generate_bookings(self, baseline_demand, campaigns, customers):
        """Generate bookings using all configured parameters"""
        bookings = []
        attribution_data = []
        
        # Create campaign lookup with configured influence periods
        campaign_lookup = {}
        for campaign in campaigns:
            start_date = campaign['start_date']
            end_date = campaign['end_date']
            
            current_date = start_date
            while current_date <= end_date:
                if current_date not in campaign_lookup:
                    campaign_lookup[current_date] = []
                campaign_lookup[current_date].append(campaign)
                current_date += timedelta(days=1)
            
            # Extend influence for early booking campaigns
            if campaign['campaign_type'] == 'Early_Booking':
                influence_days = self.config.CAMPAIGN_TYPES['Early_Booking']['influence_period_days']
                extended_end = end_date + timedelta(days=influence_days)
                current_date = end_date + timedelta(days=1)
                while current_date <= extended_end:
                    if current_date not in campaign_lookup:
                        campaign_lookup[current_date] = []
                    campaign_lookup[current_date].append(campaign)
                    current_date += timedelta(days=1)
        
        total_bookings_generated = 0
        
        for date, base_demand in baseline_demand.items():
            # Apply configured external shocks
            if random.random() < self.config.DATA_CONFIG['external_shock_probability']:
                shock_factor = random.uniform(*self.config.DATA_CONFIG['shock_impact_range'])
                base_demand *= shock_factor
            
            num_bookings = max(1, int(np.random.poisson(base_demand)))
            
            for _ in range(num_bookings):
                customer = random.choice(customers)
                
                # Campaign eligibility check
                active_campaigns = campaign_lookup.get(date, [])
                eligible_campaigns = []
                
                for campaign in active_campaigns:
                    if campaign['campaign_type'] == 'Early_Booking':
                        if (customer['segment'] in campaign.get('target_segments', []) and
                            campaign['actual_bookings'] < campaign['capacity_limit']):
                            
                            current_year = date.year
                            if date.month <= 4:
                                target_season_start = datetime(current_year, 5, 1)
                            elif date.month >= 10:
                                target_season_start = datetime(current_year + 1, 5, 1)
                            else:
                                target_season_start = datetime(current_year + 1, 5, 1)
                            
                            advance_days = (target_season_start - date).days
                            if advance_days >= campaign.get('advance_booking_requirements', 90):
                                eligible_campaigns.append(campaign)
                    else:
                        if (customer['segment'] in campaign.get('target_segments', []) and
                            campaign['actual_bookings'] < campaign['capacity_limit']):
                            eligible_campaigns.append(campaign)
                
                # Campaign selection
                selected_campaign = None
                is_promotional = False
                
                if eligible_campaigns and random.random() < self.config.CAMPAIGN_PARTICIPATION_RATE:
                    selected_campaign = random.choice(eligible_campaigns)
                    selected_campaign['actual_bookings'] += 1
                    is_promotional = True
                
                # Non-campaign promotional logic
                if not selected_campaign:
                    if (customer['channel_preference'] == 'Connected_Agent' and 
                        random.random() < self.config.CONNECTED_AGENT_PROMO_RATE):
                        is_promotional = True
                    elif (customer['channel_preference'] in ['Online_Direct'] and 
                          random.random() < self.config.ONLINE_DIRECT_PROMO_RATE):
                        is_promotional = True
                
                # Channel determination using configured weights
                if selected_campaign and selected_campaign['channel'] != 'Mixed':
                    channel = selected_campaign['channel']
                else:
                    segment_config = self.config.CUSTOMER_SEGMENTS[customer['segment']]
                    channel_weights = np.array(list(segment_config['channel_preference_weights'].values()))
                    channel_weights_normalized = channel_weights / channel_weights.sum()
                    
                    channel = np.random.choice(
                        list(segment_config['channel_preference_weights'].keys()),
                        p=channel_weights_normalized
                    )
                
                # Room type selection
                room_weights = np.array(self.config.DATA_CONFIG['room_type_distribution'])
                room_weights_normalized = room_weights / room_weights.sum()
                
                room_type = np.random.choice(
                    self.config.ROOM_TYPES, 
                    p=room_weights_normalized
                )
                
                # Pricing with configured multipliers
                base_price = self.config.BASE_PRICES[room_type]
                seasonal_multiplier = self.config.DATA_CONFIG['seasonal_pricing_multipliers'].get(date.month, 1.0)
                base_price *= seasonal_multiplier
                
                # Generate stay dates
                stay_start_date, stay_end_date, stay_length = self.generate_stay_dates(
                    date, customer, selected_campaign
                )
                
                # Apply discounts
                discount_amount = 0
                final_price = base_price
                campaign_id_final = None
                
                if selected_campaign:
                    if room_type in selected_campaign['room_types_eligible']:
                        discount_amount = base_price * selected_campaign['discount_percentage']
                        final_price = base_price - discount_amount
                        campaign_id_final = selected_campaign['campaign_id']
                elif is_promotional:
                    discount_ranges = self.config.DATA_CONFIG['non_campaign_discount_ranges']
                    if channel in discount_ranges:
                        discount_rate = random.uniform(*discount_ranges[channel])
                        discount_amount = base_price * discount_rate
                        final_price = base_price - discount_amount
                
                # Calculate attribution
                attribution_score, is_incremental = self.calculate_attribution_score(
                    {'booking_date': date, 'campaign_id': campaign_id_final}, campaigns, customer
                )
                
                # Update customer fatigue
                if selected_campaign:
                    customer['campaign_fatigue'] += 1
                
                # Create booking record
                booking = {
                    'booking_id': f'BK_{self.booking_counter}',
                    'customer_id': customer['customer_id'],
                    'booking_date': date,
                    'stay_start_date': stay_start_date,
                    'stay_end_date': stay_end_date,
                    'stay_length': stay_length,
                    'room_type': room_type,
                    'customer_segment': customer['segment'],
                    'booking_channel': channel,
                    'base_price': round(base_price, 2),
                    'final_price': round(final_price, 2),
                    'discount_amount': round(discount_amount, 2),
                    'campaign_id': campaign_id_final,
                    'attribution_score': round(attribution_score, 3),
                    'incremental_flag': is_incremental if campaign_id_final else False,
                    'is_cancelled': False,
                    'cancellation_date': None,
                    'is_overbooked': False
                }
                
                bookings.append(booking)
                
                # Attribution ground truth
                counterfactual_price = base_price if campaign_id_final else final_price
                attribution_truth = {
                    'booking_id': booking['booking_id'],
                    'true_attribution_score': round(attribution_score, 3),
                    'causal_campaign_id': campaign_id_final,
                    'counterfactual_price': round(counterfactual_price, 2),
                    'would_have_booked_anyway': not is_incremental if campaign_id_final else True
                }
                attribution_data.append(attribution_truth)
                
                self.booking_counter += 1
                total_bookings_generated += 1
        
        # Update campaign performance
        df_bookings = pd.DataFrame(bookings)
        campaign_booking_counts = df_bookings[df_bookings['campaign_id'].notna()]['campaign_id'].value_counts()
        campaign_incremental_counts = df_bookings[
            (df_bookings['campaign_id'].notna()) & (df_bookings['incremental_flag'] == True)
        ]['campaign_id'].value_counts()
        
        for campaign in campaigns:
            campaign['actual_bookings'] = campaign_booking_counts.get(campaign['campaign_id'], 0)
            campaign['incremental_bookings'] = campaign_incremental_counts.get(campaign['campaign_id'], 0)
        
        print(f"✅ Generated {total_bookings_generated} total bookings")
        return bookings, attribution_data
    
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
        
        connected_promo_actual = (connected_bookings['discount_amount'] > 0).mean()
        online_promo_actual = (online_bookings['discount_amount'] > 0).mean()
        
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
        
        print("="*80)
    
    def save_data(self, bookings, campaigns, customers, attribution_data, baseline_demand):
        """Save all generated data"""
        df_bookings = pd.DataFrame(bookings)
        df_bookings.to_csv('historical_bookings.csv', index=False)
        
        df_campaigns = pd.DataFrame(campaigns)
        df_campaigns['target_segments'] = df_campaigns['target_segments'].apply(lambda x: ';'.join(x))
        df_campaigns['room_types_eligible'] = df_campaigns['room_types_eligible'].apply(lambda x: ';'.join(x))
        df_campaigns.to_csv('campaigns_run.csv', index=False)
        
        df_customers = pd.DataFrame(customers)
        df_customers['booking_history'] = df_customers['booking_history'].apply(lambda x: ';'.join(x))
        df_customers.to_csv('customer_segments.csv', index=False)
        
        df_attribution = pd.DataFrame(attribution_data)
        df_attribution.to_csv('attribution_ground_truth.csv', index=False)
        
        with open('baseline_demand_model.pkl', 'wb') as f:
            pickle.dump(baseline_demand, f)
        
        print(f"\n✅ Saved all data files:")
        print(f"   📄 historical_bookings.csv ({len(bookings):,} records)")
        print(f"   📄 campaigns_run.csv ({len(campaigns)} records)")
        print(f"   📄 customer_segments.csv ({len(customers):,} records)")
        print(f"   📄 attribution_ground_truth.csv ({len(attribution_data):,} records)")
        print(f"   📄 baseline_demand_model.pkl")
    
    def generate_all_data(self):
        """Main method to generate all data using configuration"""
        print(f"\n🚀 Starting configurable data generation...")
        
        campaigns = self.generate_campaigns()
        baseline_demand = self.generate_baseline_demand()
        customers = self.generate_customers()
        bookings, attribution_data = self.generate_bookings(baseline_demand, campaigns, customers)
        
        print(f"\n📋 Applying cancellation and overbooking logic...")
        
        # Apply cancellations to existing bookings
        bookings = self.apply_cancellation_logic(bookings)
        
        # Generate additional overbooked bookings
        overbooked_bookings = self.generate_overbooked_bookings(bookings)
        
        # Combine original and overbooked bookings
        all_bookings = bookings + overbooked_bookings
        
        # Generate attribution data for overbooked bookings
        for overbooked_booking in overbooked_bookings:
            counterfactual_price = overbooked_booking['base_price']
            attribution_truth = {
                'booking_id': overbooked_booking['booking_id'],
                'true_attribution_score': round(overbooked_booking['attribution_score'], 3),
                'causal_campaign_id': overbooked_booking['campaign_id'],
                'counterfactual_price': round(counterfactual_price, 2),
                'would_have_booked_anyway': not overbooked_booking['incremental_flag']
            }
            attribution_data.append(attribution_truth)
        
        # Update attribution data for cancelled bookings
        for booking in all_bookings:
            if booking['is_cancelled'] and booking['campaign_id']:
                for attr_record in attribution_data:
                    if attr_record['booking_id'] == booking['booking_id']:
                        attr_record['true_attribution_score'] = round(booking['attribution_score'], 3)
                        attr_record['would_have_booked_anyway'] = True
                        break
        
        self.validate_data(all_bookings, campaigns)
        self.save_data(all_bookings, campaigns, customers, attribution_data, baseline_demand)
        
        print(f"\n🎉 Configurable data generation complete!")
        return all_bookings, campaigns, customers, attribution_data, baseline_demand