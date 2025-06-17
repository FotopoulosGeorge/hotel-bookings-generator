"""
Hotel Booking Data Generator - Improved Version

Enhanced data generation logic for creating realistic hotel booking datasets
with improved pricing, attribution, overbooking, and date consistency.
"""

import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta
import pickle
import random
import warnings
from collections import defaultdict
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
        
        # Initialize inventory tracking
        self.inventory_tracker = InventoryTracker(self.config)
        
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
                if month == 12:
                    days_in_month = 31
                else:
                    days_in_month = (datetime(year, month + 1, 1) - datetime(year, month, 1)).days
                    
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
                'campaign_exposures': [],  # Track campaign exposures for attribution
                'booking_history': []
            }
            customers.append(customer)
            self.customer_counter += 1
        
        return customers
    
    def get_base_price_for_date(self, date, room_type):
        """Get base price for a specific date and room type using ONLY periodic pricing"""
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
        
        # Handle missing periods by finding the most recent period and extrapolating
        if pricing_periods:
            # Sort periods by end date
            sorted_periods = sorted(pricing_periods, key=lambda p: p['end_date'])
            latest_period = sorted_periods[-1]
            
            # If the date is after our latest period, use the latest period's price
            latest_end = datetime.strptime(latest_period['end_date'], '%Y-%m-%d').date()
            if date > latest_end:
                return latest_period['base_price']
            
            # If date is before our earliest period, use the earliest period's price
            earliest_period = sorted_periods[0]
            earliest_start = datetime.strptime(earliest_period['start_date'], '%Y-%m-%d').date()
            if date < earliest_start:
                return earliest_period['base_price']
        
        # Final fallback to static pricing
        return self.config.BASE_PRICES[room_type]
    
    def calculate_improved_attribution_score(self, booking_date, campaign, customer):
        """
        Improved attribution model using time-decay and campaign-specific logic
        
        Returns: (attribution_score, is_incremental)
        """
        if not campaign:
            return 0.0, False
        
        # Campaign-specific attribution parameters
        attribution_params = {
            'Early_Booking': {
                'peak_attribution': 0.8,
                'decay_rate': 0.02,  # Slow decay for long-term campaigns
                'min_attribution': 0.3,
                'incremental_threshold': 0.4
            },
            'Flash_Sale': {
                'peak_attribution': 0.9,
                'decay_rate': 0.15,  # Fast decay for urgency
                'min_attribution': 0.1,
                'incremental_threshold': 0.6
            },
            'Special_Offer': {
                'peak_attribution': 0.7,
                'decay_rate': 0.05,  # Medium decay
                'min_attribution': 0.2,
                'incremental_threshold': 0.5
            }
        }
        
        params = attribution_params.get(campaign['campaign_type'], attribution_params['Special_Offer'])
        
        # Calculate time since campaign start
        days_since_start = (booking_date - campaign['start_date']).days
        
        # Time-decay attribution
        time_decay_factor = np.exp(-params['decay_rate'] * days_since_start)
        base_attribution = params['peak_attribution'] * time_decay_factor
        base_attribution = max(params['min_attribution'], base_attribution)
        
        # Segment matching bonus
        segment_bonus = 1.0
        if customer['segment'] in campaign.get('target_segments', []):
            segment_bonus = 1.3
        else:
            segment_bonus = 0.7
        
        # Channel alignment bonus
        channel_bonus = 1.0
        if campaign['channel'] == 'Mixed' or campaign['channel'] == customer['channel_preference']:
            channel_bonus = 1.1
        else:
            channel_bonus = 0.9
        
        # Customer fatigue penalty (based on recent campaign exposures)
        fatigue_factor = 1.0
        recent_exposures = [exp for exp in customer['campaign_exposures'] 
                          if (booking_date - exp['exposure_date']).days <= 30]
        if len(recent_exposures) > 0:
            fatigue_factor = max(0.3, 1.0 - (len(recent_exposures) * 0.1))
        
        # Final attribution score
        final_attribution = base_attribution * segment_bonus * channel_bonus * fatigue_factor
        final_attribution = max(0.0, min(1.0, final_attribution))
        
        # Determine incrementality
        is_incremental = final_attribution > params['incremental_threshold']
        
        return final_attribution, is_incremental
    
    def generate_robust_stay_dates(self, booking_date, customer, selected_campaign):
        """
        Robust stay date generation with proper validation and hierarchy
        
        Priority order:
        1. Ensure stay is after booking
        2. Respect customer planning horizon
        3. Apply campaign-specific logic
        4. Ensure operational season compliance
        5. Validate stay length feasibility
        """
        stay_config = self.config.DATA_CONFIG['stay_length_distribution']
        stay_weights = np.array(list(stay_config.values()))
        stay_weights_normalized = stay_weights / stay_weights.sum()
        
        stay_length = np.random.choice(
            list(stay_config.keys()),
            p=stay_weights_normalized
        )
        
        # 1. Calculate latest possible booking start (respecting planning horizon)
        max_advance_days = min(customer['planning_horizon'], 365)
        latest_start = booking_date + timedelta(days=max_advance_days)
        
        # 2. Campaign-specific stay date logic
        if selected_campaign and selected_campaign['campaign_type'] == 'Early_Booking':
            # Early booking campaigns target operational season only
            current_year = booking_date.year
            
            # Determine target operational season year
            if booking_date.month <= 4:  # Booking in Jan-Apr targets same year season
                target_year = current_year
            else:  # Booking later targets next year season  
                target_year = current_year + 1
            
            # Use configured seasonal weights but constrain to operational months
            eb_config = self.config.CAMPAIGN_TYPES['Early_Booking']
            seasonal_weights = eb_config['seasonal_stay_weights']
            
            # Filter weights to only include operational months
            # Use smart distribution that favors Jul-Aug even for early booking
            smart_weights = {5: 0.10, 6: 0.25, 7: 0.35, 8: 0.28, 9: 0.02}
            operational_weights = {month: smart_weights.get(month, 0.1) for month in self.config.OPERATIONAL_MONTHS}
            
            if operational_weights:
                weights_array = np.array(list(operational_weights.values()))
                weights_normalized = weights_array / weights_array.sum()
                
                selected_month = np.random.choice(
                    list(operational_weights.keys()),
                    p=weights_normalized
                )
            else:
                # Fallback to random operational month
                selected_month = random.choice(self.config.OPERATIONAL_MONTHS)
            
            # Generate stay start in selected month
            try:
                month_start = datetime(target_year, selected_month, 1)
                if selected_month == 9:
                    month_end = datetime(target_year, selected_month, 30)
                else:
                    # Use 30 as safe end day for all months
                    month_end = datetime(target_year, selected_month, 30)
                
                # Ensure stay can fit in the month
                max_start_in_month = month_end - timedelta(days=int(stay_length))
                if max_start_in_month >= month_start:
                    days_available = (max_start_in_month - month_start).days
                    stay_start_date = month_start + timedelta(days=random.randint(0, max(0, days_available)))
                else:
                    stay_start_date = month_start
                    
            except ValueError:
                # Fallback to start of operational season
                stay_start_date = datetime(target_year, min(self.config.OPERATIONAL_MONTHS), 1)
        
        else:
            # Regular booking logic - constrain to operational season from the start
            min_advance_days = 1
            max_advance_days = min(customer['planning_horizon'], 180)
            
            # Generate multiple candidate dates and pick the first one in operational season
            max_attempts = 10
            stay_start_date = None
            
            # Try to find candidates, but favor center months
            smart_weights = {5: 0.10, 6: 0.25, 7: 0.35, 8: 0.28, 9: 0.02}
            best_candidates = []
            regular_candidates = []

            for attempt in range(max_attempts):
                advance_days = random.randint(min_advance_days, max_advance_days)
                candidate_date = booking_date + timedelta(days=advance_days)
                
                if candidate_date.month in self.config.OPERATIONAL_MONTHS:
                    if candidate_date.month in [7, 8]:  # Prefer July-August
                        best_candidates.append(candidate_date)
                    else:
                        regular_candidates.append(candidate_date)

            # Select from best candidates first, then regular
            if best_candidates:
                stay_start_date = random.choice(best_candidates)
            elif regular_candidates:
                stay_start_date = random.choice(regular_candidates)
            else:
                stay_start_date = None
            
            # If no candidate found in operational season, force one
            if stay_start_date is None:
                # Pick random operational month in reasonable timeframe
                target_year = booking_date.year
                
                # If booking is late in year, target next year's season
                if booking_date.month >= 10:
                    target_year += 1
                # If booking is early but would land before operational season
                elif booking_date.month < min(self.config.OPERATIONAL_MONTHS):
                    pass  # Use current year
                else:
                    # Booking during operational season or just after
                    target_year += 1
                
                target_month = random.choice(self.config.OPERATIONAL_MONTHS)
                target_day = random.randint(1, 28)
                stay_start_date = datetime(target_year, target_month, target_day)
        
        # 3. MANDATORY OPERATIONAL SEASON ENFORCEMENT
        # This section runs regardless of campaign type to ensure ALL stays are in operational season
        
        # Ensure stay start is after booking date
        if stay_start_date <= booking_date:
            stay_start_date = booking_date + timedelta(days=random.randint(1, 7))
        
        # FORCE stay into operational season if it's not already
        if stay_start_date.month not in self.config.OPERATIONAL_MONTHS:
            # Determine which operational year to target
            current_year = stay_start_date.year
            segment_lead_time_caps = {
                'Last_Minute': 45,      # 1.5 months max
                'Flexible': 120,        # 4 months max  
                'Early_Planner': 180    # 6 months max
            }
            max_reasonable_lead_time = segment_lead_time_caps.get(customer['segment'], 120)
            
            # If booking is late in year, target next year's season
            if booking_date.month >= 10:
                next_year_start = datetime(current_year + 1, min(self.config.OPERATIONAL_MONTHS), 1)
                if (next_year_start - booking_date).days <= max_reasonable_lead_time:
                    target_year = current_year + 1
                else:
                    # Lead time too long - skip to current year if any operational season left
                    current_season_end = datetime(current_year, max(self.config.OPERATIONAL_MONTHS), 30)
                    if current_season_end > booking_date:
                        target_year = current_year
                    else:
                        # No good options - use next year but will be capped later
                        target_year = current_year + 1
            # If booking is early in year but stay would be before season, target current year
            elif stay_start_date.month < min(self.config.OPERATIONAL_MONTHS):
                target_year = current_year
            # If stay is after season, target next year
            else:
                next_year_start = datetime(current_year + 1, min(self.config.OPERATIONAL_MONTHS), 1)
                if (next_year_start - booking_date).days <= max_reasonable_lead_time:
                    target_year = current_year + 1
                else:
                    target_year = current_year  # Keep in current year to avoid excessive lead time
            
            # Pick operational month with smart distribution (favor center months)
            operational_month_weights = {5: 0.15, 6: 0.25, 7: 0.30, 8: 0.25, 9: 0.05}
            available_months = [m for m in self.config.OPERATIONAL_MONTHS if m in operational_month_weights]
            if available_months:
                weights = [operational_month_weights[m] for m in available_months]
                weights_normalized = np.array(weights) / sum(weights)
                target_month = np.random.choice(available_months, p=weights_normalized)
            else:
                target_month = random.choice(self.config.OPERATIONAL_MONTHS)
            target_day = random.randint(1, 28)  # Safe day for all months
            
            try:
                forced_stay_date = datetime(target_year, target_month, target_day)
                
                # Ensure the forced date is reasonable (not too far from booking)
                days_ahead = (forced_stay_date - booking_date).days
                max_reasonable_advance = min(customer['planning_horizon'], 365)
                
                if 1 <= days_ahead <= max_reasonable_advance:
                    stay_start_date = forced_stay_date
                else:
                    # If too far, pick a closer operational date
                    if days_ahead > max_reasonable_advance:
                        # Try current year operational season
                        if current_year == booking_date.year:
                            closer_date = datetime(current_year, target_month, target_day)
                            if (closer_date - booking_date).days >= 1:
                                stay_start_date = closer_date
                    else:
                        # If negative days, ensure it's at least 1 day ahead
                        stay_start_date = booking_date + timedelta(days=random.randint(1, 30))
                        # Re-force into operational season
                        if stay_start_date.month not in self.config.OPERATIONAL_MONTHS:
                            operational_start = datetime(stay_start_date.year, min(self.config.OPERATIONAL_MONTHS), 1)
                            stay_start_date = operational_start + timedelta(days=random.randint(0, 60))
                            
            except ValueError:
                # Fallback - force to start of operational season
                operational_start = datetime(current_year, min(self.config.OPERATIONAL_MONTHS), 1)
                stay_start_date = operational_start + timedelta(days=random.randint(0, 30))
        
        # Calculate stay end date
        stay_end_date = stay_start_date + timedelta(days=int(stay_length))
        
        # Ensure stay doesn't extend beyond operational season
        if stay_end_date.month > max(self.config.OPERATIONAL_MONTHS):
            # Truncate stay to end of operational season
            season_end = datetime(stay_start_date.year, max(self.config.OPERATIONAL_MONTHS), 30)
            if season_end > stay_start_date:
                adjusted_length = (season_end - stay_start_date).days
                stay_length = max(1, min(stay_length, adjusted_length))
                stay_end_date = stay_start_date + timedelta(days=int(stay_length))
        
        # Final validation - if we still have issues, force a valid date
        if stay_start_date.month not in self.config.OPERATIONAL_MONTHS:
            # Emergency fallback - force to May of appropriate year
            target_year = stay_start_date.year if stay_start_date.month < 5 else stay_start_date.year + 1
            stay_start_date = datetime(target_year, 5, random.randint(1, 28))
            stay_end_date = stay_start_date + timedelta(days=int(stay_length))

        # Final lead time validation - cap excessive lead times
        # Segment-specific final lead time validation
        segment_caps = {'Last_Minute': 45, 'Flexible': 120, 'Early_Planner': 180}
        max_lead_time = segment_caps.get(customer['segment'], 120)
        lead_time_days = (stay_start_date - booking_date).days
        if lead_time_days > max_lead_time:
            # Force to closer operational season
            max_date = booking_date + timedelta(days=180)
            if max_date.month in self.config.OPERATIONAL_MONTHS:
                stay_start_date = max_date
            else:
                # Find nearest operational month within 180 days
                for days_ahead in range(30, 181, 30):  # Check 1, 2, 3, 4, 5, 6 months ahead
                    candidate_date = booking_date + timedelta(days=days_ahead)
                    if candidate_date.month in self.config.OPERATIONAL_MONTHS:
                        stay_start_date = candidate_date
                        break
                else:
                    # Last resort - use current year operational season if available
                    current_season_start = datetime(booking_date.year, min(self.config.OPERATIONAL_MONTHS), 1)
                    if current_season_start > booking_date:
                        stay_start_date = current_season_start
            
            stay_end_date = stay_start_date + timedelta(days=int(stay_length))
            
        return stay_start_date, stay_end_date, stay_length
    
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
    
    def generate_bookings(self, baseline_demand, campaigns, customers):
        """Generate bookings using improved logic"""
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
            if date.year > max(self.config.SIMULATION_YEARS):
                 continue
            
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
                    
                    # Track campaign exposure for attribution
                    customer['campaign_exposures'].append({
                        'campaign_id': selected_campaign['campaign_id'],
                        'exposure_date': date,
                        'campaign_type': selected_campaign['campaign_type']
                    })
                
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
                
                # Generate stay dates using improved logic
                stay_start_date, stay_end_date, stay_length = self.generate_robust_stay_dates(
                    date, customer, selected_campaign
                )
                
                # Improved pricing using ONLY periodic pricing
                base_price = self.get_base_price_for_date(stay_start_date, room_type)
                
                # Apply segment-specific pricing adjustments
                segment_pricing_multipliers = {
                    'Early_Planner': 0.95,    # 5% discount for advance booking
                    'Last_Minute': 1.10,      # 10% premium for last-minute
                    'Flexible': 1.00          # Standard pricing
                }
                segment_multiplier = segment_pricing_multipliers.get(customer['segment'], 1.0)
                base_price *= segment_multiplier
                
                # Check inventory and apply acceptance probability
                acceptance_probability = self.inventory_tracker.get_acceptance_probability(
                    stay_start_date, room_type, selected_campaign
                )
                
                # If inventory management rejects the booking, skip this iteration
                if random.random() > acceptance_probability:
                    continue
                
                # Reserve inventory
                self.inventory_tracker.reserve_inventory(
                    stay_start_date, stay_end_date, room_type
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
                
                # Calculate improved attribution
                attribution_score, is_incremental = self.calculate_improved_attribution_score(
                    date, selected_campaign, customer
                )
                
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
        # Handle campaign exposures
        df_customers['campaign_exposures'] = df_customers['campaign_exposures'].apply(
            lambda x: ';'.join([f"{exp['campaign_id']}:{exp['exposure_date'].strftime('%Y-%m-%d')}" for exp in x])
        )
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
        """Main method to generate all data using improved configuration"""
        print(f"\n🚀 Starting improved data generation...")
        
        campaigns = self.generate_campaigns()
        baseline_demand = self.generate_baseline_demand()
        customers = self.generate_customers()
        bookings, attribution_data = self.generate_bookings(baseline_demand, campaigns, customers)
        
        print(f"\n📋 Applying cancellation logic...")
        # Apply cancellations to existing bookings
        bookings = self.apply_cancellation_logic(bookings)
        
        # Note: Removed old overbooking logic - now handled in inventory management
        
        self.validate_data(bookings, campaigns)
        self.save_data(bookings, campaigns, customers, attribution_data, baseline_demand)
        
        print(f"\n🎉 Improved data generation complete!")
        return bookings, campaigns, customers, attribution_data, baseline_demand


class InventoryTracker:
    """
    Inventory management system for realistic overbooking and capacity constraints
    """
    
    def __init__(self, config):
        self.config = config
        self.daily_inventory = defaultdict(lambda: defaultdict(int))
        self.daily_reservations = defaultdict(lambda: defaultdict(int))
        
        # Initialize base capacity per room type per day
        self.base_capacity = {
            'Standard': 50,
            'Deluxe': 30, 
            'Suite': 15,
            'Premium': 10
        }
    
    def get_acceptance_probability(self, stay_date, room_type, campaign=None):
        """
        Calculate probability of accepting a booking based on current inventory
        and overbooking strategy
        """
        if not self.config.OVERBOOKING_CONFIG['enable_overbooking']:
            # Conservative mode - only accept if capacity available
            current_reservations = self.daily_reservations[stay_date][room_type]
            base_cap = self.base_capacity[room_type]
            return 1.0 if current_reservations < base_cap else 0.0
        
        # Calculate current occupancy rate
        current_reservations = self.daily_reservations[stay_date][room_type]
        base_cap = self.base_capacity[room_type]
        occupancy_rate = current_reservations / base_cap
        
        # Get overbooking parameters
        base_overbooking_rate = self.config.OVERBOOKING_CONFIG['base_overbooking_rate']
        month = stay_date.month
        seasonal_multiplier = self.config.OVERBOOKING_CONFIG['seasonal_overbooking_multipliers'].get(month, 1.0)
        
        # Maximum overbooking threshold
        max_overbooking = base_cap * (1 + base_overbooking_rate * seasonal_multiplier)
        
        if current_reservations < base_cap:
            # Below capacity - always accept
            return 1.0
        elif current_reservations < max_overbooking:
            # In overbooking zone - acceptance probability decreases
            overbooking_progress = (current_reservations - base_cap) / (max_overbooking - base_cap)
            acceptance_prob = 1.0 - (overbooking_progress * 0.8)  # Decrease to 20% at max overbooking
            
            # Campaign-specific adjustments
            if campaign:
                campaign_type = campaign['campaign_type']
                campaign_adjustment = self.config.OVERBOOKING_CONFIG['campaign_overbooking_adjustment'].get(campaign_type, 1.0)
                acceptance_prob *= campaign_adjustment
            
            return max(0.1, min(1.0, acceptance_prob))
        else:
            # Beyond maximum overbooking - very low acceptance
            return 0.05
    
    def reserve_inventory(self, stay_start_date, stay_end_date, room_type):
        """Reserve inventory for the entire stay duration"""
        current_date = stay_start_date
        while current_date < stay_end_date:
            self.daily_reservations[current_date][room_type] += 1
            current_date += timedelta(days=1)
    
    def get_overbooking_stats(self):
        """Get statistics about overbooking levels"""
        stats = {}
        for date, room_data in self.daily_reservations.items():
            for room_type, reservations in room_data.items():
                base_cap = self.base_capacity[room_type]
                if reservations > base_cap:
                    overbooking_rate = (reservations - base_cap) / base_cap
                    if date not in stats:
                        stats[date] = {}
                    stats[date][room_type] = {
                        'reservations': reservations,
                        'capacity': base_cap,
                        'overbooking_rate': overbooking_rate
                    }
        return stats