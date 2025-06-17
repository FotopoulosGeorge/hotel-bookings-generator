"""
Campaign Generation Module

Handles creation and management of promotional campaigns including
Early Booking, Flash Sales, and Special Offers.
"""

import random
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict


class CampaignGenerator:
    """Handles generation and management of promotional campaigns"""
    
    def __init__(self, config, campaign_counter_start=1000):
        self.config = config
        self.campaign_counter = campaign_counter_start
    
    def generate_campaigns(self):
        """Generate promotional campaigns based on configuration"""
        campaigns = []
        
        for year in self.config.SIMULATION_YEARS:
            # Early booking campaigns
            campaigns.extend(self._generate_early_booking_campaigns(year))
            
            # Flash sales (during operational season)
            campaigns.extend(self._generate_flash_sale_campaigns(year))
            
            # Special offers (shoulder season)
            campaigns.extend(self._generate_special_offer_campaigns(year))
        
        print(f"✅ Generated {len(campaigns)} campaigns")
        return campaigns
    
    def _generate_early_booking_campaigns(self, year):
        """Generate early booking campaigns for the year"""
        campaigns = []
        early_booking_config = self.config.CAMPAIGN_TYPES['Early_Booking']
        
        for month in early_booking_config['campaign_months']:
            campaign_year = year - 1 if month > 9 else year
            if campaign_year < 2021:
                continue
                
            for _ in range(early_booking_config['campaigns_per_month']):
                start_date = datetime(campaign_year, month, random.randint(1, 15))
                duration = random.randint(*early_booking_config['duration_range'])
                end_date = start_date + timedelta(days=duration)
                
                campaign = {
                    'campaign_id': f'EB_{self.campaign_counter}',
                    'campaign_type': 'Early_Booking',
                    'start_date': start_date,
                    'end_date': end_date,
                    'discount_percentage': random.uniform(*early_booking_config['discount_range']),
                    'target_segments': early_booking_config['target_segments'],
                    'channel': early_booking_config['preferred_channel'],
                    'room_types_eligible': self.config.ROOM_TYPES,
                    'advance_booking_requirements': early_booking_config['advance_booking_requirement'],
                    'capacity_limit': random.randint(*early_booking_config['capacity_range']),
                    'actual_bookings': 0,
                    'incremental_bookings': 0
                }
                campaigns.append(campaign)
                self.campaign_counter += 1
        
        return campaigns
    
    def _generate_flash_sale_campaigns(self, year):
        """Generate flash sale campaigns for the year"""
        campaigns = []
        flash_sale_config = self.config.CAMPAIGN_TYPES['Flash_Sale']
        
        for month in self.config.OPERATIONAL_MONTHS:
            num_flash_sales = random.randint(*flash_sale_config['campaigns_per_month'])
            for _ in range(num_flash_sales):
                start_date = datetime(year, month, random.randint(1, 25))
                duration = random.randint(*flash_sale_config['duration_range'])
                end_date = start_date + timedelta(days=duration)
                
                campaign = {
                    'campaign_id': f'FS_{self.campaign_counter}',
                    'campaign_type': 'Flash_Sale',
                    'start_date': start_date,
                    'end_date': end_date,
                    'discount_percentage': random.uniform(*flash_sale_config['discount_range']),
                    'target_segments': flash_sale_config['target_segments'],
                    'channel': random.choice(['Online_Direct', 'Connected_Agent']),
                    'room_types_eligible': random.sample(self.config.ROOM_TYPES, random.randint(2, 4)),
                    'advance_booking_requirements': flash_sale_config['advance_booking_requirement'],
                    'capacity_limit': random.randint(*flash_sale_config['capacity_range']),
                    'actual_bookings': 0,
                    'incremental_bookings': 0
                }
                campaigns.append(campaign)
                self.campaign_counter += 1
        
        return campaigns
    
    def _generate_special_offer_campaigns(self, year):
        """Generate special offer campaigns for the year"""
        campaigns = []
        special_offer_config = self.config.CAMPAIGN_TYPES['Special_Offer']
        
        for month in special_offer_config['target_months']:
            for _ in range(special_offer_config['campaigns_per_month']):
                start_date = datetime(year, month, random.randint(5, 20))
                duration = random.randint(*special_offer_config['duration_range'])
                end_date = start_date + timedelta(days=duration)
                
                campaign = {
                    'campaign_id': f'SO_{self.campaign_counter}',
                    'campaign_type': 'Special_Offer',
                    'start_date': start_date,
                    'end_date': end_date,
                    'discount_percentage': random.uniform(*special_offer_config['discount_range']),
                    'target_segments': special_offer_config['target_segments'],
                    'channel': 'Mixed',
                    'room_types_eligible': self.config.ROOM_TYPES,
                    'advance_booking_requirements': special_offer_config['advance_booking_requirement'],
                    'capacity_limit': random.randint(*special_offer_config['capacity_range']),
                    'actual_bookings': 0,
                    'incremental_bookings': 0
                }
                campaigns.append(campaign)
                self.campaign_counter += 1
        
        return campaigns
    
    def create_campaign_lookup(self, campaigns):
        """Create campaign lookup with configured influence periods"""
        campaign_lookup = defaultdict(list)
        
        for campaign in campaigns:
            start_date = campaign['start_date']
            end_date = campaign['end_date']
            
            # Add campaign for active period
            current_date = start_date
            while current_date <= end_date:
                campaign_lookup[current_date].append(campaign)
                current_date += timedelta(days=1)
            
            # Extend influence for early booking campaigns
            if campaign['campaign_type'] == 'Early_Booking':
                influence_days = self.config.CAMPAIGN_TYPES['Early_Booking']['influence_period_days']
                extended_end = end_date + timedelta(days=influence_days)
                current_date = end_date + timedelta(days=1)
                while current_date <= extended_end:
                    campaign_lookup[current_date].append(campaign)
                    current_date += timedelta(days=1)
        
        return campaign_lookup
    
    def select_campaign_for_booking(self, booking_date, customer, campaign_lookup):
        """Select appropriate campaign for a booking if eligible"""
        active_campaigns = campaign_lookup.get(booking_date, [])
        eligible_campaigns = []
        
        for campaign in active_campaigns:
            if self._is_customer_eligible_for_campaign(campaign, customer, booking_date):
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
                'exposure_date': booking_date,
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
        
        return selected_campaign, is_promotional
    
    def _is_customer_eligible_for_campaign(self, campaign, customer, booking_date):
        """Check if customer is eligible for a specific campaign"""
        # Check segment targeting
        if customer['segment'] not in campaign.get('target_segments', []):
            return False
        
        # Check capacity
        if campaign['actual_bookings'] >= campaign['capacity_limit']:
            return False
        
        # Check advance booking requirements for Early Booking campaigns
        if campaign['campaign_type'] == 'Early_Booking':
            current_year = booking_date.year
            if booking_date.month <= 4:
                target_season_start = datetime(current_year, 5, 1)
            elif booking_date.month >= 10:
                target_season_start = datetime(current_year + 1, 5, 1)
            else:
                target_season_start = datetime(current_year + 1, 5, 1)
            
            advance_days = (target_season_start - booking_date).days
            if advance_days < campaign.get('advance_booking_requirements', 90):
                return False
        
        return True
    
    def update_campaign_performance(self, campaigns, bookings):
        """Update campaign performance metrics"""
        df_bookings = pd.DataFrame(bookings)
        campaign_booking_counts = df_bookings[df_bookings['campaign_id'].notna()]['campaign_id'].value_counts()
        campaign_incremental_counts = df_bookings[
            (df_bookings['campaign_id'].notna()) & (df_bookings['incremental_flag'] == True)
        ]['campaign_id'].value_counts()
        
        for campaign in campaigns:
            campaign['actual_bookings'] = campaign_booking_counts.get(campaign['campaign_id'], 0)
            campaign['incremental_bookings'] = campaign_incremental_counts.get(campaign['campaign_id'], 0)