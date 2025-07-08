"""
generators.booking_logic.py
"""

import random
import numpy as np
from datetime import datetime, timedelta


class SimpleBookingLogic:

    
    def __init__(self, config):
        self.config = config
    
    def select_room_type(self):
        """Select room type based on configured distribution"""
        room_weights = np.array(self.config.DATA_CONFIG['room_type_distribution'])
        room_weights_normalized = room_weights / room_weights.sum()
        return np.random.choice(self.config.ROOM_TYPES, p=room_weights_normalized)
    
    def generate_stay_dates(self, booking_date, customer, selected_campaign):
        """
        Generate stay dates with basic logic that works reliably
        """
        # 1. Select stay length
        stay_length = self._select_stay_length()
        
        # 2. Determine lead time based on customer segment
        lead_time = self._get_lead_time_for_customer(customer)
        
        # 3. Calculate stay start date
        stay_start_date = booking_date + timedelta(days=lead_time)
        
        # 4. Ensure stay is in operational months
        stay_start_date = self._adjust_to_operational_season(stay_start_date)
        
        # 5. Calculate end date
        stay_end_date = stay_start_date + timedelta(days=int(stay_length))
        
        return stay_start_date, stay_end_date, stay_length
    
    def _select_stay_length(self):
        """Select stay length from configured distribution"""
        stay_config = self.config.DATA_CONFIG['stay_length_distribution']
        stay_weights = np.array(list(stay_config.values()))
        stay_weights_normalized = stay_weights / stay_weights.sum()
        return np.random.choice(list(stay_config.keys()), p=stay_weights_normalized)
    
    def _get_lead_time_for_customer(self, customer):
        """Get appropriate lead time for customer segment"""
        segment_config = self.config.CUSTOMER_SEGMENTS[customer['segment']]
        min_lead, max_lead = segment_config['advance_booking_days']
        return random.randint(min_lead, min(max_lead, 180))  # Cap at 6 months
    
    def _adjust_to_operational_season(self, proposed_date):
        """Adjust date to fall within operational season"""
        operational_months = self.config.OPERATIONAL_MONTHS
        
        # If already in operational season, use as-is
        if proposed_date.month in operational_months:
            return proposed_date
        
        weights = []
        for i, month in enumerate(operational_months):
            if i == 0:  # First month
                weights.append(0.7)  # 30% reduction
            elif i == len(operational_months) - 1:  # Last month
                weights.append(0.8)  # 20% reduction  
            else:  # Middle months
                weights.append(1.0)  # Normal weight
        
        # Normalize weights
        total_weight = sum(weights)
        probabilities = [w/total_weight for w in weights]
        
        # Choose month based on weights
        chosen_month = np.random.choice(operational_months, p=probabilities)
        
        # Generate date in chosen month
        year = proposed_date.year if chosen_month >= proposed_date.month else proposed_date.year + 1
        
        try:
            return datetime(year, chosen_month, random.randint(5, 25))
        except ValueError:
            return datetime(year, chosen_month, 15)


# Simple distribution system
class BookingLogic:
    """Wrapper to maintain compatibility with existing code"""
    
    def __init__(self, config):
        self.simple_logic = SimpleBookingLogic(config)
    
    def select_room_type(self):
        return self.simple_logic.select_room_type()
    
    def generate_stay_dates(self, booking_date, customer, selected_campaign):
        return self.simple_logic.generate_stay_dates(booking_date, customer, selected_campaign)
    
    def get_distribution_stats(self):
        """Return dummy stats for compatibility"""
        return {
            'total_stays': 0,
            'target_distribution': {},
            'current_distribution': {}
        }