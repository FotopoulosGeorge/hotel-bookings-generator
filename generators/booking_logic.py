"""
Booking Logic Module

Handles core booking creation logic including room type selection,
stay date generation, and operational season enforcement.
"""

import random
import numpy as np
from datetime import datetime, timedelta


class BookingLogic:
    """Handles core booking creation and stay date logic"""
    
    def __init__(self, config):
        self.config = config
    
    def select_room_type(self):
        """Select room type based on configured distribution"""
        room_weights = np.array(self.config.DATA_CONFIG['room_type_distribution'])
        room_weights_normalized = room_weights / room_weights.sum()
        
        room_type = np.random.choice(
            self.config.ROOM_TYPES, 
            p=room_weights_normalized
        )
        
        return room_type
    
    def generate_stay_dates(self, booking_date, customer, selected_campaign):
        """
        Generate stay dates with proper validation and operational season enforcement
        
        Priority order:
        1. Ensure stay is after booking
        2. Respect customer planning horizon
        3. Apply campaign-specific logic
        4. Ensure operational season compliance (for seasonal hotels)
        5. Validate stay length feasibility
        """
        stay_length = self._select_stay_length()
        
        # Check operation mode
        if hasattr(self.config, 'OPERATION_MODE') and self.config.OPERATION_MODE == 'year_round':
            # Year-round hotels: simple logic
            stay_start_date = self._generate_year_round_stay_date(
                booking_date, customer, selected_campaign
            )
        else:
            # Seasonal hotels: must ensure stay falls within operational months
            if selected_campaign and selected_campaign['campaign_type'] == 'Early_Booking':
                stay_start_date = self._generate_early_booking_stay_dates(
                    booking_date, customer, selected_campaign, stay_length
                )
            else:
                stay_start_date = self._generate_regular_stay_dates(
                    booking_date, customer, customer['planning_horizon'], stay_length
                )
        
        # Calculate stay end date
        stay_end_date = stay_start_date + timedelta(days=int(stay_length))
        
        # Final validation for seasonal hotels
        if hasattr(self.config, 'OPERATION_MODE') and self.config.OPERATION_MODE == 'seasonal':
            stay_start_date, stay_end_date, stay_length = self._validate_seasonal_stay(
                stay_start_date, stay_end_date, stay_length, booking_date
            )
        
        return stay_start_date, stay_end_date, stay_length
    
    def _select_stay_length(self):
        """Select stay length based on configured distribution"""
        stay_config = self.config.DATA_CONFIG['stay_length_distribution']
        stay_weights = np.array(list(stay_config.values()))
        stay_weights_normalized = stay_weights / stay_weights.sum()
        
        stay_length = np.random.choice(
            list(stay_config.keys()),
            p=stay_weights_normalized
        )
        
        return stay_length
    
    def _generate_year_round_stay_date(self, booking_date, customer, selected_campaign):
        """Simple stay date generation for year-round hotels"""
        segment_config = self.config.CUSTOMER_SEGMENTS[customer['segment']]
        min_days, max_days = segment_config['advance_booking_days']
        
        # Apply campaign constraints if any
        if selected_campaign and selected_campaign['campaign_type'] == 'Early_Booking':
            min_days = max(min_days, selected_campaign.get('advance_booking_requirements', 60))
        
        advance_days = random.randint(min_days, max_days)
        return booking_date + timedelta(days=advance_days)
    
    def _generate_early_booking_stay_dates(self, booking_date, customer, selected_campaign, stay_length):
        """Generate stay dates for early booking campaigns with improved distribution"""
        current_year = booking_date.year
        
        # Determine target operational season year
        if booking_date.month <= 4:  # Booking in Jan-Apr targets same year season
            target_year = current_year
        else:  # Booking later targets next year season  
            target_year = current_year + 1
        
        # Use configured distribution or improved default weights
        if hasattr(self.config, 'SEASONAL_STAY_DISTRIBUTION'):
            operational_weights = self.config.SEASONAL_STAY_DISTRIBUTION
        else:
            # Better distribution that prevents spikes
            operational_weights = {
                5: 0.18,   # May: Moderate early season
                6: 0.22,   # June: Building up
                7: 0.26,   # July: Peak
                8: 0.24,   # August: Still busy
                9: 0.10    # September: Tail end
            }
        
        # Add variation based on booking month to spread load
        adjusted_weights = self._adjust_weights_by_booking_month(
            operational_weights, booking_date.month, stay_length
        )
        
        # Select month with adjusted weights
        months = list(adjusted_weights.keys())
        weights = list(adjusted_weights.values())
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w/total_weight for w in weights]
        
        selected_month = np.random.choice(months, p=normalized_weights)
        
        # Generate stay start within selected month
        try:
            month_start = datetime(target_year, selected_month, 1)
            
            # Calculate days in month
            if selected_month in [4, 6, 9, 11]:
                days_in_month = 30
            elif selected_month == 2:
                days_in_month = 29 if target_year % 4 == 0 else 28
            else:
                days_in_month = 31
            
            # Ensure stay can fit in the month
            max_start_day = days_in_month - stay_length + 1
            max_start_day = max(1, min(max_start_day, days_in_month - 1))
            
            # Distribute throughout the month, not just at the beginning
            if random.random() < 0.7:  # 70% chance to avoid first week
                min_start_day = 8
            else:
                min_start_day = 1
            
            min_start_day = min(min_start_day, max_start_day)
            selected_day = random.randint(min_start_day, max_start_day)
            
            stay_start_date = datetime(target_year, selected_month, selected_day)
            
        except ValueError:
            # Fallback to middle of operational season
            stay_start_date = datetime(target_year, 7, 15)
        
        return stay_start_date
    
    def _adjust_weights_by_booking_month(self, base_weights, booking_month, stay_length):
        """Adjust month selection weights based on when booking is made"""
        adjusted_weights = base_weights.copy()
        
        # Very early bookings (Jan-Feb) should spread more to mid-season
        if booking_month <= 2:
            # Reduce May weight
            if 5 in adjusted_weights:
                adjusted_weights[5] *= 0.7
            # Increase June-July weights
            if 6 in adjusted_weights:
                adjusted_weights[6] *= 1.2
            if 7 in adjusted_weights:
                adjusted_weights[7] *= 1.1
        
        # March-April bookings can be more evenly distributed
        elif booking_month in [3, 4]:
            # Slight reduction in May
            if 5 in adjusted_weights:
                adjusted_weights[5] *= 0.9
        
        # Late year bookings (Oct-Dec) for next year
        elif booking_month >= 10:
            # These can use normal distribution
            pass
        
        return adjusted_weights
    
    def _generate_regular_stay_dates(self, booking_date, customer, max_advance_days, stay_length):
        """Generate stay dates for regular bookings with better distribution"""
        min_advance_days = 1
        max_advance_days = min(customer['planning_horizon'], 180)
        
        # For seasonal hotels, find stays within operational months
        if hasattr(self.config, 'OPERATIONAL_MONTHS') and len(self.config.OPERATIONAL_MONTHS) < 12:
            stay_candidates = []
            
            # Look for valid stay dates
            for advance in range(min_advance_days, min(max_advance_days, 365)):
                candidate_date = booking_date + timedelta(days=advance)
                
                if candidate_date.month in self.config.OPERATIONAL_MONTHS:
                    # Check if full stay would fit
                    stay_end = candidate_date + timedelta(days=int(stay_length))
                    
                    # Allow stays that mostly fit in operational months
                    if stay_end.month in self.config.OPERATIONAL_MONTHS or \
                       (stay_end.month == self.config.OPERATIONAL_MONTHS[-1] + 1 and stay_end.day <= 5):
                        stay_candidates.append((advance, candidate_date))
            
            if stay_candidates:
                # Distribute across operational season
                weights = []
                for advance, date in stay_candidates:
                    month = date.month
                    
                    # Use configured distribution if available
                    if hasattr(self.config, 'SEASONAL_STAY_DISTRIBUTION'):
                        weight = self.config.SEASONAL_STAY_DISTRIBUTION.get(month, 0.2)
                    else:
                        # Default weights to reduce May spike
                        if month == 5:
                            weight = 0.7  # Reduce May
                        elif month in [6, 7]:
                            weight = 1.3  # Prefer June-July
                        elif month == 8:
                            weight = 1.1
                        else:
                            weight = 0.9
                    
                    weights.append(weight)
                
                # Normalize and select
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w/total_weight for w in weights]
                    selected_idx = np.random.choice(len(stay_candidates), p=weights)
                    return stay_candidates[selected_idx][1]
            
            # Fallback: find next operational period
            return self._find_next_operational_stay(booking_date, stay_length)
        
        # Year-round hotel: simple selection
        advance_days = random.randint(min_advance_days, max_advance_days)
        return booking_date + timedelta(days=advance_days)
    
    def _find_next_operational_stay(self, booking_date, stay_length):
        """Find the next available date in operational season"""
        current_year = booking_date.year
        
        # Find next operational period
        first_op_month = min(self.config.OPERATIONAL_MONTHS)
        last_op_month = max(self.config.OPERATIONAL_MONTHS)
        
        # Check this year first
        if booking_date.month < first_op_month:
            # Before season - use this year
            season_start = datetime(current_year, first_op_month, 1)
        elif booking_date.month > last_op_month:
            # After season - use next year
            season_start = datetime(current_year + 1, first_op_month, 1)
        else:
            # During season - use current date + 1 day
            season_start = booking_date + timedelta(days=1)
        
        # Add some randomization to avoid all bookings on same date
        if season_start.month in self.config.OPERATIONAL_MONTHS:
            days_to_add = random.randint(0, 30)
            candidate = season_start + timedelta(days=days_to_add)
            
            # Ensure we stay in operational months
            if candidate.month in self.config.OPERATIONAL_MONTHS:
                return candidate
        
        return season_start
    
    def _validate_seasonal_stay(self, stay_start_date, stay_end_date, stay_length, booking_date):
        """Ensure entire stay falls within operational season for seasonal hotels"""
        # Check if stay dates are valid
        if (stay_start_date.month in self.config.OPERATIONAL_MONTHS and 
            stay_end_date.month in self.config.OPERATIONAL_MONTHS):
            return stay_start_date, stay_end_date, stay_length
        
        # If end date spills over, truncate stay
        if (stay_start_date.month in self.config.OPERATIONAL_MONTHS and 
            stay_end_date.month not in self.config.OPERATIONAL_MONTHS):
            
            last_op_month = max(self.config.OPERATIONAL_MONTHS)
            if stay_start_date.month <= last_op_month:
                # Find last day of operational season
                if last_op_month in [4, 6, 9, 11]:
                    last_day = 30
                elif last_op_month == 2:
                    last_day = 29 if stay_start_date.year % 4 == 0 else 28
                else:
                    last_day = 31
                
                season_end = datetime(stay_start_date.year, last_op_month, last_day)
                
                # Adjust stay length
                max_nights = (season_end - stay_start_date).days
                new_stay_length = min(stay_length, max(1, max_nights))
                
                return stay_start_date, stay_start_date + timedelta(days=int(new_stay_length)), new_stay_length
        
        # If completely outside operational months, this shouldn't happen
        # but return original values
        return stay_start_date, stay_end_date, stay_length