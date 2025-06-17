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
        4. Ensure operational season compliance
        5. Validate stay length feasibility
        """
        stay_length = self._select_stay_length()
        
        # Calculate latest possible booking start (respecting planning horizon)
        max_advance_days = min(customer['planning_horizon'], 365)
        
        if selected_campaign and selected_campaign['campaign_type'] == 'Early_Booking':
            stay_start_date = self._generate_early_booking_stay_dates(
                booking_date, customer, selected_campaign, stay_length
            )
        else:
            stay_start_date = self._generate_regular_stay_dates(
                booking_date, customer, max_advance_days, stay_length
            )
        
        # MANDATORY OPERATIONAL SEASON ENFORCEMENT
        stay_start_date = self._enforce_operational_season(
            stay_start_date, booking_date, customer, stay_length
        )
        
        # Calculate stay end date
        stay_end_date = stay_start_date + timedelta(days=int(stay_length))
        
        # Ensure stay doesn't extend beyond operational season
        stay_end_date, stay_length = self._adjust_stay_end_for_season(
            stay_start_date, stay_end_date, stay_length
        )
        
        # Final lead time validation - cap excessive lead times
        stay_start_date = self._validate_final_lead_time(
            stay_start_date, booking_date, customer
        )
        
        stay_end_date = stay_start_date + timedelta(days=int(stay_length))
        
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
    
    def _generate_early_booking_stay_dates(self, booking_date, customer, selected_campaign, stay_length):
        """Generate stay dates for early booking campaigns"""
        current_year = booking_date.year
        
        # Determine target operational season year
        if booking_date.month <= 4:  # Booking in Jan-Apr targets same year season
            target_year = current_year
        else:  # Booking later targets next year season  
            target_year = current_year + 1
        
        # Use configured seasonal weights but constrain to operational months
        eb_config = self.config.CAMPAIGN_TYPES['Early_Booking']
        operational_weights = {5: 0.10, 6: 0.25, 7: 0.35, 8: 0.28, 9: 0.02}
        
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
        
        return stay_start_date
    
    def _generate_regular_stay_dates(self, booking_date, customer, max_advance_days, stay_length):
        """Generate stay dates for regular bookings"""
        min_advance_days = 1
        max_advance_days = min(customer['planning_horizon'], 180)
        
        # Generate multiple candidate dates and pick the first one in operational season
        max_attempts = 10
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
            target_year = booking_date.year
            
            # If booking is late in year, target next year's season
            if booking_date.month >= 10:
                target_year += 1
            elif booking_date.month < min(self.config.OPERATIONAL_MONTHS):
                pass  # Use current year
            else:
                target_year += 1
            
            target_month = random.choice(self.config.OPERATIONAL_MONTHS)
            target_day = random.randint(1, 28)
            stay_start_date = datetime(target_year, target_month, target_day)
        
        return stay_start_date
    
    def _enforce_operational_season(self, stay_start_date, booking_date, customer, stay_length):
        """Force stay into operational season if it's not already"""
        if stay_start_date.month in self.config.OPERATIONAL_MONTHS:
            return stay_start_date
        
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
                current_season_end = datetime(current_year, max(self.config.OPERATIONAL_MONTHS), 30)
                if current_season_end > booking_date:
                    target_year = current_year
                else:
                    target_year = current_year + 1
        elif stay_start_date.month < min(self.config.OPERATIONAL_MONTHS):
            target_year = current_year
        else:
            next_year_start = datetime(current_year + 1, min(self.config.OPERATIONAL_MONTHS), 1)
            if (next_year_start - booking_date).days <= max_reasonable_lead_time:
                target_year = current_year + 1
            else:
                target_year = current_year
        
        # Pick operational month with smart distribution
        operational_month_weights = {5: 0.15, 6: 0.25, 7: 0.30, 8: 0.25, 9: 0.05}
        available_months = [m for m in self.config.OPERATIONAL_MONTHS if m in operational_month_weights]
        
        if available_months:
            weights = [operational_month_weights[m] for m in available_months]
            weights_normalized = np.array(weights) / sum(weights)
            target_month = np.random.choice(available_months, p=weights_normalized)
        else:
            target_month = random.choice(self.config.OPERATIONAL_MONTHS)
        
        target_day = random.randint(1, 28)
        
        try:
            forced_stay_date = datetime(target_year, target_month, target_day)
            
            # Ensure the forced date is reasonable
            days_ahead = (forced_stay_date - booking_date).days
            max_reasonable_advance = min(customer['planning_horizon'], 365)
            
            if 1 <= days_ahead <= max_reasonable_advance:
                return forced_stay_date
            else:
                # If too far, pick a closer operational date
                if days_ahead > max_reasonable_advance:
                    if current_year == booking_date.year:
                        closer_date = datetime(current_year, target_month, target_day)
                        if (closer_date - booking_date).days >= 1:
                            return closer_date
                else:
                    # If negative days, ensure it's at least 1 day ahead
                    fallback_date = booking_date + timedelta(days=random.randint(1, 30))
                    if fallback_date.month not in self.config.OPERATIONAL_MONTHS:
                        operational_start = datetime(fallback_date.year, min(self.config.OPERATIONAL_MONTHS), 1)
                        return operational_start + timedelta(days=random.randint(0, 60))
                    return fallback_date
                    
        except ValueError:
            # Emergency fallback
            operational_start = datetime(current_year, min(self.config.OPERATIONAL_MONTHS), 1)
            return operational_start + timedelta(days=random.randint(0, 30))
        
        # Final fallback
        return datetime(target_year, 5, random.randint(1, 28))
    
    def _adjust_stay_end_for_season(self, stay_start_date, stay_end_date, stay_length):
        """Ensure stay doesn't extend beyond operational season"""
        if stay_end_date.month > max(self.config.OPERATIONAL_MONTHS):
            # Truncate stay to end of operational season
            season_end = datetime(stay_start_date.year, max(self.config.OPERATIONAL_MONTHS), 30)
            if season_end > stay_start_date:
                adjusted_length = (season_end - stay_start_date).days
                stay_length = max(1, min(stay_length, adjusted_length))
                stay_end_date = stay_start_date + timedelta(days=int(stay_length))
        
        return stay_end_date, stay_length
    
    def _validate_final_lead_time(self, stay_start_date, booking_date, customer):
        """Final lead time validation - cap excessive lead times"""
        segment_caps = {'Last_Minute': 45, 'Flexible': 120, 'Early_Planner': 180}
        max_lead_time = segment_caps.get(customer['segment'], 120)
        lead_time_days = (stay_start_date - booking_date).days
        
        if lead_time_days > max_lead_time:
            # Force to closer operational season
            max_date = booking_date + timedelta(days=180)
            if max_date.month in self.config.OPERATIONAL_MONTHS:
                return max_date
            else:
                # Find nearest operational month within 180 days
                for days_ahead in range(30, 181, 30):
                    candidate_date = booking_date + timedelta(days=days_ahead)
                    if candidate_date.month in self.config.OPERATIONAL_MONTHS:
                        return candidate_date
                
                # Last resort - use current year operational season if available
                current_season_start = datetime(booking_date.year, min(self.config.OPERATIONAL_MONTHS), 1)
                if current_season_start > booking_date:
                    return current_season_start
        
        return stay_start_date