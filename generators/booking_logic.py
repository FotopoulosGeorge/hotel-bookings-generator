"""
DIRECT FIX for generators/booking_logic.py

Replace your existing StayDateDistributor class with this fixed version.
The issue was that candidate generation had impossible constraints.
"""

import random
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict


class StayDateDistributor:
    """
    FIXED: Unified stay date distribution system that actually works
    """
    
    def __init__(self, config):
        self.config = config
        self.monthly_allocation_tracker = defaultdict(int)
        self.total_stays_generated = 0
        self._target_distribution = None
        self._operational_months = None
        
    def get_target_distribution(self):
        """Get the target distribution for stay months"""
        if self._target_distribution is None:
            self._target_distribution = self._calculate_target_distribution()
        return self._target_distribution
    
    def _calculate_target_distribution(self):
        """Calculate target distribution based on operation mode"""
        if hasattr(self.config, 'OPERATION_MODE') and self.config.OPERATION_MODE == 'year_round':
            return self._get_year_round_distribution()
        else:
            return self._get_seasonal_distribution()
    
    def _get_seasonal_distribution(self):
        """Get seasonal target distribution"""
        if hasattr(self.config, 'SEASONAL_STAY_DISTRIBUTION'):
            return self.config.SEASONAL_STAY_DISTRIBUTION.copy()
        
        # Default that prevents May spike
        return {
            5: 0.16,   # May - reduced
            6: 0.24,   # June - higher
            7: 0.28,   # July - peak
            8: 0.26,   # August - peak
            9: 0.06    # September - minimal
        }
    
    def _get_year_round_distribution(self):
        """Get year-round target distribution"""
        seasonal_multipliers = self.config.SEASONAL_DEMAND_MULTIPLIERS
        total_weight = sum(seasonal_multipliers.values())
        return {month: weight / total_weight for month, weight in seasonal_multipliers.items()}
    
    def get_operational_months(self):
        """Get operational months for the hotel"""
        if self._operational_months is None:
            if hasattr(self.config, 'OPERATION_MODE') and self.config.OPERATION_MODE == 'year_round':
                self._operational_months = list(range(1, 13))
            else:
                self._operational_months = self.config.OPERATIONAL_MONTHS
        return self._operational_months
    
    def generate_stay_date(self, booking_date, customer, campaign=None):
        """
        FIXED: Generate stay dates with proper candidate generation
        """
        stay_length = self._select_stay_length()
        candidates = self._generate_candidate_dates_fixed(booking_date, customer, campaign, stay_length)
        stay_start_date = self._select_optimal_date(candidates, booking_date)
        stay_end_date = stay_start_date + timedelta(days=int(stay_length))
        self._update_allocation_tracking(stay_start_date)
        return stay_start_date, stay_end_date, stay_length
    
    def _select_stay_length(self):
        """Select stay length based on configured distribution"""
        stay_config = self.config.DATA_CONFIG['stay_length_distribution']
        stay_weights = np.array(list(stay_config.values()))
        stay_weights_normalized = stay_weights / stay_weights.sum()
        return np.random.choice(list(stay_config.keys()), p=stay_weights_normalized)
    
    def _generate_candidate_dates_fixed(self, booking_date, customer, campaign, stay_length):
        """
        FIXED: Generate candidate dates with relaxed constraints for seasonal hotels
        """
        candidates = []
        operational_months = self.get_operational_months()
        
        # FIXED: Get more reasonable lead time constraints
        segment_config = self.config.CUSTOMER_SEGMENTS[customer['segment']]
        base_min_lead, base_max_lead = segment_config['advance_booking_days']
        
        # FIXED: Relax constraints for seasonal hotels
        if len(operational_months) < 12:  # Seasonal hotel
            if customer['segment'] == 'Early_Planner':
                min_lead, max_lead = 30, 365  # Allow very early booking
            elif customer['segment'] == 'Last_Minute':
                min_lead, max_lead = 1, 120   # More flexibility for last minute
            else:  # Flexible
                min_lead, max_lead = 7, 180   # Reasonable range
        else:  # Year-round hotel - use original constraints
            min_lead, max_lead = base_min_lead, base_max_lead
        
        # Adjust for campaign requirements
        if campaign and campaign.get('advance_booking_requirements', 0) > 0:
            min_lead = max(min_lead, campaign['advance_booking_requirements'])
        
        # FIXED: Generate candidates with broader search
        search_start = booking_date + timedelta(days=min_lead)
        search_end = booking_date + timedelta(days=min(max_lead, 365))
        
        current_date = search_start
        while current_date <= search_end and len(candidates) < 100:  # Limit for performance
            if self._is_valid_stay_period_fixed(current_date, stay_length, operational_months):
                candidates.append(current_date)
            current_date += timedelta(days=1)
        
        # FIXED: Emergency search if no candidates found
        if not candidates:
            candidates = self._emergency_candidate_search(booking_date, stay_length, operational_months)
        
        return candidates
    
    def _is_valid_stay_period_fixed(self, start_date, stay_length, operational_months):
        """
        FIXED: Check if stay period is valid with some flexibility
        """
        if len(operational_months) == 12:  # Year-round
            return True
        
        # For seasonal hotels, check start and end months
        end_date = start_date + timedelta(days=int(stay_length) - 1)
        
        # FIXED: Allow some spillover at end of season
        start_valid = start_date.month in operational_months
        end_valid = (end_date.month in operational_months or 
                    (end_date.month == operational_months[-1] + 1 and end_date.day <= 7))
        
        return start_valid and end_valid
    
    def _emergency_candidate_search(self, booking_date, stay_length, operational_months):
        """
        FIXED: Emergency search for candidates when normal search fails
        """
        candidates = []
        
        # Look up to 1 year ahead, jumping by weeks for speed
        current_date = booking_date + timedelta(days=1)
        max_search_date = booking_date + timedelta(days=365)
        
        while current_date <= max_search_date and len(candidates) < 20:
            if self._is_valid_stay_period_fixed(current_date, stay_length, operational_months):
                candidates.append(current_date)
            current_date += timedelta(days=7)  # Jump by weeks
        
        return candidates
    
    def _select_optimal_date(self, candidates, booking_date):
        """
        FIXED: Select optimal date with better fallback
        """
        if not candidates:
            return self._get_better_fallback(booking_date)
        
        if self.total_stays_generated < 50:
            return self._select_with_target_weights(candidates)
        else:
            return self._select_with_spike_prevention(candidates)
    
    def _select_with_target_weights(self, candidates):
        """Select date using target distribution weights"""
        target_dist = self.get_target_distribution()
        
        weights = []
        for date in candidates:
            weight = target_dist.get(date.month, 0.1)
            weights.append(weight)
        
        if sum(weights) > 0:
            weights = np.array(weights) / sum(weights)
            selected_idx = np.random.choice(len(candidates), p=weights)
            return candidates[selected_idx]
        else:
            return random.choice(candidates)
    
    def _select_with_spike_prevention(self, candidates):
        """Select date while actively preventing spikes"""
        target_dist = self.get_target_distribution()
        current_dist = self._get_current_distribution()
        
        candidate_scores = []
        for date in candidates:
            month = date.month
            target_ratio = target_dist.get(month, 0.0)
            current_ratio = current_dist.get(month, 0.0)
            
            if target_ratio > 0:
                balance_score = max(0.1, target_ratio - current_ratio + 0.2)
                randomness = random.uniform(0.8, 1.2)
                total_score = balance_score * randomness
            else:
                total_score = 0.01
            
            candidate_scores.append(total_score)
        
        if max(candidate_scores) > 0:
            scores_array = np.array(candidate_scores)
            probabilities = scores_array / scores_array.sum()
            selected_idx = np.random.choice(len(candidates), p=probabilities)
            return candidates[selected_idx]
        else:
            return random.choice(candidates)
    
    def _get_current_distribution(self):
        """Get current allocation distribution"""
        if self.total_stays_generated == 0:
            return {}
        return {month: count / self.total_stays_generated 
                for month, count in self.monthly_allocation_tracker.items()}
    
    def _get_better_fallback(self, booking_date):
        """
        FIXED: Better fallback that finds valid operational dates
        """
        operational_months = self.get_operational_months()
        current_year = booking_date.year
        
        # Find next operational season
        first_op_month = min(operational_months)
        
        # Determine target year
        if booking_date.month < first_op_month:
            target_year = current_year
        else:
            target_year = current_year + 1
        
        # Try each operational month
        for month in operational_months:
            try:
                fallback_date = datetime(target_year, month, 15)
                if fallback_date > booking_date:
                    return fallback_date
            except ValueError:
                continue
        
        # Last resort
        return datetime(current_year + 1, 7, 15)
    
    def _update_allocation_tracking(self, stay_start_date):
        """Update allocation tracking"""
        self.monthly_allocation_tracker[stay_start_date.month] += 1
        self.total_stays_generated += 1
    
    def get_allocation_stats(self):
        """Get allocation statistics for monitoring"""
        target_dist = self.get_target_distribution()
        current_dist = self._get_current_distribution()
        
        stats = {
            'total_stays': self.total_stays_generated,
            'target_distribution': target_dist,
            'current_distribution': current_dist,
            'deviations': {}
        }
        
        for month in target_dist:
            target = target_dist[month]
            actual = current_dist.get(month, 0)
            deviation = actual - target
            stats['deviations'][month] = {
                'target': target,
                'actual': actual,
                'deviation': deviation,
                'deviation_pct': (deviation / target * 100) if target > 0 else 0
            }
        
        return stats



class BookingLogic:
    """Booking logic that uses the stay date distributor"""
    
    def __init__(self, config):
        self.config = config
        self.stay_date_distributor = StayDateDistributor(config)
    
    def select_room_type(self):
        """Select room type based on configured distribution (unchanged)"""
        room_weights = np.array(self.config.DATA_CONFIG['room_type_distribution'])
        room_weights_normalized = room_weights / room_weights.sum()
        
        return np.random.choice(self.config.ROOM_TYPES, p=room_weights_normalized)
    
    def generate_stay_dates(self, booking_date, customer, selected_campaign):
        """
        Generate stay dates using the distributor
        """
        return self.stay_date_distributor.generate_stay_date(
            booking_date, customer, selected_campaign
        )
    
    def get_distribution_stats(self):
        """Get distribution statistics for monitoring"""
        return self.stay_date_distributor.get_allocation_stats()