"""
Inventory Management Module

Handles room inventory tracking, overbooking strategies, and
booking acceptance probability calculations.
"""

import numpy as np
from datetime import timedelta
from collections import defaultdict


class InventoryManager:
    """
    Inventory management system for realistic overbooking and capacity constraints
    """
    
    def __init__(self, config):
        self.config = config
        self.daily_reservations = defaultdict(lambda: defaultdict(int))
        
        # Initialize base capacity per room type per day
        self.base_capacity = self.config.INVENTORY_CONFIG['base_capacity_per_room_type']
        
        print(f"🏨 Initialized inventory management")
        print(f"   📊 Room capacities: {self.base_capacity}")
        print(f"   🔄 Overbooking enabled: {self.config.OVERBOOKING_CONFIG['enable_overbooking']}")
    
    def get_acceptance_probability(self, stay_date, room_type, campaign=None):
        """
        Calculate probability of accepting a booking based on current inventory
        and overbooking strategy
        """
        if not self.config.OVERBOOKING_CONFIG['enable_overbooking']:
            # Conservative mode - only accept if capacity available
            return self._conservative_acceptance_probability(stay_date, room_type)
        
        return self._dynamic_acceptance_probability(stay_date, room_type, campaign)
    
    def _conservative_acceptance_probability(self, stay_date, room_type):
        """Conservative acceptance: only accept if capacity available"""
        current_reservations = self.daily_reservations[stay_date][room_type]
        base_cap = self.base_capacity[room_type]
        return 1.0 if current_reservations < base_cap else 0.0
    
    def _dynamic_acceptance_probability(self, stay_date, room_type, campaign=None):
        """Dynamic acceptance with overbooking strategy"""
        # Calculate current occupancy rate
        current_reservations = self.daily_reservations[stay_date][room_type]
        base_cap = self.base_capacity[room_type]
        occupancy_rate = current_reservations / base_cap
        
        # Get overbooking parameters
        max_overbooking = self._calculate_max_overbooking(stay_date, room_type, campaign)
        
        if current_reservations < base_cap:
            # Below capacity - always accept
            return 1.0
        elif current_reservations < max_overbooking:
            # In overbooking zone - acceptance probability decreases
            overbooking_progress = (current_reservations - base_cap) / (max_overbooking - base_cap)
            acceptance_prob = 1.0 - (overbooking_progress * 0.8)  # Decrease to 20% at max overbooking
            
            # Campaign-specific adjustments
            if campaign:
                campaign_adjustment = self._get_campaign_overbooking_adjustment(campaign)
                acceptance_prob *= campaign_adjustment
            
            return max(0.1, min(1.0, acceptance_prob))
        else:
            # Beyond maximum overbooking - very low acceptance
            return 0.05
    
    def _calculate_max_overbooking(self, stay_date, room_type, campaign=None):
        """Calculate maximum overbooking threshold for given conditions"""
        base_cap = self.base_capacity[room_type]
        base_overbooking_rate = self.config.OVERBOOKING_CONFIG['base_overbooking_rate']
        
        # Seasonal adjustment
        month = stay_date.month
        seasonal_multiplier = self.config.OVERBOOKING_CONFIG['seasonal_overbooking_multipliers'].get(month, 1.0)
        
        # Calculate maximum overbooking
        max_overbooking = base_cap * (1 + base_overbooking_rate * seasonal_multiplier)
        
        return max_overbooking
    
    def _get_campaign_overbooking_adjustment(self, campaign):
        """Get campaign-specific overbooking adjustment factor"""
        campaign_type = campaign['campaign_type']
        return self.config.OVERBOOKING_CONFIG['campaign_overbooking_adjustment'].get(campaign_type, 1.0)
    
    def reserve_inventory(self, stay_start_date, stay_end_date, room_type):
        """Reserve inventory for the entire stay duration"""
        current_date = stay_start_date
        while current_date < stay_end_date:
            self.daily_reservations[current_date][room_type] += 1
            current_date += timedelta(days=1)
    
    def get_occupancy_stats(self, date_range=None):
        """Get occupancy statistics for analysis"""
        stats = {}
        
        if date_range:
            start_date, end_date = date_range
            relevant_dates = [
                date for date in self.daily_reservations.keys() 
                if start_date <= date <= end_date
            ]
        else:
            relevant_dates = self.daily_reservations.keys()
        
        for date in relevant_dates:
            room_stats = {}
            for room_type in self.base_capacity.keys():
                reservations = self.daily_reservations[date][room_type]
                base_cap = self.base_capacity[room_type]
                occupancy_rate = reservations / base_cap
                
                room_stats[room_type] = {
                    'reservations': reservations,
                    'capacity': base_cap,
                    'occupancy_rate': occupancy_rate,
                    'is_overbooked': reservations > base_cap
                }
            
            stats[date] = room_stats
        
        return stats
    
    def get_overbooking_stats(self):
        """Get statistics about overbooking levels"""
        stats = {}
        total_overbooked_nights = 0
        total_nights = 0
        
        for date, room_data in self.daily_reservations.items():
            for room_type, reservations in room_data.items():
                total_nights += 1
                base_cap = self.base_capacity[room_type]
                
                if reservations > base_cap:
                    total_overbooked_nights += 1
                    overbooking_rate = (reservations - base_cap) / base_cap
                    
                    if date not in stats:
                        stats[date] = {}
                    
                    stats[date][room_type] = {
                        'reservations': reservations,
                        'capacity': base_cap,
                        'overbooking_rate': overbooking_rate,
                        'excess_reservations': reservations - base_cap
                    }
        
        # Calculate overall overbooking statistics
        overall_stats = {
            'total_overbooked_nights': total_overbooked_nights,
            'total_nights': total_nights,
            'overall_overbooking_rate': total_overbooked_nights / total_nights if total_nights > 0 else 0,
            'detailed_stats': stats
        }
        
        return overall_stats
    
    def simulate_walk_ins_and_no_shows(self, date, room_type):
        """
        Simulate walk-ins and no-shows for more realistic inventory management
        
        Args:
            date: Date to simulate
            room_type: Room type
        
        Returns:
            Dictionary with walk-in and no-show adjustments
        """
        current_reservations = self.daily_reservations[date][room_type]
        base_cap = self.base_capacity[room_type]
        
        # No-show probability (higher for overbooked situations)
        if current_reservations > base_cap:
            no_show_rate = 0.08 + (0.02 * (current_reservations - base_cap) / base_cap)
        else:
            no_show_rate = 0.05
        
        # Calculate expected no-shows
        expected_no_shows = int(current_reservations * no_show_rate)
        
        # Walk-in probability (lower when highly occupied)
        occupancy_rate = current_reservations / base_cap
        if occupancy_rate < 0.7:
            walk_in_probability = 0.1
        elif occupancy_rate < 0.9:
            walk_in_probability = 0.05
        else:
            walk_in_probability = 0.01
        
        # Calculate expected walk-ins
        expected_walk_ins = int(base_cap * walk_in_probability)
        
        return {
            'expected_no_shows': expected_no_shows,
            'expected_walk_ins': expected_walk_ins,
            'net_adjustment': expected_walk_ins - expected_no_shows,
            'final_occupancy': current_reservations + expected_walk_ins - expected_no_shows
        }
    
    def get_revenue_optimization_insights(self):
        """
        Generate insights for revenue optimization based on inventory patterns
        
        Returns:
            Dictionary with optimization recommendations
        """
        occupancy_stats = self.get_occupancy_stats()
        overbooking_stats = self.get_overbooking_stats()
        
        insights = {
            'high_demand_periods': [],
            'low_demand_periods': [],
            'overbooking_opportunities': [],
            'capacity_constraints': []
        }
        
        # Analyze occupancy patterns
        for date, room_stats in occupancy_stats.items():
            for room_type, stats in room_stats.items():
                occupancy_rate = stats['occupancy_rate']
                
                if occupancy_rate > 0.95:
                    insights['high_demand_periods'].append({
                        'date': date,
                        'room_type': room_type,
                        'occupancy_rate': occupancy_rate,
                        'recommendation': 'Consider dynamic pricing increase'
                    })
                elif occupancy_rate < 0.6:
                    insights['low_demand_periods'].append({
                        'date': date,
                        'room_type': room_type,
                        'occupancy_rate': occupancy_rate,
                        'recommendation': 'Consider promotional campaigns'
                    })
                
                if stats['is_overbooked']:
                    insights['capacity_constraints'].append({
                        'date': date,
                        'room_type': room_type,
                        'excess_demand': stats['reservations'] - stats['capacity']
                    })
        
        return insights