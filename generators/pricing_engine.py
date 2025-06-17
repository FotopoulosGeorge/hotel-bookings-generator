"""
Pricing Engine Module

Handles all pricing logic including base pricing, discounts, campaign pricing,
and attribution modeling for hotel bookings.
"""

import random
import numpy as np
from datetime import datetime


class PricingEngine:
    """Handles pricing calculations and attribution modeling"""
    
    def __init__(self, config):
        self.config = config
    
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
    
    def calculate_pricing(self, stay_start_date, room_type, customer, selected_campaign, channel, is_promotional):
        """Calculate complete pricing for a booking"""
        # Get base price using periodic pricing
        base_price = self.get_base_price_for_date(stay_start_date, room_type)
        
        # Apply customer-specific pricing adjustments
        base_price *= self._get_customer_pricing_multiplier(customer)
        
        # Apply seasonal pricing if configured
        base_price *= self._get_seasonal_pricing_multiplier(stay_start_date)
        
        # Initialize final price and discount
        final_price = base_price
        discount_amount = 0
        campaign_id_final = None
        
        # Apply campaign discounts
        if selected_campaign:
            if room_type in selected_campaign['room_types_eligible']:
                discount_amount = base_price * selected_campaign['discount_percentage']
                final_price = base_price - discount_amount
                campaign_id_final = selected_campaign['campaign_id']
        
        # Apply non-campaign promotional discounts
        elif is_promotional:
            discount_amount, final_price = self._apply_non_campaign_discount(
                base_price, channel
            )
        
        return base_price, final_price, discount_amount, campaign_id_final
    
    def _get_customer_pricing_multiplier(self, customer):
        """Get pricing multiplier based on customer characteristics"""
        segment_pricing_multipliers = {
            'Early_Planner': 0.95,    # 5% discount for advance booking
            'Last_Minute': 1.10,      # 10% premium for last-minute
            'Flexible': 1.00          # Standard pricing
        }
        
        loyalty_multipliers = {
            'Bronze': 1.0,
            'Silver': 0.97,  # 3% loyalty discount
            'Gold': 0.95     # 5% loyalty discount
        }
        
        segment_mult = segment_pricing_multipliers.get(customer['segment'], 1.0)
        loyalty_mult = loyalty_multipliers.get(customer['loyalty_status'], 1.0)
        
        return segment_mult * loyalty_mult
    
    def _get_seasonal_pricing_multiplier(self, stay_start_date):
        """Get seasonal pricing multiplier"""
        seasonal_multipliers = self.config.DATA_CONFIG.get('seasonal_pricing_multipliers', {})
        month = stay_start_date.month
        return seasonal_multipliers.get(month, 1.0)
    
    def _apply_non_campaign_discount(self, base_price, channel):
        """Apply non-campaign promotional discounts"""
        discount_ranges = self.config.DATA_CONFIG['non_campaign_discount_ranges']
        
        if channel in discount_ranges:
            discount_rate = random.uniform(*discount_ranges[channel])
            discount_amount = base_price * discount_rate
            final_price = base_price - discount_amount
            return discount_amount, final_price
        
        return 0, base_price
    
    def calculate_attribution(self, booking_date, campaign, customer):
        """
        Calculate attribution score and incrementality using improved model
        
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
        fatigue_factor = self._calculate_fatigue_factor(customer, booking_date)
        
        # Final attribution score
        final_attribution = base_attribution * segment_bonus * channel_bonus * fatigue_factor
        final_attribution = max(0.0, min(1.0, final_attribution))
        
        # Determine incrementality
        is_incremental = final_attribution > params['incremental_threshold']
        
        return final_attribution, is_incremental
    
    def _calculate_fatigue_factor(self, customer, booking_date):
        """Calculate customer fatigue factor based on recent campaign exposures"""
        recent_exposures = [
            exp for exp in customer['campaign_exposures'] 
            if (booking_date - exp['exposure_date']).days <= 30
        ]
        
        if len(recent_exposures) == 0:
            return 1.0
        
        # Reduce effectiveness based on number of recent exposures
        fatigue_factor = max(0.3, 1.0 - (len(recent_exposures) * 0.1))
        return fatigue_factor
    
    def calculate_price_elasticity_impact(self, customer, base_price, market_conditions=None):
        """
        Calculate how price sensitivity affects booking probability
        
        Args:
            customer: Customer profile
            base_price: Base price for the booking
            market_conditions: Optional market condition modifiers
        
        Returns:
            Price elasticity impact factor
        """
        # Base price sensitivity from customer profile
        price_sensitivity = customer['price_sensitivity']
        
        # Price level impact (higher prices = higher sensitivity impact)
        price_level_factor = min(2.0, base_price / 200)  # Normalize around $200 base
        
        # Segment-specific elasticity
        segment_elasticity = {
            'Early_Planner': 0.8,    # Less price sensitive (planning ahead)
            'Last_Minute': 0.6,      # Less price sensitive (need rooms now)
            'Flexible': 1.2          # More price sensitive (can adjust plans)
        }
        
        segment_factor = segment_elasticity.get(customer['segment'], 1.0)
        
        # Combine factors
        elasticity_impact = price_sensitivity * price_level_factor * segment_factor
        
        # Convert to booking probability impact (higher elasticity = lower probability)
        booking_probability_factor = max(0.1, 1.0 - (elasticity_impact * 0.3))
        
        return booking_probability_factor