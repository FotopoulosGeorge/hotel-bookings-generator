"""
Customer Generation Module

Handles creation of customer profiles with behavioral characteristics,
channel preferences, and segment-based attributes.
"""

import random
import numpy as np


class CustomerGenerator:
    """Handles generation of customer profiles and behavioral modeling"""
    
    def __init__(self, config, customer_counter_start=1000):
        self.config = config
        self.customer_counter = customer_counter_start
    
    def generate_customers(self):
        """Generate customer profiles using configuration"""
        customers = []
        total_customers = self.config.DATA_CONFIG['total_customers']
        
        # Prepare segment selection probabilities
        segments = list(self.config.CUSTOMER_SEGMENTS.keys())
        market_shares = [seg_data['market_share'] for seg_data in self.config.CUSTOMER_SEGMENTS.values()]
        
        # Normalize to ensure probabilities sum to exactly 1.0
        market_shares = np.array(market_shares)
        market_shares = market_shares / market_shares.sum()
        
        for i in range(total_customers):
            # Select segment based on configured market shares
            segment_choice = np.random.choice(segments, p=market_shares)
            segment_config = self.config.CUSTOMER_SEGMENTS[segment_choice]
            
            customer = self._create_customer_profile(segment_choice, segment_config)
            customers.append(customer)
            self.customer_counter += 1
        
        print(f"✅ Generated {len(customers):,} customer profiles")
        return customers
    
    def _create_customer_profile(self, segment, segment_config):
        """Create individual customer profile with segment-based characteristics"""
        # Channel preference selection
        channel_weights = np.array(list(segment_config['channel_preference_weights'].values()))
        channel_weights_normalized = channel_weights / channel_weights.sum()
        
        channel_preference = np.random.choice(
            list(segment_config['channel_preference_weights'].keys()),
            p=channel_weights_normalized
        )
        
        # Loyalty status selection
        loyalty_weights = np.array(list(segment_config['loyalty_distribution'].values()))
        loyalty_weights_normalized = loyalty_weights / loyalty_weights.sum()
        
        loyalty_status = np.random.choice(
            list(segment_config['loyalty_distribution'].keys()),
            p=loyalty_weights_normalized
        )
        
        customer = {
            'customer_id': f'CUST_{self.customer_counter}',
            'segment': segment,
            'price_sensitivity': self._generate_price_sensitivity(segment_config),
            'planning_horizon': self._generate_planning_horizon(segment_config),
            'channel_preference': channel_preference,
            'loyalty_status': loyalty_status,
            'campaign_exposures': [],  # Track campaign exposures for attribution
            'booking_history': []
        }
        
        return customer
    
    def _generate_price_sensitivity(self, segment_config):
        """Generate price sensitivity with some variation around segment baseline"""
        base_sensitivity = segment_config['price_sensitivity']
        # Add some individual variation (±10% of base)
        variation = np.random.normal(0, 0.1)
        return max(0.1, min(1.0, base_sensitivity + variation))
    
    def _generate_planning_horizon(self, segment_config):
        """Generate planning horizon within segment range"""
        return random.randint(*segment_config['advance_booking_days'])
    
    def determine_booking_channel(self, customer, selected_campaign):
        """Determine booking channel based on customer preferences and campaign"""
        if selected_campaign and selected_campaign['channel'] != 'Mixed':
            return selected_campaign['channel']
        
        # Use customer segment preferences
        segment_config = self.config.CUSTOMER_SEGMENTS[customer['segment']]
        channel_weights = np.array(list(segment_config['channel_preference_weights'].values()))
        channel_weights_normalized = channel_weights / channel_weights.sum()
        
        channel = np.random.choice(
            list(segment_config['channel_preference_weights'].keys()),
            p=channel_weights_normalized
        )
        
        return channel
    
    def get_customer_behavior_multiplier(self, customer, context):
        """
        Get behavioral multipliers for various contexts (pricing, cancellation, etc.)
        
        Args:
            customer: Customer profile
            context: Context for behavior ('pricing', 'cancellation', 'campaign_response')
        
        Returns:
            Behavioral multiplier
        """
        if context == 'pricing':
            return self._get_pricing_behavior_multiplier(customer)
        elif context == 'cancellation':
            return self._get_cancellation_behavior_multiplier(customer)
        elif context == 'campaign_response':
            return self._get_campaign_response_multiplier(customer)
        else:
            return 1.0
    
    def _get_pricing_behavior_multiplier(self, customer):
        """Get pricing behavior multiplier based on segment and loyalty"""
        segment_multipliers = {
            'Early_Planner': 0.95,    # 5% discount for advance booking
            'Last_Minute': 1.10,      # 10% premium for last-minute
            'Flexible': 1.00          # Standard pricing
        }
        
        loyalty_multipliers = {
            'Bronze': 1.0,
            'Silver': 0.97,  # 3% loyalty discount
            'Gold': 0.95     # 5% loyalty discount
        }
        
        segment_mult = segment_multipliers.get(customer['segment'], 1.0)
        loyalty_mult = loyalty_multipliers.get(customer['loyalty_status'], 1.0)
        
        return segment_mult * loyalty_mult
    
    def _get_cancellation_behavior_multiplier(self, customer):
        """Get cancellation probability multiplier"""
        # More price-sensitive customers more likely to cancel
        price_sensitivity_effect = 0.8 + (customer['price_sensitivity'] * 0.4)
        
        # Loyalty reduces cancellation probability
        loyalty_effects = {
            'Bronze': 1.0,
            'Silver': 0.9,
            'Gold': 0.8
        }
        loyalty_effect = loyalty_effects.get(customer['loyalty_status'], 1.0)
        
        return price_sensitivity_effect * loyalty_effect
    
    def _get_campaign_response_multiplier(self, customer):
        """Get campaign response probability multiplier"""
        # Price-sensitive customers more responsive to campaigns
        price_sensitivity_effect = 0.7 + (customer['price_sensitivity'] * 0.6)
        
        # Channel preference affects response
        channel_effects = {
            'Connected_Agent': 1.1,  # Agents better at selling campaigns
            'Online_Direct': 0.95    # Slightly lower online response
        }
        channel_effect = channel_effects.get(customer['channel_preference'], 1.0)
        
        return price_sensitivity_effect * channel_effect