"""
Hotel Booking Data Generator - Main Orchestration Class

This is the main orchestrator that coordinates all other components
to generate comprehensive hotel booking datasets.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
from config import HotelBusinessConfig

from .campaign_generator import CampaignGenerator
from .customer_generator import CustomerGenerator
from .pricing_engine import PricingEngine
from .inventory_manager import InventoryManager
from .booking_logic import BookingLogic
from processors.data_processors import DataProcessor


class ConfigurableHotelBookingGenerator:
    """
    Main orchestrator class for hotel booking data generation.
    Coordinates all components to produce realistic booking datasets.
    """
    
    def __init__(self, config=None):
        """Initialize with configuration object"""
        self.config = config or HotelBusinessConfig()
        
        # Initialize counters
        self.campaign_counter = 1000
        self.booking_counter = 10000  
        self.customer_counter = 1000
        
        # Initialize component generators
        self.campaign_generator = CampaignGenerator(self.config, self.campaign_counter)
        self.customer_generator = CustomerGenerator(self.config, self.customer_counter)
        self.pricing_engine = PricingEngine(self.config)
        self.inventory_manager = InventoryManager(self.config)
        self.booking_logic = BookingLogic(self.config)
        self.data_processor = DataProcessor(self.config)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        print(f"🏨 Initialized Hotel Booking Generator")
        print(f"   📅 Simulation period: {min(self.config.SIMULATION_YEARS)}-{max(self.config.SIMULATION_YEARS)}")
        print(f"   🏖️  Operational months: {self.config.OPERATIONAL_MONTHS}")
        print(f"   👥 Target customers: {self.config.DATA_CONFIG['total_customers']:,}")
        print(f"   📊 Target channel split: {self.config.TARGET_CONNECTED_AGENT_SHARE:.0%}/{self.config.TARGET_ONLINE_DIRECT_SHARE:.0%}")
    
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
    
    def generate_bookings(self, baseline_demand, campaigns, customers):
        """Generate bookings using all component systems"""
        bookings = []
        attribution_data = []
        
        # Create campaign lookup with configured influence periods
        campaign_lookup = self.campaign_generator.create_campaign_lookup(campaigns)
        
        total_bookings_generated = 0
        rejected_out_of_season = 0 # debugging variable
        rejected_inventory = 0 # debugging variable
        monthly_rejection_stats = {} # debugging variable

        for date, base_demand in baseline_demand.items():
            if date.year > max(self.config.SIMULATION_YEARS):
                continue
            
            # Apply configured external shocks
            if random.random() < self.config.DATA_CONFIG['external_shock_probability']:
                shock_factor = random.uniform(*self.config.DATA_CONFIG['shock_impact_range'])
                base_demand *= shock_factor
            
            num_bookings = max(1, int(np.random.poisson(base_demand)))
            daily_attempts = 0 # debugging variable
            daily_rejections = 0 # debugging variable

            for _ in range(num_bookings):
                customer = random.choice(customers)
                
                # Campaign eligibility and selection
                selected_campaign, is_promotional = self.campaign_generator.select_campaign_for_booking(
                    date, customer, campaign_lookup
                )
                
                # Channel determination
                channel = self.customer_generator.determine_booking_channel(customer, selected_campaign)
                
                # Room type selection
                room_type = self.booking_logic.select_room_type()
                
                # Generate stay dates
                stay_start_date, stay_end_date, stay_length = self.booking_logic.generate_stay_dates(
                    date, customer, selected_campaign
                )

                if stay_start_date.month not in self.config.OPERATIONAL_MONTHS:
                    rejected_out_of_season += 1
                    daily_rejections += 1
                    continue
                
                if stay_end_date.month not in self.config.OPERATIONAL_MONTHS:
                    rejected_out_of_season += 1
                    daily_rejections += 1
                    continue

                # Check if stay is in operational season
                if stay_start_date.month not in self.config.OPERATIONAL_MONTHS:
                    continue  # Skip this booking attempt, try next one
                
                # Check if stay end is also in operational season
                if stay_end_date.month not in self.config.OPERATIONAL_MONTHS:
                    continue  # Skip this booking attempt, try next one

                # Check inventory and get acceptance probability
                acceptance_probability = self.inventory_manager.get_acceptance_probability(
                    stay_start_date, room_type, selected_campaign
                )
                
                # If inventory management rejects the booking, skip this iteration
                if random.random() > acceptance_probability:
                    rejected_inventory += 1
                    daily_rejections += 1
                    continue
                
                # Reserve inventory
                self.inventory_manager.reserve_inventory(stay_start_date, stay_end_date, room_type)
                
                # Calculate pricing
                base_price, final_price, discount_amount, campaign_id_final = self.pricing_engine.calculate_pricing(
                    stay_start_date, room_type, customer, selected_campaign, channel, is_promotional
                )
                
                # Calculate attribution
                attribution_score, is_incremental = self.pricing_engine.calculate_attribution(
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

        month_key = date.strftime('%Y-%m')
        if month_key not in monthly_rejection_stats:
            monthly_rejection_stats[month_key] = {'attempts': 0, 'rejections': 0, 'out_of_season': 0}
        
        monthly_rejection_stats[month_key]['attempts'] += daily_attempts
        monthly_rejection_stats[month_key]['rejections'] += daily_rejections        
        
        # Update campaign performance
        self.campaign_generator.update_campaign_performance(campaigns, bookings)
        print(f"📊 BOOKING GENERATION STATISTICS")
        print(f"   Out-of-season rejections: {rejected_out_of_season:,}")
        print(f"   Inventory rejections: {rejected_inventory:,}")
        print(f"   Total successful bookings: {total_bookings_generated:,}")
        print(f"✅ Generated {total_bookings_generated} total bookings")
        return bookings, attribution_data
    
    def validate_data(self, bookings, campaigns):
        """Validate generated data against configuration targets"""
        return self.data_processor.validate_data(bookings, campaigns)
    
    def save_data(self, bookings, campaigns, customers, attribution_data, baseline_demand):
        """Save all generated data"""
        return self.data_processor.save_data(
            bookings, campaigns, customers, attribution_data, baseline_demand
        )
    
    def generate_all_data(self):
        """Main method to generate all data using configuration"""
        print(f"\n🚀 Starting data generation...")
        
        # Generate all components
        campaigns = self.campaign_generator.generate_campaigns()
        baseline_demand = self.generate_baseline_demand()
        customers = self.customer_generator.generate_customers()
        bookings, attribution_data = self.generate_bookings(baseline_demand, campaigns, customers)
        
        # Apply post-processing
        print(f"\n📋 Applying cancellation logic...")
        bookings = self.data_processor.apply_cancellation_logic(bookings)
        
        # Validate and save
        self.validate_data(bookings, campaigns)
        self.save_data(bookings, campaigns, customers, attribution_data, baseline_demand)
        
        print(f"\n🎉 Data generation complete!")
        return bookings, campaigns, customers, attribution_data, baseline_demand