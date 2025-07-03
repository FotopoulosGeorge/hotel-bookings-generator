"""
Hotel Booking Data Generator - Main Orchestration Class

This is the main orchestrator that coordinates all other components
to generate comprehensive hotel booking datasets.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
from collections import defaultdict
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
        
        # Initialize monthly booking tracker for capacity management
        self.monthly_bookings = defaultdict(lambda: defaultdict(int))
        self.monthly_capacity_targets = self._calculate_monthly_capacity_targets()
        
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Determine operation mode for display
        operation_mode = getattr(self.config, 'OPERATION_MODE', 'seasonal')
        
        print(f"🏨 Initialized Hotel Booking Generator")
        print(f"   🏢 Operation mode: {operation_mode}")
        print(f"   📅 Simulation period: {min(self.config.SIMULATION_YEARS)}-{max(self.config.SIMULATION_YEARS)}")
        print(f"   🏖️  Operational months: {self.config.OPERATIONAL_MONTHS}")
        print(f"   👥 Target customers: {self.config.DATA_CONFIG['total_customers']:,}")
        print(f"   📊 Target channel split: {self.config.TARGET_CONNECTED_AGENT_SHARE:.0%}/{self.config.TARGET_ONLINE_DIRECT_SHARE:.0%}")
    
    def _calculate_monthly_capacity_targets(self):
        """Calculate target booking distribution to prevent spikes"""
        targets = {}
        
        # Check if we have configured distribution
        if hasattr(self.config, 'SEASONAL_STAY_DISTRIBUTION'):
            return self.config.SEASONAL_STAY_DISTRIBUTION
        
        # For seasonal hotels, calculate based on operational months
        if hasattr(self.config, 'OPERATIONAL_MONTHS') and len(self.config.OPERATIONAL_MONTHS) < 12:
            total_days = 0
            monthly_days = {}
            
            for month in self.config.OPERATIONAL_MONTHS:
                # Approximate days per month
                if month in [4, 6, 9, 11]:
                    days = 30
                elif month == 2:
                    days = 28
                else:
                    days = 31
                monthly_days[month] = days
                total_days += days
            
            # Set proportional targets with adjustments
            for month, days in monthly_days.items():
                base_proportion = days / total_days
                
                # Adjust to prevent May spike
                if month == 5:
                    targets[month] = base_proportion * 0.9  # Reduce May
                elif month in [6, 7]:
                    targets[month] = base_proportion * 1.1  # Increase June-July
                else:
                    targets[month] = base_proportion
        else:
            # Year-round: use demand multipliers
            for month in range(1, 13):
                targets[month] = self.config.SEASONAL_DEMAND_MULTIPLIERS.get(month, 1.0)
        
        # Normalize targets
        total = sum(targets.values())
        if total > 0:
            for month in targets:
                targets[month] = targets[month] / total
        
        return targets
    
    def generate_baseline_demand(self):
        """Generate baseline demand using configured parameters"""
        baseline_demand = {}
        
        for year in self.config.SIMULATION_YEARS:
            for month in range(1, 13):
                # Hotels accept bookings year-round
                if month not in self.config.BOOKING_ACCEPTANCE_MONTHS:
                    continue
                    
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
    
    def _should_redirect_booking(self, stay_month, year):
        """Check if booking should be redirected to prevent monthly spikes"""
        # Only redirect for seasonal hotels
        if not hasattr(self.config, 'OPERATIONAL_MONTHS') or len(self.config.OPERATIONAL_MONTHS) >= 12:
            return False
        
        # Get current distribution
        total_bookings = sum(self.monthly_bookings[year].values())
        
        # Need minimum bookings before redirecting
        if total_bookings < 100:
            return False
        
        current_month_bookings = self.monthly_bookings[year][stay_month]
        current_ratio = current_month_bookings / total_bookings if total_bookings > 0 else 0
        
        # Get target ratio
        target_ratio = self.monthly_capacity_targets.get(stay_month, 0.2)
        
        # Redirect if this month has >140% of its target share
        return current_ratio > (target_ratio * 1.4)
    
    def generate_bookings(self, baseline_demand, campaigns, customers):
        """Generate bookings using all component systems"""
        bookings = []
        attribution_data = []
        
        # Create campaign lookup with configured influence periods
        campaign_lookup = self.campaign_generator.create_campaign_lookup(campaigns)
        
        total_bookings_generated = 0
        rejected_stays = 0  # For seasonal hotels
        capacity_redirects = 0  # Track redirections

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
                
                # Campaign eligibility and selection
                selected_campaign, is_promotional = self.campaign_generator.select_campaign_for_booking(
                    date, customer, campaign_lookup
                )
                
                # Channel determination
                channel = self.customer_generator.determine_booking_channel(customer, selected_campaign)
                
                # Room type selection
                room_type = self.booking_logic.select_room_type()
                
                # Generate stay dates (handles both seasonal and year-round)
                stay_start_date, stay_end_date, stay_length = self.booking_logic.generate_stay_dates(
                    date, customer, selected_campaign
                )
                
                # For seasonal hotels, verify stays are within operational months
                if hasattr(self.config, 'OPERATION_MODE') and self.config.OPERATION_MODE == 'seasonal':
                    if (stay_start_date.month not in self.config.OPERATIONAL_MONTHS or 
                        stay_end_date.month not in self.config.OPERATIONAL_MONTHS):
                        rejected_stays += 1
                        continue
                
                # Check if we should redirect to prevent spikes
                if self._should_redirect_booking(stay_start_date.month, stay_start_date.year):
                    # Try to find alternative month
                    alternative_found = False
                    for _ in range(3):  # Try up to 3 times
                        alt_start, alt_end, alt_length = self.booking_logic.generate_stay_dates(
                            date, customer, selected_campaign
                        )
                        
                        if not self._should_redirect_booking(alt_start.month, alt_start.year):
                            stay_start_date = alt_start
                            stay_end_date = alt_end
                            stay_length = alt_length
                            capacity_redirects += 1
                            alternative_found = True
                            break
                    
                    if not alternative_found:
                        # Use original dates if no alternative found
                        pass

                # Check inventory and get acceptance probability
                acceptance_probability = self.inventory_manager.get_acceptance_probability(
                    stay_start_date, room_type, selected_campaign
                )
                
                # If inventory management rejects the booking, skip this iteration
                if random.random() > acceptance_probability:
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
                
                # Track monthly distribution
                self.monthly_bookings[stay_start_date.year][stay_start_date.month] += 1
                
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
        self.campaign_generator.update_campaign_performance(campaigns, bookings)
        
        # Print generation statistics
        print(f"\n📊 BOOKING GENERATION STATISTICS")
        if hasattr(self.config, 'OPERATION_MODE') and self.config.OPERATION_MODE == 'seasonal':
            print(f"   Out-of-season stay rejections: {rejected_stays:,}")
        print(f"   Capacity-based redirections: {capacity_redirects:,}")
        print(f"   Total successful bookings: {total_bookings_generated:,}")
        
        # Print monthly distribution for seasonal hotels
        if hasattr(self.config, 'OPERATIONAL_MONTHS') and len(self.config.OPERATIONAL_MONTHS) < 12:
            print(f"\n📅 Monthly Stay Distribution:")
            for year in sorted(self.monthly_bookings.keys()):
                year_total = sum(self.monthly_bookings[year].values())
                if year_total > 0:
                    print(f"   Year {year}:")
                    for month in self.config.OPERATIONAL_MONTHS:
                        count = self.monthly_bookings[year][month]
                        percentage = (count / year_total * 100) if year_total > 0 else 0
                        target = self.monthly_capacity_targets.get(month, 0) * 100
                        print(f"     Month {month}: {count:,} stays ({percentage:.1f}% actual vs {target:.1f}% target)")
                        
        distribution_stats = self.booking_logic.get_distribution_stats()
    
        print(f"\n📅 STAY DISTRIBUTION ANALYSIS")
        print("=" * 50)
        print(f"   Total stays generated: {distribution_stats['total_stays']:,}")
        
        target_dist = distribution_stats['target_distribution']
        current_dist = distribution_stats['current_distribution']
        
        print(f"\n   Target vs Actual Distribution:")
        month_names = {5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 
                    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 
                    10: 'Oct', 11: 'Nov', 12: 'Dec'}
        
        for month in sorted(target_dist.keys()):
            target_pct = target_dist[month] * 100
            actual_pct = current_dist.get(month, 0) * 100
            deviation = actual_pct - target_pct
            
            month_name = month_names.get(month, f'Month {month}')
            status = "✅" if abs(deviation) < 3 else "⚠️" if abs(deviation) < 6 else "❌"
            
            print(f"   {month_name}: {actual_pct:5.1f}% (target: {target_pct:4.1f}%, deviation: {deviation:+5.1f}%) {status}")
        
        # Check for May spike specifically
        may_actual = current_dist.get(5, 0) * 100
        may_target = target_dist.get(5, 0) * 100
        
        if may_actual > 25:
            print(f"\n   ❌ MAY SPIKE DETECTED: {may_actual:.1f}% (should be ~{may_target:.1f}%)")
        elif may_actual > may_target + 5:
            print(f"\n   ⚠️ May slightly elevated: {may_actual:.1f}% (target: {may_target:.1f}%)")
        else:
            print(f"\n   ✅ May distribution looks good: {may_actual:.1f}% (target: {may_target:.1f}%)")

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
        
        # Load the generated data
        df = pd.read_csv('output/historical_bookings.csv')
        df['stay_start_date'] = pd.to_datetime(df['stay_start_date'])
        df['booking_date'] = pd.to_datetime(df['booking_date'])

        # Check STAY month distribution (this is what matters)
        df['stay_month'] = df['stay_start_date'].dt.month
        print("STAY Month Distribution:")
        print(df['stay_month'].value_counts().sort_index())
        print("\nAs percentages:")
        print(df['stay_month'].value_counts(normalize=True).sort_index() * 100)

        # Check BOOKING month distribution (for comparison)
        df['booking_month'] = df['booking_date'].dt.month
        print("\n\nBOOKING Month Distribution:")
        print(df['booking_month'].value_counts().sort_index())

        print(f"\n🎉 Data generation complete!")
        return bookings, campaigns, customers, attribution_data, baseline_demand
