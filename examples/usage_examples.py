"""
Usage Examples for Hotel Booking Generator

This file demonstrates various ways to use the modular hotel booking generator,
from basic usage to advanced customization scenarios.
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HotelBusinessConfig
from generators import ConfigurableHotelBookingGenerator
from generators.campaign_generator import CampaignGenerator
from generators.customer_generator import CustomerGenerator
from generators.pricing_engine import PricingEngine
from generators.inventory_manager import InventoryManager
from scenarios import create_test_scenarios


def example_basic_generation():
    """Example 1: Basic data generation using default configuration"""
    print("🏨 Example 1: Basic Data Generation")
    print("-" * 50)
    
    # Create default configuration
    config = HotelBusinessConfig()
    
    # Initialize generator
    generator = ConfigurableHotelBookingGenerator(config)
    
    # Generate all data
    bookings, campaigns, customers, attribution_data, baseline_demand = generator.generate_all_data()
    
    print(f"✅ Generated:")
    print(f"   - {len(bookings):,} bookings")
    print(f"   - {len(campaigns)} campaigns") 
    print(f"   - {len(customers):,} customers")
    print(f"   - {len(attribution_data):,} attribution records")
    print()


def example_scenario_usage():
    """Example 2: Using predefined scenarios"""
    print("🎯 Example 2: Scenario-Based Generation")
    print("-" * 50)
    
    # Load available scenarios
    scenarios = create_test_scenarios()
    
    # Generate data for luxury hotel scenario
    luxury_config = scenarios['luxury']
    generator = ConfigurableHotelBookingGenerator(luxury_config)
    
    # Generate smaller dataset for demonstration
    luxury_config.DATA_CONFIG['total_customers'] = 1000
    luxury_config.SIMULATION_YEARS = [2024]  # Single year
    
    bookings, campaigns, customers, attribution_data, baseline_demand = generator.generate_all_data()
    
    # Analyze luxury hotel characteristics
    import pandas as pd
    df = pd.DataFrame(bookings)
    
    print(f"Luxury Hotel Characteristics:")
    print(f"   - Average price: ${df['final_price'].mean():.2f}")
    print(f"   - Premium room share: {(df['room_type'] == 'Premium').mean():.1%}")
    print(f"   - Early planner share: {(df['customer_segment'] == 'Early_Planner').mean():.1%}")
    print()


def example_custom_configuration():
    """Example 3: Custom configuration creation"""
    print("⚙️ Example 3: Custom Configuration")
    print("-" * 50)
    
    # Start with base configuration
    config = HotelBusinessConfig()
    
    # Customize for a boutique hotel
    config.BASE_PRICES = {
        'Standard': 180,
        'Deluxe': 250, 
        'Suite': 350,
        'Premium': 450
    }
    
    # Adjust customer segments for boutique experience
    config.CUSTOMER_SEGMENTS['Early_Planner']['market_share'] = 0.65  # More planners
    config.CUSTOMER_SEGMENTS['Last_Minute']['market_share'] = 0.15   # Fewer last-minute
    config.CUSTOMER_SEGMENTS['Flexible']['market_share'] = 0.20
    
    # Lower volume, higher quality
    config.DATA_CONFIG['base_daily_demand'] = 20
    config.DATA_CONFIG['total_customers'] = 2000
    
    # Generate data
    generator = ConfigurableHotelBookingGenerator(config)
    bookings, campaigns, customers, attribution_data, baseline_demand = generator.generate_all_data()
    
    print(f"Boutique Hotel Results:")
    print(f"   - Total bookings: {len(bookings):,}")
    print(f"   - Average daily demand: ~{len(bookings) / 365:.1f}")
    print()


def example_component_usage():
    """Example 4: Using individual components"""
    print("🔧 Example 4: Individual Component Usage")
    print("-" * 50)
    
    config = HotelBusinessConfig()
    
    # Use campaign generator independently
    campaign_gen = CampaignGenerator(config)
    campaigns = campaign_gen.generate_campaigns()
    
    print(f"Generated {len(campaigns)} campaigns:")
    for campaign in campaigns[:3]:  # Show first 3
        print(f"   - {campaign['campaign_id']}: {campaign['campaign_type']} "
              f"({campaign['discount_percentage']:.1%} discount)")
    
    # Use customer generator independently
    customer_gen = CustomerGenerator(config, customer_counter_start=5000)
    config.DATA_CONFIG['total_customers'] = 100  # Small sample
    customers = customer_gen.generate_customers()
    
    print(f"\nGenerated {len(customers)} customers:")
    segments = {}
    for customer in customers:
        segment = customer['segment']
        segments[segment] = segments.get(segment, 0) + 1
    
    for segment, count in segments.items():
        print(f"   - {segment}: {count} customers")
    
    # Use pricing engine independently
    pricing_engine = PricingEngine(config)
    stay_date = datetime(2024, 7, 15)  # Peak season
    base_price = pricing_engine.get_base_price_for_date(stay_date, 'Standard')
    
    print(f"\nPricing for July 15, 2024 (Standard room): ${base_price}")
    print()


def example_custom_pricing_strategy():
    """Example 5: Custom pricing strategy implementation"""
    print("💰 Example 5: Custom Pricing Strategy")
    print("-" * 50)
    
    from generators.pricing_engine import PricingEngine
    
    class DynamicPricingEngine(PricingEngine):
        """Custom pricing engine with dynamic weekend pricing"""
        
        def get_base_price_for_date(self, date, room_type):
            # Get standard base price
            base_price = super().get_base_price_for_date(date, room_type)
            
            # Apply weekend premium
            if date.weekday() in [4, 5, 6]:  # Fri, Sat, Sun
                base_price *= 1.25  # 25% weekend premium
            
            # Apply holiday premium (example: July 4th week)
            if date.month == 7 and 1 <= date.day <= 7:
                base_price *= 1.4  # 40% holiday premium
            
            return base_price
    
    # Use custom pricing engine
    config = HotelBusinessConfig()
    generator = ConfigurableHotelBookingGenerator(config)
    
    # Replace pricing engine with custom one
    generator.pricing_engine = DynamicPricingEngine(config)
    
    # Test pricing for different dates
    test_dates = [
        datetime(2024, 7, 1),   # Monday, holiday week
        datetime(2024, 7, 6),   # Saturday, holiday week  
        datetime(2024, 8, 15),  # Thursday, regular
        datetime(2024, 8, 17)   # Saturday, regular
    ]
    
    print("Custom pricing examples:")
    for date in test_dates:
        price = generator.pricing_engine.get_base_price_for_date(date, 'Standard')
        day_type = "Weekend" if date.weekday() >= 4 else "Weekday"
        holiday = "Holiday Week" if date.month == 7 and date.day <= 7 else "Regular"
        print(f"   - {date.strftime('%Y-%m-%d')} ({day_type}, {holiday}): ${price:.2f}")
    print()


def example_inventory_analysis():
    """Example 6: Inventory management and analysis"""
    print("📊 Example 6: Inventory Analysis")
    print("-" * 50)
    
    config = HotelBusinessConfig()
    
    # Create inventory manager
    inventory_manager = InventoryManager(config)
    
    # Simulate some bookings
    from datetime import datetime, timedelta
    
    peak_date = datetime(2024, 7, 15)  # Peak season
    
    # Simulate bookings for peak date
    for i in range(55):  # More than standard capacity (50)
        inventory_manager.reserve_inventory(
            peak_date, 
            peak_date + timedelta(days=2), 
            'Standard'
        )
    
    # Check acceptance probability for new booking
    acceptance_prob = inventory_manager.get_acceptance_probability(peak_date, 'Standard')
    
    print(f"Peak date ({peak_date.strftime('%Y-%m-%d')}) analysis:")
    print(f"   - Standard room reservations: 55 (capacity: 50)")
    print(f"   - Acceptance probability for new booking: {acceptance_prob:.1%}")
    
    # Get overbooking statistics
    overbooking_stats = inventory_manager.get_overbooking_stats()
    print(f"   - Overall overbooking rate: {overbooking_stats['overall_overbooking_rate']:.1%}")
    
    # Get revenue optimization insights
    insights = inventory_manager.get_revenue_optimization_insights()
    print(f"   - High demand periods identified: {len(insights['high_demand_periods'])}")
    print()


def example_attribution_analysis():
    """Example 7: Attribution modeling analysis"""
    print("🎯 Example 7: Attribution Analysis")
    print("-" * 50)
    
    config = HotelBusinessConfig()
    pricing_engine = PricingEngine(config)
    customer_gen = CustomerGenerator(config)
    campaign_gen = CampaignGenerator(config)
    
    # Create sample customer
    config.DATA_CONFIG['total_customers'] = 1
    customers = customer_gen.generate_customers()
    customer = customers[0]
    
    # Create sample campaign
    campaigns = campaign_gen.generate_campaigns()
    early_booking_campaigns = [c for c in campaigns if c['campaign_type'] == 'Early_Booking']
    if early_booking_campaigns:
        campaign = early_booking_campaigns[0]
        
        # Test attribution for different time periods
        booking_dates = [
            campaign['start_date'],
            campaign['start_date'] + timedelta(days=7),
            campaign['start_date'] + timedelta(days=30),
            campaign['start_date'] + timedelta(days=60)
        ]
        
        print("Attribution analysis for Early Booking campaign:")
        for booking_date in booking_dates:
            if booking_date <= campaign['end_date'] + timedelta(days=120):  # Within influence period
                attribution_score, is_incremental = pricing_engine.calculate_attribution(
                    booking_date, campaign, customer
                )
                days_since_start = (booking_date - campaign['start_date']).days
                print(f"   - Day {days_since_start:2d}: Attribution={attribution_score:.3f}, "
                      f"Incremental={'Yes' if is_incremental else 'No'}")
    print()


def example_performance_comparison():
    """Example 8: Performance comparison across scenarios"""
    print("🏆 Example 8: Scenario Performance Comparison")
    print("-" * 50)
    
    scenarios = create_test_scenarios()
    results = {}
    
    # Test subset of scenarios for speed
    test_scenarios = ['standard', 'luxury', 'budget']
    
    for scenario_name in test_scenarios:
        print(f"Testing {scenario_name} scenario...")
        
        config = scenarios[scenario_name]
        config.DATA_CONFIG['total_customers'] = 500  # Smaller for speed
        config.SIMULATION_YEARS = [2024]  # Single year
        
        generator = ConfigurableHotelBookingGenerator(config)
        bookings, campaigns, customers, attribution_data, baseline_demand = generator.generate_all_data()
        
        import pandas as pd
        df = pd.DataFrame(bookings)
        
        results[scenario_name] = {
            'total_bookings': len(bookings),
            'avg_price': df['final_price'].mean(),
            'cancellation_rate': df['is_cancelled'].mean(),
            'campaign_participation': (df['campaign_id'].notna()).mean(),
            'premium_room_share': (df['room_type'] == 'Premium').mean()
        }
    
    # Display comparison
    print("\nScenario Comparison Results:")
    print(f"{'Metric':<20} {'Standard':<12} {'Luxury':<12} {'Budget':<12}")
    print("-" * 60)
    
    metrics = [
        ('Avg Price ($)', 'avg_price', '${:.0f}'),
        ('Cancellation %', 'cancellation_rate', '{:.1%}'),
        ('Campaign Part. %', 'campaign_participation', '{:.1%}'),
        ('Premium Share %', 'premium_room_share', '{:.1%}')
    ]
    
    for metric_name, metric_key, format_str in metrics:
        row = f"{metric_name:<20}"
        for scenario in test_scenarios:
            value = results[scenario][metric_key]
            row += f" {format_str.format(value):<11}"
        print(row)
    print()


def main():
    """Run all examples"""
    print("🎨 Hotel Booking Generator - Usage Examples")
    print("=" * 60)
    
    examples = [
        example_basic_generation,
        example_scenario_usage,
        example_custom_configuration,
        example_component_usage,
        example_custom_pricing_strategy,
        example_inventory_analysis,
        example_attribution_analysis,
        example_performance_comparison
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("🎉 All examples completed!")


if __name__ == "__main__":
    main()