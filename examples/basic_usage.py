"""
Basic Usage Example for Hotel Booking Data Generator

This example shows the most common usage patterns for generating 
and analyzing hotel booking data.
"""

import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HotelBusinessConfig
from data_generator import ConfigurableHotelBookingGenerator
from scenarios import create_test_scenarios
from utils import load_generated_data, analyze_booking_patterns, create_summary_report


def basic_generation_example():
    """Example 1: Basic data generation with default settings"""
    print("🚀 Example 1: Basic Data Generation")
    print("=" * 50)
    
    # Create generator with default configuration
    generator = ConfigurableHotelBookingGenerator()
    
    # Generate all data
    bookings, campaigns, customers, attribution_data, baseline_demand = generator.generate_all_data()
    
    print(f"✅ Generated {len(bookings)} bookings")
    print(f"✅ Generated {len(campaigns)} campaigns")
    print(f"✅ Generated {len(customers)} customers")
    
    return bookings, campaigns, customers, attribution_data, baseline_demand


def scenario_based_generation():
    """Example 2: Using pre-defined scenarios"""
    print("\n🎯 Example 2: Scenario-Based Generation")
    print("=" * 50)
    
    # Get available scenarios
    scenarios = create_test_scenarios()
    
    # Generate data for luxury hotel scenario
    luxury_config = scenarios['luxury']
    generator = ConfigurableHotelBookingGenerator(luxury_config)
    
    # Override save method to use luxury prefix
    original_save = generator.save_data
    def save_with_prefix(bookings, campaigns, customers, attribution_data, baseline_demand):
        import pandas as pd
        import pickle
        
        # Save with luxury_ prefix
        pd.DataFrame(bookings).to_csv('luxury_historical_bookings.csv', index=False)
        
        df_campaigns = pd.DataFrame(campaigns)
        df_campaigns['target_segments'] = df_campaigns['target_segments'].apply(lambda x: ';'.join(x))
        df_campaigns['room_types_eligible'] = df_campaigns['room_types_eligible'].apply(lambda x: ';'.join(x))
        df_campaigns.to_csv('luxury_campaigns_run.csv', index=False)
        
        pd.DataFrame(customers).to_csv('luxury_customer_segments.csv', index=False)
        pd.DataFrame(attribution_data).to_csv('luxury_attribution_ground_truth.csv', index=False)
        
        with open('luxury_baseline_demand_model.pkl', 'wb') as f:
            pickle.dump(baseline_demand, f)
        
        print("📁 Saved luxury hotel data files")
    
    generator.save_data = save_with_prefix
    
    # Generate luxury hotel data
    luxury_data = generator.generate_all_data()
    
    return luxury_data


def custom_configuration_example():
    """Example 3: Creating custom configuration"""
    print("\n⚙️ Example 3: Custom Configuration")
    print("=" * 50)
    
    # Start with default configuration
    config = HotelBusinessConfig()
    
    # Customize for a specific use case: High-end resort
    print("Creating custom high-end resort configuration...")
    
    # Higher base prices
    config.BASE_PRICES = {
        'Standard': 400,
        'Deluxe': 600, 
        'Suite': 900,
        'Premium': 1200
    }
    
    # Lower volume, higher quality
    config.DATA_CONFIG['base_daily_demand'] = 20
    config.DATA_CONFIG['total_customers'] = 2000
    
    # More early planners (luxury guests plan ahead)
    config.CUSTOMER_SEGMENTS['Early_Planner']['market_share'] = 0.65
    config.CUSTOMER_SEGMENTS['Last_Minute']['market_share'] = 0.15
    config.CUSTOMER_SEGMENTS['Flexible']['market_share'] = 0.20
    
    # Lower cancellation rates (more committed luxury guests)
    for segment in config.CANCELLATION_CONFIG:
        config.CANCELLATION_CONFIG[segment]['base_cancellation_rate'] *= 0.6
    
    # Conservative overbooking (protect luxury experience)
    config.OVERBOOKING_CONFIG['base_overbooking_rate'] = 0.03
    
    # Less aggressive promotions
    config.CONNECTED_AGENT_PROMO_RATE = 0.60
    config.ONLINE_DIRECT_PROMO_RATE = 0.50
    
    print("Custom configuration created with:")
    print(f"  - Standard room price: ${config.BASE_PRICES['Standard']}")
    print(f"  - Daily demand: {config.DATA_CONFIG['base_daily_demand']}")
    print(f"  - Early planner share: {config.CUSTOMER_SEGMENTS['Early_Planner']['market_share']:.0%}")
    print(f"  - Overbooking rate: {config.OVERBOOKING_CONFIG['base_overbooking_rate']:.1%}")
    
    # Generate data with custom configuration
    generator = ConfigurableHotelBookingGenerator(config)
    
    # Save with custom prefix
    original_save = generator.save_data
    def save_with_custom_prefix(bookings, campaigns, customers, attribution_data, baseline_demand):
        import pandas as pd
        import pickle
        
        pd.DataFrame(bookings).to_csv('custom_resort_historical_bookings.csv', index=False)
        
        df_campaigns = pd.DataFrame(campaigns)
        df_campaigns['target_segments'] = df_campaigns['target_segments'].apply(lambda x: ';'.join(x))
        df_campaigns['room_types_eligible'] = df_campaigns['room_types_eligible'].apply(lambda x: ';'.join(x))
        df_campaigns.to_csv('custom_resort_campaigns_run.csv', index=False)
        
        pd.DataFrame(customers).to_csv('custom_resort_customer_segments.csv', index=False)
        pd.DataFrame(attribution_data).to_csv('custom_resort_attribution_ground_truth.csv', index=False)
        
        with open('custom_resort_baseline_demand_model.pkl', 'wb') as f:
            pickle.dump(baseline_demand, f)
        
        print("📁 Saved custom resort data files")
    
    generator.save_data = save_with_custom_prefix
    
    custom_data = generator.generate_all_data()
    return custom_data


def data_analysis_example():
    """Example 4: Analyzing generated data"""
    print("\n📊 Example 4: Data Analysis")
    print("=" * 50)
    
    try:
        # Load previously generated data (assumes standard data exists)
        data = load_generated_data()  # No prefix = standard files
        
        if not data:
            print("⚠️ No data files found. Run basic generation first.")
            return
        
        # Analyze booking patterns
        analysis = analyze_booking_patterns(data['bookings'])
        
        print("Key metrics from analysis:")
        print(f"  📈 Total bookings: {analysis['total_bookings']:,}")
        print(f"  💰 Total revenue: ${analysis['total_revenue']:,.2f}")
        print(f"  🎯 Campaign participation: {analysis['campaign_participation_rate']:.1%}")
        print(f"  ❌ Cancellation rate: {analysis['cancellation_rate']:.1%}")
        print(f"  📋 Overbooking rate: {analysis['overbooking_rate']:.1%}")
        
        print("\nChannel distribution:")
        for channel, share in analysis['channel_distribution'].items():
            print(f"  {channel}: {share:.1%}")
        
        print("\nCustomer segment distribution:")
        for segment, share in analysis['segment_distribution'].items():
            print(f"  {segment}: {share:.1%}")
        
        # Create comprehensive report
        report = create_summary_report(data, 'analysis_report.txt')
        print(f"\n📄 Detailed report saved to 'analysis_report.txt'")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        print("Make sure you've generated data first using the basic example.")


def comparison_example():
    """Example 5: Comparing different scenarios"""
    print("\n🔄 Example 5: Scenario Comparison")
    print("=" * 50)
    
    from utils import compare_scenarios
    
    # List of scenario prefixes to compare
    scenario_prefixes = ['', 'luxury_', 'custom_resort_']  # Empty string = no prefix (standard)
    scenario_names = ['Standard', 'Luxury', 'Custom Resort']
    
    print("Attempting to compare scenarios...")
    
    # Try to load and compare data
    try:
        comparison_df = compare_scenarios(scenario_prefixes)
        
        if not comparison_df.empty:
            print("\n📊 Scenario Comparison Results:")
            print("-" * 60)
            
            # Print formatted comparison
            for idx, (scenario, row) in enumerate(comparison_df.iterrows()):
                name = scenario_names[idx] if idx < len(scenario_names) else scenario
                print(f"\n{name}:")
                if 'total_bookings' in row:
                    print(f"  Total bookings: {row['total_bookings']:,}")
                if 'total_revenue' in row:
                    print(f"  Total revenue: ${row['total_revenue']:,.2f}")
                if 'avg_booking_value' in row:
                    print(f"  Avg booking value: ${row['avg_booking_value']:.2f}")
                if 'campaign_participation_rate' in row:
                    print(f"  Campaign participation: {row['campaign_participation_rate']:.1%}")
        else:
            print("⚠️ No scenario data found for comparison.")
            print("Generate data for multiple scenarios first.")
    
    except Exception as e:
        print(f"❌ Error in comparison: {e}")


def main():
    """Run all examples"""
    print("🏨 Hotel Booking Data Generator - Usage Examples")
    print("=" * 60)
    
    # Example 1: Basic generation
    basic_data = basic_generation_example()
    
    # Example 2: Scenario-based generation  
    luxury_data = scenario_based_generation()
    
    # Example 3: Custom configuration
    custom_data = custom_configuration_example()
    
    # Example 4: Data analysis
    analysis_results = data_analysis_example()
    
    # Example 5: Scenario comparison
    comparison_example()
    
    print("\n🎉 All examples completed!")
    print("\nNext steps:")
    print("  1. Examine the generated CSV files")
    print("  2. Check the analysis report (analysis_report.txt)")
    print("  3. Try modifying configurations for your specific needs")
    print("  4. Use the utils module for custom analysis")


if __name__ == "__main__":
    main()