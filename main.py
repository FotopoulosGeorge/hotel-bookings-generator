"""
Hotel Booking Data Generator - Main Execution Script

Run this script to generate hotel booking datasets using different scenarios.
"""

import sys
import argparse
from datetime import datetime
from config import HotelBusinessConfig
from data_generator import ConfigurableHotelBookingGenerator
from scenarios import create_test_scenarios, get_scenario_description, validate_scenario, print_scenario_comparison


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Generate hotel booking datasets with configurable scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with default 'standard' scenario
  python main.py --scenario luxury        # Run luxury hotel scenario
  python main.py --list-scenarios         # List all available scenarios
  python main.py --compare-scenarios      # Compare all scenario configurations
  python main.py --validate-only          # Just validate configurations without generating data
        """
    )
    
    parser.add_argument(
        '--scenario', 
        type=str, 
        default='standard',
        help='Scenario to run (default: standard)'
    )
    
    parser.add_argument(
        '--list-scenarios',
        action='store_true',
        help='List all available scenarios and exit'
    )
    
    parser.add_argument(
        '--compare-scenarios',
        action='store_true',
        help='Compare scenario configurations and exit'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate scenario configurations without generating data'
    )
    
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='',
        help='Prefix for output files (e.g., "luxury_" will create "luxury_historical_bookings.csv")'
    )
    
    args = parser.parse_args()
    
    # Load scenarios
    scenarios = create_test_scenarios()
    
    # Handle list scenarios
    if args.list_scenarios:
        print("🎯 Available Test Scenarios:")
        print("=" * 80)
        for name in scenarios.keys():
            description = get_scenario_description(name)
            print(f"• {name:15}: {description}")
        print("=" * 80)
        return
    
    # Handle compare scenarios
    if args.compare_scenarios:
        print_scenario_comparison(scenarios)
        return
    
    # Validate scenario choice
    if args.scenario not in scenarios:
        print(f"❌ Error: Scenario '{args.scenario}' not found.")
        print(f"Available scenarios: {', '.join(scenarios.keys())}")
        print("Use --list-scenarios to see descriptions.")
        sys.exit(1)
    
    # Get selected scenario configuration
    config = scenarios[args.scenario]
    
    # Validate configuration
    print(f"🔍 Validating scenario '{args.scenario}'...")
    if not validate_scenario(config, args.scenario):
        print(f"❌ Scenario validation failed!")
        sys.exit(1)
    
    # If validate-only, stop here
    if args.validate_only:
        print(f"✅ Validation complete - scenario '{args.scenario}' is valid!")
        return
    
    # Print scenario info
    print(f"\n🎯 Running scenario: {args.scenario.upper()}")
    print(f"📝 Description: {get_scenario_description(args.scenario)}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize generator
    generator = ConfigurableHotelBookingGenerator(config)
    
    # Update output file names if prefix provided
    if args.output_prefix:
        original_save_data = generator.save_data
        
        def save_data_with_prefix(bookings, campaigns, customers, attribution_data, baseline_demand):
            """Modified save_data method with prefix"""
            import pandas as pd
            import pickle
            
            prefix = args.output_prefix
            if not prefix.endswith('_'):
                prefix += '_'
            
            df_bookings = pd.DataFrame(bookings)
            df_bookings.to_csv(f'{prefix}historical_bookings.csv', index=False)
            
            df_campaigns = pd.DataFrame(campaigns)
            df_campaigns['target_segments'] = df_campaigns['target_segments'].apply(lambda x: ';'.join(x))
            df_campaigns['room_types_eligible'] = df_campaigns['room_types_eligible'].apply(lambda x: ';'.join(x))
            df_campaigns.to_csv(f'{prefix}campaigns_run.csv', index=False)
            
            df_customers = pd.DataFrame(customers)
            df_customers['booking_history'] = df_customers['booking_history'].apply(lambda x: ';'.join(x))
            df_customers.to_csv(f'{prefix}customer_segments.csv', index=False)
            
            df_attribution = pd.DataFrame(attribution_data)
            df_attribution.to_csv(f'{prefix}attribution_ground_truth.csv', index=False)
            
            with open(f'{prefix}baseline_demand_model.pkl', 'wb') as f:
                pickle.dump(baseline_demand, f)
            
            print(f"\n✅ Saved all data files with prefix '{prefix}':")
            print(f"   📄 {prefix}historical_bookings.csv ({len(bookings):,} records)")
            print(f"   📄 {prefix}campaigns_run.csv ({len(campaigns)} records)")
            print(f"   📄 {prefix}customer_segments.csv ({len(customers):,} records)")
            print(f"   📄 {prefix}attribution_ground_truth.csv ({len(attribution_data):,} records)")
            print(f"   📄 {prefix}baseline_demand_model.pkl")
        
        generator.save_data = save_data_with_prefix
    
    # Generate data
    try:
        bookings, campaigns, customers, attribution_data, baseline_demand = generator.generate_all_data()
        
        # Print summary
        print(f"\n📊 GENERATION SUMMARY")
        print(f"=" * 50)
        print(f"Scenario: {args.scenario}")
        print(f"Total bookings: {len(bookings):,}")
        print(f"Total campaigns: {len(campaigns)}")
        print(f"Total customers: {len(customers):,}")
        print(f"Cancelled bookings: {sum(1 for b in bookings if b.get('is_cancelled', False)):,}")
        print(f"Overbooked bookings: {sum(1 for b in bookings if b.get('is_overbooked', False)):,}")
        print(f"Campaign bookings: {sum(1 for b in bookings if b.get('campaign_id')):,}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"=" * 50)
        
    except Exception as e:
        print(f"❌ Error during data generation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_scenario_comparison():
    """Utility function to run all scenarios and compare outputs"""
    scenarios = create_test_scenarios()
    
    print("🔄 Running comparison across all scenarios...")
    print("This will generate data for each scenario with prefixed filenames")
    
    results = {}
    
    for scenario_name, config in scenarios.items():
        print(f"\n🎯 Processing scenario: {scenario_name}")
        
        try:
            generator = ConfigurableHotelBookingGenerator(config)
            
            # Override save_data to use scenario prefix
            original_save_data = generator.save_data
            
            def save_data_with_scenario_prefix(bookings, campaigns, customers, attribution_data, baseline_demand):
                import pandas as pd
                import pickle
                
                prefix = f"{scenario_name}_"
                
                df_bookings = pd.DataFrame(bookings)
                df_bookings.to_csv(f'{prefix}historical_bookings.csv', index=False)
                
                df_campaigns = pd.DataFrame(campaigns)
                df_campaigns['target_segments'] = df_campaigns['target_segments'].apply(lambda x: ';'.join(x))
                df_campaigns['room_types_eligible'] = df_campaigns['room_types_eligible'].apply(lambda x: ';'.join(x))
                df_campaigns.to_csv(f'{prefix}campaigns_run.csv', index=False)
                
                df_customers = pd.DataFrame(customers)
                df_customers['booking_history'] = df_customers['booking_history'].apply(lambda x: ';'.join(x))
                df_customers.to_csv(f'{prefix}customer_segments.csv', index=False)
                
                df_attribution = pd.DataFrame(attribution_data)
                df_attribution.to_csv(f'{prefix}attribution_ground_truth.csv', index=False)
                
                with open(f'{prefix}baseline_demand_model.pkl', 'wb') as f:
                    pickle.dump(baseline_demand, f)
            
            generator.save_data = save_data_with_scenario_prefix
            
            bookings, campaigns, customers, attribution_data, baseline_demand = generator.generate_all_data()
            
            # Store results for summary
            results[scenario_name] = {
                'total_bookings': len(bookings),
                'total_campaigns': len(campaigns),
                'cancelled_bookings': sum(1 for b in bookings if b.get('is_cancelled', False)),
                'overbooked_bookings': sum(1 for b in bookings if b.get('is_overbooked', False)),
                'campaign_bookings': sum(1 for b in bookings if b.get('campaign_id')),
                'avg_price': sum(b['final_price'] for b in bookings) / len(bookings)
            }
            
        except Exception as e:
            print(f"❌ Error processing scenario {scenario_name}: {str(e)}")
            results[scenario_name] = {'error': str(e)}
    
    # Print comparison summary
    print(f"\n📊 SCENARIO COMPARISON RESULTS")
    print(f"=" * 100)
    print(f"{'Scenario':<15} {'Bookings':<10} {'Campaigns':<10} {'Cancelled':<10} {'Overbooked':<10} {'Avg Price':<10}")
    print(f"-" * 100)
    
    for scenario_name, results_data in results.items():
        if 'error' in results_data:
            print(f"{scenario_name:<15} ERROR: {results_data['error']}")
        else:
            print(f"{scenario_name:<15} {results_data['total_bookings']:<10,} {results_data['total_campaigns']:<10} "
                  f"{results_data['cancelled_bookings']:<10,} {results_data['overbooked_bookings']:<10,} "
                  f"${results_data['avg_price']:<9.2f}")
    
    print(f"=" * 100)


if __name__ == "__main__":
    main()