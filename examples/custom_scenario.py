"""
Custom Scenario Creation Example

This example demonstrates how to create completely custom business scenarios
for specific testing or analysis needs.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HotelBusinessConfig
from data_generator import ConfigurableHotelBookingGenerator
from utils import analyze_booking_patterns, create_summary_report, validate_data_quality


class CustomScenarioBuilder:
    """Helper class for building custom scenarios"""
    
    def __init__(self, base_config=None):
        self.config = base_config or HotelBusinessConfig()
    
    def create_covid_impact_scenario(self):
        """Create a scenario simulating COVID-19 impact on hotel bookings"""
        print("🦠 Creating COVID Impact Scenario")
        print("-" * 40)
        
        # Dramatically reduced demand
        self.config.DATA_CONFIG['base_daily_demand'] = 8  # 75% reduction
        
        # Much higher cancellation rates
        for segment in self.config.CANCELLATION_CONFIG:
            self.config.CANCELLATION_CONFIG[segment]['base_cancellation_rate'] *= 3.0
            # Cap at reasonable maximum
            self.config.CANCELLATION_CONFIG[segment]['base_cancellation_rate'] = min(
                self.config.CANCELLATION_CONFIG[segment]['base_cancellation_rate'], 0.6
            )
        
        # More last-minute bookings (uncertainty)
        self.config.CUSTOMER_SEGMENTS['Last_Minute']['market_share'] = 0.50
        self.config.CUSTOMER_SEGMENTS['Early_Planner']['market_share'] = 0.25
        self.config.CUSTOMER_SEGMENTS['Flexible']['market_share'] = 0.25
        
        # Aggressive promotional pricing
        self.config.CONNECTED_AGENT_PROMO_RATE = 0.95
        self.config.ONLINE_DIRECT_PROMO_RATE = 0.90
        
        # Deeper campaign discounts
        for campaign_type in self.config.CAMPAIGN_TYPES:
            discount_range = self.config.CAMPAIGN_TYPES[campaign_type]['discount_range']
            # Increase discounts by 50%
            new_range = (discount_range[0] * 1.5, discount_range[1] * 1.5)
            self.config.CAMPAIGN_TYPES[campaign_type]['discount_range'] = new_range
        
        # Reduced overbooking (risk averse)
        self.config.OVERBOOKING_CONFIG['base_overbooking_rate'] = 0.02
        
        print("✅ COVID impact scenario configured")
        return self.config
    
    def create_economic_boom_scenario(self):
        """Create a scenario for economic boom period"""
        print("📈 Creating Economic Boom Scenario")
        print("-" * 40)
        
        # Increased demand
        self.config.DATA_CONFIG['base_daily_demand'] = 45  # 50% increase
        
        # Higher prices across all room types
        for room_type in self.config.BASE_PRICES:
            self.config.BASE_PRICES[room_type] *= 1.4  # 40% price increase
        
        # Update periodic pricing too
        if self.config.PERIODIC_BASE_PRICING['enabled']:
            for room_type in self.config.PERIODIC_BASE_PRICING['pricing_periods']:
                for period in self.config.PERIODIC_BASE_PRICING['pricing_periods'][room_type]:
                    period['base_price'] = int(period['base_price'] * 1.4)
        
        # Lower cancellation rates (more committed customers)
        for segment in self.config.CANCELLATION_CONFIG:
            self.config.CANCELLATION_CONFIG[segment]['base_cancellation_rate'] *= 0.7
        
        # More early planners (confident about future)
        self.config.CUSTOMER_SEGMENTS['Early_Planner']['market_share'] = 0.60
        self.config.CUSTOMER_SEGMENTS['Last_Minute']['market_share'] = 0.15
        self.config.CUSTOMER_SEGMENTS['Flexible']['market_share'] = 0.25
        
        # Less promotional activity needed
        self.config.CONNECTED_AGENT_PROMO_RATE = 0.60
        self.config.ONLINE_DIRECT_PROMO_RATE = 0.50
        
        # More aggressive overbooking (high demand)
        self.config.OVERBOOKING_CONFIG['base_overbooking_rate'] = 0.18
        
        print("✅ Economic boom scenario configured")
        return self.config
    
    def create_new_hotel_scenario(self):
        """Create a scenario for a newly opened hotel"""
        print("🏗️ Creating New Hotel Scenario")
        print("-" * 40)
        
        # Lower initial demand (building reputation)
        self.config.DATA_CONFIG['base_daily_demand'] = 15
        
        # Aggressive promotional strategy to build market share
        self.config.CONNECTED_AGENT_PROMO_RATE = 0.85
        self.config.ONLINE_DIRECT_PROMO_RATE = 0.80
        self.config.CAMPAIGN_PARTICIPATION_RATE = 0.95
        
        # More flash sales and special offers
        self.config.CAMPAIGN_TYPES['Flash_Sale']['campaigns_per_month'] = (4, 6)
        self.config.CAMPAIGN_TYPES['Special_Offer']['campaigns_per_month'] = 2
        
        # Deeper discounts to attract customers
        for campaign_type in self.config.CAMPAIGN_TYPES:
            if campaign_type != 'Early_Booking':  # Keep early booking conservative
                discount_range = self.config.CAMPAIGN_TYPES[campaign_type]['discount_range']
                new_range = (discount_range[0] + 0.1, discount_range[1] + 0.1)  # +10% deeper
                self.config.CAMPAIGN_TYPES[campaign_type]['discount_range'] = new_range
        
        # Higher proportion of last-minute bookings (people trying new hotel)
        self.config.CUSTOMER_SEGMENTS['Last_Minute']['market_share'] = 0.40
        self.config.CUSTOMER_SEGMENTS['Early_Planner']['market_share'] = 0.30
        self.config.CUSTOMER_SEGMENTS['Flexible']['market_share'] = 0.30
        
        # Conservative overbooking (building reputation)
        self.config.OVERBOOKING_CONFIG['base_overbooking_rate'] = 0.05
        
        # Lower cancellation rates (discounted rates have stricter policies)
        for segment in self.config.CANCELLATION_CONFIG:
            self.config.CANCELLATION_CONFIG[segment]['base_cancellation_rate'] *= 0.8
        
        print("✅ New hotel scenario configured")
        return self.config
    
    def create_festival_destination_scenario(self):
        """Create a scenario for a hotel in a festival/event destination"""
        print("🎪 Creating Festival Destination Scenario")
        print("-" * 40)
        
        # Extreme seasonal variation
        self.config.SEASONAL_DEMAND_MULTIPLIERS = {
            5: 0.3,   # May - very low
            6: 0.5,   # June - building
            7: 1.8,   # July - festival season peak
            8: 1.6,   # August - still high
            9: 0.4    # September - back to low
        }
        
        # Extreme seasonal pricing
        self.config.DATA_CONFIG['seasonal_pricing_multipliers'] = {
            5: 0.7,   # May: -30%
            6: 0.9,   # June: -10%
            7: 2.5,   # July: +150% (festival premium)
            8: 2.0,   # August: +100%
            9: 0.8    # September: -20%
        }
        
        # Different overbooking strategy by season
        self.config.OVERBOOKING_CONFIG['seasonal_overbooking_multipliers'] = {
            5: 0.3,   # Very conservative in low season
            6: 0.7,   # Moderate
            7: 2.0,   # Very aggressive during festival
            8: 1.8,   # Still aggressive
            9: 0.4    # Conservative again
        }
        
        # More early planners for festival season
        self.config.CUSTOMER_SEGMENTS['Early_Planner']['market_share'] = 0.70
        self.config.CUSTOMER_SEGMENTS['Last_Minute']['market_share'] = 0.10
        self.config.CUSTOMER_SEGMENTS['Flexible']['market_share'] = 0.20
        
        # Higher cancellation rates (festivals can be cancelled)
        for segment in self.config.CANCELLATION_CONFIG:
            self.config.CANCELLATION_CONFIG[segment]['base_cancellation_rate'] *= 1.2
        
        # Minimal promotions during peak (no need)
        self.config.CONNECTED_AGENT_PROMO_RATE = 0.40
        self.config.ONLINE_DIRECT_PROMO_RATE = 0.30
        
        print("✅ Festival destination scenario configured")
        return self.config


def run_custom_scenario_analysis():
    """Run analysis comparing multiple custom scenarios"""
    
    print("🔬 Custom Scenario Analysis Pipeline")
    print("=" * 60)
    
    # Create scenario builder
    builder = CustomScenarioBuilder()
    
    scenarios = {
        'covid_impact': builder.create_covid_impact_scenario(),
        'economic_boom': CustomScenarioBuilder().create_economic_boom_scenario(),
        'new_hotel': CustomScenarioBuilder().create_new_hotel_scenario(),
        'festival_destination': CustomScenarioBuilder().create_festival_destination_scenario()
    }
    
    results = {}
    
    # Generate data for each scenario
    for scenario_name, config in scenarios.items():
        print(f"\n🎯 Generating data for {scenario_name} scenario...")
        
        try:
            generator = ConfigurableHotelBookingGenerator(config)
            
            # Override save method with scenario prefix
            def create_save_method(prefix):
                def save_data_with_prefix(bookings, campaigns, customers, attribution_data, baseline_demand):
                    import pickle
                    
                    pd.DataFrame(bookings).to_csv(f'{prefix}_historical_bookings.csv', index=False)
                    
                    df_campaigns = pd.DataFrame(campaigns)
                    df_campaigns['target_segments'] = df_campaigns['target_segments'].apply(lambda x: ';'.join(x))
                    df_campaigns['room_types_eligible'] = df_campaigns['room_types_eligible'].apply(lambda x: ';'.join(x))
                    df_campaigns.to_csv(f'{prefix}_campaigns_run.csv', index=False)
                    
                    pd.DataFrame(customers).to_csv(f'{prefix}_customer_segments.csv', index=False)
                    pd.DataFrame(attribution_data).to_csv(f'{prefix}_attribution_ground_truth.csv', index=False)
                    
                    with open(f'{prefix}_baseline_demand_model.pkl', 'wb') as f:
                        pickle.dump(baseline_demand, f)
                
                return save_data_with_prefix
            
            generator.save_data = create_save_method(scenario_name)
            
            # Generate data
            bookings, campaigns, customers, attribution_data, baseline_demand = generator.generate_all_data()
            
            # Analyze results
            analysis = analyze_booking_patterns(pd.DataFrame(bookings))
            quality = validate_data_quality(pd.DataFrame(bookings))
            
            results[scenario_name] = {
                'bookings': len(bookings),
                'campaigns': len(campaigns),
                'revenue': analysis['total_revenue'],
                'avg_price': analysis['avg_booking_value'],
                'campaign_rate': analysis['campaign_participation_rate'],
                'cancellation_rate': analysis['cancellation_rate'],
                'overbooking_rate': analysis['overbooking_rate'],
                'quality_score': quality['overall_quality_score']
            }
            
        except Exception as e:
            print(f"❌ Error generating {scenario_name}: {e}")
            results[scenario_name] = {'error': str(e)}
    
    # Create comparison report
    print(f"\n📊 CUSTOM SCENARIO COMPARISON")
    print("=" * 80)
    
    # Print header
    metrics = ['bookings', 'revenue', 'avg_price', 'campaign_rate', 'cancellation_rate', 'quality_score']
    header = f"{'Scenario':<20}"
    for metric in metrics:
        header += f"{metric.replace('_', ' ').title():<15}"
    print(header)
    print("-" * len(header))
    
    # Print results
    for scenario_name, data in results.items():
        if 'error' not in data:
            row = f"{scenario_name:<20}"
            row += f"{data['bookings']:<15,}"
            row += f"${data['revenue']:<14,.0f}"
            row += f"${data['avg_price']:<14.2f}"
            row += f"{data['campaign_rate']:<15.1%}"
            row += f"{data['cancellation_rate']:<15.1%}"
            row += f"{data['quality_score']:<15.1%}"
            print(row)
        else:
            print(f"{scenario_name:<20}ERROR: {data['error']}")
    
    print("=" * 80)
    
    # Generate detailed reports for each scenario
    print(f"\n📄 Generating detailed reports...")
    
    for scenario_name in results.keys():
        if 'error' not in results[scenario_name]:
            try:
                from utils import load_generated_data, create_summary_report
                
                data = load_generated_data(f'{scenario_name}_')
                if data:
                    report = create_summary_report(data, f'{scenario_name}_detailed_report.txt')
                    print(f"   ✅ {scenario_name}_detailed_report.txt")
                    
            except Exception as e:
                print(f"   ❌ Error creating report for {scenario_name}: {e}")
    
    return results


def create_ab_test_scenario():
    """Create A/B test scenarios for campaign effectiveness"""
    print("\n🧪 A/B Test Scenario Creation")
    print("=" * 50)
    
    # Scenario A: Conservative campaign strategy
    config_a = HotelBusinessConfig()
    config_a.CAMPAIGN_PARTICIPATION_RATE = 0.60  # Lower participation
    config_a.CONNECTED_AGENT_PROMO_RATE = 0.70
    config_a.ONLINE_DIRECT_PROMO_RATE = 0.60
    
    # Scenario B: Aggressive campaign strategy  
    config_b = HotelBusinessConfig()
    config_b.CAMPAIGN_PARTICIPATION_RATE = 0.90  # Higher participation
    config_b.CONNECTED_AGENT_PROMO_RATE = 0.90
    config_b.ONLINE_DIRECT_PROMO_RATE = 0.85
    
    # Deeper discounts in scenario B
    for campaign_type in config_b.CAMPAIGN_TYPES:
        discount_range = config_b.CAMPAIGN_TYPES[campaign_type]['discount_range']
        new_range = (discount_range[0] + 0.05, discount_range[1] + 0.05)
        config_b.CAMPAIGN_TYPES[campaign_type]['discount_range'] = new_range
    
    scenarios = {'ab_test_conservative': config_a, 'ab_test_aggressive': config_b}
    
    ab_results = {}
    
    for test_name, config in scenarios.items():
        print(f"Running {test_name}...")
        
        generator = ConfigurableHotelBookingGenerator(config)
        
        # Save with test prefix
        def create_ab_save_method(prefix):
            def save_data_with_prefix(bookings, campaigns, customers, attribution_data, baseline_demand):
                import pickle
                
                pd.DataFrame(bookings).to_csv(f'{prefix}_historical_bookings.csv', index=False)
                
                df_campaigns = pd.DataFrame(campaigns)
                df_campaigns['target_segments'] = df_campaigns['target_segments'].apply(lambda x: ';'.join(x))
                df_campaigns['room_types_eligible'] = df_campaigns['room_types_eligible'].apply(lambda x: ';'.join(x))
                df_campaigns.to_csv(f'{prefix}_campaigns_run.csv', index=False)
                
                pd.DataFrame(customers).to_csv(f'{prefix}_customer_segments.csv', index=False)
                pd.DataFrame(attribution_data).to_csv(f'{prefix}_attribution_ground_truth.csv', index=False)
                
                with open(f'{prefix}_baseline_demand_model.pkl', 'wb') as f:
                    pickle.dump(baseline_demand, f)
            
            return save_data_with_prefix
        
        generator.save_data = create_ab_save_method(test_name)
        
        # Generate data
        bookings, campaigns, customers, attribution_data, baseline_demand = generator.generate_all_data()
        
        # Calculate key metrics
        df_bookings = pd.DataFrame(bookings)
        campaign_bookings = df_bookings[df_bookings['campaign_id'].notna()]
        
        ab_results[test_name] = {
            'total_revenue': df_bookings['final_price'].sum(),
            'total_bookings': len(df_bookings),
            'campaign_bookings': len(campaign_bookings),
            'avg_discount': campaign_bookings['discount_amount'].mean() if len(campaign_bookings) > 0 else 0,
            'incremental_bookings': df_bookings['incremental_flag'].sum(),
            'campaign_participation': len(campaign_bookings) / len(df_bookings)
        }
    
    # Compare A/B test results
    print(f"\n📊 A/B TEST RESULTS COMPARISON")
    print("-" * 60)
    
    for test_name, metrics in ab_results.items():
        strategy = "Conservative" if "conservative" in test_name else "Aggressive"
        print(f"\n{strategy} Strategy:")
        print(f"  Total Revenue: ${metrics['total_revenue']:,.2f}")
        print(f"  Campaign Participation: {metrics['campaign_participation']:.1%}")
        print(f"  Average Discount: ${metrics['avg_discount']:.2f}")
        print(f"  Incremental Bookings: {metrics['incremental_bookings']:,}")
    
    # Calculate lift
    if len(ab_results) == 2:
        conservative = ab_results['ab_test_conservative']
        aggressive = ab_results['ab_test_aggressive']
        
        revenue_lift = (aggressive['total_revenue'] - conservative['total_revenue']) / conservative['total_revenue']
        booking_lift = (aggressive['total_bookings'] - conservative['total_bookings']) / conservative['total_bookings']
        
        print(f"\n📈 Aggressive vs Conservative Lift:")
        print(f"  Revenue Lift: {revenue_lift:+.1%}")
        print(f"  Booking Volume Lift: {booking_lift:+.1%}")
    
    return ab_results


def main():
    """Run custom scenario examples"""
    print("🎨 Custom Scenario Creation Examples")
    print("=" * 60)
    
    # Run comprehensive custom scenario analysis
    scenario_results = run_custom_scenario_analysis()
    
    # Run A/B test example
    ab_results = create_ab_test_scenario()
    
    print(f"\n🎉 Custom scenario analysis complete!")
    print(f"📁 Generated files for each scenario")
    print(f"📄 Detailed reports saved for analysis")
    
    print(f"\nKey insights from custom scenarios:")
    if scenario_results:
        # Find scenario with highest and lowest revenue
        revenues = {name: data.get('revenue', 0) for name, data in scenario_results.items() if 'error' not in data}
        if revenues:
            highest_revenue = max(revenues, key=revenues.get)
            lowest_revenue = min(revenues, key=revenues.get)
            
            print(f"  💰 Highest revenue scenario: {highest_revenue}")
            print(f"  📉 Lowest revenue scenario: {lowest_revenue}")
            
            revenue_diff = revenues[highest_revenue] - revenues[lowest_revenue]
            print(f"  📊 Revenue difference: ${revenue_diff:,.2f}")


if __name__ == "__main__":
    main()