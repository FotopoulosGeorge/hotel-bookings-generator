"""
Hotel Booking Test Scenarios

Different configuration scenarios for testing various hotel business models
and market conditions.
"""

from config import HotelBusinessConfig


def create_test_scenarios():
    """Create different configuration scenarios for testing"""
    
    # Scenario 1: High Competition Market (more promotions, higher cancellations)
    high_competition_config = HotelBusinessConfig()
    high_competition_config.CONNECTED_AGENT_PROMO_RATE = 0.90  # 90% promotional
    high_competition_config.ONLINE_DIRECT_PROMO_RATE = 0.85    # 85% promotional
    high_competition_config.CAMPAIGN_PARTICIPATION_RATE = 0.90  # Higher participation
    
    # Higher cancellation rates in competitive market
    for segment in high_competition_config.CANCELLATION_CONFIG:
        high_competition_config.CANCELLATION_CONFIG[segment]['base_cancellation_rate'] *= 1.3
    
    # More aggressive overbooking
    high_competition_config.OVERBOOKING_CONFIG['base_overbooking_rate'] = 0.15
    
    # Scenario 2: Luxury Property (higher prices, lower volume, lower cancellations)
    luxury_config = HotelBusinessConfig()
    luxury_config.BASE_PRICES = {
        'Standard': 300, 'Deluxe': 450, 'Suite': 650, 'Premium': 900
    }
    
    # Update periodic pricing for luxury property
    if luxury_config.PERIODIC_BASE_PRICING['enabled']:
        for room_type in luxury_config.PERIODIC_BASE_PRICING['pricing_periods']:
            for period in luxury_config.PERIODIC_BASE_PRICING['pricing_periods'][room_type]:
                period['base_price'] = int(period['base_price'] * 2.5)  # 2.5x luxury multiplier
    
    luxury_config.DATA_CONFIG['base_daily_demand'] = 15  # Lower volume
    luxury_config.CUSTOMER_SEGMENTS['Early_Planner']['market_share'] = 0.60  # More planners
    
    # Lower cancellation rates for luxury (more committed guests)
    for segment in luxury_config.CANCELLATION_CONFIG:
        luxury_config.CANCELLATION_CONFIG[segment]['base_cancellation_rate'] *= 0.7
    
    # Conservative overbooking for luxury
    luxury_config.OVERBOOKING_CONFIG['base_overbooking_rate'] = 0.05
    
    # Scenario 3: Budget Property (lower prices, higher volume, higher cancellations)
    budget_config = HotelBusinessConfig()
    budget_config.BASE_PRICES = {
        'Standard': 80, 'Deluxe': 120, 'Suite': 180, 'Premium': 220
    }
    
    # Update periodic pricing for budget property
    if budget_config.PERIODIC_BASE_PRICING['enabled']:
        for room_type in budget_config.PERIODIC_BASE_PRICING['pricing_periods']:
            for period in budget_config.PERIODIC_BASE_PRICING['pricing_periods'][room_type]:
                period['base_price'] = int(period['base_price'] * 0.65)  # 65% of standard rates
    
    budget_config.DATA_CONFIG['base_daily_demand'] = 50  # Higher volume
    budget_config.CUSTOMER_SEGMENTS['Last_Minute']['market_share'] = 0.40  # More last-minute
    
    # Higher cancellation rates for budget (less committed)
    for segment in budget_config.CANCELLATION_CONFIG:
        budget_config.CANCELLATION_CONFIG[segment]['base_cancellation_rate'] *= 1.5
    
    # Aggressive overbooking for budget
    budget_config.OVERBOOKING_CONFIG['base_overbooking_rate'] = 0.20
    
    # Scenario 4: Conservative Property (minimal overbooking, low cancellations)
    conservative_config = HotelBusinessConfig()
    
    # Lower cancellation rates
    for segment in conservative_config.CANCELLATION_CONFIG:
        conservative_config.CANCELLATION_CONFIG[segment]['base_cancellation_rate'] *= 0.5
    
    # Minimal overbooking
    conservative_config.OVERBOOKING_CONFIG['base_overbooking_rate'] = 0.03
    
    # Scenario 5: Seasonal Resort (extreme seasonality)
    seasonal_config = HotelBusinessConfig()
    seasonal_config.SEASONAL_DEMAND_MULTIPLIERS = {
        5: 0.4,   # May - very low
        6: 0.6,   # June - building
        7: 1.3,   # July - extreme peak
        8: 1.3,   # August - extreme peak  
        9: 0.4    # September - very low
    }
    
    # Higher pricing during peak season
    seasonal_config.DATA_CONFIG['seasonal_pricing_multipliers'] = {
        7: 1.8,   # July: +80%
        8: 1.8,   # August: +80%
        5: 0.8,   # May: -20%
        9: 0.8,   # September: -20%
        6: 1.2    # June: +20%
    }
    
    # More aggressive overbooking during peak
    seasonal_config.OVERBOOKING_CONFIG['seasonal_overbooking_multipliers'] = {
        5: 0.5, 6: 0.8, 7: 1.5, 8: 1.5, 9: 0.5
    }
    
    # Scenario 6: Business Hotel (weekday focused)
    business_config = HotelBusinessConfig()
    business_config.WEEKLY_DEMAND_MULTIPLIERS = {
        0: 1.2,   # Monday - high business travel
        1: 1.3,   # Tuesday - high business travel
        2: 1.3,   # Wednesday - high business travel
        3: 1.2,   # Thursday - high business travel
        4: 0.9,   # Friday - lower business travel
        5: 0.6,   # Saturday - low business travel
        6: 0.7    # Sunday - low business travel
    }
    
    # More connected agent bookings for business travelers
    business_config.TARGET_CONNECTED_AGENT_SHARE = 0.75
    business_config.TARGET_ONLINE_DIRECT_SHARE = 0.25
    
    # Higher early planner share (corporate bookings)
    business_config.CUSTOMER_SEGMENTS['Early_Planner']['market_share'] = 0.55
    business_config.CUSTOMER_SEGMENTS['Last_Minute']['market_share'] = 0.20
    business_config.CUSTOMER_SEGMENTS['Flexible']['market_share'] = 0.25
    
    # Lower cancellation rates for business (corporate policies)
    for segment in business_config.CANCELLATION_CONFIG:
        business_config.CANCELLATION_CONFIG[segment]['base_cancellation_rate'] *= 0.8
    
    return {
        'standard': HotelBusinessConfig(),
        'high_competition': high_competition_config,
        'luxury': luxury_config,
        'budget': budget_config,
        'conservative': conservative_config,
        'seasonal_resort': seasonal_config,
        'business_hotel': business_config
    }


def get_scenario_description(scenario_name):
    """Get a description of what each scenario represents"""
    descriptions = {
        'standard': "Baseline hotel configuration with standard industry parameters",
        'high_competition': "Highly competitive market with aggressive promotions and higher cancellation rates",
        'luxury': "High-end luxury property with premium pricing, lower volume, and conservative policies",
        'budget': "Budget-friendly property with lower prices, higher volume, and aggressive overbooking",
        'conservative': "Risk-averse property with minimal overbooking and low cancellation rates",
        'seasonal_resort': "Seasonal resort with extreme demand variations and dynamic pricing",
        'business_hotel': "Business-focused hotel with weekday demand patterns and corporate booking behavior"
    }
    return descriptions.get(scenario_name, "Unknown scenario")


def print_scenario_comparison(scenarios):
    """Print a comparison of key metrics across scenarios"""
    print("\n" + "="*100)
    print("📊 SCENARIO COMPARISON")
    print("="*100)
    
    metrics = [
        ('Base Daily Demand', lambda c: c.DATA_CONFIG['base_daily_demand']),
        ('Standard Room Base Price', lambda c: c.BASE_PRICES['Standard']),
        ('Premium Room Base Price', lambda c: c.BASE_PRICES['Premium']),
        ('Agent Promo Rate', lambda c: f"{c.CONNECTED_AGENT_PROMO_RATE:.0%}"),
        ('Online Promo Rate', lambda c: f"{c.ONLINE_DIRECT_PROMO_RATE:.0%}"),
        ('Base Overbooking Rate', lambda c: f"{c.OVERBOOKING_CONFIG['base_overbooking_rate']:.1%}"),
        ('Early Planner Cancellation', lambda c: f"{c.CANCELLATION_CONFIG['Early_Planner']['base_cancellation_rate']:.1%}"),
        ('Last Minute Cancellation', lambda c: f"{c.CANCELLATION_CONFIG['Last_Minute']['base_cancellation_rate']:.1%}"),
    ]
    
    # Print header
    header = f"{'Metric':<25}"
    for scenario_name in scenarios.keys():
        header += f"{scenario_name:<15}"
    print(header)
    print("-" * len(header))
    
    # Print each metric
    for metric_name, metric_func in metrics:
        row = f"{metric_name:<25}"
        for scenario_name, config in scenarios.items():
            try:
                value = metric_func(config)
                row += f"{str(value):<15}"
            except:
                row += f"{'N/A':<15}"
        print(row)
    
    print("="*100)


def validate_scenario(config, scenario_name):
    """Validate that a scenario configuration is valid"""
    issues = []
    
    # Check basic configuration validity
    if not isinstance(config.SIMULATION_YEARS, list) or len(config.SIMULATION_YEARS) == 0:
        issues.append("SIMULATION_YEARS must be a non-empty list")
    
    if not isinstance(config.OPERATIONAL_MONTHS, list) or len(config.OPERATIONAL_MONTHS) == 0:
        issues.append("OPERATIONAL_MONTHS must be a non-empty list")
    
    # Check price validity
    for room_type, price in config.BASE_PRICES.items():
        if price <= 0:
            issues.append(f"BASE_PRICES[{room_type}] must be positive")
    
    # Check probability validity
    for segment, data in config.CUSTOMER_SEGMENTS.items():
        if not (0 <= data['market_share'] <= 1):
            issues.append(f"CUSTOMER_SEGMENTS[{segment}]['market_share'] must be between 0 and 1")
    
    # Check market share sums to 1
    total_market_share = sum(data['market_share'] for data in config.CUSTOMER_SEGMENTS.values())
    if abs(total_market_share - 1.0) > 0.01:
        issues.append(f"CUSTOMER_SEGMENTS market shares must sum to 1.0 (currently {total_market_share:.3f})")
    
    # Check cancellation rates
    for segment, cancel_config in config.CANCELLATION_CONFIG.items():
        if not (0 <= cancel_config['base_cancellation_rate'] <= 1):
            issues.append(f"CANCELLATION_CONFIG[{segment}]['base_cancellation_rate'] must be between 0 and 1")
    
    # Check overbooking rates
    if not (0 <= config.OVERBOOKING_CONFIG['base_overbooking_rate'] <= 1):
        issues.append("OVERBOOKING_CONFIG['base_overbooking_rate'] must be between 0 and 1")
    
    if issues:
        print(f"\n⚠️  VALIDATION ISSUES FOR SCENARIO '{scenario_name}':")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"✅ Scenario '{scenario_name}' validation passed")
        return True


if __name__ == "__main__":
    # Test scenario creation and validation
    scenarios = create_test_scenarios()
    
    print("🎯 Available Test Scenarios:")
    print("-" * 50)
    for name, config in scenarios.items():
        description = get_scenario_description(name)
        print(f"• {name}: {description}")
    
    # Validate all scenarios
    print(f"\n🔍 Validating scenarios...")
    all_valid = True
    for name, config in scenarios.items():
        if not validate_scenario(config, name):
            all_valid = False
    
    if all_valid:
        print(f"\n✅ All scenarios validated successfully!")
        print_scenario_comparison(scenarios)
    else:
        print(f"\n❌ Some scenarios have validation issues!")