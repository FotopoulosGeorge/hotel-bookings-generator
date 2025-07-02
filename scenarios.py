"""
Hotel Booking Test Scenarios

Different configuration scenarios for testing various hotel business models
and market conditions.
"""

from config import HotelBusinessConfig


def create_test_scenarios():
    """Create different configuration scenarios for testing"""
    
    # Scenario 1: Standard Seasonal Hotel (improved distribution)
    standard_config = HotelBusinessConfig()
    standard_config.OPERATION_MODE = 'seasonal'
    # Uses default SEASONAL_STAY_DISTRIBUTION from config to prevent spikes
    
    # Scenario 2: Year-Round Hotel
    year_round_config = HotelBusinessConfig()
    year_round_config.OPERATION_MODE = 'year_round'
    year_round_config.OPERATIONAL_MONTHS = list(range(1, 13))  # All months
    
    # Adjust demand patterns for year-round operations
    year_round_config.SEASONAL_DEMAND_MULTIPLIERS = {
        1: 0.70,   # January - post-holiday slowdown
        2: 0.75,   # February - winter low season
        3: 0.85,   # March - spring pickup
        4: 0.90,   # April - pre-summer
        5: 0.95,   # May - shoulder season
        6: 1.00,   # June - summer begins
        7: 1.10,   # July - peak summer
        8: 1.10,   # August - peak summer
        9: 0.90,   # September - fall shoulder
        10: 0.85,  # October - fall season
        11: 0.75,  # November - pre-holiday low
        12: 0.95   # December - holiday season
    }
    
    # Year-round pricing adjustments
    year_round_config.DATA_CONFIG['seasonal_pricing_multipliers'] = {
        1: 0.85,   # January: -15%
        2: 0.85,   # February: -15%
        3: 0.95,   # March: -5%
        4: 1.00,   # April: base
        5: 1.05,   # May: +5%
        6: 1.10,   # June: +10%
        7: 1.20,   # July: +20% (peak)
        8: 1.20,   # August: +20% (peak)
        9: 1.00,   # September: base
        10: 0.90,  # October: -10%
        11: 0.85,  # November: -15%
        12: 1.15   # December: +15% (holidays)
    }
    
    # Scenario 3: High Competition Market (more promotions, higher cancellations)
    high_competition_config = HotelBusinessConfig()
    high_competition_config.CONNECTED_AGENT_PROMO_RATE = 0.90  # 90% promotional
    high_competition_config.ONLINE_DIRECT_PROMO_RATE = 0.85    # 85% promotional
    high_competition_config.CAMPAIGN_PARTICIPATION_RATE = 0.90  # Higher participation
    
    # Higher cancellation rates in competitive market
    for segment in high_competition_config.CANCELLATION_CONFIG:
        high_competition_config.CANCELLATION_CONFIG[segment]['base_cancellation_rate'] *= 1.3
    
    # More aggressive overbooking
    high_competition_config.OVERBOOKING_CONFIG['base_overbooking_rate'] = 0.15
    
    # Scenario 4: Luxury Seasonal Property (higher prices, lower volume, lower cancellations)
    luxury_config = HotelBusinessConfig()
    luxury_config.OPERATION_MODE = 'seasonal'
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
    
    # Scenario 5: Budget Year-Round Property (lower prices, higher volume, higher cancellations)
    budget_config = HotelBusinessConfig()
    budget_config.OPERATION_MODE = 'year_round'
    budget_config.OPERATIONAL_MONTHS = list(range(1, 13))
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
    
    # Budget year-round demand patterns
    budget_config.SEASONAL_DEMAND_MULTIPLIERS = {
        1: 0.80, 2: 0.85, 3: 0.90, 4: 0.95,
        5: 1.00, 6: 1.05, 7: 1.10, 8: 1.10,
        9: 1.00, 10: 0.90, 11: 0.85, 12: 0.95
    }
    
    # Scenario 6: Conservative Property (minimal overbooking, low cancellations)
    conservative_config = HotelBusinessConfig()
    
    # Lower cancellation rates
    for segment in conservative_config.CANCELLATION_CONFIG:
        conservative_config.CANCELLATION_CONFIG[segment]['base_cancellation_rate'] *= 0.5
    
    # Minimal overbooking
    conservative_config.OVERBOOKING_CONFIG['base_overbooking_rate'] = 0.03
    
    # Scenario 7: Seasonal Resort with Better Distribution
    seasonal_resort_config = HotelBusinessConfig()
    seasonal_resort_config.OPERATION_MODE = 'seasonal'
    
    # Extreme seasonality in booking patterns (when people book)
    seasonal_resort_config.SEASONAL_DEMAND_MULTIPLIERS = {
        1: 0.50,   # January - steady early bookings
        2: 0.60,   # February - gradual increase
        3: 0.75,   # March - building up
        4: 0.90,   # April - pre-season rush
        5: 0.60,   # May - lower because many already booked
        6: 0.40,   # June - mostly full, fewer bookings
        7: 0.25,   # July - very few spots left
        8: 0.20,   # August - minimal availability
        9: 0.30,   # September - last minute deals
        10: 0.55,  # October - early birds for next year
        11: 0.70,  # November - campaign season
        12: 0.65   # December - holiday bookings for next summer
    }
    
    # Custom stay distribution to prevent May spike
    seasonal_resort_config.SEASONAL_STAY_DISTRIBUTION = {
        5: 0.15,   # May - reduced to prevent spike
        6: 0.20,   # June - moderate
        7: 0.30,   # July - peak
        8: 0.28,   # August - still peak
        9: 0.07    # September - tail
    }
    
    # Higher pricing during peak season
    seasonal_resort_config.DATA_CONFIG['seasonal_pricing_multipliers'] = {
        7: 1.8,   # July: +80%
        8: 1.8,   # August: +80%
        5: 0.8,   # May: -20%
        9: 0.8,   # September: -20%
        6: 1.2    # June: +20%
    }
    
    # More aggressive overbooking during peak
    seasonal_resort_config.OVERBOOKING_CONFIG['seasonal_overbooking_multipliers'] = {
        5: 0.5, 6: 0.8, 7: 1.5, 8: 1.5, 9: 0.5
    }
    
    # Scenario 8: Business Hotel (year-round, weekday focused)
    business_config = HotelBusinessConfig()
    business_config.OPERATION_MODE = 'year_round'
    business_config.OPERATIONAL_MONTHS = list(range(1, 13))
    
    business_config.WEEKLY_DEMAND_MULTIPLIERS = {
        0: 1.2,   # Monday - high business travel
        1: 1.3,   # Tuesday - high business travel
        2: 1.3,   # Wednesday - high business travel
        3: 1.2,   # Thursday - high business travel
        4: 0.9,   # Friday - lower business travel
        5: 0.6,   # Saturday - low business travel
        6: 0.7    # Sunday - low business travel
    }
    
    # Stable year-round demand with slight variations
    business_config.SEASONAL_DEMAND_MULTIPLIERS = {
        1: 0.90, 2: 0.95, 3: 1.00, 4: 1.00,
        5: 0.95, 6: 0.90, 7: 0.80, 8: 0.75,  # Lower summer (vacation season)
        9: 1.00, 10: 1.05, 11: 0.95, 12: 0.85
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
        'standard': standard_config,
        'year_round': year_round_config,
        'high_competition': high_competition_config,
        'luxury': luxury_config,
        'budget': budget_config,
        'conservative': conservative_config,
        'seasonal_resort': seasonal_resort_config,
        'business_hotel': business_config
    }


def get_scenario_description(scenario_name):
    """Get a description of what each scenario represents"""
    descriptions = {
        'standard': "Seasonal hotel (May-Sep) with balanced distribution to prevent spikes",
        'year_round': "Year-round hotel with natural seasonal demand variations",
        'high_competition': "Highly competitive market with aggressive promotions and higher cancellation rates",
        'luxury': "High-end luxury seasonal property with premium pricing and conservative policies",
        'budget': "Budget-friendly year-round property with lower prices and higher volume",
        'conservative': "Risk-averse property with minimal overbooking and low cancellation rates",
        'seasonal_resort': "Seasonal resort with optimized stay distribution and extreme demand variations",
        'business_hotel': "Year-round business hotel with weekday demand patterns and corporate behavior"
    }
    return descriptions.get(scenario_name, "Unknown scenario")


def print_scenario_comparison(scenarios):
    """Print a comparison of key metrics across scenarios"""
    print("\n" + "="*100)
    print("📊 SCENARIO COMPARISON")
    print("="*100)
    
    metrics = [
        ('Operation Mode', lambda c: getattr(c, 'OPERATION_MODE', 'seasonal')),
        ('Operational Months', lambda c: f"{len(c.OPERATIONAL_MONTHS)} months" if len(c.OPERATIONAL_MONTHS) < 12 else "Year-round"),
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
    header = f"{'Metric':<28}"
    for scenario_name in scenarios.keys():
        header += f"{scenario_name:<15}"
    print(header)
    print("-" * len(header))
    
    # Print each metric
    for metric_name, metric_func in metrics:
        row = f"{metric_name:<28}"
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