# Hotel Booking Data Generator

A sophisticated synthetic data generation system that creates realistic hotel booking datasets with complex business logic, customer behavior patterns, and campaign attribution modeling.

## 🏗️ Architecture Overview

The system uses a **modular architecture** designed for maintainability, testability, and extensibility:

```
hotel-booking-generator/
├── 📋 config.py                    # Business configuration and parameters
├── 🚀 main.py                     # CLI entry point and scenario execution
├── 🎯 scenarios.py                # Predefined business scenarios
├── 📊 utils.py                    # Analysis and utility functions
├── 📈 visualize_data.py           # Comprehensive visualization tool
├── 🏗️ generators/                 # Core generation logic
│   ├── booking_generator.py       # Main orchestration class
│   ├── campaign_generator.py      # Campaign creation and management
│   ├── customer_generator.py      # Customer profiles and segments
│   ├── pricing_engine.py          # Pricing, discounts, attribution
│   ├── inventory_manager.py       # Room inventory and overbooking
│   └── booking_logic.py           # Core booking creation logic
├── ⚙️ processors/                  # Data processing and validation
│   └── data_processors.py         # Post-processing, validation, I/O
├── 📚 examples/                   # Usage examples and tutorials
│   └── usage_examples.py         # Comprehensive usage demonstrations
└── 🧪 tests/                      # Test suite
    └── test_refactoring.py        # Refactoring verification tests
```

## 🚀 Quick Start

### Basic Usage

```bash
# Generate data with default scenario
python main.py

# Use a specific scenario
python main.py --scenario luxury

# Generate with custom file prefix
python main.py --scenario budget --output-prefix "budget_hotel_"
```

### Available Scenarios

```bash
# List all available scenarios
python main.py --list-scenarios

# Compare scenario configurations
python main.py --compare-scenarios

# Validate configurations only
python main.py --validate-only
```

## 🎯 Available Scenarios

| Scenario | Description | Key Characteristics |
|----------|-------------|-------------------|
| **standard** | Baseline hotel operations | Standard industry parameters |
| **luxury** | High-end luxury property | Premium pricing, conservative policies |
| **budget** | Budget-friendly property | Lower prices, aggressive overbooking |
| **high_competition** | Highly competitive market | Aggressive promotions, higher cancellations |
| **seasonal_resort** | Seasonal resort operations | Extreme demand variations |
| **business_hotel** | Business-focused hotel | Weekday demand patterns |
| **conservative** | Risk-averse property | Minimal overbooking, low cancellations |

## 🔧 Programmatic Usage

### Basic Generation

```python
from generators import ConfigurableHotelBookingGenerator
from config import HotelBusinessConfig

# Create configuration
config = HotelBusinessConfig()

# Generate data
generator = ConfigurableHotelBookingGenerator(config)
bookings, campaigns, customers, attribution, demand = generator.generate_all_data()
```

### Using Scenarios

```python
from scenarios import create_test_scenarios

# Load predefined scenarios
scenarios = create_test_scenarios()
luxury_config = scenarios['luxury']

# Generate luxury hotel data
generator = ConfigurableHotelBookingGenerator(luxury_config)
data = generator.generate_all_data()
```

### Using Individual Components

```python
from generators.campaign_generator import CampaignGenerator
from generators.customer_generator import CustomerGenerator
from generators.pricing_engine import PricingEngine

config = HotelBusinessConfig()

# Use components independently
campaign_gen = CampaignGenerator(config)
campaigns = campaign_gen.generate_campaigns()

customer_gen = CustomerGenerator(config)
customers = customer_gen.generate_customers()

pricing = PricingEngine(config)
base_price = pricing.get_base_price_for_date(datetime.now(), 'Standard')
```

## 📊 Output Data Schema

### Historical Bookings (`historical_bookings.csv`)
```
booking_id, customer_id, booking_date, stay_start_date, stay_end_date,
stay_length, room_type, customer_segment, booking_channel,
base_price, final_price, discount_amount, campaign_id,
attribution_score, incremental_flag, is_cancelled, cancellation_date
```

### Campaign Performance (`campaigns_run.csv`)
```
campaign_id, campaign_type, start_date, end_date, discount_percentage,
target_segments, channel, capacity_limit, actual_bookings, incremental_bookings
```

### Customer Profiles (`customer_segments.csv`)
```
customer_id, segment, price_sensitivity, planning_horizon,
channel_preference, loyalty_status, campaign_exposures, booking_history
```

### Attribution Ground Truth (`attribution_ground_truth.csv`)
```
booking_id, true_attribution_score, causal_campaign_id,
counterfactual_price, would_have_booked_anyway
```

## 🏨 Business Logic Features

### Customer Segments
- **Early Planner**: Books 60-150 days in advance, price-sensitive, prefers agents
- **Last Minute**: Books 1-30 days in advance, less price-sensitive, prefers online
- **Flexible**: Books 14-90 days in advance, moderate price sensitivity

### Campaign Types
- **Early Booking**: Long-term campaigns targeting operational season
- **Flash Sale**: Short-term urgency campaigns with high discounts
- **Special Offer**: Medium-term shoulder season promotions

### Pricing Strategy
- **Periodic Base Pricing**: Time-based rate changes across years
- **Seasonal Adjustments**: Peak season pricing (July-August)
- **Segment-Specific Pricing**: Early planner discounts, last-minute premiums
- **Campaign Discounts**: Structured promotional pricing

### Revenue Management
- **Dynamic Inventory**: Real-time capacity tracking
- **Overbooking Strategy**: Configurable risk-based overbooking
- **Cancellation Modeling**: Segment-specific cancellation patterns
- **Attribution Modeling**: Time-decay campaign attribution

## 📈 Data Analysis and Visualization

### Generate Visualizations

```bash
# Create comprehensive visualizations
python visualize_data.py

# Visualize specific scenario data
python visualize_data.py --prefix "luxury_"
```

### Analysis Tools

```python
from utils import load_generated_data, analyze_booking_patterns, create_summary_report

# Load and analyze data
data = load_generated_data('luxury_')
analysis = analyze_booking_patterns(data['bookings'])
report = create_summary_report(data, 'luxury_analysis.txt')
```

## 🎨 Examples and Tutorials

See the `examples/usage_examples.py` file for comprehensive examples including:

1. **Basic Data Generation**: Simple end-to-end generation
2. **Scenario Usage**: Using predefined business scenarios
3. **Custom Configuration**: Creating custom hotel configurations
4. **Component Usage**: Using individual components independently
5. **Custom Pricing**: Implementing custom pricing strategies
6. **Inventory Analysis**: Advanced inventory management
7. **Attribution Analysis**: Campaign attribution modeling
8. **Performance Comparison**: Comparing scenarios

```bash
# Run all examples
python examples/usage_examples.py
```

## 🧪 Testing

### Run Tests

```bash
# Run refactoring verification tests
python tests/test_refactoring.py

# Run specific test class
python -m unittest tests.test_refactoring.TestRefactoredArchitecture
```

### Test Categories
- **Component Tests**: Individual module functionality
- **Integration Tests**: Cross-component interactions
- **End-to-End Tests**: Complete generation pipeline
- **Backward Compatibility**: Existing interface preservation

## ⚙️ Configuration

### Key Configuration Categories

1. **Property Operations**: Seasons, room types, capacity
2. **Pricing Strategy**: Base rates, seasonal adjustments, discount ranges
3. **Customer Segments**: Market shares, behaviors, preferences
4. **Campaign Strategy**: Types, timing, discount levels, targeting
5. **Risk Management**: Cancellation rates, overbooking policies

### Custom Configuration Example

```python
config = HotelBusinessConfig()

# Customize pricing
config.BASE_PRICES = {
    'Standard': 180,
    'Deluxe': 250, 
    'Suite': 350,
    'Premium': 450
}

# Adjust customer segments
config.CUSTOMER_SEGMENTS['Early_Planner']['market_share'] = 0.65

# Modify operational parameters
config.DATA_CONFIG['base_daily_demand'] = 25
config.OVERBOOKING_CONFIG['base_overbooking_rate'] = 0.12
```

## 🔄 Migration from Previous Version

If upgrading from the monolithic version, see the detailed [Migration Guide](MIGRATION.md) for step-by-step instructions.

### Key Changes
- Modular architecture with focused components
- Improved testability and extensibility
- Enhanced performance and maintainability
- Backward-compatible main interface

## 📋 Data Quality Features

### Validation Framework
- **Business Logic Validation**: Price hierarchies, date consistency
- **Data Quality Checks**: Missing values, outliers, constraint violations
- **Statistical Validation**: Distribution checks, correlation analysis
- **ML Readiness Assessment**: Feature balance, normality tests

### Quality Metrics
- Automatically validates against configuration targets
- Generates quality reports with actionable insights
- Provides ML readiness scores and recommendations

## 🎯 Use Cases

### Revenue Management
- **Dynamic Pricing Analysis**: Test pricing strategies across scenarios
- **Campaign Attribution**: Measure promotional campaign effectiveness
- **Overbooking Optimization**: Analyze risk vs. revenue trade-offs
- **Demand Forecasting**: Generate realistic demand patterns

### Data Science and ML
- **Feature Engineering**: Rich dataset with temporal and behavioral features
- **Model Training**: Clean, labeled data for predictive modeling
- **A/B Testing**: Compare strategies across different scenarios
- **Synthetic Data Generation**: Privacy-safe dataset for development

### Business Analytics
- **Customer Segmentation**: Realistic customer behavior patterns
- **Channel Analysis**: Multi-channel booking behavior
- **Seasonal Analysis**: Operational season demand patterns
- **Performance Benchmarking**: Compare against industry scenarios

## 🚀 Performance Characteristics

- **Data Volume**: 5,000-15,000 bookings per scenario (configurable)
- **Generation Time**: 30-60 seconds for standard scenario
- **Memory Usage**: ~100MB for large datasets
- **File Sizes**: 2-10MB CSV files depending on volume

## 🤝 Contributing

The modular architecture makes it easy to extend the system:

- **New Customer Segments**: Add behavioral patterns in `customer_generator.py`
- **Additional Campaign Types**: Extend campaign logic in `campaign_generator.py`
- **Complex Pricing Rules**: Enhance pricing engine with new strategies
- **Advanced Attribution**: Implement sophisticated attribution models
- **Custom Scenarios**: Create domain-specific configurations

## 📄 License

[Add your license information here]

## 🆘 Support

For questions and support:
1. Check the [examples](examples/usage_examples.py) for common use cases
2. Review the [migration guide](MIGRATION.md) if upgrading
3. Run the test suite to verify your setup
4. Create an issue for bugs or feature requests

---

**Built with **