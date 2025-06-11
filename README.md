# Hotel Booking Data Generator

A comprehensive, configurable system for generating realistic hotel booking datasets with campaigns, customer behavior, cancellations, and overbooking patterns.

## 🎯 Purpose

This tool generates synthetic hotel booking data for:
- **Revenue Management Testing**: Test pricing and campaign strategies
- **Data Science Projects**: Practice with realistic hospitality datasets  
- **Attribution Modeling**: Ground truth data for campaign effectiveness analysis
- **Business Intelligence**: Scenario planning and forecasting model development

## 📁 Project Structure

```
hotel-booking-generator/
├── config.py              # Business configuration and parameters
├── data_generator.py       # Core data generation logic
├── scenarios.py           # Pre-defined test scenarios
├── main.py                # Command-line interface
├── utils.py               # Data analysis and visualization utilities
├── visualize_data.py      # Comprehensive data visualization tool
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
└── examples/             # Usage examples and tutorials
    ├── basic_usage.py
    ├── custom_scenario.py
    └── analysis_example.py
```

## 🚀 Quick Start

### Installation

```bash
# Clone or download the files
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Generate data with default settings
python main.py

# Use a specific scenario
python main.py --scenario luxury

# List all available scenarios
python main.py --list-scenarios

# Compare scenario configurations
python main.py --compare-scenarios
```

### Python API Usage

```python
from config import HotelBusinessConfig
from data_generator import ConfigurableHotelBookingGenerator

# Use default configuration
generator = ConfigurableHotelBookingGenerator()
bookings, campaigns, customers, attribution, demand = generator.generate_all_data()

# Use custom configuration
config = HotelBusinessConfig()
config.BASE_PRICES['Standard'] = 150  # Modify pricing
generator = ConfigurableHotelBookingGenerator(config)
data = generator.generate_all_data()
```

## 🏨 Available Scenarios

| Scenario | Description | Key Characteristics |
|----------|-------------|-------------------|
| `standard` | Baseline hotel configuration | Industry-standard parameters |
| `luxury` | High-end luxury property | Premium pricing, lower volume, conservative policies |
| `budget` | Budget-friendly property | Lower prices, higher volume, aggressive overbooking |
| `high_competition` | Competitive market | Aggressive promotions, higher cancellations |
| `conservative` | Risk-averse property | Minimal overbooking, low cancellation rates |
| `seasonal_resort` | Seasonal resort | Extreme demand variations, dynamic pricing |
| `business_hotel` | Business-focused hotel | Weekday patterns, corporate booking behavior |

## 📊 Generated Data Files

The generator creates several CSV files and analysis objects:

### Core Data Files
- **`historical_bookings.csv`**: Main booking records with pricing, campaigns, cancellations
- **`campaigns_run.csv`**: Campaign details and performance metrics  
- **`customer_segments.csv`**: Customer profiles and behavior patterns
- **`attribution_ground_truth.csv`**: True causal relationships for testing attribution models

### Analysis Objects
- **`baseline_demand_model.pkl`**: Underlying demand patterns (pickled Python object)

## 🔧 Configuration System

### Key Configuration Areas

**Business Operations**
```python
OPERATIONAL_MONTHS = [5, 6, 7, 8, 9]  # May-September
SIMULATION_YEARS = [2022, 2023, 2024]
ROOM_TYPES = ['Standard', 'Deluxe', 'Suite', 'Premium']
```

**Pricing Strategy**
```python
BASE_PRICES = {
    'Standard': 120, 'Deluxe': 180, 'Suite': 280, 'Premium': 350
}
SEASONAL_PRICING_MULTIPLIERS = {
    7: 1.3,  # July: +30%
    8: 1.3,  # August: +30%
}
```

**Customer Segments**
```python
CUSTOMER_SEGMENTS = {
    'Early_Planner': {
        'market_share': 0.45,
        'advance_booking_days': (120, 240),
        'price_sensitivity': 0.8
    }
}
```

**Campaign Types**
- **Early Booking**: Long-term advance purchase campaigns
- **Flash Sales**: Short-duration urgent promotions  
- **Special Offers**: Shoulder season targeted promotions

### Creating Custom Scenarios

```python
from config import HotelBusinessConfig

# Create custom configuration
custom_config = HotelBusinessConfig()

# Modify specific parameters
custom_config.BASE_PRICES = {'Standard': 200, 'Deluxe': 300}
custom_config.CONNECTED_AGENT_PROMO_RATE = 0.95  # Very aggressive promotions
custom_config.OVERBOOKING_CONFIG['base_overbooking_rate'] = 0.15

# Use with generator
from data_generator import ConfigurableHotelBookingGenerator
generator = ConfigurableHotelBookingGenerator(custom_config)
data = generator.generate_all_data()
```

## 📈 Data Analysis Features

### Built-in Analysis Tools

```python
from utils import load_generated_data, analyze_booking_patterns, create_summary_report

# Load previously generated data
data = load_generated_data('luxury_')  # For luxury scenario files

# Analyze patterns
analysis = analyze_booking_patterns(data['bookings'])
print(f"Campaign participation rate: {analysis['campaign_participation_rate']:.1%}")

# Generate comprehensive report
report = create_summary_report(data, 'luxury_hotel_report.txt')
print(report)
```

### Visualization Creation

```python
from utils import create_visualizations

# Generate plots and charts
create_visualizations(data, output_dir='luxury_analysis/')
```

### Comprehensive Data Visualization

```python
# Generate extensive visualizations for data quality assessment
python visualize_data.py

# Generate visualizations for specific scenario
python visualize_data.py --prefix luxury_ --output-dir luxury_plots/

# Creates 9 different plot types + HTML dashboard:
# - Distribution analysis
# - Temporal patterns  
# - Campaign performance
# - Correlation matrices
# - ML readiness assessment
# - Business logic validation
# - Summary dashboard
```

### Scenario Comparison

```python
from utils import compare_scenarios

# Compare multiple scenarios
comparison = compare_scenarios(['standard_', 'luxury_', 'budget_'])
print(comparison)
```

## 🎲 Key Features

### Realistic Business Logic
- **Dynamic Pricing**: Seasonal and day-of-week pricing variations
- **Customer Segmentation**: Early planners, last-minute bookers, flexible travelers
- **Channel Management**: Connected agents vs. direct online bookings
- **Campaign Attribution**: Realistic attribution modeling with uncertainty

### Advanced Operations Modeling  
- **Cancellation Patterns**: Segment-specific cancellation rates and timing
- **Overbooking Strategy**: Seasonal and channel-based overbooking rates
- **Revenue Management**: Promotional rate targeting and capacity limits

### Data Science Ready
- **Ground Truth Labels**: Known causal relationships for model validation
- **Realistic Noise**: Model uncertainty and attribution errors
- **Comprehensive Metrics**: Business KPIs and operational statistics

## 🔍 Validation and Quality Checks

The system includes built-in validation:

```python
from utils import validate_data_quality

# Check data quality
quality_report = validate_data_quality(bookings_df)
print(f"Overall quality score: {quality_report['overall_quality_score']:.2%}")
```

**Validation Checks:**
- Date consistency (stay dates after booking dates)
- Price validity (no negative prices or excessive discounts)
- Attribution score bounds (0-1 range)
- Cancellation logic consistency

## 🛠️ Command Line Interface

```bash
# Basic generation
python main.py --scenario standard

# Generate with custom file prefix
python main.py --scenario luxury --output-prefix "q1_2024_"

# Validation only (no data generation)
python main.py --validate-only --scenario budget

# Compare all scenarios
python main.py --compare-scenarios
```

## 📋 Use Cases

### Revenue Management
- Test dynamic pricing strategies across different demand patterns
- Evaluate campaign effectiveness and attribution models
- Analyze cancellation impact on revenue optimization

### Data Science Practice
- Practice feature engineering with realistic hospitality data
- Build predictive models for booking demand and pricing
- Develop customer segmentation and lifetime value models

### Business Intelligence
- Create dashboards and reporting systems
- Scenario planning for different market conditions  
- Benchmark attribution model performance

## 🤝 Contributing

This tool is designed to be easily extensible:

1. **Add New Scenarios**: Create configurations in `scenarios.py`
2. **Extend Business Logic**: Modify rules in `config.py`  
3. **Add Analysis Features**: Extend functions in `utils.py`
4. **Improve Generation**: Enhance algorithms in `data_generator.py`

## 📝 Technical Notes

**Performance**: Generates ~10,000-50,000 bookings per run depending on configuration
**Memory Usage**: Moderate - stores all data in memory during generation
**Output Size**: CSV files typically 1-10MB depending on scenario
**Dependencies**: pandas, numpy, matplotlib, seaborn, pickle

## 🎯 Next Steps

After generating data, you can:
1. **Explore the data** using the analysis utilities
2. **Build predictive models** for demand forecasting
3. **Test attribution models** against ground truth
4. **Create dashboards** for business intelligence
5. **Experiment with pricing strategies** using different scenarios

## 🆘 Support

For issues or questions:
1. Check the validation output for configuration errors
2. Review the generated summary reports for data quality
3. Use the built-in analysis tools to understand patterns
4. Modify scenarios gradually to understand parameter impacts

---

*Generated with ❤️*