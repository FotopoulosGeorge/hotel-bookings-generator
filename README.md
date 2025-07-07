# Hotel Booking Data Generator

A sophisticated synthetic data generation system that creates realistic hotel booking datasets with complex business logic, customer behavior patterns, and campaign attribution modeling. Perfect for data science, machine learning, and revenue management applications.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate standard hotel data
python main.py

# Generate specific scenario
python main.py --scenario luxury

# See all available scenarios
python main.py --list-scenarios
```

This creates:
- `output/historical_bookings.csv` - Main booking dataset (5K-15K records)
- `output/campaigns_run.csv` - Campaign performance data
- `output/customer_segments.csv` - Customer profiles
- `output/attribution_ground_truth.csv` - ML attribution labels

## 🎯 Available Scenarios

| Scenario | Description | Key Features |
|----------|-------------|--------------|
| **standard** | Seasonal hotel (May-Sep) | Balanced distribution, realistic patterns |
| **luxury** | High-end property | Premium pricing, conservative policies |
| **year_round** | Business hotel | 12-month operations, varied demand |

## 🏗️ Architecture

The system uses a modular architecture for maintainability and extensibility:

```
generators/          # Core data generation logic
├── booking_generator.py    # Main orchestration
├── campaign_generator.py   # Promotional campaigns  
├── customer_generator.py   # Customer profiles & behavior
├── pricing_engine.py       # Dynamic pricing & attribution
├── inventory_manager.py    # Capacity & overbooking logic
└── booking_logic.py        # Stay dates & room selection

processors/          # Data processing & validation
└── data_processors.py      # Post-processing, I/O, validation

examples/           # Usage examples & tutorials
utils.py           # Analysis & utility functions
validator.py       # Comprehensive data validation
```

## 💡 Key Features

### 🏨 Realistic Business Logic
- **Customer Segments**: Early Planner, Last Minute, Flexible (with distinct behaviors)
- **Seasonal Patterns**: Operational seasons, demand fluctuations
- **Revenue Management**: Dynamic pricing, overbooking strategies
- **Campaign Types**: Early Booking, Flash Sales, Special Offers

### 📊 ML-Ready Outputs
- **Clean Datasets**: No missing critical values, proper data types
- **Attribution Ground Truth**: True causal impact for model validation
- **Feature Engineering**: Lead times, seasonality, customer behaviors
- **Quality Validation**: Automated checks with 0-100 ML readiness score

### 🎛️ Highly Configurable
- **Pricing Strategies**: Periodic pricing, seasonal adjustments
- **Customer Behavior**: Segment distributions, cancellation patterns  
- **Campaign Strategy**: Discount ranges, targeting rules, capacity limits
- **Operational Parameters**: Capacity, overbooking, seasonal operations

## 📈 Data Analysis & Validation

### Generate Comprehensive Validation Report
```bash
python validator.py --prefix "luxury_"
```

Creates detailed validation suite with:
- ✅ Temporal pattern analysis
- ✅ Business logic validation  
- ✅ Campaign effectiveness review
- ✅ Statistical property checks
- ✅ ML readiness assessment

### Quick Data Analysis
```python
from utils import load_generated_data, analyze_booking_patterns

# Load and analyze data
data = load_generated_data('luxury_')
analysis = analyze_booking_patterns(data['bookings'])

print(f"Total Revenue: ${analysis['total_revenue']:,.2f}")
print(f"Campaign Participation: {analysis['campaign_participation_rate']:.1%}")
```

## 🔧 Programmatic Usage

### Basic Generation
```python
from config import HotelBusinessConfig
from generators import ConfigurableHotelBookingGenerator

# Create and customize configuration
config = HotelBusinessConfig()
config.BASE_PRICES = {'Standard': 200, 'Premium': 400}
config.DATA_CONFIG['base_daily_demand'] = 25

# Generate data
generator = ConfigurableHotelBookingGenerator(config)
bookings, campaigns, customers, attribution, demand = generator.generate_all_data()
```

### Using Predefined Scenarios
```python
from scenarios import create_test_scenarios

scenarios = create_test_scenarios()
luxury_config = scenarios['luxury']

generator = ConfigurableHotelBookingGenerator(luxury_config)
data = generator.generate_all_data()
```

### Custom Scenarios
```python
# Create boutique hotel scenario
config = HotelBusinessConfig()
config.BASE_PRICES = {'Standard': 300, 'Suite': 600}
config.CUSTOMER_SEGMENTS['Early_Planner']['market_share'] = 0.70
config.OVERBOOKING_CONFIG['base_overbooking_rate'] = 0.03

# Generate with custom prefix
python main.py --scenario luxury --output-prefix "boutique_"
```

## 🎓 Use Cases

### Data Science & ML
- **Predictive Modeling**: Revenue forecasting, demand prediction
- **A/B Testing**: Campaign strategy comparison
- **Feature Engineering**: Customer behavior analysis
- **Model Validation**: Attribution model testing with ground truth

### Revenue Management
- **Pricing Strategy**: Dynamic pricing optimization
- **Campaign Analysis**: Promotional effectiveness measurement  
- **Capacity Management**: Overbooking strategy evaluation
- **Customer Segmentation**: Behavior pattern analysis

### Education & Training
- **ML Practice**: Clean, labeled datasets for learning
- **Business Analytics**: Realistic hospitality data
- **Data Quality**: Validation methodology examples

## 📊 Data Schema

### Bookings Dataset (`historical_bookings.csv`)
```
booking_id, customer_id, booking_date, stay_start_date, stay_end_date,
room_type, customer_segment, booking_channel, base_price, final_price,
discount_amount, campaign_id, attribution_score, incremental_flag,
is_cancelled, cancellation_date
```

### Key Metrics Generated
- **5,000-15,000 bookings** per scenario
- **Revenue**: $2-10M+ depending on scenario
- **Campaign Participation**: 60-80% promotional rate
- **Cancellation Rate**: 8-15% depending on segment
- **Seasonal Distribution**: Realistic patterns with Jul-Aug peaks

## 🚨 Data Quality Assurance

Every generated dataset includes:
- ✅ **Business Rule Validation**: Price hierarchies, date consistency
- ✅ **Statistical Checks**: Distribution analysis, outlier detection  
- ✅ **ML Readiness Score**: 0-100 scale with actionable feedback
- ✅ **Attribution Validation**: Ground truth for model evaluation

Score ≥75 = Ready for ML applications

## 🛠️ Advanced Examples

See `examples/` directory for:
- **`usage_examples.py`** - Comprehensive usage patterns
- **`analysis_example.py`** - Data analysis workflows  
- **`custom_scenario.py`** - Creating custom business scenarios

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.7.0
python-dateutil>=2.8.0
```

## 🧪 Testing

```bash
python tests/test_refactoring.py
```

Validates modular architecture and component integration.


---

**Built with ❤️**