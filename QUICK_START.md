# Quick Start Guide - Hotel Booking Data Generator

Complete workflow for generating, analyzing, and validating ML-ready hotel booking datasets.

## 🚀 5-Minute Quick Start

### Step 1: Generate Your First Dataset
```bash
# Install dependencies
pip install -r requirements.txt

# Generate standard hotel data
python main.py

# This creates:
# - historical_bookings.csv (main dataset)
# - campaigns_run.csv (campaign details)
# - customer_segments.csv (customer profiles)
# - attribution_ground_truth.csv (ML labels)
```

### Step 2: Validate Data Quality
```bash
# Run comprehensive analysis
python examples/analysis_example.py

# This generates:
# - comprehensive_analysis_report.txt (detailed analysis)
# - ML readiness score (0-100)
# - Data quality validation
```

### Step 3: Create Visual Assessment
```bash
# Generate all visualizations
python visualize_data.py

# This creates:
# - visualizations/ directory with 9 plot types
# - visualization_index.html (interactive dashboard)
```

**Total time: ~2-3 minutes for standard dataset generation + analysis**

## 🎯 Choose Your Use Case

### For Revenue Management Practice
```bash
# Generate luxury hotel scenario
python main.py --scenario luxury --output-prefix "luxury_"

# Analyze pricing patterns
python examples/analysis_example.py

# Create revenue-focused visualizations
python visualize_data.py --prefix luxury_
```

### For Campaign Attribution Modeling
```bash
# Generate high-competition scenario (more campaigns)
python main.py --scenario high_competition --output-prefix "campaigns_"

# Focus on campaign analysis
python examples/analysis_example.py

# Visualize campaign effectiveness
python visualize_data.py --prefix campaigns_
```

### For Demand Forecasting
```bash
# Generate seasonal resort data
python main.py --scenario seasonal_resort --output-prefix "seasonal_"

# Analyze temporal patterns
python examples/analysis_example.py

# Create demand forecasting visuals
python visualize_data.py --prefix seasonal_
```

### For Custom Business Scenarios
```bash
# Create and test custom scenarios
python examples/custom_scenario.py

# This generates multiple scenarios:
# - covid_impact_*.csv
# - economic_boom_*.csv
# - new_hotel_*.csv
# - festival_destination_*.csv
```

## 📊 Data Quality Validation Checklist

After generating data, verify these key aspects:

### ✅ Business Logic Validation
- [ ] Room type prices increase: Standard < Deluxe < Suite < Premium
- [ ] Lead times match segments: Last_Minute < Flexible < Early_Planner  
- [ ] Seasonal patterns show Jul-Aug peaks
- [ ] Campaign attribution scores are realistic (0-1 range)

### ✅ ML Readiness Assessment
- [ ] **ML Score ≥ 75**: Data is suitable for machine learning
- [ ] **No missing values** in critical fields
- [ ] **Realistic price ranges** (no negative values)
- [ ] **Proper date relationships** (stay dates after booking dates)
- [ ] **Balanced categorical variables** (no class has <5% representation)

### ✅ Statistical Properties
- [ ] **Correlation structure** makes business sense
- [ ] **Distribution shapes** are realistic (not perfectly normal)
- [ ] **Outliers are reasonable** (luxury suites can be expensive)
- [ ] **Temporal patterns** show expected seasonality

## 🔧 Troubleshooting Common Issues

### Issue: ML Score < 75
**Solution**: Check the analysis report for specific issues
```bash
# Re-run analysis with detailed validation
python examples/analysis_example.py

# Check the comprehensive_analysis_report.txt for issues
```

### Issue: Unrealistic Patterns
**Solution**: Adjust configuration parameters
```python
# Edit config.py to modify business rules
from config import HotelBusinessConfig
config = HotelBusinessConfig()
config.BASE_PRICES['Standard'] = 120  # Adjust as needed
```

### Issue: Poor Visualizations
**Solution**: Regenerate with specific focus
```bash
# Generate scenario-specific visualizations
python visualize_data.py --prefix your_scenario_ --output-dir focused_analysis/
```

## 📈 Next Steps for ML Applications

### 1. Feature Engineering
```python
import pandas as pd

# Load generated data
df = pd.read_csv('historical_bookings.csv')

# Create additional features
df['booking_weekday'] = pd.to_datetime(df['booking_date']).dt.dayofweek
df['is_weekend_stay'] = pd.to_datetime(df['stay_start_date']).dt.dayofweek.isin([5,6])
df['discount_percentage'] = df['discount_amount'] / df['base_price']
df['price_per_night'] = df['final_price'] / df['stay_length']
```

### 2. Train ML Models
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Example: Revenue prediction model
features = ['room_type', 'customer_segment', 'lead_time', 'stay_length', 'booking_month']
target = 'final_price'

# Encode categorical variables
le = LabelEncoder()
for col in ['room_type', 'customer_segment']:
    df[f'{col}_encoded'] = le.fit_transform(df[col])

# Train model
X = df[features + [f'{col}_encoded' for col in ['room_type', 'customer_segment']]]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 3. Validate Attribution Models
```python
# Use ground truth for attribution model validation
attribution_df = pd.read_csv('attribution_ground_truth.csv')

# Compare your attribution model predictions with ground truth
# This is unique to synthetic data - you have the "true" causal relationships!
```

## 🎓 Learning Objectives

This tool helps you practice:

### Data Science Skills
- **Data Quality Assessment**: Learning to identify and fix data issues
- **Exploratory Data Analysis**: Understanding data through visualization
- **Feature Engineering**: Creating meaningful variables for ML
- **Model Validation**: Using ground truth data for model evaluation

### Business Skills  
- **Revenue Management**: Understanding hotel pricing strategies
- **Campaign Analysis**: Measuring marketing effectiveness
- **Customer Segmentation**: Analyzing different customer behaviors
- **Operational Analytics**: Optimizing cancellation and overbooking policies

### Technical Skills
- **Python Data Stack**: pandas, numpy, matplotlib, seaborn, scikit-learn
- **Software Engineering**: Modular code design, configuration management
- **Documentation**: Creating clear project documentation
- **Version Control**: Managing data science projects

## 📁 File Organization for Projects

Recommended project structure for using this data:

```
my_hotel_analysis_project/
├── data/
│   ├── raw/                    # Generated CSV files
│   ├── processed/              # Cleaned data for ML
│   └── external/               # Any additional data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_attribution_analysis.ipynb
├── src/
│   ├── data_processing.py
│   ├── model_training.py
│   └── evaluation.py
├── visualizations/             # Generated plots
├── reports/                    # Analysis reports
└── README.md                   # Project documentation
```

## 🚀 Ready to Start?

1. **Clone/download** the hotel booking generator files
2. **Run the quick start** commands above  
3. **Examine the generated data** using the analysis tools
4. **Create your first ML model** using the clean, labeled data
5. **Validate your models** against the ground truth attribution data

The synthetic data is designed to be immediately usable for machine learning while being realistic enough to teach real-world data science skills.

**Happy analyzing! 🎉**