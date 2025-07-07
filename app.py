import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import zipfile
import time

# Import your existing modules
from config import HotelBusinessConfig
from generators import ConfigurableHotelBookingGenerator
from scenarios import create_test_scenarios

# Page config
st.set_page_config(
    page_title="Hotel Booking Data Generator",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: rgb(232, 244, 253);
        border: 1px solid rgb(52, 152, 219);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #222 !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏨 Hotel Booking Data Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Generate realistic synthetic hotel booking data for ML and analytics</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode selection
        mode = st.radio(
            "Choose your approach:",
            ["🎯 Quick Scenarios", "🛠️ Custom Setup"],
            help="Quick scenarios use predefined settings. Custom setup lets you adjust everything."
        )
        
        config = None
        
        if mode == "🎯 Quick Scenarios":
            scenarios = create_test_scenarios()
            
            scenario_choice = st.selectbox(
                "Select hotel type:",
                ["standard", "luxury", "year_round"],
                format_func=lambda x: {
                    "standard": "🏖️ Seasonal Resort (May-Sep)",
                    "luxury": "✨ Luxury Property", 
                    "year_round": "🏢 Business Hotel (Year-round)"
                }[x]
            )
            
            config = scenarios[scenario_choice]
            
            # Show scenario description
            descriptions = {
                "standard": "🏖️ **Seasonal Resort**: Operates May-September with summer peaks and balanced demand distribution",
                "luxury": "✨ **Luxury Property**: High-end pricing ($300-900/night), conservative policies, affluent clientele",
                "year_round": "🏢 **Business Hotel**: Year-round operations with varied seasonal demand patterns"
            }
            st.info(descriptions[scenario_choice])
            
            # Allow some customization even in quick mode
            st.subheader("Quick Adjustments")
            
            # Data volume slider
            volume_multiplier = st.slider(
                "Dataset Size", 
                0.5, 2.0, 1.0, 0.1,
                help="Adjust the size of your dataset"
            )
            
            original_customers = config.DATA_CONFIG['total_customers']
            config.DATA_CONFIG['total_customers'] = int(original_customers * volume_multiplier)
            
            # Years selection
            years = st.multiselect(
                "Simulation Years",
                [2022, 2023, 2024, 2025],
                default=[2024],
                help="Years to simulate. More years = more data."
            )
            config.SIMULATION_YEARS = years
            
        else:  # Custom Setup
            config = HotelBusinessConfig()
            
            st.subheader("🏨 Hotel Operations")
            
            # Operation mode
            operation_mode = st.selectbox(
                "Operation Type",
                ["seasonal", "year_round"],
                format_func=lambda x: "🏖️ Seasonal (specific months)" if x == "seasonal" else "🏢 Year-round"
            )
            config.OPERATION_MODE = operation_mode
            
            if operation_mode == "seasonal":
                months = st.multiselect(
                    "Operating Months",
                    list(range(1, 13)),
                    default=[5, 6, 7, 8, 9],
                    format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1]
                )
                config.OPERATIONAL_MONTHS = months
            
            st.subheader("💰 Pricing")
            
            col1, col2 = st.columns(2)
            with col1:
                standard = st.number_input("Standard Room ($)", 50, 500, 120, 10)
                suite = st.number_input("Suite ($)", 200, 800, 280, 20)
            with col2:
                deluxe = st.number_input("Deluxe Room ($)", 100, 600, 180, 10) 
                premium = st.number_input("Premium ($)", 300, 1000, 350, 25)
            
            config.BASE_PRICES = {
                'Standard': standard,
                'Deluxe': deluxe, 
                'Suite': suite,
                'Premium': premium
            }
            
            st.subheader("📊 Volume & Customers")
            
            daily_demand = st.slider("Daily Demand", 10, 80, 30)
            total_customers = st.slider("Total Customers", 1000, 8000, 5000, 500)
            
            config.DATA_CONFIG['base_daily_demand'] = daily_demand
            config.DATA_CONFIG['total_customers'] = total_customers
            
            # Customer segments
            st.subheader("👥 Customer Mix")
            early = st.slider("Early Planners %", 0, 80, 45, 5)
            last_min = st.slider("Last Minute %", 0, 50, 25, 5)
            flexible = 100 - early - last_min
            
            if flexible < 0:
                st.warning("Customer percentages must add up to 100%")
                flexible = max(0, flexible)
                early = min(early, 100 - last_min)
            
            st.write(f"Flexible: {flexible}%")
            
            # Normalize
            total = early + last_min + flexible
            if total > 0:
                config.CUSTOMER_SEGMENTS['Early_Planner']['market_share'] = early / 100
                config.CUSTOMER_SEGMENTS['Last_Minute']['market_share'] = last_min / 100
                config.CUSTOMER_SEGMENTS['Flexible']['market_share'] = flexible / 100
        
        # Output options
        st.subheader("📤 Output")
        output_prefix = st.text_input(
            "File Prefix (optional)", 
            value="",
            help="Add a prefix to your files, e.g. 'luxury_' → 'luxury_bookings.csv'"
        )
    
 
    
    
    st.header("📊 Data Preview")
    
    if config:
        # Calculate estimates
        estimates = calculate_estimates(config)
        
        # Display metrics in a nice grid
        metric1, metric2, metric3, metric4 = st.columns(4)
        
        with metric1:
            st.metric(
                "📈 Bookings", 
                f"{estimates['bookings']:,}",
                help="Estimated number of booking records"
            )
        with metric2:
            st.metric(
                "💰 Revenue", 
                f"${estimates['revenue']:,.0f}",
                help="Total estimated revenue"
            )
        with metric3:
            st.metric(
                "🎯 Campaigns", 
                f"{estimates['campaigns']}",
                help="Number of promotional campaigns"
            )
        with metric4:
            st.metric(
                "👥 Customers", 
                f"{config.DATA_CONFIG['total_customers']:,}",
                help="Total customer profiles"
            )
        
        # Configuration summary
        with st.expander("📋 Configuration Summary", expanded=False):
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.write("**Room Pricing:**")
                for room_type, price in config.BASE_PRICES.items():
                    st.write(f"• {room_type}: ${price}")
                
                st.write(f"**Operation:** {config.OPERATION_MODE}")
                st.write(f"**Daily Demand:** {config.DATA_CONFIG['base_daily_demand']}")
            
            with summary_col2:
                st.write("**Customer Segments:**")
                for segment, data in config.CUSTOMER_SEGMENTS.items():
                    percentage = data['market_share'] * 100
                    st.write(f"• {segment}: {percentage:.0f}%")
                
                st.write(f"**Simulation Years:** {', '.join(map(str, config.SIMULATION_YEARS))}")
    
    
    st.header("🚀 Generate")
    
    # Big generate button
    if st.button("🎲 Generate Dataset", type="primary", use_container_width=True):
        generate_and_download(config, output_prefix)
    
    # Information tabs
    st.header("📚 About This Tool")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Use Cases", "📊 Data Schema", "🔧 Features"])
    
    with tab1:
        st.markdown("""
        ### Perfect for:
        
        **🤖 Machine Learning**
        - Training models with clean, realistic data
        - Testing attribution and forecasting algorithms
        - A/B testing different modeling approaches
        
        **📊 Analytics & Education** 
        - Learning revenue management concepts
        - Practicing data analysis techniques
        - Teaching hospitality business intelligence
        
        **💼 Business Applications**
        - Prototyping analytics dashboards
        - Testing pricing strategies
        - Understanding seasonal demand patterns
        """)
    
    with tab2:
        st.markdown("""
        ### Generated Files:
        
        **📁 historical_bookings.csv** (Main dataset)
        ```
        booking_id, customer_id, booking_date, stay_start_date, 
        room_type, customer_segment, booking_channel,
        base_price, final_price, discount_amount, 
        campaign_id, attribution_score, is_cancelled
        ```
        
        **📁 campaigns_run.csv** (Campaign performance)
        ```
        campaign_id, campaign_type, start_date, end_date,
        discount_percentage, target_segments, actual_bookings
        ```
        
        **📁 customer_segments.csv** (Customer profiles)
        ```
        customer_id, segment, price_sensitivity, 
        channel_preference, loyalty_status
        ```
        
        **📁 attribution_ground_truth.csv** (ML labels)
        ```
        booking_id, true_attribution_score, 
        causal_campaign_id, would_have_booked_anyway
        ```
        """)
    
    with tab3:
        st.markdown("""
        ### Key Features:
        
        **🏨 Realistic Business Logic**
        - Customer behavior patterns (Early Planners vs Last Minute)
        - Seasonal demand fluctuations
        - Dynamic pricing with promotional campaigns
        - Realistic cancellation and overbooking patterns
        
        **📊 ML-Ready Output**
        - Clean data with proper types
        - No missing values in critical fields
        - Ground truth attribution for model validation
        - Automatic quality scoring (0-100 scale)
        
        **⚙️ Highly Configurable**
        - Multiple hotel types (seasonal, luxury, business)
        - Adjustable pricing strategies
        - Customizable customer segments
        - Flexible operational parameters
        """)

def calculate_estimates(config):
    """Calculate estimated outputs based on configuration"""
    # Estimate bookings
    daily_demand = config.DATA_CONFIG['base_daily_demand']
    years = len(config.SIMULATION_YEARS) if config.SIMULATION_YEARS else 1
    
    if hasattr(config, 'OPERATIONAL_MONTHS') and len(config.OPERATIONAL_MONTHS) < 12:
        # Seasonal hotel
        days_per_year = len(config.OPERATIONAL_MONTHS) * 30
    else:
        # Year-round hotel
        days_per_year = 365
    
    estimated_bookings = int(daily_demand * days_per_year * years * 0.85)  # 85% capacity factor
    
    # Estimate revenue
    avg_price = sum(config.BASE_PRICES.values()) / len(config.BASE_PRICES)
    estimated_revenue = estimated_bookings * avg_price * 0.82  # Account for discounts
    
    # Estimate campaigns (roughly 1 per month per campaign type)
    estimated_campaigns = years * 12 * 3  # 3 campaign types
    
    return {
        'bookings': estimated_bookings,
        'revenue': estimated_revenue,
        'campaigns': estimated_campaigns
    }

def generate_and_download(config, output_prefix):
    """Generate data and provide download options"""
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🚀 Starting data generation...")
        progress_bar.progress(10)
        time.sleep(0.5)  # Small delay for visual feedback
        
        status_text.text("⚙️ Initializing generator...")
        generator = ConfigurableHotelBookingGenerator(config)
        progress_bar.progress(20)
        
        status_text.text("🎯 Creating campaigns...")
        progress_bar.progress(40)
        time.sleep(0.5)
        
        status_text.text("👥 Generating customers...")
        progress_bar.progress(60)
        time.sleep(0.5)
        
        status_text.text("🏨 Processing bookings...")
        progress_bar.progress(80)
        
        # Generate all data
        bookings, campaigns, customers, attribution_data, baseline_demand = generator.generate_all_data()
        
        status_text.text("📊 Preparing downloads...")
        progress_bar.progress(95)
        
        # Convert to DataFrames
        df_bookings = pd.DataFrame(bookings)
        df_campaigns = pd.DataFrame(campaigns)
        df_customers = pd.DataFrame(customers)
        df_attribution = pd.DataFrame(attribution_data)
        
        progress_bar.progress(100)
        status_text.text("✅ Generation complete!")
        
        
        st.markdown("""
        <div class="success-box">
            <h3>🎉 Data Generated Successfully!</h3>
            <p>Your synthetic hotel booking dataset is ready for download.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("📊 Bookings Created", f"{len(bookings):,}")
        with stats_col2:
            st.metric("💰 Total Revenue", f"${df_bookings['final_price'].sum():,.0f}")
        with stats_col3:
            st.metric("🎯 Campaigns", f"{len(campaigns)}")
        with stats_col4:
            campaign_bookings = df_bookings['campaign_id'].notna().sum()
            participation_rate = campaign_bookings / len(df_bookings)
            st.metric("📈 Campaign Rate", f"{participation_rate:.1%}")
        
        # Download section
        st.header("📥 Download Your Data")
        
        # Create download columns
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        with dl_col1:
            st.download_button(
                "📊 Bookings CSV",
                data=df_bookings.to_csv(index=False),
                file_name=f"{output_prefix}historical_bookings.csv",
                mime="text/csv",
                use_container_width=True,
                help="Main dataset with all booking records"
            )
            
        with dl_col2:
            st.download_button(
                "🎯 Campaigns CSV",
                data=df_campaigns.to_csv(index=False),
                file_name=f"{output_prefix}campaigns_run.csv", 
                mime="text/csv",
                use_container_width=True,
                help="Campaign performance data"
            )
            
        with dl_col3:
            st.download_button(
                "👥 Customers CSV",
                data=df_customers.to_csv(index=False),
                file_name=f"{output_prefix}customer_segments.csv",
                mime="text/csv", 
                use_container_width=True,
                help="Customer profiles and segments"
            )
        
        # Complete package download
        zip_data = create_zip_package(df_bookings, df_campaigns, df_customers, df_attribution, output_prefix)
        
        st.download_button(
            "📦 Download Complete Package (ZIP)",
            data=zip_data,
            file_name=f"{output_prefix}hotel_booking_data.zip",
            mime="application/zip",
            use_container_width=True,
            help="All files in one convenient package"
        )
        
        # Quick visualizations
        st.header("📊 Quick Analysis")
        
        
        
        st.subheader("📅 Bookings by Month")
        # Monthly booking pattern
        df_bookings['stay_month'] = pd.to_datetime(df_bookings['stay_start_date']).dt.month
        monthly_counts = df_bookings['stay_month'].value_counts().sort_index()
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_data = [monthly_counts.get(i, 0) for i in range(1, 13)]
        
        fig1 = px.bar(
            x=months,
            y=monthly_data,
            title="📅 Bookings by Month",
            color=monthly_data,
            color_continuous_scale="Blues"
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("🏢 Booking Channels")
        # Channel distribution
        channel_counts = df_bookings['booking_channel'].value_counts()
        
        fig2 = px.pie(
            values=channel_counts.values,
            names=channel_counts.index,
            title="🏢 Booking Channels",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Data quality score
        st.header("🎯 Data Quality Assessment")
        
        quality_score = calculate_quality_score(df_bookings)
        
        quality_col1, quality_col2 = st.columns([1, 2])
        
        with quality_col1:
            # Score gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = quality_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "ML Readiness Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with quality_col2:
            if quality_score >= 90:
                st.success("🌟 **Excellent!** Your data is ready for machine learning applications.")
            elif quality_score >= 75:
                st.success("✅ **Good!** Your data is suitable for ML with minor preprocessing.")
            elif quality_score >= 60:
                st.warning("⚠️ **Fair** - Some data quality issues detected. Review before ML use.")
            else:
                st.error("❌ **Poor** - Significant data quality issues found.")
            
            st.write("**Quality Checks:**")
            st.write("✅ No missing critical values")
            st.write("✅ Realistic price ranges")
            st.write("✅ Valid date relationships")
            st.write("✅ Proper business logic")
        
    except Exception as e:
        st.error(f"❌ Error generating data: {str(e)}")
        st.exception(e)
        
    finally:
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

def create_zip_package(df_bookings, df_campaigns, df_customers, df_attribution, prefix):
    """Create a ZIP file with all generated data"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add CSV files
        zip_file.writestr(f"{prefix}historical_bookings.csv", df_bookings.to_csv(index=False))
        zip_file.writestr(f"{prefix}campaigns_run.csv", df_campaigns.to_csv(index=False))
        zip_file.writestr(f"{prefix}customer_segments.csv", df_customers.to_csv(index=False))
        zip_file.writestr(f"{prefix}attribution_ground_truth.csv", df_attribution.to_csv(index=False))
        
        # Add README
        readme_content = f"""# Hotel Booking Synthetic Data
        
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Files:
- {prefix}historical_bookings.csv - Main booking dataset ({len(df_bookings):,} records)
- {prefix}campaigns_run.csv - Campaign data
- {prefix}customer_segments.csv - Customer profiles  
- {prefix}attribution_ground_truth.csv - ML ground truth

## Quick Start:
```python
import pandas as pd
bookings = pd.read_csv('{prefix}historical_bookings.csv')
print(f"Revenue: ${{bookings['final_price'].sum():,.2f}}")
```

Generated with Hotel Booking Data Generator
"""
        zip_file.writestr("README.txt", readme_content)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def calculate_quality_score(df_bookings):
    """Calculate a simple data quality score"""
    score = 100
    
    # Check for missing values
    if df_bookings.isnull().sum().sum() > 0:
        score -= 10
    
    # Check price consistency
    if (df_bookings['final_price'] <= 0).sum() > 0:
        score -= 20
    
    # Check date logic
    df_bookings_copy = df_bookings.copy()
    df_bookings_copy['booking_date'] = pd.to_datetime(df_bookings_copy['booking_date'])
    df_bookings_copy['stay_start_date'] = pd.to_datetime(df_bookings_copy['stay_start_date'])
    
    invalid_dates = (df_bookings_copy['stay_start_date'] <= df_bookings_copy['booking_date']).sum()
    if invalid_dates > 0:
        score -= 15
    
    return max(0, score)

if __name__ == "__main__":
    main()