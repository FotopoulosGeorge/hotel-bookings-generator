"""
Fixed ML Visualizations - Backend Issue Resolved
This version ensures matplotlib uses a non-interactive backend
"""

# CRITICAL: Set backend BEFORE any matplotlib imports
import os
import sys

# Force matplotlib to use non-interactive backend
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg', force=True)  # Force non-interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from scipy import stats
from scipy.stats import normaltest, jarque_bera
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import argparse

warnings.filterwarnings('ignore')

# Verify backend is set correctly
print(f"Matplotlib backend: {matplotlib.get_backend()}")

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100


class MLDataVisualizer:
    """Fixed ML-focused visualization tool for hotel booking data"""
    
    def __init__(self, data_prefix='', output_dir='ml_visualizations/'):
        self.data_prefix = data_prefix
        self.output_dir = output_dir
        self.bookings_df = None
        self.campaigns_df = None
        self.customers_df = None
        self.attribution_df = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load hotel booking data from CSV files"""
        print(f"Loading data with prefix: '{self.data_prefix}'")
        
        try:
            # Load CSV files
            self.bookings_df = pd.read_csv(f"output/{self.data_prefix}historical_bookings.csv")
            self.campaigns_df = pd.read_csv(f"output/{self.data_prefix}campaigns_run.csv")
            self.customers_df = pd.read_csv(f"output/{self.data_prefix}customer_segments.csv")
            self.attribution_df = pd.read_csv(f"output/{self.data_prefix}attribution_ground_truth.csv")
            
            # Convert date columns
            date_columns = ['booking_date', 'stay_start_date', 'stay_end_date', 'cancellation_date']
            for col in date_columns:
                if col in self.bookings_df.columns:
                    self.bookings_df[col] = pd.to_datetime(self.bookings_df[col], errors='coerce')
            
            # Add derived features
            self.bookings_df['lead_time'] = (self.bookings_df['stay_start_date'] - self.bookings_df['booking_date']).dt.days
            self.bookings_df['booking_month'] = self.bookings_df['booking_date'].dt.month
            self.bookings_df['stay_month'] = self.bookings_df['stay_start_date'].dt.month
            self.bookings_df['booking_weekday'] = self.bookings_df['booking_date'].dt.dayofweek
            self.bookings_df['stay_weekday'] = self.bookings_df['stay_start_date'].dt.dayofweek
            
            print(f"Data loaded successfully")
            print(f"   Bookings: {len(self.bookings_df):,} records")
            print(f"   Campaigns: {len(self.campaigns_df):,} records")
            print(f"   Customers: {len(self.customers_df):,} records")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_critical_ml_metrics(self):
        """Create visualization focusing on critical ML data quality metrics"""
        print("📊 Creating critical ML metrics dashboard...")
        
        # Force backend verification
        current_backend = matplotlib.get_backend()
        if current_backend.lower() != 'agg':
            print(f"⚠️ Warning: Using {current_backend} backend instead of Agg")
            matplotlib.use('Agg', force=True)
        
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 3, hspace=0.3, wspace=0.25)
        
        fig.suptitle('ML Data Quality Dashboard - Critical Metrics', fontsize=18, fontweight='bold')
        
        # 1. STAY Month Distribution (THE KEY ISSUE)
        ax1 = fig.add_subplot(gs[0, :])
        stay_counts = self.bookings_df['stay_month'].value_counts().sort_index()
        stay_pct = (stay_counts / stay_counts.sum() * 100)
        
        bars = ax1.bar(stay_counts.index, stay_pct.values, color='lightcoral', alpha=0.7, edgecolor='black')
        
        # Add target lines for seasonal hotels
        if len(stay_counts) <= 5:  # Likely seasonal
            targets = {5: 18, 6: 22, 7: 26, 8: 24, 9: 10}
            for month, target in targets.items():
                if month in stay_counts.index:
                    ax1.axhline(y=target, color='green', linestyle='--', alpha=0.5)
                    ax1.text(month + 0.3, target + 0.5, f'Target: {target}%', fontsize=9, color='green')
        
        # Highlight problematic months
        for i, (month, pct) in enumerate(stay_pct.items()):
            ax1.text(month, pct + 0.5, f'{pct:.1f}%', ha='center', fontweight='bold')
            if len(stay_counts) <= 5 and month in targets and abs(pct - targets[month]) > 5:
                if i < len(bars):  # Safety check
                    bars[i].set_color('red')
                    bars[i].set_alpha(0.8)
        
        ax1.set_title('STAY Month Distribution - Critical for Seasonal Hotels', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Percentage of Stays (%)')
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax1.grid(True, alpha=0.3)
        
        # 2. Price Distribution Check (ML Critical)
        ax4 = fig.add_subplot(gs[1, 2])
        
        # Get room types that actually exist in the data
        available_room_types = self.bookings_df['room_type'].unique()
        expected_order = ['Standard', 'Deluxe', 'Suite', 'Premium']
        existing_rooms = [rt for rt in expected_order if rt in available_room_types]
        
        if len(existing_rooms) >= 2:
            price_stats = self.bookings_df.groupby('room_type')['final_price'].agg(['mean', 'std'])
            means = [price_stats.loc[rt, 'mean'] for rt in existing_rooms]
            stds = [price_stats.loc[rt, 'std'] for rt in existing_rooms]
            
            bars = ax4.bar(existing_rooms, means, yerr=stds, capsize=5, alpha=0.7)
            
            # Check if prices increase monotonically (with tolerance)
            prices_correct = True
            for i in range(len(means)-1):
                if means[i] > means[i+1] * 1.05:  # 5% tolerance
                    prices_correct = False
                    break
            
            bar_color = 'green' if prices_correct else 'red'
            for bar in bars:
                bar.set_color(bar_color)
                
            ax4.set_title(f'Price Hierarchy {"✅ OK" if prices_correct else "❌ ERROR"}')
        else:
            ax4.text(0.5, 0.5, 'Need 2+ Room Types\nfor Validation', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Price Hierarchy - Insufficient Data')
            
        ax4.set_xlabel('Room Type')
        ax4.set_ylabel('Average Price ($)')
        ax4.grid(True, alpha=0.3)
        
        # 3. Missing Values Heatmap (FIXED)
        ax5 = fig.add_subplot(gs[2, 0])
        missing_data = self.bookings_df.isnull().sum()
        
        # FIXED: Proper expected missing columns
        expected_missing_cols = ['campaign_id', 'cancellation_date', 'attribution_score']
        critical_missing = missing_data[
            (missing_data > 0) & 
            (~missing_data.index.isin(expected_missing_cols))
        ]
        missing_pct = (critical_missing / len(self.bookings_df) * 100)
        
        if len(missing_pct) > 0:
            colors = ['red' if pct > 5 else 'orange' if pct > 1 else 'yellow' for pct in missing_pct.values]
            bars = ax5.barh(range(len(missing_pct)), missing_pct.values, color=colors, alpha=0.7)
            ax5.set_yticks(range(len(missing_pct)))
            ax5.set_yticklabels(missing_pct.index)
            ax5.set_xlabel('Missing %')
            
            for i, pct in enumerate(missing_pct.values):
                ax5.text(pct + 0.1, i, f'{pct:.2f}%', va='center')
        else:
            ax5.text(0.5, 0.5, 'No Unexpected\nMissing Values', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=16, color='green')
        
        ax5.set_title('Unexpected Missing Data')
        ax5.grid(True, alpha=0.3)
        
        # 4. Data Quality Score
        ax9 = fig.add_subplot(gs[3, 2])
        ax9.axis('off')
        
        # Calculate ML readiness score
        ml_score = 100
        issues = []
        
        # Check for issues
        if len(missing_pct) > 0 and missing_pct.max() > 5:
            ml_score -= 20
            issues.append("High missing values")
        
        # Check price hierarchy
        if len(existing_rooms) >= 2:
            if not prices_correct:
                ml_score -= 15
                issues.append("Price hierarchy violation")
        
        # Check for negative prices
        negative_prices = (self.bookings_df['final_price'] <= 0).sum()
        if negative_prices > 0:
            ml_score -= 20
            issues.append(f"Negative prices ({negative_prices})")
        
        # Check date consistency
        invalid_dates = (self.bookings_df['stay_start_date'] <= self.bookings_df['booking_date']).sum()
        if invalid_dates > 0:
            ml_score -= 15
            issues.append(f"Invalid dates ({invalid_dates})")
        
        # FIXED: Dynamic May spike check
        if len(stay_counts) <= 5:  # Seasonal hotel
            may_actual = stay_pct.get(5, 0)
            may_target = 18  # Expected for seasonal
            if may_actual > may_target * 1.4:  # 40% above target
                ml_score -= 15
                issues.append(f"May spike: {may_actual:.1f}%")
        
        # Display score
        if ml_score >= 90:
            score_color = 'darkgreen'
            status = 'EXCELLENT'
        elif ml_score >= 75:
            score_color = 'green'
            status = 'GOOD'
        elif ml_score >= 60:
            score_color = 'orange'
            status = 'FAIR'
        else:
            score_color = 'red'
            status = 'POOR'
        
        ax9.text(0.5, 0.7, f'ML Readiness Score', ha='center', fontsize=16, fontweight='bold')
        ax9.text(0.5, 0.5, f'{ml_score}/100', ha='center', fontsize=48, fontweight='bold', color=score_color)
        ax9.text(0.5, 0.25, status, ha='center', fontsize=14, fontweight='bold', color=score_color)
        
        if issues:
            ax9.text(0.5, 0.1, 'Issues:', ha='center', fontsize=12, fontweight='bold')
            issues_text = '\n'.join([f'• {issue}' for issue in issues[:3]])
            ax9.text(0.5, 0.02, issues_text, ha='center', fontsize=10, va='top')
        
        # Force close any open figures to prevent memory issues
        plt.tight_layout()
        
        # Save with explicit path and close immediately
        output_path = f'{self.output_dir}01_ml_critical_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Explicitly close the figure
        
        print(f"   ✅ Critical metrics saved to {output_path}")
        
        return ml_score, issues
    
    def generate_all_visualizations(self):
        """Generate all ML-focused visualizations with proper backend handling"""
        print("GENERATING ML-FOCUSED VISUALIZATIONS")
        print("=" * 60)
        
        # Verify backend
        print(f"Using matplotlib backend: {matplotlib.get_backend()}")
        
        if not self.load_data():
            return False
        
        print(f"Output directory: {self.output_dir}")
        
        try:
            # Generate critical metrics (the main one)
            ml_score, issues = self.create_critical_ml_metrics()
            print("Created critical ML metrics dashboard")
            
            # Create a simple summary HTML
            self.create_simple_summary(ml_score, issues)
            
            print(f"\n🎉 Visualizations completed successfully!")
            print(f"📁 Check the '{self.output_dir}' directory")
            print(f"📊 ML Readiness Score: {ml_score}/100")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_simple_summary(self, ml_score, issues):
        """Create a simple HTML summary"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Data Quality Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .score {{ font-size: 48px; font-weight: bold; text-align: center; 
                         color: {'green' if ml_score >= 75 else 'orange' if ml_score >= 60 else 'red'}; }}
                .summary {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
            </style>
        </head>
        <body>
            <h1>Hotel Booking Data - ML Quality Summary</h1>
            <div class="summary">
                <h2>ML Readiness Score</h2>
                <div class="score">{ml_score}/100</div>
                <h3>Status: {'EXCELLENT' if ml_score >= 90 else 'GOOD' if ml_score >= 75 else 'FAIR' if ml_score >= 60 else 'POOR'}</h3>
                
                <h3>Dataset Info:</h3>
                <ul>
                    <li>Total Bookings: {len(self.bookings_df):,}</li>
                    <li>Date Range: {self.bookings_df['booking_date'].min().strftime('%Y-%m-%d')} to {self.bookings_df['booking_date'].max().strftime('%Y-%m-%d')}</li>
                    <li>Total Revenue: ${self.bookings_df['final_price'].sum():,.2f}</li>
                </ul>
                
                {"<h3>Issues Found:</h3><ul>" + "".join([f"<li>{issue}</li>" for issue in issues]) + "</ul>" if issues else "<h3>✅ No critical issues found!</h3>"}
            </div>
            
            <h2>Visualizations</h2>
            <img src="01_ml_critical_metrics.png" alt="Critical ML Metrics" style="max-width: 100%;">
        </body>
        </html>
        """
        
        with open(f'{self.output_dir}summary.html', 'w', encoding='utf-8') as f:
            f.write(html_content)


def main():
    """Main function with backend verification"""
    print("🎨 ML-Focused Data Visualizations (Backend Fixed)")
    print("=" * 60)
    
    # Verify matplotlib backend
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    if matplotlib.get_backend().lower() != 'agg':
        print("⚠️ Warning: Backend not set to Agg, forcing...")
        matplotlib.use('Agg', force=True)
        print(f"Backend now: {matplotlib.get_backend()}")
    
    parser = argparse.ArgumentParser(description='Generate ML visualizations for hotel booking data')
    parser.add_argument('--prefix', type=str, default='', help='Data file prefix')
    parser.add_argument('--output-dir', type=str, default='ml_visualizations/', help='Output directory')
    
    args = parser.parse_args()
    
    visualizer = MLDataVisualizer(data_prefix=args.prefix, output_dir=args.output_dir)
    success = visualizer.generate_all_visualizations()
    
    if success:
        print(f"\n✨ Success! Check '{args.output_dir}summary.html' for results")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())