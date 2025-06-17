#!/usr/bin/env python3
"""
Hotel Booking Generator - Setup Verification Script

Run this script after refactoring to verify everything is working correctly.
"""

import sys
import os
import traceback
from datetime import datetime


def check_imports():
    """Verify all imports work correctly"""
    print("🔍 Checking imports...")
    
    try:
        # Core imports
        from config import HotelBusinessConfig
        print("   ✅ config.py imported successfully")
        
        from scenarios import create_test_scenarios
        print("   ✅ scenarios.py imported successfully")
        
        # New modular imports
        from generators import ConfigurableHotelBookingGenerator
        print("   ✅ generators package imported successfully")
        
        from generators.campaign_generator import CampaignGenerator
        from generators.customer_generator import CustomerGenerator
        from generators.pricing_engine import PricingEngine
        from generators.inventory_manager import InventoryManager
        from generators.booking_logic import BookingLogic
        print("   ✅ All generator components imported successfully")
        
        from processors import DataProcessor
        print("   ✅ processors package imported successfully")
        
        # Utility imports
        from utils import load_generated_data
        print("   ✅ utils.py imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False


def check_component_initialization():
    """Verify all components can be initialized"""
    print("\n🏗️  Checking component initialization...")
    
    try:
        from config import HotelBusinessConfig
        from generators import ConfigurableHotelBookingGenerator
        
        config = HotelBusinessConfig()
        generator = ConfigurableHotelBookingGenerator(config)
        
        # Check all components are initialized
        components = [
            ('campaign_generator', generator.campaign_generator),
            ('customer_generator', generator.customer_generator),
            ('pricing_engine', generator.pricing_engine),
            ('inventory_manager', generator.inventory_manager),
            ('booking_logic', generator.booking_logic),
            ('data_processor', generator.data_processor)
        ]
        
        for name, component in components:
            if component is not None:
                print(f"   ✅ {name} initialized successfully")
            else:
                print(f"   ❌ {name} failed to initialize")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Component initialization failed: {e}")
        traceback.print_exc()
        return False


def check_individual_components():
    """Test individual components work independently"""
    print("\n🔧 Testing individual components...")
    
    try:
        from config import HotelBusinessConfig
        from generators.campaign_generator import CampaignGenerator
        from generators.customer_generator import CustomerGenerator
        from generators.pricing_engine import PricingEngine
        
        config = HotelBusinessConfig()
        
        # Test campaign generator
        campaign_gen = CampaignGenerator(config)
        campaigns = campaign_gen.generate_campaigns()
        if len(campaigns) > 0:
            print("   ✅ Campaign generator works independently")
        else:
            print("   ❌ Campaign generator returned no campaigns")
            return False
        
        # Test customer generator
        config.DATA_CONFIG['total_customers'] = 10  # Small test
        customer_gen = CustomerGenerator(config)
        customers = customer_gen.generate_customers()
        if len(customers) == 10:
            print("   ✅ Customer generator works independently")
        else:
            print("   ❌ Customer generator returned wrong number of customers")
            return False
        
        # Test pricing engine
        pricing_engine = PricingEngine(config)
        base_price = pricing_engine.get_base_price_for_date(datetime(2024, 7, 15), 'Standard')
        if isinstance(base_price, (int, float)) and base_price > 0:
            print("   ✅ Pricing engine works independently")
        else:
            print("   ❌ Pricing engine returned invalid price")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Individual component test failed: {e}")
        traceback.print_exc()
        return False


def check_scenarios():
    """Verify scenarios work correctly"""
    print("\n🎯 Testing scenarios...")
    
    try:
        from scenarios import create_test_scenarios, validate_scenario
        
        scenarios = create_test_scenarios()
        
        if len(scenarios) > 0:
            print(f"   ✅ Found {len(scenarios)} scenarios")
        else:
            print("   ❌ No scenarios found")
            return False
        
        # Test scenario validation
        for scenario_name, config in scenarios.items():
            if validate_scenario(config, scenario_name):
                print(f"   ✅ {scenario_name} scenario validates correctly")
            else:
                print(f"   ❌ {scenario_name} scenario validation failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Scenario test failed: {e}")
        traceback.print_exc()
        return False


def check_minimal_generation():
    """Test minimal data generation"""
    print("\n📊 Testing minimal data generation...")
    
    try:
        from config import HotelBusinessConfig
        from generators import ConfigurableHotelBookingGenerator
        
        # Create minimal config for fast testing
        config = HotelBusinessConfig()
        config.DATA_CONFIG['total_customers'] = 50
        config.SIMULATION_YEARS = [2024]
        
        generator = ConfigurableHotelBookingGenerator(config)
        
        print("   🔄 Generating minimal dataset (this may take 10-30 seconds)...")
        bookings, campaigns, customers, attribution_data, baseline_demand = generator.generate_all_data()
        
        # Verify outputs
        if (len(bookings) > 0 and 
            len(campaigns) > 0 and 
            len(customers) == 50 and
            len(attribution_data) > 0 and
            len(baseline_demand) > 0):
            print(f"   ✅ Generated {len(bookings)} bookings, {len(campaigns)} campaigns")
            print("   ✅ Minimal data generation successful")
            return True
        else:
            print("   ❌ Generated data has unexpected structure")
            return False
        
    except Exception as e:
        print(f"   ❌ Minimal generation failed: {e}")
        traceback.print_exc()
        return False


def check_backward_compatibility():
    """Test backward compatibility with old interface"""
    print("\n🔄 Testing backward compatibility...")
    
    try:
        # Test that old import style still works
        from generators import ConfigurableHotelBookingGenerator
        from config import HotelBusinessConfig
        
        config = HotelBusinessConfig()
        generator = ConfigurableHotelBookingGenerator(config)
        
        # Test that main method exists and has correct signature
        if hasattr(generator, 'generate_all_data') and callable(generator.generate_all_data):
            print("   ✅ Main interface method exists")
        else:
            print("   ❌ Main interface method missing")
            return False
        
        # Test that it returns expected structure (without running full generation)
        if hasattr(generator, 'config') and generator.config is not None:
            print("   ✅ Configuration accessible")
        else:
            print("   ❌ Configuration not accessible")
            return False
        
        print("   ✅ Backward compatibility maintained")
        return True
        
    except Exception as e:
        print(f"   ❌ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False


def check_dependencies():
    """Check required dependencies are available"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} available")
        except ImportError:
            print(f"   ❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   📋 To install missing packages, run:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True


def main():
    """Run all verification checks"""
    print("🏨 Hotel Booking Generator - Setup Verification")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Running from: {os.getcwd()}")
    print()
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Imports", check_imports),
        ("Component Initialization", check_component_initialization),
        ("Individual Components", check_individual_components),
        ("Scenarios", check_scenarios),
        ("Backward Compatibility", check_backward_compatibility),
        ("Minimal Generation", check_minimal_generation)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ {check_name} check crashed: {e}")
            results.append((check_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:<25}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Total: {passed + failed}, Passed: {passed}, Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All checks passed! Your refactored setup is working correctly.")
        print("\nNext steps:")
        print("1. Try running: python main.py --scenario standard")
        print("2. Run examples: python examples/usage_examples.py")
        print("3. Run tests: python tests/test_refactoring.py")
    else:
        print(f"\n⚠️  {failed} check(s) failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all files are in the correct directories")
        print("2. Check that all imports use the new module structure")
        print("3. Verify dependencies are installed: pip install -r requirements.txt")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)