"""
Test Suite for Refactored Hotel Booking Generator

Simple tests to verify the modular architecture works correctly
and maintains backward compatibility.
"""

import sys
import os
import unittest
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HotelBusinessConfig
from generators import ConfigurableHotelBookingGenerator
from generators.campaign_generator import CampaignGenerator
from generators.customer_generator import CustomerGenerator
from generators.pricing_engine import PricingEngine
from generators.inventory_manager import InventoryManager
from generators.booking_logic import BookingLogic
from processors.data_processors import DataProcessor


class TestRefactoredArchitecture(unittest.TestCase):
    """Test the refactored modular architecture"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = HotelBusinessConfig()
        # Reduce data size for faster tests
        self.config.DATA_CONFIG['total_customers'] = 100
        self.config.SIMULATION_YEARS = [2024]
    
    def test_main_generator_initialization(self):
        """Test that main generator initializes with all components"""
        generator = ConfigurableHotelBookingGenerator(self.config)
        
        # Check all components are initialized
        self.assertIsNotNone(generator.campaign_generator)
        self.assertIsNotNone(generator.customer_generator)
        self.assertIsNotNone(generator.pricing_engine)
        self.assertIsNotNone(generator.inventory_manager)
        self.assertIsNotNone(generator.booking_logic)
        self.assertIsNotNone(generator.data_processor)
        
        # Check component types
        self.assertIsInstance(generator.campaign_generator, CampaignGenerator)
        self.assertIsInstance(generator.customer_generator, CustomerGenerator)
        self.assertIsInstance(generator.pricing_engine, PricingEngine)
        self.assertIsInstance(generator.inventory_manager, InventoryManager)
        self.assertIsInstance(generator.booking_logic, BookingLogic)
        self.assertIsInstance(generator.data_processor, DataProcessor)
    
    def test_campaign_generator(self):
        """Test campaign generator works independently"""
        campaign_gen = CampaignGenerator(self.config)
        campaigns = campaign_gen.generate_campaigns()
        
        self.assertIsInstance(campaigns, list)
        self.assertGreater(len(campaigns), 0)
        
        # Check campaign structure
        campaign = campaigns[0]
        required_fields = [
            'campaign_id', 'campaign_type', 'start_date', 'end_date',
            'discount_percentage', 'target_segments', 'channel'
        ]
        for field in required_fields:
            self.assertIn(field, campaign)
        
        # Check campaign types
        campaign_types = {c['campaign_type'] for c in campaigns}
        expected_types = {'Early_Booking', 'Flash_Sale', 'Special_Offer'}
        self.assertTrue(campaign_types.intersection(expected_types))
    
    def test_customer_generator(self):
        """Test customer generator works independently"""
        customer_gen = CustomerGenerator(self.config)
        customers = customer_gen.generate_customers()
        
        self.assertIsInstance(customers, list)
        self.assertEqual(len(customers), self.config.DATA_CONFIG['total_customers'])
        
        # Check customer structure
        customer = customers[0]
        required_fields = [
            'customer_id', 'segment', 'price_sensitivity', 'planning_horizon',
            'channel_preference', 'loyalty_status'
        ]
        for field in required_fields:
            self.assertIn(field, customer)
        
        # Check segment distribution
        segments = {c['segment'] for c in customers}
        expected_segments = set(self.config.CUSTOMER_SEGMENTS.keys())
        self.assertEqual(segments, expected_segments)
    
    def test_pricing_engine(self):
        """Test pricing engine works independently"""
        pricing_engine = PricingEngine(self.config)
        
        # Test base price calculation
        test_date = datetime(2024, 7, 15)
        base_price = pricing_engine.get_base_price_for_date(test_date, 'Standard')
        
        self.assertIsInstance(base_price, (int, float))
        self.assertGreater(base_price, 0)
        
        # Test pricing for different room types
        room_types = ['Standard', 'Deluxe', 'Suite', 'Premium']
        prices = []
        for room_type in room_types:
            price = pricing_engine.get_base_price_for_date(test_date, room_type)
            prices.append(price)
        
        # Prices should generally increase with room tier
        self.assertLessEqual(prices[0], prices[-1])  # Standard <= Premium
    
    def test_inventory_manager(self):
        """Test inventory manager works independently"""
        inventory_manager = InventoryManager(self.config)
        
        # Test acceptance probability
        test_date = datetime(2024, 7, 15)
        prob = inventory_manager.get_acceptance_probability(test_date, 'Standard')
        
        self.assertIsInstance(prob, (int, float))
        self.assertGreaterEqual(prob, 0)
        self.assertLessEqual(prob, 1)
        
        # Test inventory reservation
        inventory_manager.reserve_inventory(
            test_date, 
            test_date + timedelta(days=2), 
            'Standard'
        )
        
        # Check that reservation affects future acceptance probability
        new_prob = inventory_manager.get_acceptance_probability(test_date, 'Standard')
        self.assertLessEqual(new_prob, prob)
    
    def test_booking_logic(self):
        """Test booking logic works independently"""
        booking_logic = BookingLogic(self.config)
        
        # Test room type selection
        room_type = booking_logic.select_room_type()
        self.assertIn(room_type, self.config.ROOM_TYPES)
        
        # Test stay date generation (simplified)
        customer_gen = CustomerGenerator(self.config)
        customers = customer_gen.generate_customers()
        customer = customers[0]
        
        booking_date = datetime(2024, 3, 15)
        stay_start, stay_end, stay_length = booking_logic.generate_stay_dates(
            booking_date, customer, None
        )
        
        self.assertIsInstance(stay_start, datetime)
        self.assertIsInstance(stay_end, datetime)
        self.assertGreater(stay_start, booking_date)
        self.assertGreater(stay_end, stay_start)
        self.assertGreater(stay_length, 0)
        
        # Check operational season enforcement
        self.assertIn(stay_start.month, self.config.OPERATIONAL_MONTHS)
    
    def test_data_processor(self):
        """Test data processor works independently"""
        data_processor = DataProcessor(self.config)
        
        # Create sample booking data
        sample_bookings = [
            {
                'booking_id': 'BK_1',
                'customer_segment': 'Early_Planner',
                'booking_date': datetime(2024, 3, 1),
                'stay_start_date': datetime(2024, 7, 15),
                'campaign_id': 'EB_1001',
                'attribution_score': 0.8,
                'is_cancelled': False,
                'cancellation_date': None,
                'incremental_flag': True
            }
        ]
        
        # Test cancellation logic
        processed_bookings = data_processor.apply_cancellation_logic(sample_bookings)
        
        self.assertIsInstance(processed_bookings, list)
        self.assertEqual(len(processed_bookings), 1)
        
        booking = processed_bookings[0]
        self.assertIn('is_cancelled', booking)
        self.assertIsInstance(booking['is_cancelled'], bool)
    
    def test_end_to_end_generation(self):
        """Test complete end-to-end data generation"""
        generator = ConfigurableHotelBookingGenerator(self.config)
        
        # Generate data
        bookings, campaigns, customers, attribution_data, baseline_demand = generator.generate_all_data()
        
        # Verify outputs
        self.assertIsInstance(bookings, list)
        self.assertIsInstance(campaigns, list)
        self.assertIsInstance(customers, list)
        self.assertIsInstance(attribution_data, list)
        self.assertIsInstance(baseline_demand, dict)
        
        # Check data consistency
        self.assertGreater(len(bookings), 0)
        self.assertGreater(len(campaigns), 0)
        self.assertEqual(len(customers), self.config.DATA_CONFIG['total_customers'])
        
        # Verify booking structure
        if bookings:
            booking = bookings[0]
            required_fields = [
                'booking_id', 'customer_id', 'booking_date', 'stay_start_date',
                'room_type', 'customer_segment', 'booking_channel', 'final_price'
            ]
            for field in required_fields:
                self.assertIn(field, booking)
    
    def test_backward_compatibility(self):
        """Test that the main interface remains backward compatible"""
        # This test ensures existing code using the generator still works
        generator = ConfigurableHotelBookingGenerator(self.config)
        
        # Test main method exists and works
        self.assertTrue(hasattr(generator, 'generate_all_data'))
        self.assertTrue(callable(generator.generate_all_data))
        
        # Test that it returns the expected structure
        result = generator.generate_all_data()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)  # bookings, campaigns, customers, attribution, demand


class TestComponentIntegration(unittest.TestCase):
    """Test integration between components"""
    
    def setUp(self):
        self.config = HotelBusinessConfig()
        self.config.DATA_CONFIG['total_customers'] = 50
        self.config.SIMULATION_YEARS = [2024]
    
    def test_campaign_customer_integration(self):
        """Test campaign and customer generators work together"""
        campaign_gen = CampaignGenerator(self.config)
        customer_gen = CustomerGenerator(self.config)
        
        campaigns = campaign_gen.generate_campaigns()
        customers = customer_gen.generate_customers()
        
        # Test campaign eligibility logic
        campaign_lookup = campaign_gen.create_campaign_lookup(campaigns)
        
        # Find a campaign date
        campaign_date = list(campaign_lookup.keys())[0]
        customer = customers[0]
        
        # Test campaign selection
        selected_campaign, is_promotional = campaign_gen.select_campaign_for_booking(
            campaign_date, customer, campaign_lookup
        )
        
        # Should return either a campaign or None
        if selected_campaign:
            self.assertIsInstance(selected_campaign, dict)
            self.assertIn('campaign_id', selected_campaign)
        self.assertIsInstance(is_promotional, bool)


def run_tests():
    """Run all tests"""
    print("🧪 Running Hotel Booking Generator Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestRefactoredArchitecture))
    suite.addTest(unittest.makeSuite(TestComponentIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed! Refactoring successful.")
    else:
        print("❌ Some tests failed. Check the output above.")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)