"""
Hotel Business Configuration Module

This module contains all business rules, parameters, and configuration
classes for the hotel booking data generator.
"""

class HotelBusinessConfig:
    """
    Centralized configuration for all business parameters.
    Modify these values to test different scenarios.
    """
    
    # 🏨 PROPERTY OPERATIONS
    OPERATIONAL_MONTHS = [5, 6, 7, 8, 9]  # May-September
    SIMULATION_YEARS = [2022, 2023, 2024]  # Years to generate data for
    ROOM_TYPES = ['Standard', 'Deluxe', 'Suite', 'Premium']
    
    # 💰 PERIODIC BASE PRICING CONFIGURATION
    PERIODIC_BASE_PRICING = {
        'enabled': True,  # Set to False to use static BASE_PRICES
        'pricing_periods': {
            'Standard': [
                {'start_date': '2022-01-01', 'end_date': '2022-03-31', 'base_price': 110},
                {'start_date': '2022-04-01', 'end_date': '2022-06-30', 'base_price': 125},
                {'start_date': '2022-07-01', 'end_date': '2022-09-30', 'base_price': 140},
                {'start_date': '2022-10-01', 'end_date': '2022-12-31', 'base_price': 120},
                {'start_date': '2023-01-01', 'end_date': '2023-03-31', 'base_price': 115},
                {'start_date': '2023-04-01', 'end_date': '2023-06-30', 'base_price': 130},
                {'start_date': '2023-07-01', 'end_date': '2023-09-30', 'base_price': 145},
                {'start_date': '2023-10-01', 'end_date': '2023-12-31', 'base_price': 125},
                {'start_date': '2024-01-01', 'end_date': '2024-03-31', 'base_price': 120},
                {'start_date': '2024-04-01', 'end_date': '2024-06-30', 'base_price': 135},
                {'start_date': '2024-07-01', 'end_date': '2024-09-30', 'base_price': 150},
                {'start_date': '2024-10-01', 'end_date': '2024-12-31', 'base_price': 130}
            ],
            'Deluxe': [
                {'start_date': '2022-01-01', 'end_date': '2022-03-31', 'base_price': 165},
                {'start_date': '2022-04-01', 'end_date': '2022-06-30', 'base_price': 185},
                {'start_date': '2022-07-01', 'end_date': '2022-09-30', 'base_price': 210},
                {'start_date': '2022-10-01', 'end_date': '2022-12-31', 'base_price': 180},
                {'start_date': '2023-01-01', 'end_date': '2023-03-31', 'base_price': 170},
                {'start_date': '2023-04-01', 'end_date': '2023-06-30', 'base_price': 190},
                {'start_date': '2023-07-01', 'end_date': '2023-09-30', 'base_price': 215},
                {'start_date': '2023-10-01', 'end_date': '2023-12-31', 'base_price': 185},
                {'start_date': '2024-01-01', 'end_date': '2024-03-31', 'base_price': 175},
                {'start_date': '2024-04-01', 'end_date': '2024-06-30', 'base_price': 195},
                {'start_date': '2024-07-01', 'end_date': '2024-09-30', 'base_price': 220},
                {'start_date': '2024-10-01', 'end_date': '2024-12-31', 'base_price': 190}
            ],
            'Suite': [
                {'start_date': '2022-01-01', 'end_date': '2022-03-31', 'base_price': 260},
                {'start_date': '2022-04-01', 'end_date': '2022-06-30', 'base_price': 285},
                {'start_date': '2022-07-01', 'end_date': '2022-09-30', 'base_price': 320},
                {'start_date': '2022-10-01', 'end_date': '2022-12-31', 'base_price': 280},
                {'start_date': '2023-01-01', 'end_date': '2023-03-31', 'base_price': 270},
                {'start_date': '2023-04-01', 'end_date': '2023-06-30', 'base_price': 295},
                {'start_date': '2023-07-01', 'end_date': '2023-09-30', 'base_price': 330},
                {'start_date': '2023-10-01', 'end_date': '2023-12-31', 'base_price': 290},
                {'start_date': '2024-01-01', 'end_date': '2024-03-31', 'base_price': 280},
                {'start_date': '2024-04-01', 'end_date': '2024-06-30', 'base_price': 305},
                {'start_date': '2024-07-01', 'end_date': '2024-09-30', 'base_price': 340},
                {'start_date': '2024-10-01', 'end_date': '2024-12-31', 'base_price': 300}
            ],
            'Premium': [
                {'start_date': '2022-01-01', 'end_date': '2022-03-31', 'base_price': 320},
                {'start_date': '2022-04-01', 'end_date': '2022-06-30', 'base_price': 350},
                {'start_date': '2022-07-01', 'end_date': '2022-09-30', 'base_price': 400},
                {'start_date': '2022-10-01', 'end_date': '2022-12-31', 'base_price': 350},
                {'start_date': '2023-01-01', 'end_date': '2023-03-31', 'base_price': 335},
                {'start_date': '2023-04-01', 'end_date': '2023-06-30', 'base_price': 365},
                {'start_date': '2023-07-01', 'end_date': '2023-09-30', 'base_price': 415},
                {'start_date': '2023-10-01', 'end_date': '2023-12-31', 'base_price': 365},
                {'start_date': '2024-01-01', 'end_date': '2024-03-31', 'base_price': 350},
                {'start_date': '2024-04-01', 'end_date': '2024-06-30', 'base_price': 380},
                {'start_date': '2024-07-01', 'end_date': '2024-09-30', 'base_price': 430},
                {'start_date': '2024-10-01', 'end_date': '2024-12-31', 'base_price': 380}
            ]
        }
    }
    
    # 💰 BASE PRICING ($ per night) - Used when periodic pricing is disabled
    BASE_PRICES = {
        'Standard': 120,
        'Deluxe': 180, 
        'Suite': 280,
        'Premium': 350
    }
    
    # 📈 SEASONAL DEMAND PATTERNS (relative to peak)
    SEASONAL_DEMAND_MULTIPLIERS = {
        5: 0.70,  # May - shoulder season
        6: 0.80,  # June - building to peak  
        7: 1.00,  # July - PEAK season
        8: 1.00,  # August - PEAK season
        9: 0.70   # September - shoulder season
    }
    
    # 📅 WEEKLY DEMAND PATTERNS (Monday=0, Sunday=6)
    WEEKLY_DEMAND_MULTIPLIERS = {
        0: 0.6,   # Monday
        1: 0.7,   # Tuesday  
        2: 0.8,   # Wednesday
        3: 0.9,   # Thursday
        4: 1.2,   # Friday
        5: 1.4,   # Saturday
        6: 1.1    # Sunday
    }
    
    # 🏷️ PROMOTIONAL BUSINESS RULES
    TARGET_CONNECTED_AGENT_SHARE = 0.60      # 60% of bookings via agents
    TARGET_ONLINE_DIRECT_SHARE = 0.40        # 40% of bookings online
    CONNECTED_AGENT_PROMO_RATE = 0.80        # 80% of agent bookings promotional
    ONLINE_DIRECT_PROMO_RATE = 0.70          # 70% of online bookings promotional
    CAMPAIGN_PARTICIPATION_RATE = 0.80       # 80% participation when eligible
    
    # 👥 CUSTOMER SEGMENT DEFINITIONS
    CUSTOMER_SEGMENTS = {
        'Early_Planner': {
            'market_share': 0.45,
            'advance_booking_days': (120, 240),
            'price_sensitivity': 0.8,
            'channel_preference_weights': {
                'Connected_Agent': 0.85,
                'Online_Direct': 0.15
            },
            'loyalty_distribution': {
                'Bronze': 0.5, 'Silver': 0.3, 'Gold': 0.2
            }
        },
        'Last_Minute': {
            'market_share': 0.25,
            'advance_booking_days': (1, 30),
            'price_sensitivity': 0.6,
            'channel_preference_weights': {
                'Connected_Agent': 0.35,
                'Online_Direct': 0.65
            },
            'loyalty_distribution': {
                'Bronze': 0.7, 'Silver': 0.2, 'Gold': 0.1
            }
        },
        'Flexible': {
            'market_share': 0.30,
            'advance_booking_days': (30, 120),
            'price_sensitivity': 0.7,
            'channel_preference_weights': {
                'Connected_Agent': 0.55,
                'Online_Direct': 0.45
            },
            'loyalty_distribution': {
                'Bronze': 0.6, 'Silver': 0.25, 'Gold': 0.15
            }
        }
    }
    
    # 🚫 CANCELLATION & OVERBOOKING CONFIGURATION
    CANCELLATION_CONFIG = {
        'Early_Planner': {
            'base_cancellation_rate': 0.20,
            'lead_time_multiplier': 1.5,
            'min_days_before_stay': 7,
            'cancellation_window_days': (14, 90)
        },
        'Last_Minute': {
            'base_cancellation_rate': 0.08,
            'lead_time_multiplier': 0.5,
            'min_days_before_stay': 1,
            'cancellation_window_days': (1, 14)
        },
        'Flexible': {
            'base_cancellation_rate': 0.15,
            'lead_time_multiplier': 1.0,
            'min_days_before_stay': 3,
            'cancellation_window_days': (7, 60)
        }
    }
    
    OVERBOOKING_CONFIG = {
        'enable_overbooking': True,
        'base_overbooking_rate': 0.10,
        'seasonal_overbooking_multipliers': {
            5: 1.0, 6: 1.1, 7: 1.2, 8: 1.2, 9: 1.0
        },
        'channel_overbooking_rates': {
            'Connected_Agent': 1.0,
            'Online_Direct': 1.1
        },
        'campaign_overbooking_adjustment': {
            'Early_Booking': 0.8,
            'Flash_Sale': 1.3,
            'Special_Offer': 1.0
        }
    }
    
    # 🎯 CAMPAIGN TYPE CONFIGURATIONS
    CAMPAIGN_TYPES = {
        'Early_Booking': {
            'campaign_months': [11, 12, 1, 2, 3, 4],
            'campaigns_per_month': 1,
            'duration_range': (14, 45),
            'discount_range': (0.25, 0.40),
            'target_segments': ['Early_Planner', 'Flexible'],
            'preferred_channel': 'Connected_Agent',
            'advance_booking_requirement': 90,
            'capacity_range': (200, 500),
            'influence_period_days': 120,
            'seasonal_stay_weights': {
                5: 0.15, 6: 0.20, 7: 0.30, 8: 0.30, 9: 0.05
            }
        },
        'Flash_Sale': {
            'campaigns_per_month': (2, 4),
            'duration_range': (2, 7),
            'discount_range': (0.15, 0.30),
            'target_segments': ['Last_Minute', 'Flexible'],
            'preferred_channel': 'Mixed',
            'advance_booking_requirement': 0,
            'capacity_range': (50, 150),
            'urgency_decay': {
                0: 1.2, 1: 1.2, 2: 1.0, 3: 0.8, 4: 0.8, 5: 0.6, 6: 0.3, 7: 0.3
            }
        },
        'Special_Offer': {
            'target_months': [5, 9],
            'campaigns_per_month': 1,
            'duration_range': (10, 21),
            'discount_range': (0.20, 0.35),
            'target_segments': ['Early_Planner', 'Flexible'],
            'preferred_channel': 'Mixed',
            'advance_booking_requirement': 30,
            'capacity_range': (100, 300)
        }
    }
    
    # 🎲 ATTRIBUTION MODEL PARAMETERS
    ATTRIBUTION_CONFIG = {
        'base_attribution_score': 0.5,
        'model_uncertainty_std': 0.15,
        'high_error_probability': 0.10,
        'high_error_range': (0.7, 1.4),
        'cannibalization_threshold_range': (0.2, 0.4),
        'segment_match_boost': 1.3,
        'segment_mismatch_penalty': 0.7,
        'channel_alignment_boost': 1.1,
        'fatigue_penalty_per_exposure': 0.15,
        'min_fatigue_factor': 0.3,
        'temporal_decay': {
            30: 0.9, 60: 0.7, 999: 0.4
        }
    }
    
    # 📊 DATA GENERATION PARAMETERS
    DATA_CONFIG = {
        'total_customers': 5000,
        'base_daily_demand': 30,
        'demand_noise_std': 0.10,
        'min_daily_bookings': 8,
        'external_shock_probability': 0.02,
        'shock_impact_range': (0.3, 0.7),
        'seasonal_pricing_multipliers': {
            7: 1.3, 8: 1.3, 5: 1.1, 9: 1.1, 6: 1.0
        },
        'stay_length_distribution': {
            1: 0.15, 2: 0.25, 3: 0.25, 4: 0.15, 5: 0.10, 7: 0.08, 14: 0.02
        },
        'room_type_distribution': [0.4, 0.35, 0.15, 0.1],
        'non_campaign_discount_ranges': {
            'Connected_Agent': (0.12, 0.28),
            'Online_Direct': (0.08, 0.18)
        }
    }