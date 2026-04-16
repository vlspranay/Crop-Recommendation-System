"""
Combined Crop and Soil Recommendation System

This script combines the soil classification model and crop recommendation model
to provide crop suggestions based on soil type and environmental conditions.
"""

import numpy as np
import pandas as pd
import os
import json

# Apply quiet/reproducible TensorFlow defaults unless user overrides them.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import tensorflow as tf
from tensorflow import keras
import joblib
from PIL import Image
from backend.services.weather_service import get_weather_data
from backend.services.groundwater_service import get_groundwater_level
import warnings
warnings.filterwarnings('ignore')

class CombinedCropSoilRecommender:
    def __init__(self, soil_model_path, crop_model_path, crop_encoder_path):
        """
        Initialize the combined recommendation system.
        
        Args:
            soil_model_path: Path to the trained soil classification model
            crop_model_path: Path to the trained crop recommendation model
            crop_encoder_path: Path to the crop label encoder
        """
        # Load soil classification model
        self.soil_model = keras.models.load_model(soil_model_path)
        self.soil_classes = ['Alluvial Soil', 'Black Soil', 'Cinder Soil', 'Clay Soil', 
                            'Laterite Soil', 'Peat Soil', 'Red Soil', 'Yellow Soil']
        
        # Load crop recommendation model and encoder
        self.crop_model = joblib.load(crop_model_path)
        self.crop_encoder = joblib.load(crop_encoder_path)
        
        # Soil-to-crop mapping based on agricultural knowledge
        self.soil_crop_mapping = {
            'Alluvial Soil': ['rice', 'wheat', 'sugarcane', 'cotton', 'jute', 'maize', 'pulses'],
            'Black Soil': ['cotton', 'sugarcane', 'wheat', 'jowar', 'sunflower', 'groundnut', 'pulses'],
            'Cinder Soil': ['coffee', 'tea', 'cardamom', 'pepper', 'coconut', 'cashew'],
            'Clay Soil': ['rice', 'wheat', 'barley', 'oats', 'potatoes', 'onions'],
            'Laterite Soil': ['cashew', 'coconut', 'rubber', 'tea', 'coffee', 'cardamom'],
            'Peat Soil': ['rice', 'vegetables', 'fruits', 'flowers', 'herbs'],
            'Red Soil': ['groundnut', 'potato', 'rice', 'ragi', 'tobacco', 'oilseeds', 'pulses'],
            'Yellow Soil': ['wheat', 'barley', 'potato', 'rice', 'maize', 'pulses']
        }
        
        # Environmental parameter ranges for different soil types
        self.soil_environmental_ranges = {
            'Alluvial Soil': {
                'N': (50, 100), 'P': (30, 60), 'K': (30, 60),
                'temperature': (20, 35), 'humidity': (60, 90), 'pH': (6.0, 8.0), 'rainfall': (100, 300)
            },
            'Black Soil': {
                'N': (40, 80), 'P': (20, 50), 'K': (40, 80),
                'temperature': (25, 40), 'humidity': (50, 80), 'pH': (7.0, 8.5), 'rainfall': (50, 200)
            },
            'Cinder Soil': {
                'N': (30, 60), 'P': (15, 40), 'K': (20, 50),
                'temperature': (15, 30), 'humidity': (70, 95), 'pH': (5.5, 7.0), 'rainfall': (200, 400)
            },
            'Clay Soil': {
                'N': (60, 100), 'P': (40, 70), 'K': (50, 90),
                'temperature': (15, 30), 'humidity': (60, 85), 'pH': (6.5, 8.0), 'rainfall': (100, 250)
            },
            'Laterite Soil': {
                'N': (20, 50), 'P': (10, 30), 'K': (15, 40),
                'temperature': (20, 35), 'humidity': (60, 90), 'pH': (5.0, 6.5), 'rainfall': (150, 300)
            },
            'Peat Soil': {
                'N': (80, 120), 'P': (30, 60), 'K': (20, 50),
                'temperature': (10, 25), 'humidity': (70, 95), 'pH': (4.0, 6.0), 'rainfall': (200, 500)
            },
            'Red Soil': {
                'N': (30, 70), 'P': (20, 50), 'K': (25, 60),
                'temperature': (20, 35), 'humidity': (50, 80), 'pH': (5.5, 7.5), 'rainfall': (50, 200)
            },
            'Yellow Soil': {
                'N': (40, 80), 'P': (25, 55), 'K': (30, 70),
                'temperature': (15, 30), 'humidity': (60, 85), 'pH': (6.0, 7.5), 'rainfall': (100, 300)
            }
        }

        req_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'crop_requirements_ap.json')
        try:
            with open(req_path, 'r', encoding='utf-8') as f:
                self.crop_requirements = json.load(f)
            print("✅ Crop requirements loaded")
        except Exception as e:
            print(f"⚠️ Failed to load crop requirements: {e}")
            self.crop_requirements = {}

        economics_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'crop_economics.json')
        try:
            with open(economics_path, 'r', encoding='utf-8') as f:
                self.crop_economics = json.load(f)
            print("✅ Crop economics loaded")
        except Exception as e:
            print(f"⚠️ Failed to load crop economics: {e}")
            self.crop_economics = {}
    
    def classify_soil(self, image_path):
        """
        Classify soil type from image.
        
        Args:
            image_path: Path to the soil image
            
        Returns:
            tuple: (predicted_soil_type, confidence_score)
        """
        try:
            # Load and preprocess image
            img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_batch = tf.expand_dims(img_array, 0)
            
            # Make prediction
            prediction = self.soil_model.predict(img_batch, verbose=0)
            score = tf.nn.softmax(prediction[0])
            
            predicted_class_index = np.argmax(score)
            predicted_soil_type = self.soil_classes[predicted_class_index]
            confidence = float(np.max(score)) * 100
            
            return predicted_soil_type, confidence
            
        except Exception as e:
            print(f"Error in soil classification: {e}")
            return None, 0.0
    
    def get_environmental_parameters(self, soil_type, custom_params=None):
        """
        Get environmental parameters for a given soil type.
        Ensures all required keys are present, filling missing ones with defaults.
        """
        required_keys = ['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall']
        # Get default ranges for the soil type
        ranges = self.soil_environmental_ranges.get(soil_type, {})
        # Generate typical values within the ranges
        default_params = {param: (min_val + max_val) / 2 for param, (min_val, max_val) in ranges.items()}
        if custom_params:
            # Fill in missing keys from defaults
            params = {**default_params, **custom_params}
        else:
            params = default_params.copy()
        # Ensure all required keys are present
        for key in required_keys:
            if key not in params:
                # Fallback to 0 if no default is available
                params[key] = 0
        return params
    
    def compute_constraint_score(self, crop_name, rainfall, temperature, groundwater_level):
        """
        Score crop viability against rainfall, temperature, and groundwater constraints.
        """
        req = self.crop_requirements.get(str(crop_name).lower())

        if not req:
            return 1.0

        score = 1.0

        if rainfall is not None:
            min_r, max_r = req['rainfall']
            if rainfall < min_r:
                score *= 0.6
            elif rainfall > max_r:
                score *= 0.7

        if temperature is not None:
            min_t, max_t = req['temperature']
            if temperature < min_t or temperature > max_t:
                score *= 0.7

        gw_map = {'low': 1, 'medium': 2, 'high': 3}
        crop_req = gw_map.get(req['groundwater'], 2)
        available = gw_map.get(str(groundwater_level).lower(), 2)

        if available < crop_req:
            score *= 0.6

        return score

    def compute_profit_score(self, crop_name):
        data = self.crop_economics.get(str(crop_name).lower())
        if not data:
            return 0.5

        cost = float(data.get('cost', 0))
        price = float(data.get('price', 0))
        if price <= 0:
            return 0.5

        margin = max(price - cost, 0.0)
        ratio = margin / price
        return max(0.0, min(1.0, ratio))

    def compute_risk_score(self, crop_name, risk_preference=0.5):
        data = self.crop_economics.get(str(crop_name).lower())
        if not data:
            return 0.5

        crop_risk = float(data.get('risk', 0.5))
        pref = max(0.0, min(1.0, float(risk_preference)))

        # Match score: closer crop risk to user preference means better fit.
        return 1.0 - min(1.0, abs(crop_risk - pref))

    def compute_water_factor(self, crop_name, groundwater_level):
        data = self.crop_economics.get(str(crop_name).lower())
        if not data:
            return 1.0

        gw_map = {'low': 1, 'medium': 2, 'high': 3}
        need = gw_map.get(str(data.get('water_need', 'medium')).lower(), 2)
        available = gw_map.get(str(groundwater_level).lower(), 2)

        if available < need:
            return 0.85
        if available > need:
            return 1.05
        return 1.0

    def recommend_crops(
        self,
        soil_type,
        environmental_params=None,
        top_n=5,
        location=None,
        risk_preference=0.5,
        groundwater_level='medium'
    ):
        """
        Recommend crops based on soil type and environmental conditions.
        
        Args:
            soil_type: The classified soil type
            environmental_params: Environmental parameters (optional)
            top_n: Number of top recommendations to return
            
        Returns:
            list: List of recommended crops with scores
        """
        try:
            # Get environmental parameters
            env_params = self.get_environmental_parameters(soil_type, environmental_params)
            rainfall = env_params.get('rainfall')
            temperature = env_params.get('temperature')
            
            # Prepare input for crop model
            crop_input = np.array([[
                env_params['N'],
                env_params['P'],
                env_params['K'],
                env_params['temperature'],
                env_params['humidity'],
                env_params['pH'],
                env_params['rainfall']
            ]])
            
            # Get crop predictions
            crop_probabilities = self.crop_model.predict_proba(crop_input)[0]
            crop_names = self.crop_encoder.classes_
            
            # Get soil-specific crop recommendations
            soil_specific_crops = self.soil_crop_mapping.get(soil_type, [])
            
            # Create recommendations with scores, filter out crops with no name or non-positive score
            recommendations = []
            for i, (crop_name, prob) in enumerate(zip(crop_names, crop_probabilities)):
                ml_score = float(prob)
                soil_score = 1.0 if crop_name in soil_specific_crops else 0.6
                profit_score = self.compute_profit_score(crop_name)
                risk_score = self.compute_risk_score(crop_name, risk_preference)
                constraint_score = self.compute_constraint_score(
                    crop_name,
                    rainfall,
                    temperature,
                    groundwater_level
                )
                water_factor = self.compute_water_factor(crop_name, groundwater_level)

                final_score = (
                    0.35 * ml_score +
                    0.2 * soil_score +
                    0.2 * profit_score +
                    0.15 * risk_score +
                    0.1 * constraint_score
                ) * water_factor

                reasons = []
                if crop_name in soil_specific_crops:
                    reasons.append('Good soil match')
                if constraint_score < 0.7:
                    reasons.append('May face environmental constraints')

                # Only include crops with a valid name and positive score
                if crop_name and final_score > 0:
                    recommendations.append({
                        'crop': crop_name,
                        'score': final_score,
                        'soil_suitable': crop_name in soil_specific_crops,
                        'original_probability': prob,
                        'constraint_score': constraint_score,
                        'ml_score': ml_score,
                        'soil_score': soil_score,
                        'profit_score': profit_score,
                        'risk_score': risk_score,
                        'water_factor': water_factor,
                        'reasons': reasons
                    })
            # Sort by adjusted score and return top N
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            top_recommendations = recommendations[:top_n]
            # If no valid recommendations, return a random soil-specific crop
            if not top_recommendations:
                import random
                soil_specific_crops = self.soil_crop_mapping.get(soil_type, [])
                if soil_specific_crops:
                    random_crop = random.choice(soil_specific_crops)
                    return [{
                        'crop': random_crop,
                        'score': 1.0,
                        'soil_suitable': True,
                        'original_probability': 1.0,
                        'constraint_score': 1.0,
                        'ml_score': 1.0,
                        'soil_score': 1.0,
                        'profit_score': 0.5,
                        'risk_score': 0.5,
                        'water_factor': 1.0,
                        'reasons': ['Fallback soil-specific recommendation']
                    }]
                
            print("Top Recommendations:")
            for rec in top_recommendations:
                print(f"   {rec['crop'].title()}: {rec['score']:.3f}")

            return top_recommendations
            
        except Exception as e:
            print(f"Error in crop recommendation: {e}")
            return []
    
    def get_comprehensive_recommendation(
        self,
        image_path,
        custom_env_params=None,
        top_n=5,
        location=None,
        risk_preference=0.5
    ):
        """
        Get comprehensive crop recommendation based on soil image and optional environmental parameters.
        
        Args:
            image_path: Path to the soil image
            custom_env_params: Custom environmental parameters (optional)
            top_n: Number of top recommendations to return
            
        Returns:
            dict: Comprehensive recommendation results
        """
        # Classify soil
        soil_type, soil_confidence = self.classify_soil(image_path)
        
        if soil_type is None:
            return {
                'error': 'Failed to classify soil type',
                'soil_type': None,
                'soil_confidence': 0.0,
                'recommendations': []
            }
        
        groundwater_level = 'medium'

        if location:
            weather = get_weather_data(location['lat'], location['lon'])

            if weather:
                if custom_env_params is None:
                    custom_env_params = {}

                custom_env_params['rainfall'] = weather['rainfall']
                custom_env_params['temperature'] = weather['temperature']

                groundwater_level = get_groundwater_level(
                    location['lat'],
                    location['lon']
                )

        # Get environmental parameters
        env_params = self.get_environmental_parameters(soil_type, custom_env_params)
        
        # Get crop recommendations
        recommendations = self.recommend_crops(
            soil_type,
            env_params,
            top_n,
            location=location,
            risk_preference=risk_preference,
            groundwater_level=groundwater_level
        )
        
        return {
            'soil_type': soil_type,
            'soil_confidence': soil_confidence,
            'environmental_parameters': env_params,
            'recommendations': recommendations,
            'soil_specific_crops': self.soil_crop_mapping.get(soil_type, []),
            'groundwater_level': groundwater_level,
            'risk_preference': risk_preference
        }
    
    def print_recommendation(self, result):
        """
        Print formatted recommendation results.
        
        Args:
            result: Result from get_comprehensive_recommendation
        """
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        print("=" * 60)
        print("🌱 COMBINED CROP & SOIL RECOMMENDATION SYSTEM")
        print("=" * 60)
        
        print(f"\n🌍 SOIL ANALYSIS:")
        print(f"   Soil Type: {result['soil_type']}")
        print(f"   Confidence: {result['soil_confidence']:.1f}%")
        
        print(f"\n🌡️  ENVIRONMENTAL CONDITIONS:")
        env = result['environmental_parameters']
        print(f"   Nitrogen (N): {env['N']:.1f}")
        print(f"   Phosphorus (P): {env['P']:.1f}")
        print(f"   Potassium (K): {env['K']:.1f}")
        print(f"   Temperature: {env['temperature']:.1f}°C")
        print(f"   Humidity: {env['humidity']:.1f}%")
        print(f"   pH Level: {env['pH']:.1f}")
        print(f"   Rainfall: {env['rainfall']:.1f}mm")
        
        print(f"\n🌾 RECOMMENDED CROPS:")
        for i, rec in enumerate(result['recommendations'], 1):
            soil_indicator = "✅" if rec['soil_suitable'] else "⚠️"
            print(f"   {i}. {rec['crop'].title()} {soil_indicator}")
            print(f"      Score: {rec['score']:.3f} (Original: {rec['original_probability']:.3f})")
        
        print(f"\n💡 SOIL-SPECIFIC CROPS FOR {result['soil_type'].upper()}:")
        for crop in result['soil_specific_crops']:
            print(f"   • {crop.title()}")
        
        print("=" * 60)

def main():
    """
    Example usage of the CombinedCropSoilRecommender.
    """
    # Initialize the recommender
    recommender = CombinedCropSoilRecommender(
        soil_model_path='../../model_outputs/soil_classifier_model.keras',
        crop_model_path='../../model_outputs/crop_model.pkl',
        crop_encoder_path='../../model_outputs/crop_label_encoder.pkl'
    )
    
    # Example 1: Using soil image only
    print("Example 1: Soil Image Analysis")
    image_path = '../../dataset/dummies/black.jpg'  # Replace with your image path
    
    result = recommender.get_comprehensive_recommendation(image_path)
    recommender.print_recommendation(result)
    
    # Example 2: Using custom environmental parameters
    print("\n\nExample 2: Custom Environmental Parameters")
    custom_params = {
        'N': 75, 'P': 45, 'K': 50,
        'temperature': 28, 'humidity': 75, 'pH': 6.8, 'rainfall': 180
    }
    
    result2 = recommender.get_comprehensive_recommendation(
        image_path, custom_env_params=custom_params
    )
    recommender.print_recommendation(result2)

if __name__ == "__main__":
    main()
