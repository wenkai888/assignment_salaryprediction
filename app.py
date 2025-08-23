import streamlit as st
import numpy as np
import pandas as pd
import os

# Your exact Linear Regression model parameters (extracted from training)
MODEL_PARAMS = {
    'feature_names': ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'price_per_sqft', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished', 'area_per_bed', 'bath_per_bed', 'parking_per_bed', 'amenity_count', 'area_pref', 'area_ac', 'area_sqrt'],
    'feature_means': [2473.0, 2.968, 1.29, 1.63, 0.822, 0.274, 0.168, 0.432, 0.658, 0.91, 0.256, 199.827, 0.406, 0.268, 924.91, 0.491, 0.418, 3.61, 633.77, 1627.67, 49.638],
    'feature_scales': [671.676, 0.868, 0.693, 0.601, 0.383, 0.446, 0.374, 0.496, 0.475, 0.968, 0.437, 52.959, 0.491, 0.443, 388.68, 0.329, 0.264, 1.658, 1133.29, 1159.96, 6.703],
    'coefficients': [0.940392, -0.213932, 0.201837, 0.121816, -0.013853, 0.013926, 0.037952, 0.003746, 0.023701, 0.037654, 0.007059, 0.067831, 0.036134, 0.016593, -0.536326, 0.041831, -0.024639, 0.074062, 0.00638, 0.089359, -0.168089],
    'intercept': 15.301071,
    'model_type': 'Linear Regression',
    'r2_score': 0.9673
}

def linear_regression_predict(features):
    """Exact replication of your trained Linear Regression model"""
    feature_means = np.array(MODEL_PARAMS['feature_means'])
    feature_scales = np.array(MODEL_PARAMS['feature_scales'])
    coefficients = np.array(MODEL_PARAMS['coefficients'])
    intercept = MODEL_PARAMS['intercept']
    
    # Standardize features (same as StandardScaler)
    features_standardized = (features - feature_means) / feature_scales
    
    # Linear regression prediction: y = intercept + sum(coef * x)
    log_price = intercept + np.dot(features_standardized, coefficients)
    
    # Convert from log space to actual price
    predicted_price = np.exp(log_price)
    
    return predicted_price

def calculate_features(area, bedrooms, bathrooms, stories, parking, 
                      mainroad, guestroom, basement, hotwaterheating, 
                      airconditioning, prefarea, furnishing):
    """Calculate features in the EXACT same order as training"""
    
    # Basic engineered features
    area_per_bed = area / max(bedrooms, 1)
    bath_per_bed = bathrooms / max(bedrooms, 1)
    parking_per_bed = parking / (bedrooms + 1)
    
    # Amenity count
    amenities = [mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea]
    amenity_count = sum(amenities)
    
    # Interaction terms
    area_pref = area * prefarea
    area_ac = area * airconditioning
    area_sqrt = np.sqrt(area)
    
    # Furnishing encoding (furnished is reference, so both are 0)
    furnishingstatus_semi_furnished = 1 if furnishing == 'Semi-Furnished' else 0
    furnishingstatus_unfurnished = 1 if furnishing == 'Unfurnished' else 0
    
    # Price per sqft - start with placeholder
    price_per_sqft = 200
    
    # Create feature array in EXACT training order
    features = np.array([
        area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
        hotwaterheating, airconditioning, parking, prefarea, price_per_sqft,
        furnishingstatus_semi_furnished, furnishingstatus_unfurnished,
        area_per_bed, bath_per_bed, parking_per_bed, amenity_count,
        area_pref, area_ac, area_sqrt
    ])
    
    return features

def predict_with_price_iteration(area, bedrooms, bathrooms, stories, parking, 
                                mainroad, guestroom, basement, hotwaterheating, 
                                airconditioning, prefarea, furnishing):
    """Predict price with iterative price_per_sqft calculation"""
    
    features = calculate_features(area, bedrooms, bathrooms, stories, parking, 
                                mainroad, guestroom, basement, hotwaterheating, 
                                airconditioning, prefarea, furnishing)
    
    # Iterative refinement of price_per_sqft
    for i in range(3):  # 3 iterations should converge
        predicted_price = linear_regression_predict(features)
        new_price_per_sqft = predicted_price / area
        features[11] = new_price_per_sqft  # Update price_per_sqft
    
    return predicted_price

def main():
    st.set_page_config(page_title="🏠 AI House Price Predictor", layout="wide")
    
    st.title("🏠 AI House Price Predictor")
    st.write("Enter your house details to get an AI-powered price estimate")
    
    # Model info
    st.success(f"✅ Using trained {MODEL_PARAMS['model_type']} (R² = {MODEL_PARAMS['r2_score']:.4f})")
    
    # Create input interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📏 House Physical Details")
        area = st.number_input("House Area (sq ft)", min_value=500, max_value=10000, value=2000, step=50)
        bedrooms = st.selectbox("Number of Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
        bathrooms = st.selectbox("Number of Bathrooms", options=[1, 2, 3, 4, 5], index=1)
        stories = st.selectbox("Number of Stories", options=[1, 2, 3], index=0)
        parking = st.selectbox("Parking Spaces", options=[0, 1, 2, 3, 4], index=1)
    
    with col2:
        st.subheader("🏘️ Amenities & Features")
        mainroad = st.checkbox("Main Road Access", value=True)
        guestroom = st.checkbox("Guest Room")
        basement = st.checkbox("Basement")
        hotwaterheating = st.checkbox("Hot Water Heating")
        airconditioning = st.checkbox("Air Conditioning", value=True)
        prefarea = st.checkbox("Preferred Area")
        
        furnishing = st.selectbox("Furnishing Status", 
                                options=["Unfurnished", "Semi-Furnished", "Furnished"],
                                index=0)
    
    # Input validation warnings
    if bathrooms > bedrooms + 1:
        st.warning("⚠️ Unusual: More bathrooms than bedrooms + 1")
    
    if area < bedrooms * 200:
        st.warning("⚠️ Small area for number of bedrooms")
    
    # Prediction
    if st.button("🔮 Predict House Price", type="primary"):
        try:
            # Make prediction using exact Linear Regression model
            predicted_price = predict_with_price_iteration(
                area, bedrooms, bathrooms, stories, parking,
                int(mainroad), int(guestroom), int(basement), 
                int(hotwaterheating), int(airconditioning), int(prefarea),
                furnishing
            )
            
            # Display results
            st.success("✅ Prediction Complete!")
            st.info("🤖 Using your trained Linear Regression model")
            
            # Main result
            st.metric(
                label="🏠 Estimated House Price",
                value=f"${predicted_price:,.0f}",
                delta=f"±{predicted_price*0.1:,.0f} (±10%)"
            )
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                price_per_sqft = predicted_price / area
                st.metric("💰 Price per Sq Ft", f"${price_per_sqft:.2f}")
            
            with col2:
                amenity_score = sum([mainroad, guestroom, basement, 
                                   hotwaterheating, airconditioning, prefarea])
                st.metric("⭐ Amenity Score", f"{amenity_score}/6")
            
            with col3:
                efficiency = area / bedrooms if bedrooms > 0 else 0
                st.metric("📐 Area per Bedroom", f"{efficiency:.0f} sq ft")
            
            # Detailed breakdown
            with st.expander("🔍 See Prediction Breakdown"):
                st.write("**Input Summary:**")
                st.write(f"• Area: {area:,} sq ft")
                st.write(f"• Bedrooms: {bedrooms}, Bathrooms: {bathrooms}")
                st.write(f"• Stories: {stories}, Parking: {parking}")
                st.write(f"• Amenities: {amenity_score}/6 features")
                st.write(f"• Furnishing: {furnishing}")
                
                st.write("**Calculated Features:**")
                st.write(f"• Area per bedroom: {area/max(bedrooms,1):.1f} sq ft")
                st.write(f"• Bathrooms per bedroom: {bathrooms/max(bedrooms,1):.2f}")
                st.write(f"• Price per sq ft: ${price_per_sqft:.2f}")
                
                st.write("**Model Information:**")
                st.write(f"• Model: {MODEL_PARAMS['model_type']}")
                st.write(f"• R² Score: {MODEL_PARAMS['r2_score']:.4f}")
                st.write(f"• Features: {len(MODEL_PARAMS['feature_names'])}")
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.write("Please check that all inputs are valid.")
    
    # Model status in sidebar
    with st.sidebar:
        st.subheader("🤖 Model Status")
        st.success("✅ Linear Regression Model")
        st.write(f"**R² Score:** {MODEL_PARAMS['r2_score']:.4f}")
        st.write(f"**Features:** {len(MODEL_PARAMS['feature_names'])}")
        st.write("**Status:** Using exact trained coefficients")

if __name__ == "__main__":
    main()
