import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('house_price_model.joblib')
        return model
    except Exception as e:
        st.error(f"Model file not found: {str(e)}")
        st.error("Please ensure 'house_price_model.joblib' is uploaded to your repository")
        return None

def calculate_features(area, bedrooms, bathrooms, stories, parking, 
                      mainroad, guestroom, basement, hotwaterheating, 
                      airconditioning, prefarea, furnishing):
    """
    Calculate all engineered features automatically from user inputs
    This must match your training feature engineering exactly
    """
    
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
    
    # Furnishing encoding (drop_first=True, so unfurnished is reference)
    furnishingstatus_furnished = 1 if furnishing == 'Furnished' else 0
    furnishingstatus_semi_furnished = 1 if furnishing == 'Semi-Furnished' else 0
    
    # Price per sqft placeholder (will be updated after prediction)
    price_per_sqft = 200  # Placeholder
    
    # Return feature array in the EXACT order as your training data
    features = np.array([[
        area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
        hotwaterheating, airconditioning, parking, prefarea,
        furnishingstatus_furnished, furnishingstatus_semi_furnished,
        price_per_sqft, area_per_bed, bath_per_bed, parking_per_bed,
        amenity_count, area_pref, area_ac, area_sqrt
    ]])
    
    return features

def main():
    st.set_page_config(page_title="🏠 AI House Price Predictor", layout="wide")
    
    st.title("🏠 AI House Price Predictor")
    st.write("Enter your house details to get an AI-powered price estimate")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
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
            # Calculate features
            features = calculate_features(
                area, bedrooms, bathrooms, stories, parking,
                int(mainroad), int(guestroom), int(basement), 
                int(hotwaterheating), int(airconditioning), int(prefarea),
                furnishing
            )
            
            # Make prediction
            prediction_log = model.predict(features)[0]
            predicted_price = np.exp(prediction_log)  # Convert from log scale
            
            # Display results
            st.success("✅ Prediction Complete!")
            
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
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Please check that all inputs are valid.")

if __name__ == "__main__":
    main()
