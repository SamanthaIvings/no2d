import pandas as pd
import numpy as np
import threadpoolctl
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Function to load and preprocess OSMNx data
def preprocess_road_data(df):
    """
    Preprocess direct OSMNx road data with missing values and potential inconsistencies
    """
    # Load the OSMNx data
    #df = pd.read_csv(filepath)
    
    # Basic data exploration
    print(f"Initial data shape: {df.shape}")
    print(f"Missing values per column:\n{df.isna().sum()}")
    
    # Step 1: Check for and handle anomalies in width data
    # Identify implausible width values (too narrow for lane count)
    min_width_per_lane = 2.5  # Minimum reasonable width per lane in meters
    
    # Flag suspicious width values where width << lanes * min_width_per_lane
    mask = (df['width'].notna()) & (df['lanes'].notna())
    df.loc[mask, 'suspicious_width'] = df.loc[mask, 'width'] < (df.loc[mask, 'lanes'] * min_width_per_lane * 0.7)
    
    print(f"Suspicious width values detected: {df['suspicious_width'].sum()}")
    
    # For suspicious values, set width to NaN to be imputed later
    df.loc[df['suspicious_width'] == True, 'width'] = np.nan
    
    # Step 2: Handle missing speed limits based on road hierarchies
    # Create default speed limits dictionary based on UK norms by road type
    default_speed_limits = {
        'motorway': 70,
        'motorway_link': 70,
        'trunk': 60,
        'trunk_link': 60,
        'primary': 60,
        'primary_link': 60,
        'secondary': 50,
        'secondary_link': 50,
        'tertiary': 30,
        'tertiary_link': 30,
        'unclassified': 30,
        'residential': 30,
        'living_street': 20,
        'service': 15,
        'UT': 30
    }
    
    # Fill missing speed limits with defaults when missing
    for road_type, default_speed in default_speed_limits.items():
        mask = (df['highway_key'] == road_type) & (df['speedlim'].isna())
        df.loc[mask, 'speedlim'] = default_speed
    
    # Step 3: Handle missing lane counts

    # Use median lane count by road type for imputation
    # lane_medians = df.groupby('highway_key')['lanes'].median().to_dict()
    
    # for road_type, median_lanes in lane_medians.items():
    #     if not pd.isna(median_lanes):
    #         mask = (df['highway_key'] == road_type) & (df['lanes'].isna())
    #         df.loc[mask, 'lanes'] = median_lanes
    
    # For any remaining missing lanes, use road hierarchy-based defaults
    default_lanes = {
        'motorway': 3,
        'motorway_link': 2,
        'trunk': 2,
        'trunk_link': 1,
        'primary': 2,
        'primary_link': 1,
        'secondary': 2,
        'secondary_link': 1,
        'tertiary': 1,
        'tertiary_link': 1,
        'unclassified': 1,
        'residential': 1,
        'living_street': 1,
        'service': 1,
        'UT': 1
    }
    
    for road_type, default_lane in default_lanes.items():
        mask = (df['highway_key'] == road_type) & (df['lanes'].isna())
        df.loc[mask, 'lanes'] = default_lane
    
    # Step 4: Width imputation using a Random Forest model for non-suspicious missing widths
    # Only use non-suspicious width data for training
    # width_model_df = df[df['suspicious_width'] != True].dropna(subset=['width', 'lanes', 'speedlim'])
    
    # if len(width_model_df) > 20:  # Only if we have enough training data
    #     print("Training width imputation model...")
        
    #     # Prepare features
    #     X = width_model_df[['lanes', 'speedlim']]
    #     enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    #     road_types = enc.fit_transform(width_model_df[['highway_key']])
    #     X_encoded = np.hstack([X, road_types])
        
    #     # Target
    #     y = width_model_df['width']
        
    #     # Train model
    #     rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    #     rf_model.fit(X_encoded, y)
        
    #     # Prepare data for prediction
    #     missing_width_df = df[df['width'].isna()]
    #     if len(missing_width_df) > 0:
    #         X_missing = missing_width_df[['lanes', 'speedlim']]
    #         road_types_missing = enc.transform(missing_width_df[['highway_key']])
    #         X_missing_encoded = np.hstack([X_missing, road_types_missing])
            
    #         # Predict widths
    #         predicted_widths = rf_model.predict(X_missing_encoded)
    #         df.loc[df['width'].isna(), 'width'] = predicted_widths
    # else:
    #     print("Insufficient data for machine learning width imputation, using rule-based method...")
    # Use rule-based width estimation as fallback
        
    # Step 5: Rule-based width imputation for any remaining missing values
    # Create a function based on our adjusted estimation model
    # def estimate_missing_width(row):
    #     highway_type = row['highway_key']
    #     lanes = row['lanes']
    #     speed = row['speedlim']
        
    #     # Base lane width depends on road type and speed
    #     if highway_type.startswith('motorway'):
    #         base_lane_width = 3.4
    #     elif speed >= 50:
    #         if highway_type in ['trunk', 'primary']:
    #             base_lane_width = 3.4
    #         elif highway_type in ['secondary', 'tertiary']:
    #             base_lane_width = 3.2
    #         else:
    #             base_lane_width = 2.8
    #     else:
    #         if highway_type in ['trunk', 'primary']:
    #             base_lane_width = 3.0
    #         elif highway_type in ['secondary']:
    #             base_lane_width = 2.8
    #         elif highway_type in ['tertiary']:
    #             base_lane_width = 2.4
    #         else:
    #             base_lane_width = 2.0
        
    #     # Calculate effective lanes
    #     if highway_type in ['unclassified', 'tertiary', 'residential'] and lanes >= 3 and speed <= 30:
    #         effective_lanes = 1.5
    #     else:
    #         effective_lanes = lanes
            
    #     # Calculate carriageway width
    #     carriageway_width = effective_lanes * base_lane_width
        
    #     # Add additional width based on road type
    #     if highway_type.startswith('motorway'):
    #         additional_width = 3.0
    #     elif highway_type in ['trunk', 'primary']:
    #         additional_width = 1.0
    #     elif highway_type in ['secondary', 'tertiary']:
    #         additional_width = 0.5
    #     else:
    #         additional_width = 0.2
            
    #     total_width = carriageway_width + additional_width
        
    #     # Apply correction factor
    #     if highway_type in ['unclassified', 'residential'] and speed <= 30:
    #         correction_factor = 0.8
    #     elif highway_type == 'tertiary' and speed <= 30:
    #         correction_factor = 0.85
    #     else:
    #         correction_factor = 0.9
            
    #     return round(total_width * correction_factor, 1)
    
    # # Apply rule-based estimation to any remaining missing width values
    # mask = df['width'].isna()
    # if mask.any():
    #     df.loc[mask, 'width'] = df.loc[mask].apply(estimate_missing_width, axis=1)
        
    # print(f"Remaining missing values after imputation:\n{df.isna().sum()}")
    
    # # Remove temporary columns
    # df = df.drop(columns=['suspicious_width'], errors='ignore')
    
    # return df
    # def estimate_missing_width(row):
    #     highway_type = row['highway_key']
    #     lanes = row['lanes'] if pd.notna(row['lanes']) else 2  # Default to 2 if missing
    #     speed = row['speedlim'] if pd.notna(row['speedlim']) else 30  # Default speed limit

    #     # Define base lane width based on road type and speed
    #     if highway_type.startswith('motorway'):
    #         base_lane_width = 3.4
    #     elif speed >= 60:  # High-speed A-roads
    #         base_lane_width = 3.4 if highway_type in ['trunk', 'primary'] else 3.2
    #     elif speed >= 40:  # Mid-speed urban roads
    #         base_lane_width = 3.2 if highway_type in ['trunk', 'primary'] else 2.8
    #     elif speed >= 30:  # Low-speed roads
    #         base_lane_width = 3.0 if highway_type in ['primary', 'secondary'] else 2.4
    #     else:  # Very low-speed streets
    #         base_lane_width = 2.2 if highway_type in ['tertiary', 'unclassified'] else 2.0

    #     # Adjust lanes for constrained roads
    #     if highway_type in ['unclassified', 'tertiary', 'residential'] and lanes >= 3 and speed <= 30:
    #         effective_lanes = 1.5
    #     else:
    #         effective_lanes = lanes

    #     # Calculate base carriageway width
    #     carriageway_width = effective_lanes * base_lane_width

    #     # Additional space for different road types
    #     additional_width = {
    #         'motorway': 3.0, 'trunk': 1.5, 'primary': 1.0,
    #         'secondary': 0.5, 'tertiary': 0.3, 'unclassified': 0.2, 'residential': 0.2
    #     }.get(highway_type, 0.2)  # Default fallback

    #     # Account for cycle lanes if applicable
    #     if row.get('cycleway') in ['lane', 'track']:
    #         additional_width += 1.5  # Assume standard 1.5m cycle lane

    #     # Account for parking lanes if likely present
    #     if highway_type in ['residential', 'tertiary'] and speed <= 30:
    #         additional_width += 2.0  # Standard UK on-street parking width

    #     total_width = carriageway_width + additional_width

    #     # Apply correction factor for low-speed and minor roads
    #     correction_factor = 0.85 if highway_type in ['residential', 'tertiary'] and speed <= 30 else 0.9

    #     return round(total_width * correction_factor, 1)

    # Define an improved width estimation function
    def estimate_missing_width(row):
        highway_type = row['highway_key']
        lanes = row['lanes'] if pd.notna(row['lanes']) else 2  # Default to 2 lanes if missing
        speed = row['speedlim'] if pd.notna(row['speedlim']) else 30  # Default to 30 km/h
        
        # Adjusted base lane width
        if highway_type.startswith('motorway'):
            base_lane_width = 3.75  # Motorway lanes are wider
        elif highway_type in ['trunk', 'primary']:
            base_lane_width = 3.5
        elif highway_type in ['secondary', 'tertiary']:
            base_lane_width = 3.25
        else:
            base_lane_width = 2.75  # Residential/unclassified roads

        # Calculate effective lanes
        if highway_type in ['unclassified', 'tertiary', 'residential'] and lanes >= 3 and speed <= 30:
            effective_lanes = 1.5  # Allow for shared spaces
        else:
            effective_lanes = lanes

        # Calculate carriageway width
        carriageway_width = effective_lanes * base_lane_width

        # Additional width for shoulders and central reservations
        if highway_type.startswith('motorway'):
            additional_width = 3.5  # Shoulders and barriers
        elif highway_type in ['trunk', 'primary']:
            additional_width = 1.5  # Includes some shoulder width
        elif highway_type in ['secondary', 'tertiary']:
            additional_width = 0.5
        else:
            additional_width = 0.2  # Minor local roads

        total_width = carriageway_width + additional_width

        # Correction factors for minor roads
        if highway_type in ['unclassified', 'residential'] and speed <= 30:
            correction_factor = 0.85  # Slightly reduced width for constrained spaces
        elif highway_type == 'tertiary' and speed <= 30:
            correction_factor = 0.9
        else:
            correction_factor = 1.0  # No correction for major roads

        return round(total_width * correction_factor, 2)

    # Apply the updated function to fill missing width values
    mask = df['width'].isna()
    if mask.any():
        df.loc[mask, 'width'] = df.loc[mask].apply(estimate_missing_width, axis=1)

    print(f"Remaining missing values after imputation:\n{df.isna().sum()}")

    return df

# Example usage
# df = preprocess_road_data('osmnx_direct_output.csv')
# df.to_csv('processed_road_data.csv', index=False)

# # Demonstrate width distribution visualization
# def visualize_width_distribution(df):
#     import matplotlib.pyplot as plt
    
#     plt.figure(figsize=(12, 6))
    
#     # Width by road type
#     road_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified']
    
#     for i, road_type in enumerate(road_types):
#         subset = df[df['highway_key'] == road_type]
#         if len(subset) > 0:
#             plt.subplot(2, 3, i+1)
#             plt.hist(subset['width'], bins=20, alpha=0.7)
#             plt.title(f'{road_type} (n={len(subset)})')
#             plt.xlabel('Width (m)')
#             plt.ylabel('Count')
    
#     plt.tight_layout()
#     plt.savefig('width_distribution.png')
#     plt.close()
    
#     # Width vs lanes scatter plot
#     plt.figure(figsize=(10, 6))
#     for road_type in road_types:
#         subset = df[df['highway_key'] == road_type]
#         if len(subset) > 0:
#             plt.scatter(subset['lanes'], subset['width'], alpha=0.5, label=road_type)
    
#     plt.xlabel('Lanes')
#     plt.ylabel('Width (m)')
#     plt.title('Road Width vs Lane Count')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.savefig('width_vs_lanes.png')

# # If this were an actual script, you could call it like:
# # df = preprocess_road_data('your_osmnx_data.csv')
# # visualize_width_distribution(df)
# # df.to_csv('processed_road_data_with_width.csv', index=False)