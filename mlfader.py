import streamlit as st
import sys, subprocess, importlib, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color
from skimage.color import deltaE_ciede2000
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Package Installation Function (can be omitted in Streamlit) ---
def install_packages(packages):
    for package in packages:
        try:
            importlib.import_module(package)
            logging.info(f"Package '{package}' is already installed.")
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# --- Data and Image Upload Helpers ---
def upload_file_streamlit(prompt_message, file_types):
    st.write(prompt_message)
    uploaded_file = st.file_uploader("", type=file_types)
    if uploaded_file is not None:
        return uploaded_file
    else:
        st.error("No file uploaded!")
        st.stop()

# --- Load and Clean Dataset ---
def load_and_clean_dataset(file_obj):
    try:
        dataset = pd.read_csv(file_obj)
        required_columns = {'L', 'A', 'B', 'Color Name'}
        if not required_columns.issubset(dataset.columns):
            missing = required_columns - set(dataset.columns)
            logging.error(f"Dataset is missing required columns: {missing}")
            st.stop()
        dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=['L', 'A', 'B'])
        logging.info(f"Dataset loaded with {len(dataset)} entries after cleaning.")
        return dataset
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        st.stop()

# --- Process Image ---
def process_image(file_obj):
    try:
        image = Image.open(file_obj).convert('RGB')
        image_array = np.array(image).astype(np.float32) / 255.0
        lab_image = color.rgb2lab(image_array)
        logging.info(f"Image loaded and converted to LAB color space.")
        return image, lab_image
    except Exception as e:
        logging.error(f"Failed to process image: {e}")
        st.stop()

# --- Synthetic Data Generation with Caching ---
@st.cache_data
def create_synthetic_data(art_types, material_types, dye_types, valid_combinations, num_samples_per_combination=500):
    np.random.seed(42)
    data_list = []
    for art_type in art_types:
        for material_type in material_types:
            dye_type_options = [dye for art, material, dye in valid_combinations if art == art_type and material == material_type]
            for dye_type in dye_type_options:
                lux_hours = np.random.uniform(low=1000, high=100000, size=num_samples_per_combination)
                uv_exposure = np.random.uniform(low=0.0, high=1.0, size=num_samples_per_combination)
                temperature = np.random.uniform(low=-10, high=50, size=num_samples_per_combination)
                humidity = np.random.uniform(low=0, high=100, size=num_samples_per_combination)
                pollution = np.random.uniform(low=0, high=1.0, size=num_samples_per_combination)
                year_of_manufacture = np.random.randint(low=1455, high=2020, size=num_samples_per_combination)
                L_fading, A_fading, B_fading = generate_fading_data(
                    art_type, material_type, dye_type, lux_hours, uv_exposure, temperature, humidity, pollution, year_of_manufacture, num_samples_per_combination
                )
                data = pd.DataFrame({
                    'art_type': art_type,
                    'material_type': material_type,
                    'dye_type': dye_type,
                    'lux_hours': lux_hours,
                    'uv_exposure': uv_exposure,
                    'temperature': temperature,
                    'humidity': humidity,
                    'pollution': pollution,
                    'year_of_manufacture': year_of_manufacture,
                    'L_fading': L_fading,
                    'A_fading': A_fading,
                    'B_fading': B_fading
                })
                data_list.append(data)
    return pd.concat(data_list, ignore_index=True)

def generate_fading_data(art_type, material_type, dye_type, lux_hours, uv_exposure, temperature, humidity, pollution, year_of_manufacture, num_samples):
    L_fading = np.zeros(num_samples)
    A_fading = np.zeros(num_samples)
    B_fading = np.zeros(num_samples)
    lux_normalized = lux_hours / 100000
    uv_normalized = np.minimum(uv_exposure, 1.0)
    pollution_normalized = pollution
    uv_threshold = 0.075
    exposure_factor = lux_normalized + uv_normalized + pollution_normalized
    year_factor = (2020 - year_of_manufacture) / 220.0
    if material_type == 'Textiles' and dye_type == 'Natural':
        L_fading += np.random.normal(loc=-5, scale=1.5, size=num_samples) * exposure_factor * year_factor
        A_fading += np.random.normal(loc=-2, scale=1, size=num_samples) * exposure_factor * year_factor
        B_fading += np.random.normal(loc=-2, scale=1, size=num_samples) * exposure_factor * year_factor
    elif material_type == 'Paper with Black Text':
        L_fading += np.random.normal(loc=-1, scale=0.5, size=num_samples) * lux_normalized * year_factor
    if np.any(uv_exposure > uv_threshold):
        uv_impact = (uv_exposure - uv_threshold) * 10
        L_fading -= uv_impact
        A_fading -= uv_impact / 2
        B_fading -= uv_impact / 2
    if 'Acidic' in material_type:
        L_fading -= np.random.normal(loc=-2, scale=1, size=num_samples) * pollution_normalized
        B_fading += np.random.normal(loc=3, scale=1, size=num_samples) * pollution_normalized
    L_fading = np.clip(L_fading, -20, 0)
    A_fading = np.clip(A_fading, -10, 10)
    B_fading = np.clip(B_fading, -10, 10)
    return L_fading, A_fading, B_fading

def prepare_features(synthetic_data):
    X_numeric = synthetic_data[['lux_hours', 'uv_exposure', 'temperature', 'humidity', 'pollution', 'year_of_manufacture']]
    X_categorical = synthetic_data[['art_type', 'material_type', 'dye_type']].fillna('None')
    encoder = OneHotEncoder(sparse_output=False)
    X_categorical_encoded = encoder.fit_transform(X_categorical)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_numeric_poly = poly.fit_transform(X_numeric)
    X = np.hstack((X_numeric_poly, X_categorical_encoded))
    Y = synthetic_data[['L_fading', 'A_fading', 'B_fading']].values
    return X, Y, encoder, poly

def train_ml_model(X, Y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    multi_xgb = MultiOutputRegressor(xgb)
    param_grid = {
        'estimator__n_estimators': [100, 200],
        'estimator__max_depth': [3, 5, 7],
        'estimator__learning_rate': [0.01, 0.1, 0.2]
    }
    grid_search = GridSearchCV(multi_xgb, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0)
    grid_search.fit(X_scaled, Y)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    st.write(f"Best parameters found: {best_params}")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = cross_val_score(best_model, X_scaled, Y, cv=kf, scoring='neg_mean_squared_error')
    avg_mse = -np.mean(mse_scores)
    st.write(f"Cross-validated MSE: {avg_mse:.4f}")
    return best_model, scaler, avg_mse

def simulate_exposure_by_material(lab_image, art_type, material_type, dye_type, uv_exposure, lux_hours, humidity, temperature):
    lab_exposed = lab_image.copy()
    lux_normalized = lux_hours / 100000
    uv_normalized = uv_exposure
    if art_type == 'Chromolithograph Print':
        lab_exposed[:, :, 0] -= ((lux_normalized * 10) + (uv_normalized * 10))
        lab_exposed[:, :, 1] -= ((lux_normalized * 5) + (uv_normalized * 5))
        lab_exposed[:, :, 2] -= ((lux_normalized * 5) + (uv_normalized * 5))
    elif art_type == 'Sanguine Etching':
        lab_exposed[:, :, 1] -= (lux_normalized * 10)
    elif art_type == 'Steel Engraving':
        lab_exposed[:, :, 0] -= (lux_normalized * 5)
    if 'Acidic' in material_type:
        lab_exposed[:, :, 0] -= uv_normalized * 10
        lab_exposed[:, :, 2] += uv_normalized * 10
    elif 'Alkaline' in material_type:
        lab_exposed[:, :, 0] -= lux_normalized * 5
    elif material_type == 'Textiles':
        if dye_type == 'Natural':
            fading_multiplier = np.log(lux_hours + 1) / np.log(100000 + 1)
            lab_exposed[:, :, 0] -= uv_normalized * 15 * fading_multiplier
            lab_exposed[:, :, 1] -= uv_normalized * 15 * fading_multiplier
            lab_exposed[:, :, 2] -= uv_normalized * 15 * fading_multiplier
        elif dye_type == 'Synthetic':
            lab_exposed[:, :, 0] -= uv_normalized * 10
            lab_exposed[:, :, 1] -= uv_normalized * 10
            lab_exposed[:, :, 2] -= uv_normalized * 10
    elif material_type == 'Paper with Black Text':
        lab_exposed[:, :, 0] -= lux_normalized * 2
    lab_exposed[:, :, 0] = np.clip(lab_exposed[:, :, 0], 0, 100)
    lab_exposed[:, :, 1] = np.clip(lab_exposed[:, :, 1], -128, 127)
    lab_exposed[:, :, 2] = np.clip(lab_exposed[:, :, 2], -128, 127)
    logging.info(f"Simulated exposure for {art_type} on {material_type} with dye type {dye_type}.")
    return lab_exposed

def lab_to_rgb(lab_image):
    rgb_image = color.lab2rgb(lab_image)
    rgb_image = np.clip(rgb_image, 0, 1)
    rgb_image = (rgb_image * 255).astype(np.uint8)
    return rgb_image

def display_image_streamlit(image, title='Image'):
    st.image(image, caption=title, use_column_width=True)

def apply_fading(lab_image, predicted_fading):
    lab_faded = lab_image.copy()
    lab_faded[:, :, 0] += predicted_fading[0]
    lab_faded[:, :, 1] += predicted_fading[1]
    lab_faded[:, :, 2] += predicted_fading[2]
    lab_faded[:, :, 0] = np.clip(lab_faded[:, :, 0], 0, 100)
    lab_faded[:, :, 1] = np.clip(lab_faded[:, :, 1], -128, 127)
    lab_faded[:, :, 2] = np.clip(lab_faded[:, :, 2], -128, 127)
    logging.info("Applied predicted fading to the image.")
    return lab_faded

def compute_delta_e(lab1, lab2):
    delta_e = deltaE_ciede2000(lab1, lab2)
    logging.info(f"Delta-E between the two images calculated.")
    return delta_e

def plot_histograms_streamlit(image1, image2, title_suffix=''):
    import textwrap
    image1_array = np.array(image1)
    image2_array = np.array(image2)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['Red', 'Green', 'Blue']
    for i, color_name in enumerate(colors):
        axs[i].hist(image1_array[..., i].flatten(), bins=256, alpha=0.5, label=f'{color_name} (Image 1)', color=color_name.lower())
        axs[i].hist(image2_array[..., i].flatten(), bins=256, alpha=0.5, label=f'{color_name} (Image 2)', color=f'dark{color_name.lower()}')
        wrapped_title = textwrap.fill(f'{color_name} Channel {title_suffix}', width=25)
        axs[i].set_title(wrapped_title, fontsize=10)
        axs[i].legend()
    plt.tight_layout()
    st.pyplot(fig)

def display_average_color(image_lab, title='Average Color'):
    average_lab = image_lab.mean(axis=(0,1))
    average_rgb = color.lab2rgb(np.reshape(average_lab, (1,1,3))).reshape(1,1,3)
    average_rgb = np.clip(average_rgb, 0, 1)
    fig, ax = plt.subplots(figsize=(2,2))
    ax.imshow(np.ones((100,100,3)) * average_rgb)
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)
    logging.info(f"{title}: L={average_lab[0]:.2f}, A={average_lab[1]:.2f}, B={average_lab[2]:.2f}")
    return average_lab

def compute_average_delta_e(avg_lab1, avg_lab2):
    lab1 = np.array([avg_lab1])
    lab2 = np.array([avg_lab2])
    delta_e = deltaE_ciede2000(lab1, lab2)[0]
    logging.info(f"Delta-E between average colors: {delta_e:.2f}")
    return delta_e

# --- Define Art, Material, and Dye Types ---
art_types = [
    'Chromolithograph Print',
    'Sanguine Etching',
    'Steel Engraving',
    'None',
]
material_types = [
    'Acidic Wove Paper',
    'Acidic Rag Paper',
    'Alkaline Wove Paper',
    'Alkaline Rag Paper',
    'Textiles',
    'Paper with Black Text',
]
dye_types = ['Natural', 'Synthetic']
valid_combinations = [
    ('Chromolithograph Print', 'Acidic Wove Paper', None),
    ('Sanguine Etching', 'Acidic Wove Paper', None),
    ('Sanguine Etching', 'Acidic Rag Paper', None),
    ('Sanguine Etching', 'Alkaline Wove Paper', None),
    ('Sanguine Etching', 'Alkaline Rag Paper', None),
    ('Steel Engraving', 'Acidic Wove Paper', None),
    ('None', 'Textiles', 'Natural'),
    ('None', 'Textiles', 'Synthetic'),
    ('None', 'Paper with Black Text', None),
    ('None', 'Acidic Wove Paper', None),
    ('None', 'Acidic Rag Paper', None),
    ('None', 'Alkaline Wove Paper', None),
    ('None', 'Alkaline Rag Paper', None),
]

# --- Streamlit UI ---
st.title("Machine Learning Fading Simulator")
st.write("Upload required files and adjust parameters below:")

# Upload dataset and image
csv_file = upload_file_streamlit("Please upload your LAB color dataset CSV file.", ['csv'])
dataset = load_and_clean_dataset(csv_file)

image_file = upload_file_streamlit("Please upload the image file you want to analyze.", ['png', 'jpg', 'jpeg'])
original_image, original_lab = process_image(image_file)
display_image_streamlit(original_image, title="Original Image")

# UI for selecting art/material/dye and sliders
art_type = st.selectbox('Art Type:', art_types)
valid_materials = sorted({material for art, material, dye in valid_combinations if art == art_type})
material_type = st.selectbox('Material Type:', valid_materials)

dye_type = None
if material_type == 'Textiles':
    valid_dyes = sorted({dye for art, material, dye in valid_combinations if material == material_type and dye})
    dye_type = st.selectbox('Dye Type:', valid_dyes)

exposure_years = st.slider('Years of Aging:', min_value=0, max_value=100, value=5)  # Currently unused
uv_exposure = st.slider('UV Exposure:', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
lux_hours = st.slider('Lux Hours:', min_value=0, max_value=100000, value=50000, step=1000)
humidity = st.slider('Humidity (%):', min_value=0, max_value=100, value=50, step=1)
temp_value = st.slider('Temperature (°C):', min_value=-10, max_value=50, value=20, step=1)
pollution = st.slider('Pollution Level:', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
year_of_manufacture = st.slider('Year of Manufacture:', min_value=1455, max_value=2020, value=2000, step=1)

if st.button('Run Simulation'):
    # Display original average color
    avg_lab_before = display_average_color(original_lab, title='Average Color - Original Image')

    # Create synthetic data, prepare features, and train model
    synthetic_data = create_synthetic_data(art_types, material_types, dye_types, valid_combinations, num_samples_per_combination=500)
    X, Y, encoder, poly = prepare_features(synthetic_data)
    model, scaler, mse = train_ml_model(X, Y)
    st.write(f"Cross-validated Mean Squared Error: {mse:.4f}")

    # Simulate exposure
    lab_exposed = simulate_exposure_by_material(original_lab, art_type, material_type, dye_type, uv_exposure, lux_hours, humidity, temp_value)
    exposed_image = lab_to_rgb(lab_exposed)
    display_image_streamlit(exposed_image, title=f'Simulated Exposure: {art_type} on {material_type}')

    avg_lab_exposed = display_average_color(lab_exposed, title='Average Color - Simulated Exposure')

    # Compute Delta-E for simulation
    delta_e_simulation = compute_delta_e(original_lab, lab_exposed)
    st.write(f"Delta-E (Original vs Simulated Exposure): {delta_e_simulation}")
    delta_e_avg_simulation = compute_average_delta_e(avg_lab_before, avg_lab_exposed)
    st.write(f"Delta-E between average colors: {delta_e_avg_simulation:.2f}")

    # Plot histograms for Original vs Simulated Exposure
    plot_histograms_streamlit(original_image, exposed_image, title_suffix='Original vs Simulated Exposure')

    # Prepare features for prediction
    categorical_input = pd.DataFrame({'art_type': [art_type], 'material_type': [material_type], 'dye_type': [dye_type if dye_type else 'None']})
    categorical_input = categorical_input.fillna('None')
    categorical_encoded = encoder.transform(categorical_input)
    X_input_numeric = np.array([[lux_hours, uv_exposure, temp_value, humidity, pollution, year_of_manufacture]])
    X_input_numeric_poly = poly.transform(X_input_numeric)
    X_input = np.hstack((X_input_numeric_poly, categorical_encoded))
    X_input_scaled = scaler.transform(X_input)

    predicted_fading = model.predict(X_input_scaled)[0]
    lab_faded = apply_fading(lab_exposed, predicted_fading)
    faded_image = lab_to_rgb(lab_faded)
    display_image_streamlit(faded_image, title='Faded Image After ML Prediction')

    avg_lab_after = display_average_color(lab_faded, title='Average Color - After ML Prediction')

    delta_e_ml = compute_delta_e(lab_exposed, lab_faded)
    st.write(f"Delta-E (Simulated Exposure vs ML Prediction): {delta_e_ml}")
    delta_e_avg_ml = compute_average_delta_e(avg_lab_exposed, avg_lab_after)
    st.write(f"Delta-E between average colors (Simulated vs ML): {delta_e_avg_ml:.2f}")

    delta_e_total = compute_delta_e(original_lab, lab_faded)
    st.write(f"Delta-E (Original vs Final Faded): {delta_e_total}")
    delta_e_avg_total = compute_average_delta_e(avg_lab_before, avg_lab_after)
    st.write(f"Delta-E between average colors (Original vs Final Faded): {delta_e_avg_total:.2f}")

    plot_histograms_streamlit(original_image, faded_image, title_suffix='Original vs Final Faded')