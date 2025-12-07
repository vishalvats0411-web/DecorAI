import os
import cv2
import random
import numpy as np
import torch
from flask import Flask, render_template, request, redirect, url_for, session
from PIL import Image
from werkzeug.utils import secure_filename

# ==========================================
# CONFIGURATION
# ==========================================
app = Flask(__name__)
app.secret_key = 'decorai_demo_secret_key'  # Required for session storage

# Configure folders
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static/results')
SUGGESTIONS_FOLDER = os.path.join(BASE_DIR, 'static/suggestions')
TEXTURES_FOLDER = os.path.join(BASE_DIR, 'static/textures')

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(SUGGESTIONS_FOLDER, exist_ok=True)

# ==========================================
# 1. LOAD AI MODEL (SAM) - OPTIMIZED
# ==========================================
print("------------------------------------------------")
print("Initializing AI Model...")

PREDICTOR = None
try:
    from segment_anything import sam_model_registry, SamPredictor

    # 1. Select Device (Automatically finds your RTX 3050)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Using Device: {DEVICE.upper()}")

    # 2. Load Model (Using Base Model)
    #SAM_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints/sam_vit_b_01ec64.pth")
    #MODEL_TYPE = "vit_b"

    # 2. Load Model (Using Large Model)
    SAM_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints/sam_vit_l_0b3195.pth")
    MODEL_TYPE = "vit_l"

    # 2. Load Model (Using Heavy Model)
    #SAM_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints/sam_vit_h_4b8939.pth")
    #MODEL_TYPE = "vit_h" 

    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    PREDICTOR = SamPredictor(sam)
    
    print("✅ AI Model (ViT-L) loaded successfully.")
except Exception as e:
    print(f"❌ Error loading AI Model: {e}")
    print("Please ensure 'segment-anything' is installed and the .pth checkpoint exists.")

print("------------------------------------------------")

# Color Themes
COLOR_THEMES = [
    ["eec9d2", "f4b6c2", "f6abb6"], ["011f4b", "005b96", "b3cde0"],
    ["3da4ab", "f6cd61", "fe8a71"], ["adcbe3", "e7eff6", "63ace5"],
    ["fec8c1", "fad9c1", "f9caa7"], ["009688", "35a79c", "54b2a9"],
    ["fdf498", "7bc043", "0392cf"], ["ee4035", "f37736", "fdf498"],
    ["ffffff", "d0e1f9", "4d648d"], ["eeeeee", "dddddd", "cccccc"],
    ["ff6f69", "ffcc5c", "88d8b0"], ["008744", "0057e7", "ffa700"]
]

# ==========================================
# 2. ROUTES
# ==========================================

@app.route('/', methods=['GET', 'POST'])
def home():
    """Handle image upload and point selection."""
    if request.method == 'POST':
        # 1. Check for file
        if 'roomimage' not in request.files:
            return redirect(request.url)
        file = request.files['roomimage']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # 2. Save File
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # 3. Process Coordinates (FIXED: Keep as floats)
            # Input format: "0.5,0.2|0.8,0.3"
            coords_str = request.form.get('coords', '')
            coords = []
            if coords_str:
                try:
                    # We utilize float() here instead of int() to keep precision
                    coords = [[float(j) for j in i.split(',')] for i in coords_str.split('|') if i]
                except ValueError:
                    print("Error parsing coordinates")

            if not coords:
                return redirect(request.url)

            # 4. Save to Session
            session['current_image'] = filename
            session['coords'] = coords
            
            # Reset colors in session
            session.pop('current_colors', None) 

            return redirect(url_for('segment'))

    return render_template('home.html')


@app.route('/segment', methods=['GET', 'POST'])
def segment():
    """Generate the segmented image and handle color updates."""
    if not PREDICTOR:
        return "AI Model not loaded. Check server logs.", 500

    filename = session.get('current_image')
    coords = session.get('coords')
    
    if not filename or not coords:
        return redirect(url_for('home'))

    img_abs_path = os.path.join(UPLOAD_FOLDER, filename)
    
    # 1. Determine Colors
    current_colors = []
    
    if request.method == 'POST':
        is_picker_active = request.form.get('colorchecker') == "on"
        
        if is_picker_active:
            raw_colors = request.form.getlist('color')
            current_colors = raw_colors 
        else:
            raw_colors = request.form.getlist('col1')
            for c in raw_colors:
                if c:
                    rgb = tuple(map(int, c.split(',')))
                    current_colors.append('#%02x%02x%02x' % rgb)
                else:
                    current_colors.append('#000000')
        
        session['current_colors'] = current_colors

    else:
        # Initial random colors
        if 'current_colors' in session:
            current_colors = session['current_colors']
        else:
            current_colors = []
            for _ in coords:
                rand_color = '#%02x%02x%02x' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                current_colors.append(rand_color)
            session['current_colors'] = current_colors

    # 2. AI Processing
    image = cv2.imread(img_abs_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    PREDICTOR.set_image(image)
    
    pil_image = Image.fromarray(image)
    
    # FIXED SCALING LOGIC
    # Get actual image dimensions
    real_height = image.shape[0]
    real_width = image.shape[1]

    scaled_points = []
    for pt in coords:
        # pt[0] is X percentage, pt[1] is Y percentage
        # We multiply by real dimensions to get exact pixel location
        actual_x = int(pt[0] * real_width)
        actual_y = int(pt[1] * real_height)
        scaled_points.append([actual_x, actual_y])

    # 3. Apply Masks and Colors
    for i, point in enumerate(scaled_points):
        input_point = np.array([point])
        input_label = np.array([1]) 
        
        mask, _, _ = PREDICTOR.predict(
            point_coords=input_point, 
            point_labels=input_label, 
            multimask_output=False
        )
        mask = np.squeeze(mask, axis=0)

        hex_color = current_colors[i].lstrip('#')
        rgb_color = tuple(int(hex_color[j:j+2], 16) for j in (0, 2, 4))
        
        pillow_mask = Image.fromarray(mask)
        color_fill = Image.new("RGB", pil_image.size, rgb_color)
        pil_image.paste(color_fill, mask=pillow_mask)

    # 4. Save Result
    result_filename = f"seg_{filename}"
    result_path = os.path.join(RESULTS_FOLDER, result_filename)
    pil_image.save(result_path)

    # 5. Render Template
    return render_template('decor.html', 
                           img={'image': {'url': url_for('static', filename=f'uploads/{filename}')}}, 
                           segImg={'segmentedImage': {'url': url_for('static', filename=f'results/{result_filename}')}},
                           colorMap=current_colors,
                           pk=1)


@app.route('/suggestions/<int:pk>', methods=['GET'])
def suggestions(pk):
    """Generate curated AI themes."""
    filename = session.get('current_image')
    coords = session.get('coords')
    
    if not filename:
        return redirect(url_for('home'))
        
    img_abs_path = os.path.join(UPLOAD_FOLDER, filename)
    
    image = cv2.imread(img_abs_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    PREDICTOR.set_image(image)
    
    # FIXED SCALING LOGIC (Same as above)
    real_height = image.shape[0]
    real_width = image.shape[1]
    scaled_points = [[int(pt[0] * real_width), int(pt[1] * real_height)] for pt in coords]
    
    suggestions_paths = []

    # 1. Generate Color Themes
    for idx, theme in enumerate(COLOR_THEMES):
        pil_image = Image.fromarray(image)
        
        for i, point in enumerate(scaled_points):
            input_point = np.array([point])
            input_label = np.array([1])
            
            mask, _, _ = PREDICTOR.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
            mask = np.squeeze(mask, axis=0)
            
            hex_c = theme[i % len(theme)]
            rgb_c = tuple(int(hex_c[j:j+2], 16) for j in (0, 2, 4))
            
            pillow_mask = Image.fromarray(mask)
            color_fill = Image.new("RGB", pil_image.size, rgb_c)
            pil_image.paste(color_fill, mask=pillow_mask)

        fname = f"theme_{idx}_{filename}"
        save_path = os.path.join(SUGGESTIONS_FOLDER, fname)
        pil_image.save(save_path)
        
        suggestions_paths.append({'sugImage': {'url': url_for('static', filename=f'suggestions/{fname}')}})

    # 2. Generate Textures
    if os.path.exists(TEXTURES_FOLDER):
        textures = os.listdir(TEXTURES_FOLDER)
        if scaled_points and textures:
            target_point = random.choice(scaled_points)
            for i, texture_name in enumerate(textures):
                try:
                    pil_image = Image.fromarray(image)
                    input_point = np.array([target_point])
                    input_label = np.array([1])
                    
                    mask, _, _ = PREDICTOR.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
                    mask = np.squeeze(mask, axis=0)
                    
                    tex_path = os.path.join(TEXTURES_FOLDER, texture_name)
                    texture_img = Image.open(tex_path).convert("RGB").resize(pil_image.size)
                    
                    pillow_mask = Image.fromarray(mask)
                    pil_image = Image.composite(texture_img, pil_image, pillow_mask)
                    
                    fname = f"texture_{i}_{filename}"
                    save_path = os.path.join(SUGGESTIONS_FOLDER, fname)
                    pil_image.save(save_path)
                    
                    suggestions_paths.append({'sugImage': {'url': url_for('static', filename=f'suggestions/{fname}')}})
                except Exception as e:
                    print(f"Skipping texture {texture_name}: {e}")

    return render_template('suggestions.html', paths=suggestions_paths, pk=pk)

if __name__ == '__main__':
    app.run(debug=True, port=5000)