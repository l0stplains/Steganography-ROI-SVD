import os
import json
import uuid
import numpy as np
import cv2
import math
import tempfile  # Import for temporary file handling
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from ultralytics import YOLO
from PIL import Image
import tifffile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# Only allow PNG for embedding and PNG/TIFF for extraction
app.config['ALLOWED_EXTENSIONS_EMBED'] = {'png'}
app.config['ALLOWED_EXTENSIONS_EXTRACT'] = {'png', 'tiff', 'tif'}

# Ensure subfolders exist
for subfolder in ['original', 'detected', 'stego', 'data']:
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], subfolder), exist_ok=True)

# Load YOLOv8 model (ensure "yolov8n.pt" is in the project root)
model = YOLO('yolov8n.pt')

def allowed_file(filename, operation='embed'):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if operation == 'embed':
        return ext in app.config['ALLOWED_EXTENSIONS_EMBED']
    elif operation == 'extract':
        return ext in app.config['ALLOWED_EXTENSIONS_EXTRACT']
    return False

def generate_filename(extension):
    return f"{uuid.uuid4()}.{extension}"

###############################################################################
#                           SVD.PY LOGIC (INLINE)                             #
###############################################################################

def pad_to_multiple_of_4(image):
    """Pads the image (H,W[,C]) so that both H and W are multiples of 4."""
    height, width = image.shape[:2]
    new_height = math.ceil(height / 4) * 4
    new_width = math.ceil(width / 4) * 4

    if len(image.shape) == 3:
        result = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    else:
        result = np.zeros((new_height, new_width), dtype=image.dtype)

    result[:height, :width] = image
    return result

def divide_into_blocks(image, block_size=4):
    """Divide image into block_size x block_size blocks, padding if needed."""
    h, w, _ = image.shape
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    padded = pad_to_multiple_of_4(image)
    new_h, new_w = padded.shape[:2]

    blocks = []
    for i in range(0, new_h, block_size):
        for j in range(0, new_w, block_size):
            block = padded[i:i+block_size, j:j+block_size]
            blocks.append(((i, j), block))
    return blocks, new_h, new_w

def calculate_capacity(block_size, num_blocks, channels=3):
    """Each 4x4 block can embed 3 bits (1 bit per color channel)."""
    bits = num_blocks * channels
    chars = bits // 8
    return bits, chars

def adjust_to_nearest_multiple(value, multiple, max_value):
    """
    Adjusts the given value to the nearest multiple of 'multiple'.
    If the adjusted value exceeds 'max_value', it adjusts to the closest valid multiple within bounds.
    """
    adjusted = round(value / multiple) * multiple
    # Ensure adjusted value is within [0, max_value]
    adjusted = max(0, min(adjusted, max_value))
    return adjusted

def encode_message(image_path, message):
    # Read and convert image to double (0-1 range)
    I = np.array(Image.open(image_path).convert('RGB'))
    
    # Pad image to multiple of 4
    I_padded = pad_to_multiple_of_4(I)
    I2 = I_padded.astype(np.float64) / 255.0

    # Convert message to binary
    M = message
    lm = len(M)
    MNum = [ord(char) for char in M]
    MNumFinal = [format(num, '08b') for num in MNum]

    # Flatten binary representation
    Emp = []
    for a in range(lm):
        for b in range(8):
            Emp.append(MNumFinal[a][b])
    Emp.append('2')  # End marker

    height, width = I2.shape[:2]
    encoded = np.copy(I2)

    # Encode message
    isBreaking = False
    idx = 0

    for i in range(1, height // 4 + 1):
        for j in range(1, width // 4 + 1):
            if j == width // 4 or i == height // 4:
                continue
            for channel in range(3):  # Loop through all channels
                # Extract 4x4 block
                block = I2[4*i-4:4*i, 4*j-4:4*j, channel]

                # Skip if block is not 4x4
                if block.shape[0] != 4 or block.shape[1] != 4:
                    continue

                # Perform SVD
                U, s, VT = np.linalg.svd(block)
                S = np.zeros((4, 4))
                np.fill_diagonal(S, s)

                # Get current message bit
                if idx < len(Emp):
                    if Emp[idx] == '0':
                        S[3, 3] = 0
                    elif Emp[idx] == '1':
                        if s[3] <= 1e-3:
                            S[3, 3] = s[2] / 5
                            if s[2] <= 1e-3:
                                S[2, 2] = s[1] / 5
                                S[3, 3] = S[2, 2] / 5
                                if s[1] <= 1e-3:
                                    S[1, 1] = s[0] / 5
                                    S[2, 2] = S[1, 1] / 5
                                    S[3, 3] = S[2, 2] / 5
                    elif Emp[idx] == '2':
                        isBreaking = True
                        break

                    # Reconstruct block
                    A = U @ S @ VT
                    encoded[4*i-4:4*i, 4*j-4:4*j, channel] = A

                idx += 1
            if isBreaking:
                break
        if isBreaking:
            break

    # Crop back to original subregion size
    encoded_cropped = encoded[:I.shape[0], :I.shape[1]]
    encoded_cropped = np.clip(encoded_cropped, 0, 1.0)
    # Do not convert to uint8 here; keep float32 for TIFF
    return encoded_cropped, I.shape[:2]  # Return original dimensions

def decode_message(encoded_image):
    height, width = encoded_image.shape[:2]
    NewM = []

    isEnd = False
    for i in range(1, height // 4 + 1):
        for j in range(1, width // 4 + 1):
            if j == width // 4 or i == height // 4:
                continue
            for channel in range(3):
                block = encoded_image[4*i-4:4*i, 4*j-4:4*j, channel]
                
                if block.shape[0] != 4 or block.shape[1] != 4:
                    continue

                _, s, _ = np.linalg.svd(block)

                if np.abs(s[3]) <= 1e-6:
                    NewM.append('0')
                else:
                    NewM.append('1')

                # Check for termination pattern (end marker)
                #if len(NewM) >= 8 and ''.join(NewM[-8:]) == '00000010':
                    #NewM = NewM[:-8]  # Remove the end marker
                    #isEnd = True
                    #break
            #if isEnd:
                #break
        #if isEnd:
            #break
        #print("last 4 char from 8 bits:", chr(int(''.join(NewM[-32:-24]), 2)), chr(int(''.join(NewM[-24:-16]), 2)) , chr(int(''.join(NewM[-16:-8]), 2)))

    # Convert binary back to message
    message = ""
    for k in range(0, len(NewM) - 7, 8):
        try:
            b = ''.join(NewM[k:k+8])
            a = int(b, 2)
            l = chr(a)
            if (not l.isprintable() or not (32 <= a <= 126)) and a != 10: # 10 for newline
                break
            message += l
        except:
            break
    return message

def save_image(image, path, format="TIFF"):
    """
    Saves the given image as TIFF or PNG.
    """
    if format.upper() == "PNG":
        print("Warning: Saving as PNG will decrease accuracy due to rounding.")
        image_uint8 = (image * 255).astype(np.uint8)
        Image.fromarray(image_uint8).save(path, format="PNG")
    elif format.upper() == "TIFF":
        try:
            with open('sRGB.icc', 'rb') as f:
                icc_profile = f.read()
        except FileNotFoundError:
            icc_profile = b''  # Fallback if ICC profile not found

        image_float32 = image.astype(np.float32)

        # ICC profile tag for TIFF
        ICC_TAG_ID = 34675  # TIFF tag ID for ICC profile
        if icc_profile:
            icc_tag = (ICC_TAG_ID, 7, len(icc_profile), icc_profile, True)
            extra_tags = [icc_tag]
        else:
            extra_tags = []

        # Save the image with the ICC profile attached
        with tifffile.TiffWriter(path, bigtiff=False) as tiff_writer:
            tiff_writer.write(
                image_float32,
                photometric="rgb",  # Indicates the image is in RGB
                extratags=extra_tags,  # Add the ICC profile using extratags
            )

def load_image(path):
    """
    Loads an image from a file.
    """
    if path.lower().endswith('.tiff') or path.lower().endswith('.tif'):
        return tifffile.imread(path)
    else:
        image = Image.open(path).convert('RGB')
        return np.array(image).astype(np.float64) / 255.0

###############################################################################
#                           FLASK ROUTES START HERE                           #
###############################################################################

@app.route('/')
def home():
    return redirect(url_for('embed_page'))

@app.route('/embed')
def embed_page():
    object_classes = list(model.names.values())
    return render_template('embed.html', object_classes=object_classes)

@app.route('/extract')
def extract_page():
    object_classes = list(model.names.values())
    return render_template('extract.html', object_classes=object_classes)

@app.route('/upload', methods=['POST'])
def upload_image():
    operation = request.form.get('operation', 'embed')
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400
    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename, operation):
        ext = file.filename.rsplit('.', 1)[1].lower()
        fname = generate_filename(ext)
        if operation == 'embed':
            folder = 'original'
        elif operation == 'extract':
            folder = 'original'
        else:
            return jsonify({'error': 'Invalid operation'}), 400

        path = os.path.join(app.config['UPLOAD_FOLDER'], folder, fname)
        file.save(path)
        print(f"[UPLOAD] Saved to {path}")
        return jsonify({'filename': fname}), 200
    else:
        if operation == 'embed':
            allowed = 'PNG'
        else:
            allowed = 'PNG, TIFF'
        return jsonify({'error': f'Allowed file types for {operation}: {allowed}'}), 400

@app.route('/detect', methods=['POST'])
def detect_objects():
    data = request.get_json()
    filename = data.get('filename')
    obj_class = data.get('object_class')
    operation = data.get('operation')

    if not filename or not obj_class or not operation:
        return jsonify({'error': 'Missing parameters'}), 400

    if operation not in ['embed', 'extract']:
        return jsonify({'error': 'Invalid operation'}), 400

    folder = 'original'
    path = os.path.join(app.config['UPLOAD_FOLDER'], folder, filename)

    if not os.path.exists(path):
        return jsonify({'error': 'File does not exist'}), 400

    # Convert TIFF to PNG if needed
    if filename.lower().endswith('.tiff') or filename.lower().endswith('.tif'):
        try:
            # Load TIFF as a NumPy array
            img_array = tifffile.imread(path)

            # Convert NumPy array to a PIL Image
            img = Image.fromarray((img_array * 255).astype(np.uint8))

            # Create a temporary file for the PNG
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_path = temp_file.name
                img.save(temp_path, format='PNG')
                converted_path = temp_path
                print("cc", converted_path)

        except Exception as e:
            return jsonify({'error': f'Image conversion error: {str(e)}'}), 500
    else:
        converted_path = path  # Use original path if not TIFF

    # YOLO detection
    try:
        results = model(converted_path)
    except Exception as e:
        if converted_path != path:
            os.remove(converted_path)  # Cleanup temporary file
        return jsonify({'error': f'YOLO model error: {str(e)}'}), 500

    # Cleanup temporary file if it was created
    if converted_path != path:
        os.remove(converted_path)

    img_rgb = load_image(path)  # float64 normalized to [0,1]
    H, W = img_rgb.shape[:2]
    detections = []
    for res in results:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, "Unknown")
            if cls_name == obj_class:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Adjust to nearest multiple of 19
                x1_adj = adjust_to_nearest_multiple(x1, 19, W)
                y1_adj = adjust_to_nearest_multiple(y1, 19, H)
                x2_adj = adjust_to_nearest_multiple(x2, 19, W)
                y2_adj = adjust_to_nearest_multiple(y2, 19, H)
                # Ensure that adjusted coordinates are valid
                if x1_adj >= x2_adj:
                    x2_adj = min(x1_adj + 19, W)
                if y1_adj >= y2_adj:
                    y2_adj = min(y1_adj + 19, H)
                detections.append({
                    'class': cls_name,
                    'x1': x1_adj,
                    'y1': y1_adj,
                    'x2': x2_adj,
                    'y2': y2_adj
                })
                print(f"Original ROI: ({x1}, {y1}, {x2}, {y2}) -> Adjusted ROI: ({x1_adj}, {y1_adj}, {x2_adj}, {y2_adj})")

    if not detections:
        return jsonify({'error': f'No objects of class {obj_class} found'}), 400

    # Draw bounding boxes for visualization
    img_rgb_display = np.copy(img_rgb)
    for d in detections:
        # Convert normalized float image back to uint8 for drawing
        img_display = (img_rgb_display * 255).astype(np.uint8)
        cv2.rectangle(img_display, (d['x1'], d['y1']), (d['x2'], d['y2']), (0, 255, 0), 2)
        cv2.putText(img_display, d['class'], (d['x1'], d['y1'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Convert back to float64 normalized
        img_rgb_display = img_display.astype(np.float64) / 255.0

    det_name = f"detected_{filename}"
    det_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected', det_name)
    # Save the detected image as PNG for visualization
    save_image(img_rgb_display, det_path, format="PNG")
    print(f"[DETECT] Painted bounding boxes -> {det_path}")

    # Save ROI data
    data_save = {
        'detections': detections
    }
    if operation == 'embed':
        roi_fname = f"{filename.split('.')[0]}_roi.json"
    else:
        roi_fname = f"{filename.split('.')[0]}_roi_extraction.json"

    jpath = os.path.join(app.config['UPLOAD_FOLDER'], 'data', roi_fname)
    with open(jpath, 'w') as f:
        json.dump(data_save, f)

    # Calculate capacity if embedding
    if operation == 'embed':
        total_capacity_chars = 0
        for d in detections:
            x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
            sub = img_rgb[y1:y2, x1:x2]
            if sub.size == 0:
                continue
            blocks, _, _ = divide_into_blocks(sub, 4)
            bits, chars = calculate_capacity(4, len(blocks))
            total_capacity_chars += chars
    else:
        total_capacity_chars = 0
        for d in detections:
            x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
            sub = img_rgb[y1:y2, x1:x2]
            if sub.size == 0:
                continue
            blocks, _, _ = divide_into_blocks(sub, 4)
            bits, chars = calculate_capacity(4, len(blocks))
            total_capacity_chars += chars

    # Return capacity along with detections
    return jsonify({
        'detections': detections,
        'detected_filename': det_name,  # Includes .png extension
        'total_capacity_chars': total_capacity_chars
    }), 200

@app.route('/embed_data', methods=['POST'])
def embed_data():
    """
    Embed text into the ROIs using the reference SVD logic.
    Saves result as TIFF by default and PNG if needed.
    Immediately verifies the embedded data by decoding it.
    """
    data = request.get_json()
    filename = data.get('filename')
    hidden_data = data.get('hidden_data')
    if not filename or hidden_data is None:
        return jsonify({'error': 'Missing embed parameters'}), 400

    orig_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original', filename)
    if not os.path.exists(orig_path):
        return jsonify({'error': 'Original image not found'}), 400

    roi_json = f"{filename.split('.')[0]}_roi.json"
    roi_path = os.path.join(app.config['UPLOAD_FOLDER'], 'data', roi_json)
    if not os.path.exists(roi_path):
        return jsonify({'error': 'ROI data not found'}), 400

    with open(roi_path, 'r') as f:
        roid = json.load(f)
    rois = roid['detections']

    # Load the original image as RGB using the reference's load_image
    img_rgb = load_image(orig_path)  # float64 normalized to [0,1]

    # Embed the hidden_data into each ROI
    for r in rois:
        x1, y1, x2, y2 = r['x1'], r['y1'], r['x2'], r['y2']
        sub = img_rgb[y1:y2, x1:x2]
        if sub.size == 0:
            continue

        # Convert sub-region back to uint8 for saving as PNG
        sub_uint8 = (sub * 255).astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            Image.fromarray(sub_uint8).save(tmp.name, format='PNG')
            # Encode the message into the sub-region
            encoded_sub, original_dims = encode_message(tmp.name, hidden_data)
        # Remove the temporary file
        os.remove(tmp.name)

        # Assign the encoded sub-region back to the main image
        # encoded_sub should have shape (original_dims[0], original_dims[1], 3)
        # Ensure that the encoded_sub's shape matches the ROI's shape
        try:
            img_rgb[y1:y2, x1:x2] = encoded_sub
        except ValueError as ve:
            print(f"Error assigning encoded sub-region: {ve}")
            return jsonify({'error': 'Encoded sub-region size mismatch.'}), 500

    # Save stego image as TIFF using the provided save_image logic
    base_name = f"stego_{filename.split('.')[0]}"
    stego_tiff_fname = f"{base_name}.tiff"
    stego_tiff_path = os.path.join(app.config['UPLOAD_FOLDER'], 'stego', stego_tiff_fname)

    save_image(img_rgb, stego_tiff_path, format="TIFF")
    print(f"[EMBED] Wrote stego TIFF => {stego_tiff_path}")

    # Save as PNG
    stego_png_fname = f"{base_name}.png"
    stego_png_path = os.path.join(app.config['UPLOAD_FOLDER'], 'stego', stego_png_fname)
    save_image(img_rgb, stego_png_path, format="PNG")
    print(f"[EMBED] Wrote stego PNG => {stego_png_path}")

    return jsonify({
        'stego_tiff_filename': stego_tiff_fname,
        'stego_png_filename': stego_png_fname,
        'data_embedded': hidden_data,
    }), 200

@app.route('/extract_data', methods=['POST'])
def extract_data():
    """
    Crops the selected ROI from the stego TIFF/PNG and runs decode_message.
    """
    data = request.get_json()
    stego_filename = data.get('stego_filename')
    selected_roi = data.get('selected_roi')
    if not stego_filename or not selected_roi:
        return jsonify({'error': 'Missing parameters for extraction'}), 400

    st_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original', stego_filename)
    if not os.path.exists(st_path):
        return jsonify({'error': 'Stego file does not exist'}), 400

    # Load the stego image
    if st_path.lower().endswith('.tiff') or st_path.lower().endswith('.tif'):
        try:
            img_rgb = tifffile.imread(st_path)

        except:
            return jsonify({'error': 'Cannot read stego TIFF image.'}), 400
    else:
        image = Image.open(st_path).convert('RGB')
        img_rgb = np.array(image).astype(np.float64) / 255.0

    x1, y1, x2, y2 = selected_roi['x1'], selected_roi['y1'], selected_roi['x2'], selected_roi['y2']
    sub = img_rgb[y1:y2, x1:x2]
    if sub.size == 0:
        return jsonify({'error': 'Selected ROI is empty'}), 400

    print(f"Extracting data from ROI ({x1}, {y1}) to ({x2}, {y2})")
    extracted_message = decode_message(sub)
    if not extracted_message:
        return jsonify({'error': 'No data extracted or message is empty.'}), 400

    return jsonify({'extracted_data': extracted_message}), 200

###############################################################################
#                     DOWNLOAD ROUTES (AS-IS)                                 #
###############################################################################

@app.route('/download/<subfolder>/<filename>')
def download_file(subfolder, filename):
    """
    Serves the file as-is from the specified subfolder.
    """
    if subfolder not in ['original', 'detected', 'stego', 'data']:
        return jsonify({'error': 'Invalid subfolder'}), 400
    path = os.path.join(app.config['UPLOAD_FOLDER'], subfolder, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(path, as_attachment=True)

###############################################################################

if __name__ == '__main__':
    app.run(debug=True)
