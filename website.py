from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

print("Loading model... This may take a few minutes.")
start_time = time.time()

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipe.to(device)

end_time = time.time()
print(f"Model loaded in {end_time - start_time:.2f} seconds.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        # Check if both files are present
        if 'selfie' not in request.files or 'clothing' not in request.files:
            return 'Please upload both files'
        
        selfie = request.files['selfie']
        clothing = request.files['clothing']
        
        # Check if filenames are valid
        if selfie.filename == '' or clothing.filename == '':
            return 'No selected file'
        
        if selfie and allowed_file(selfie.filename) and clothing and allowed_file(clothing.filename):
            selfie_filename = secure_filename(selfie.filename)
            clothing_filename = secure_filename(clothing.filename)
            
            selfie_path = os.path.join(app.config['UPLOAD_FOLDER'], selfie_filename)
            clothing_path = os.path.join(app.config['UPLOAD_FOLDER'], clothing_filename)
            
            selfie.save(selfie_path)
            clothing.save(clothing_path)
            
            # Process images with pix2pix
            result_path = process_images(selfie_path, clothing_path)
            
            return render_template('result.html', result_image=result_path)
    
    return render_template('upload.html')

def process_images(selfie_path, clothing_path):
    selfie_image = Image.open(selfie_path).convert("RGB")
    clothing_image = Image.open(clothing_path).convert("RGB")
    
    # Resize images to 512x512 (or another size that works with your model)
    selfie_image = selfie_image.resize((512, 512))
    clothing_image = clothing_image.resize((512, 512))
    
    prompt = f"Replace the clothing in the selfie with the clothing from the other image"
    
    print("Starting image processing...")
    start_time = time.time()
    
    with torch.no_grad():
        images = pipe(prompt=prompt, image=selfie_image, num_inference_steps=20, image_guidance_scale=1).images
    
    end_time = time.time()
    print(f"Image processed in {end_time - start_time:.2f} seconds.")
    
    result_filename = f"result_{os.path.basename(selfie_path)}"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    images[0].save(result_path)
    
    return result_filename

@app.route('/process_status')
def process_status():
    # This is a placeholder. In a real application, you'd track the status.
    return jsonify({"status": "processing"})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
