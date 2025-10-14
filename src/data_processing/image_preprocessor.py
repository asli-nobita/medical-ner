"""Image preprocessing module for medical document processing."""

import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from pdf2image import convert_from_path
import glob

def preprocess_directory(image_dir, output_dir='cleaned_images', dpi=300, base_filename='cleaned_image'):
    """
    Preprocess all supported files in a directory and save processed images.
    Uses the existing `preprocess_images` and `save_images` functions defined above.

    Args:
        image_dir (str): Directory containing PDFs / JPG / PNG files.
        output_dir (str): Directory to save processed images.
        dpi (int): DPI for PDF conversion (forwarded to preprocess_images).
        base_filename (str): Base filename used by save_images.

    Returns:
        list: List of processed PIL images saved (or None if none processed).
    """
    if not os.path.exists(image_dir):
        print(f"❌ Directory not found: {image_dir}")
        return None

    
    patterns = ['*.pdf', '*.PDF', '*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(image_dir, p)))
    files = sorted(set(files))

    if not files:
        print(f"⚠️ No supported files found in {image_dir}")
        return None

    print(f"Found {len(files)} files in {image_dir} — processing...")

    all_processed = []
    for file_path in files:
        print(f"➡️  Processing file: {os.path.basename(file_path)}")
        try:
            processed = preprocess_images(file_path, dpi=dpi)
            if processed:
                all_processed.extend(processed)
                print(f"   ✅ {len(processed)} page(s) processed from {os.path.basename(file_path)}")
            else:
                print(f"   ⚠️ No pages returned for {os.path.basename(file_path)}")
        except Exception as e:
            print(f"   ❌ Error processing {os.path.basename(file_path)}: {e}")

    if not all_processed:
        print("❌ No images were processed successfully.")
        return None

    
    save_images(all_processed, output_dir, base_filename=base_filename)
    print(f"📁 Saved {len(all_processed)} processed images to {output_dir}")
    return all_processed

def preprocess_images(image_path, dpi=300): 
    """
    Preprocesses an image for OCR, handling PDF, JPG and PNG format. 

    Args:
        image_path (str): Path to the images
        dpi (int): DPI for PDF conversion. Defaults to 300. 
        
    Returns: 
        list: A list of PIL Image objects, one per page. 
    """ 
    images = [] 
    if image_path.lower().endswith('.pdf'): 
        
        pages = convert_from_path(image_path, dpi=dpi)  
        for page in pages: 
            images.append(page.rotate(90, expand=True)) 
    else: 
        
        try: 
            image = Image.open(image_path) 
            image = ImageOps.exif_transpose(image) 
            images = [image] 
        except FileNotFoundError: 
            print(f'Error: File not found at {image_path}') 
            return None 
        except Exception as e: 
            print(f'Unknown error opening file: {e}') 
            return None 
    
    processed_images = [] 
    for image in images: 
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) 
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        
        try: 
            if gray.dtype != np.uint8: 
                gray = (gray * 255).astype(np.uint8) 
            coords = np.column_stack(np.where(gray>0)) 
            if len(coords) > 0:  
                angle = cv2.minAreaRect(coords)[-1] 
                if angle < -45: 
                    angle = -(90+angle) 
                else: 
                    angle = -angle 
                
                
                if abs(angle) > 0.5:
                    (h,w) = gray.shape[:2]  
                    center = (w//2, h//2) 
                    
                    
                    cos = np.abs(np.cos(np.radians(angle)))
                    sin = np.abs(np.sin(np.radians(angle)))
                    new_w = int((h * sin) + (w * cos))
                    new_h = int((h * cos) + (w * sin))
                    
                    
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    M[0, 2] += (new_w / 2) - center[0]
                    M[1, 2] += (new_h / 2) - center[1]
                    
                    
                    gray = cv2.warpAffine(gray, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) 
        except Exception as e: 
            print(f"Deskewing failed: {e}") 
        
        denoised = cv2.GaussianBlur(gray, (5,5), 0) 
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) 
        sharpened = cv2.filter2D(denoised,-1,kernel) 
        thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
        processed_image = Image.fromarray(thresh) 
        processed_images.append(processed_image) 

    return processed_images


def save_images(images, output_dir, base_filename='cleaned_image'): 
    """Saves the processed images to the specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f"{base_filename}_{i+1}.png")
        image.save(output_path, "PNG")
        print(f"Saved: {output_path}")