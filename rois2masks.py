import numpy as np
import tifffile
import roifile
import cv2
import os

def get_image_shape(image_path):

    with tifffile.TiffFile(image_path) as tif:
        image_data = tif.asarray()
    return image_data.shape

def generate_mask_from_roi(roi, image_shape, mask_id):

    mask = np.zeros(image_shape, dtype=np.uint16)
    
    coordinates = roi.coordinates()
    
    if coordinates is not None and len(coordinates) > 0:
        coordinates = np.array(coordinates, dtype=np.int32)
        coordinates = coordinates.reshape((-1, 1, 2))
        
        cv2.fillPoly(mask, [coordinates], mask_id)
    
    return mask

def generate_mask_from_zip(roi_zip_path, image_shape):

    mask = np.zeros(image_shape, dtype=np.uint16)
    mask_id = 1
    
    rois = roifile.roiread(roi_zip_path)
    
    for roi in rois:
        roi_mask = generate_mask_from_roi(roi, image_shape, mask_id)
        
        mask[roi_mask > 0] = mask_id
        mask_id += 1
    
    return mask

def save_mask_as_image(mask, output_file):

    tifffile.imwrite(output_file, mask)

def process_image(image_path):

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    roi_zip_path = os.path.join(
        os.path.dirname(image_path),
        "results",
        base_name,
        "ROIs",
        "ROIs_Podosomes.zip"
    )
    
    if not os.path.exists(roi_zip_path):
        print(f"ROI zip file not found: {roi_zip_path}")
        return
    
    image_shape = get_image_shape(image_path)
    
    mask = generate_mask_from_zip(roi_zip_path, image_shape)
    
    output_file = os.path.splitext(image_path)[0] + "_segrun.tif"
    
    save_mask_as_image(mask, output_file)
    print(f"Mask saved to {output_file}")

def process_images_in_folder(folder_path=None):

    if folder_path is None:
        folder_path = input("Please enter the folder path containing the TIFF files: ")

    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return
    
    tif_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.tif')]

    if not tif_files:
        print(f"No TIFF files found in the folder '{folder_path}'.")
        return
    
    for image_path in tif_files:
        print(f"Processing {image_path}...")
        process_image(image_path)

if __name__ == "__main__":
    process_images_in_folder()

