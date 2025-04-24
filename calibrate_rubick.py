import cv2
import numpy as np
import os
import glob

def load_calibration_data(camera_matrix_path, distortion_coeffs_path):
    """Load camera calibration parameters from files."""
    # Load camera matrix
    camera_matrix = np.loadtxt(camera_matrix_path)
    camera_matrix = camera_matrix.reshape(3, 3)
    
    # Load distortion coefficients
    dist_coeffs = np.loadtxt(distortion_coeffs_path)
    
    return camera_matrix, dist_coeffs

def undistort_image(image, camera_matrix, dist_coeffs):
    """Undistort an image using camera calibration parameters."""
    h, w = image.shape[:2]
    
    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    # Undistort the image
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Crop the image based on ROI
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted

def main():
    # Paths to calibration files
    camera_matrix_path = '/home/varun/Projects/Prague_ml/calibration_results/camera_matrix_20250424_165446.txt'
    distortion_coeffs_path = '/home/varun/Projects/Prague_ml/calibration_results/distortion_coefficients_20250424_165446.txt'
    
    # Load calibration data
    camera_matrix, dist_coeffs = load_calibration_data(camera_matrix_path, distortion_coeffs_path)
    print("Camera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)
    
    # Path to rubic folder
    rubic_folder = '/home/varun/Projects/Prague_ml/rubic'
    output_folder = '/home/varun/Projects/Prague_ml/rubic_calibrated'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Get all image files in the rubic folder
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(rubic_folder, f'*.{ext}')))
        image_files.extend(glob.glob(os.path.join(rubic_folder, f'*.{ext.upper()}')))
    
    if not image_files:
        print(f"No images found in {rubic_folder}")
        return
    
    print(f"Found {len(image_files)} images to calibrate")
    
    # Process each image
    for img_path in image_files:
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
            
        # Get the filename
        filename = os.path.basename(img_path)
        
        # Undistort the image
        undistorted_img = undistort_image(img, camera_matrix, dist_coeffs)
        
        # Save the undistorted image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, undistorted_img)
        
        print(f"Processed: {filename}")
    
    print(f"Calibration complete. Calibrated images saved to {output_folder}")

if __name__ == "__main__":
    main()
