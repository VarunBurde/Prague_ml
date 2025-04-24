import numpy as np
import cv2
import glob
import os
import pickle
from datetime import datetime

def calibrate_camera(images_folder, chessboard_size=(9, 6), square_size=1.0, visualize=False):
    """
    Perform intrinsic camera calibration using images from a folder.
    
    Parameters:
    -----------
    images_folder : str
        Path to the folder containing calibration images
    chessboard_size : tuple
        Number of inner corners in the chessboard (width, height)
    square_size : float
        Size of the chessboard square in real-world units (default is 1.0)
    visualize : bool
        Whether to visualize the corner detection results (Note: this parameter is kept for compatibility but visualization is disabled)
    
    Returns:
    --------
    ret : float
        Reprojection error
    mtx : ndarray
        Camera matrix
    dist : ndarray
        Distortion coefficients
    rvecs : list
        Rotation vectors
    tvecs : list
        Translation vectors
    """
    # Prepare object points: (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Get list of image files
    image_files = glob.glob(os.path.join(images_folder, '*.jpg')) + \
                  glob.glob(os.path.join(images_folder, '*.png')) + \
                  glob.glob(os.path.join(images_folder, '*.bmp'))
    
    if not image_files:
        raise ValueError(f"No image files found in folder: {images_folder}")
    
    print(f"Found {len(image_files)} images for calibration")
    
    successful_images = 0
    
    # Process each image
    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is None:
            print(f"Failed to load image: {image_file}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        # If found, add object points, image points
        if ret:
            successful_images += 1
            print(f"Successfully detected corners in: {os.path.basename(image_file)}")
            
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            
            # Draw and display the corners if visualization is enabled
            if visualize:
                img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)le)}")
                
                # Resize image to 1/3 of its original size for visualization
                h, w = img.shape[:2]ard patterns were detected in any of the provided images")
                display_img = cv2.resize(img, (w//3, h//3))
                g {successful_images} images for calibration")
                cv2.imshow('Chessboard Detection', display_img)
                cv2.waitKey(500)  # Show each image for 500ms
        else: dist, rvecs, tvecs = cv2.calibrateCamera(
            print(f"Could not find chessboard corners in: {os.path.basename(image_file)}")
    )
    if visualize:
        cv2.destroyAllWindows()ed with reprojection error: {ret}")
    
    if successful_images == 0:cs, tvecs, successful_images
        raise ValueError("No chessboard patterns were detected in any of the provided images")
    save_calibration_results(mtx, dist, output_folder="calibration_results"):
    print(f"Using {successful_images} images for calibration")
    # Create output folder if it doesn't exist
    # Perform camera calibration_folder):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    ) Save camera matrix and distortion coefficients
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Calibration completed with reprojection error: {ret}")tamp}.pkl")
    
    return ret, mtx, dist, rvecs, tvecs, successful_images
        'camera_matrix': mtx,
def save_calibration_results(mtx, dist, output_folder="calibration_results"):
    """Save the calibration results to files"""%d %H:%M:%S")
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)s f:
        pickle.dump(calibration_data, f)
    # Save camera matrix and distortion coefficients
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_folder, f"calibration_{timestamp}.pkl") mtx)
    np.savetxt(os.path.join(output_folder, f"distortion_coefficients_{timestamp}.txt"), dist)
    calibration_data = {
        'camera_matrix': mtx,ts saved to {output_folder}")
        'dist_coeffs': dist,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }ndistort_test_image(image_path, mtx, dist):
    """Test the calibration by undistorting an image"""
    with open(output_file, 'wb') as f:
        pickle.dump(calibration_data, f)
        print(f"Failed to load test image: {image_path}")
    # Also save as text files for easy viewing
    np.savetxt(os.path.join(output_folder, f"camera_matrix_{timestamp}.txt"), mtx)
    np.savetxt(os.path.join(output_folder, f"distortion_coefficients_{timestamp}.txt"), dist)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print(f"Calibration results saved to {output_folder}")
    return output_file
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
def undistort_test_image(image_path, mtx, dist):
    """Test the calibration by undistorting an image"""
    img = cv2.imread(image_path)
    if img is None:, h]):  # Check if ROI is valid
        print(f"Failed to load test image: {image_path}")
        return None
    return dst
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    """Undistort all images and save them to the output folder"""
    # Undistortath.exists(output_folder):
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
    # Crop the image (optional)saving all images to {output_folder}...")
    x, y, w, h = roi
    if all([x, y, w, h]):  # Check if ROI is validlder
        dst = dst[y:y+h, x:x+w]strftime("%Y%m%d_%H%M%S")
    calib_params_path = os.path.join(output_folder, f"calibration_params_{timestamp}.txt")
    return dst
    with open(calib_params_path, 'w') as f:
def undistort_all_images(image_files, mtx, dist, output_folder):
    """Undistort all images and save them to the output folder"""
    if not os.path.exists(output_folder):time.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        os.makedirs(output_folder)trinsics):\n")
        f.write(str(mtx))
    print(f"\nUndistorting and saving all images to {output_folder}...")
        f.write("Distortion Coefficients:\n")
    # Save calibration parameters in the output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calib_params_path = os.path.join(output_folder, f"calibration_params_{timestamp}.txt")
    print(f"Calibration parameters saved to {calib_params_path}")
    with open(calib_params_path, 'w') as f:
        f.write("Camera Calibration Parameters\n")
        f.write("============================\n\n")
        f.write(f"Calibration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Camera Matrix (Intrinsics):\n")
        f.write(str(mtx)) undistort_test_image(image_path, mtx, dist)
        f.write("\n\n")
        f.write("Distortion Coefficients:\n")
        f.write(str(dist.ravel()))rted image with same filename (without the "undistorted_" prefix)
        f.write("\n")ame = os.path.basename(image_path)
                output_path = os.path.join(output_folder, basename)
    print(f"Calibration parameters saved to {calib_params_path}")
                processed_count += 1
    processed_count = 0essed_count % 10 == 0:  # Print status every 10 images
    for image_path in image_files:ss: {processed_count}/{len(image_files)} images processed")
        try:pt Exception as e:
            # Undistort the imageing {image_path}: {str(e)}")
            undistorted = undistort_test_image(image_path, mtx, dist)
            Finished undistorting {processed_count} of {len(image_files)} images.")
            if undistorted is not None:aved in: {output_folder}")
                # Save the undistorted image with same filename (without the "undistorted_" prefix)
                basename = os.path.basename(image_path)
                output_path = os.path.join(output_folder, basename)
                cv2.imwrite(output_path, undistorted)_ml/calibration_images"
                processed_count += 1ult chessboard size
                if processed_count % 10 == 0:  # Print status every 10 images
                    print(f"Progress: {processed_count}/{len(image_files)} images processed")
        except Exception as e:on_results'  # Default output folder
            print(f"Error processing {image_path}: {str(e)}")istorted images
            
    print(f"Finished undistorting {processed_count} of {len(image_files)} images.")
    print(f"All calibrated images are saved in: {output_folder}")
        print(f"Calibration folder '{calibration_folder}' not found. Looking in current directory.")
if __name__ == "__main__": = '.'
    # Default parameters
    calibration_folder = "/home/varun/Projects/Prague_ml/calibration_images"
    chessboard_size = (8, 5)  # Default chessboard size
    square_size = 1.0  # Default square sizesful_images = calibrate_camera(
    visualize = False  # Disable visualization
    output_folder = 'calibration_results'  # Default output folder
    calibrated_folder = 'calibrated_images'  # Folder for undistorted images
            visualize
    # Check if calibration folder exists, use current directory as fallback
    if not os.path.exists(calibration_folder):
        print(f"Calibration folder '{calibration_folder}' not found. Looking in current directory.")
        calibration_folder = '.'ults:")
        print(f"Reprojection Error: {ret}")
    try:print(f"Camera Matrix:\n{mtx}")
        # Perform calibrationefficients:\n{dist.ravel()}")
        ret, mtx, dist, rvecs, tvecs, successful_images = calibrate_camera(
            calibration_folder, 
            chessboard_size, 
            square_size,ve_calibration_results(mtx, dist, output_folder)
            visualize
        ) Test undistortion using all calibration images
        image_files = glob.glob(os.path.join(calibration_folder, '*.jpg')) + \
        # Display resultsb.glob(os.path.join(calibration_folder, '*.png')) + \
        print("\nCalibration Results:").join(calibration_folder, '*.bmp'))
        print(f"Reprojection Error: {ret}")
        print(f"Camera Matrix:\n{mtx}")
        print(f"Distortion Coefficients:\n{dist.ravel()}")alibrated_folder)
        print(f"Calibration used {successful_images} images")
        pt Exception as e:
        # Save resultsduring calibration: {str(e)}")
        output_file = save_calibration_results(mtx, dist, output_folder)                # Test undistortion using all calibration images        image_files = glob.glob(os.path.join(calibration_folder, '*.jpg')) + \                      glob.glob(os.path.join(calibration_folder, '*.png')) + \                      glob.glob(os.path.join(calibration_folder, '*.bmp'))                if image_files:            undistort_all_images(image_files, mtx, dist, calibrated_folder)            except Exception as e:        print(f"Error during calibration: {str(e)}")