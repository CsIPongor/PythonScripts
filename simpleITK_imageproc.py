import SimpleITK as sitk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_image_objects(image_path, segmentation_method='Otsu'):
    """
    Converts an image to grayscale, segments it, and analyzes object parameters.

    Args:
        image_path (str): Path to the input image (TIFF or JPEG).
        segmentation_method (str): Segmentation method to use. 'Otsu' or 'Huang'.

    Returns:
        pandas.DataFrame: A DataFrame containing object parameters, or None if an error occurs.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    try:
        # 1. Load the image
        print(f"Loading image: {image_path}")
        image = sitk.ReadImage(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # 2. Convert to grayscale if not already single channel
    if image.GetNumberOfComponentsPerPixel() > 1:
        print("Converting image to grayscale...")
        image_gray = sitk.Cast(sitk.GetImageFromArray(sitk.GetArrayFromImage(image)[:,:,0]), sitk.sitkUInt8) # Taking first channel as grayscale
    else:
        image_gray = sitk.Cast(image, sitk.sitkUInt8)

    # 3. Perform segmentation
    print(f"Performing segmentation using {segmentation_method} method...")
    if segmentation_method == 'Otsu':
        # Otsu's thresholding
        # The OtsuThreshold method returns the threshold value. We use BinaryThreshold to apply it.
        otsu_threshold = sitk.OtsuThreshold(image_gray)
        segmented_image = sitk.BinaryThreshold(image_gray,
                                               lowerThreshold=0,
                                               upperThreshold=otsu_threshold,
                                               insideValue=1,
                                               outsideValue=0)
    elif segmentation_method == 'Huang':
        # Huang's thresholding
        huang_threshold = sitk.HuangThreshold(image_gray)
        segmented_image = sitk.BinaryThreshold(image_gray,
                                               lowerThreshold=0,
                                               upperThreshold=huang_threshold,
                                               insideValue=1,
                                               outsideValue=0)
    else:
        print("Invalid segmentation method. Choose 'Otsu' or 'Huang'.")
        return None

    # 4. Label connected components
    print("Labeling connected components...")
    label_image = sitk.ConnectedComponent(segmented_image)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_intensity_filter = sitk.LabelIntensityStatisticsImageFilter()

    label_shape_filter.Execute(label_image)
    label_intensity_filter.Execute(label_image, image_gray)

    # Prepare lists to store object properties
    object_data = []

    # Iterate through each labeled object
    print("Analyzing object parameters...")
    for label in label_shape_filter.GetLabels():
        if label == 0:  # Skip background
            continue

        # Geometric properties
        object_size_pixels = label_shape_filter.GetNumberOfPixels(label)
        object_area_physical = label_shape_filter.GetPhysicalSize(label)
        centroid = label_shape_filter.GetCentroid(label)
        
   
        # A value closer to 0 indicates a more circular object.
        elongation = label_shape_filter.GetElongation(label)

        # Intensity properties
        mean_intensity = label_intensity_filter.GetMean(label)
        variance_intensity = label_intensity_filter.GetVariance(label)
        min_intensity = label_intensity_filter.GetMinimum(label)
        max_intensity = label_intensity_filter.GetMaximum(label)
        std_intensity = label_intensity_filter.GetStandardDeviation(label)
        sum_intensity = label_intensity_filter.GetSum(label)

        object_data.append({
            'Label': label,
            'Size_Pixels': object_size_pixels,
            'Area_Physical_Units': object_area_physical,
            'Centroid_X': centroid[0] if centroid else np.nan,
            'Centroid_Y': centroid[1] if centroid else np.nan,
            'Elongation': elongation, # Proxy for circularity (0 = perfectly circular)
            'Mean_Intensity': mean_intensity,
            'Variance_Intensity': variance_intensity,
            'Min_Intensity': min_intensity,
            'Max_Intensity': max_intensity,
            'Std_Dev_Intensity': std_intensity,
            'Sum_Intensity': sum_intensity
        })

    if not object_data:
        print("No objects found after segmentation.")
        return pd.DataFrame() # Return empty DataFrame

    df = pd.DataFrame(object_data)
    
    print("Analysis complete. Displaying results and visualizations...")
    # Optional: Display images
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(sitk.GetArrayViewFromImage(image_gray), cmap='gray')
    axes[0].set_title('Grayscale Image')
    axes[0].axis('off')

    axes[1].imshow(sitk.GetArrayViewFromImage(segmented_image), cmap='gray')
    axes[1].set_title(f'Segmented Image ({segmentation_method})')
    axes[1].axis('off')

    axes[2].imshow(sitk.GetArrayViewFromImage(sitk.LabelToRGB(label_image)), cmap='nipy_spectral')
    axes[2].set_title('Labeled Objects')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

    return df

if __name__ == "__main__":
    # Example Usage:
    # Create a dummy image for demonstration if no actual image is provided
    dummy_image_path = "dummy_image.png"
    if not os.path.exists(dummy_image_path):
        print(f"Creating a dummy image for demonstration: {dummy_image_path}")
        dummy_image_array = np.zeros((100, 100), dtype=np.uint8)
        # Add some "objects"
        dummy_image_array[20:40, 20:40] = 200 # Square object
        dummy_image_array[60:70, 50:80] = 150 # Rectangular object
        dummy_image_array[10:15, 80:90] = 255 # Small bright object
        dummy_image_array[75:85, 10:20] = 100 # Another small object
        
        dummy_image_sitk = sitk.GetImageFromArray(dummy_image_array)
        sitk.WriteImage(dummy_image_sitk, dummy_image_path)
    
    # Test with Otsu's method
    print("\n--- Running analysis with Otsu's method ---")
    object_stats_otsu = analyze_image_objects(dummy_image_path, segmentation_method='Otsu')
    if object_stats_otsu is not None:
        print("\nObject Statistics (Otsu's Method):")
        print(object_stats_otsu)

    # Test with Huang's method
    print("\n--- Running analysis with Huang's method ---")
    object_stats_huang = analyze_image_objects(dummy_image_path, segmentation_method='Huang')
    if object_stats_huang is not None:
        print("\nObject Statistics (Huang's Method):")
        print(object_stats_huang)

    # You can replace 'dummy_image.png' with your actual image file path
    # For example:
    # my_image_path = "path/to/your/image.tiff"
    # object_stats_my_image = analyze_image_objects(my_image_path, segmentation_method='Otsu')
    # if object_stats_my_image is not None:
    #     print("\nObject Statistics for your image:")
    #     print(object_stats_my_image)
