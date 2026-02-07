#!/usr/bin/env python3
"""
Medical Image Analysis System
Detects abnormalities in MRI/CT/X-ray scans

NOTE: This is for educational/research purposes only.
NOT approved for clinical diagnosis. Always consult medical professionals.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse

class MedicalImageAnalyzer:
    """
    Medical image analysis using various techniques
    
    DISCLAIMER: This is a demonstration tool for learning purposes.
    It is NOT validated for clinical use and should NOT be used for 
    actual medical diagnosis. Always consult qualified healthcare professionals.
    """
    
    def __init__(self, model_type='basic'):
        """
        Initialize medical image analyzer
        
        Args:
            model_type: 'basic' (preprocessing only) or 'deep' (if models available)
        """
        self.model_type = model_type
        
        print("=" * 70)
        print("MEDICAL IMAGE ANALYSIS SYSTEM - EDUCATIONAL USE ONLY")
        print("=" * 70)
        print("⚠️  WARNING: This tool is for educational/research purposes only")
        print("⚠️  NOT validated for clinical diagnosis")
        print("⚠️  Always consult qualified medical professionals")
        print("=" * 70)
        
    def preprocess_medical_image(self, image_path):
        """
        Preprocess medical images (MRI/CT/X-ray)
        
        Args:
            image_path: Path to medical image
            
        Returns:
            Dictionary with processed images
        """
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None
        
        results = {'original': img}
        
        # 1. Contrast Limited Adaptive Histogram Equalization (CLAHE)
        # Enhances contrast in medical images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        results['clahe'] = clahe.apply(img)
        
        # 2. Gaussian blur to reduce noise
        results['denoised'] = cv2.GaussianBlur(img, (5, 5), 0)
        
        # 3. Edge detection (useful for finding boundaries)
        results['edges'] = cv2.Canny(results['denoised'], 50, 150)
        
        # 4. Adaptive thresholding (separates regions)
        results['threshold'] = cv2.adaptiveThreshold(
            results['clahe'], 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 5. Morphological operations (clean up image)
        kernel = np.ones((3, 3), np.uint8)
        results['morphology'] = cv2.morphologyEx(
            results['threshold'], 
            cv2.MORPH_CLOSE, 
            kernel
        )
        
        return results
    
    def detect_abnormalities(self, image_path, sensitivity=0.5):
        """
        Basic abnormality detection using computer vision
        
        This is NOT a replacement for trained medical AI models.
        It uses basic image processing to highlight potential areas of interest.
        
        Args:
            image_path: Path to medical scan
            sensitivity: Detection sensitivity (0.0 - 1.0)
            
        Returns:
            Image with highlighted regions
        """
        # Preprocess
        processed = self.preprocess_medical_image(image_path)
        
        if processed is None:
            return None
        
        original = processed['original']
        enhanced = processed['clahe']
        
        # Find contours (potential abnormal regions)
        edges = cv2.Canny(enhanced, 50, 150)
        contours, _ = cv2.findContours(
            edges, 
            cv2.RETR_EXTERNAL, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        )
        
        # Convert to color for visualization
        output = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        
        # Filter contours by size and draw
        min_area = int(original.size * 0.001 * sensitivity)  # Adjust based on sensitivity
        max_area = int(original.size * 0.1)
        
        detected_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area < area < max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate circularity (tumors often circular)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                else:
                    circularity = 0
                
                detected_regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'circularity': circularity
                })
        
        # Sort by area (larger potentially more significant)
        detected_regions.sort(key=lambda x: x['area'], reverse=True)
        
        # Draw top regions
        for i, region in enumerate(detected_regions[:10]):  # Top 10
            x, y, w, h = region['bbox']
            
            # Color code by circularity (more circular = more red)
            if region['circularity'] > 0.7:
                color = (0, 0, 255)  # Red - high circularity
                label = "High interest"
            elif region['circularity'] > 0.5:
                color = (0, 165, 255)  # Orange
                label = "Medium interest"
            else:
                color = (0, 255, 255)  # Yellow - low circularity
                label = "Low interest"
            
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output, f"{label}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output, detected_regions, processed
    
    def analyze_scan(self, image_path, save_output=True):
        """
        Complete analysis of a medical scan
        
        Args:
            image_path: Path to scan
            save_output: Whether to save results
        """
        print(f"\nAnalyzing: {os.path.basename(image_path)}")
        print("-" * 50)
        
        result, regions, processed = self.detect_abnormalities(image_path)
        
        if result is None:
            return
        
        print(f"Detected {len(regions)} regions of interest")
        
        # Show top 5
        print("\nTop regions by size:")
        for i, region in enumerate(regions[:5], 1):
            print(f"  {i}. Area: {region['area']:.0f} pixels, "
                  f"Circularity: {region['circularity']:.2f}")
        
        if save_output:
            # Save annotated image
            output_dir = Path("medical_analysis_output")
            output_dir.mkdir(exist_ok=True)
            
            base_name = Path(image_path).stem
            
            # Save main result
            output_path = output_dir / f"{base_name}_analyzed.jpg"
            cv2.imwrite(str(output_path), result)
            print(f"\n✓ Saved annotated image: {output_path}")
            
            # Save processed versions
            cv2.imwrite(str(output_dir / f"{base_name}_enhanced.jpg"), 
                       processed['clahe'])
            cv2.imwrite(str(output_dir / f"{base_name}_edges.jpg"), 
                       processed['edges'])
            
            print(f"✓ Saved enhanced versions in {output_dir}/")
        
        # Display results
        cv2.imshow('Original', processed['original'])
        cv2.imshow('Enhanced (CLAHE)', processed['clahe'])
        cv2.imshow('Detected Regions', result)
        
        print("\nPress any key to continue to next image...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return result, regions


def main():
    parser = argparse.ArgumentParser(
        description='Medical Image Analysis - EDUCATIONAL USE ONLY'
    )
    parser.add_argument('images', nargs='+', help='Path(s) to medical images')
    parser.add_argument('--sensitivity', type=float, default=0.5,
                       help='Detection sensitivity (0.0-1.0, default: 0.5)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output images')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MedicalImageAnalyzer()
    
    # Process each image
    for image_path in args.images:
        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            continue
        
        analyzer.analyze_scan(image_path, save_output=not args.no_save)
    
    print("\n" + "=" * 70)
    print("REMINDER: This is an educational tool only.")
    print("For actual medical diagnosis, consult qualified healthcare professionals.")
    print("=" * 70)


if __name__ == "__main__":
    main()
