"""Unit tests for face ROI annotation pipeline."""

import unittest
import numpy as np

from scripts.face_roi_annotation import (
    BBox,
    CropTransform,
    FaceDetection,
    RegionDetection,
    assign_bilateral_regions,
)


class TestCoordinateTransform(unittest.TestCase):
    """Test coordinate transformation logic."""
    
    def test_inverse_transform_point_no_padding(self):
        """Test inverse transform with no padding."""
        transform = CropTransform(
            crop_xyxy=[100, 200, 300, 400],
            scale=0.5,
            pad_x=0,
            pad_y=0,
            input_size=512
        )
        
        # Point at (50, 100) in ROI space
        # Unpad: (50, 100)
        # Unscale: (100, 200)
        # Uncrop: (200, 400)
        x, y = transform.inverse_transform_point(50, 100)
        self.assertAlmostEqual(x, 200, places=1)
        self.assertAlmostEqual(y, 400, places=1)
    
    def test_inverse_transform_point_with_padding(self):
        """Test inverse transform with padding."""
        transform = CropTransform(
            crop_xyxy=[100, 200, 300, 400],
            scale=0.5,
            pad_x=50,
            pad_y=30,
            input_size=512
        )
        
        # Point at (100, 80) in ROI space
        # Unpad: (50, 50)
        # Unscale: (100, 100)
        # Uncrop: (200, 300)
        x, y = transform.inverse_transform_point(100, 80)
        self.assertAlmostEqual(x, 200, places=1)
        self.assertAlmostEqual(y, 300, places=1)
    
    def test_inverse_transform_polygon(self):
        """Test polygon transformation."""
        transform = CropTransform(
            crop_xyxy=[0, 0, 200, 200],
            scale=1.0,
            pad_x=0,
            pad_y=0,
            input_size=512
        )
        
        polygon = [[10, 20], [30, 40], [50, 60]]
        result = transform.inverse_transform_polygon(polygon)
        
        expected = [[10, 20], [30, 40], [50, 60]]
        for (rx, ry), (ex, ey) in zip(result, expected):
            self.assertAlmostEqual(rx, ex, places=1)
            self.assertAlmostEqual(ry, ey, places=1)


class TestBilateralRegionAssignment(unittest.TestCase):
    """Test left/right region assignment logic."""
    
    def test_assign_two_eyes_by_x_position(self):
        """Test that two eyes are correctly assigned to left/right."""
        detections = {
            'Eye': [
                {'confidence': 0.9, 'bbox_xyxy': [100, 200, 120, 220], 'centroid': [110, 210], 'polygon': [], 'area': 400},
                {'confidence': 0.85, 'bbox_xyxy': [300, 200, 320, 220], 'centroid': [310, 210], 'polygon': [], 'area': 400},
            ]
        }
        face_bbox = [0, 0, 400, 400]
        
        assigned = assign_bilateral_regions(detections, face_bbox)
        
        self.assertIn('left_eye', assigned)
        self.assertIn('right_eye', assigned)
        self.assertEqual(assigned['left_eye']['centroid'][0], 110)
        self.assertEqual(assigned['right_eye']['centroid'][0], 310)
    
    def test_assign_single_eye_based_on_face_center(self):
        """Test that single eye is assigned based on face center."""
        detections = {
            'Eye': [
                {'confidence': 0.9, 'bbox_xyxy': [100, 200, 120, 220], 'centroid': [110, 210], 'polygon': [], 'area': 400},
            ]
        }
        face_bbox = [0, 0, 400, 400]  # Center at x=200
        
        assigned = assign_bilateral_regions(detections, face_bbox)
        
        # Eye at x=110 is left of center (200), so should be left_eye
        self.assertIn('left_eye', assigned)
        self.assertNotIn('right_eye', assigned)
        self.assertEqual(assigned['left_eye']['centroid'][0], 110)
    
    def test_assign_two_cheeks_by_x_position(self):
        """Test that two cheeks are correctly assigned to left/right."""
        detections = {
            'Cheek': [
                {'confidence': 0.8, 'bbox_xyxy': [50, 300, 80, 350], 'centroid': [65, 325], 'polygon': [], 'area': 1500},
                {'confidence': 0.82, 'bbox_xyxy': [320, 300, 350, 350], 'centroid': [335, 325], 'polygon': [], 'area': 1500},
            ]
        }
        face_bbox = [0, 0, 400, 400]
        
        assigned = assign_bilateral_regions(detections, face_bbox)
        
        self.assertIn('left_cheek', assigned)
        self.assertIn('right_cheek', assigned)
        self.assertEqual(assigned['left_cheek']['centroid'][0], 65)
        self.assertEqual(assigned['right_cheek']['centroid'][0], 335)
    
    def test_assign_nose_by_highest_confidence(self):
        """Test that nose is assigned by highest confidence."""
        detections = {
            'Nose': [
                {'confidence': 0.75, 'bbox_xyxy': [180, 250, 220, 300], 'centroid': [200, 275], 'polygon': [], 'area': 2000},
                {'confidence': 0.9, 'bbox_xyxy': [190, 260, 210, 290], 'centroid': [200, 275], 'polygon': [], 'area': 600},
            ]
        }
        face_bbox = [0, 0, 400, 400]
        
        assigned = assign_bilateral_regions(detections, face_bbox)
        
        self.assertIn('nose', assigned)
        self.assertEqual(assigned['nose']['confidence'], 0.9)
    
    def test_assign_forehead_by_highest_confidence(self):
        """Test that forehead is assigned by highest confidence."""
        detections = {
            'Forehead': [
                {'confidence': 0.88, 'bbox_xyxy': [100, 50, 300, 150], 'centroid': [200, 100], 'polygon': [], 'area': 20000},
                {'confidence': 0.7, 'bbox_xyxy': [120, 60, 280, 140], 'centroid': [200, 100], 'polygon': [], 'area': 12800},
            ]
        }
        face_bbox = [0, 0, 400, 400]
        
        assigned = assign_bilateral_regions(detections, face_bbox)
        
        self.assertIn('forehead', assigned)
        self.assertEqual(assigned['forehead']['confidence'], 0.88)
    
    def test_no_cross_assignment_for_bilateral_regions(self):
        """Test that left/right don't get crossed."""
        # Right eye is more to the left, left eye is more to the right (unlikely but test edge case)
        detections = {
            'Eye': [
                {'confidence': 0.9, 'bbox_xyxy': [300, 200, 320, 220], 'centroid': [310, 210], 'polygon': [], 'area': 400},
                {'confidence': 0.85, 'bbox_xyxy': [100, 200, 120, 220], 'centroid': [110, 210], 'polygon': [], 'area': 400},
            ]
        }
        face_bbox = [0, 0, 400, 400]
        
        assigned = assign_bilateral_regions(detections, face_bbox)
        
        # Should still assign by x position: smaller x is left
        self.assertEqual(assigned['left_eye']['centroid'][0], 110)
        self.assertEqual(assigned['right_eye']['centroid'][0], 310)


if __name__ == '__main__':
    unittest.main()
