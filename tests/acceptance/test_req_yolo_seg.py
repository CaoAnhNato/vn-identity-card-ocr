import os
import sys
import unittest
import numpy as np
import albumentations as A

# Add project root to sys.path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.train_yolo26_segment import CardEdgeOcclusion

class TestYOLOSegmentationFineTuning(unittest.TestCase):
    def test_card_edge_occlusion_behavior(self):
        # Create an image (100x100 white square)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Create a card mask (50x50 card in the center: y in [25, 75], x in [25, 75])
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        
        bboxes = [[0.5, 0.5, 0.5, 0.5, 0]] # Center 0.5, size 0.5
        
        # Instantiate CardEdgeOcclusion
        dropout = A.Compose(
            [CardEdgeOcclusion(
                num_holes_range=(1, 1),
                hole_height_range=(0.1, 0.1),
                hole_width_range=(0.1, 0.1),
                fill=0,
                p=1.0
            )],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])
        )
        
        res = dropout(image=img, mask=mask, bboxes=bboxes, class_labels=["card"])
        
        # Verify image is modified (has black holes)
        self.assertTrue(np.any(res["image"] == 0), "Image pixels were not dropped.")
        
        # Verify mask is NOT modified (no holes)
        self.assertTrue(np.array_equal(res["mask"], mask), "Mask was modified.")
        
        # Verify bboxes are NOT modified/dropped
        self.assertEqual(len(res["bboxes"]), 1, "Bounding box was dropped.")
        
        # Verify that the hole is centered on the card boundary (25 or 75)
        # Find coordinates of 0s in image
        zeros = np.argwhere(res["image"][:, :, 0] == 0)
        self.assertTrue(len(zeros) > 0)
        
        # Find the center of the black hole
        cy = int(np.mean(zeros[:, 0]))
        cx = int(np.mean(zeros[:, 1]))
        
        # Center should be close to either x=25, x=75, y=25, or y=75
        dist_to_edges = [abs(cy - 25), abs(cy - 75), abs(cx - 25), abs(cx - 75)]
        min_dist = min(dist_to_edges)
        
        # It should be very close to the card edge (within 5 pixels due to rounding)
        self.assertTrue(min_dist < 5, f"Hole center ({cy}, {cx}) is too far from card edges [25, 75]. Min distance: {min_dist}")

if __name__ == "__main__":
    unittest.main()
