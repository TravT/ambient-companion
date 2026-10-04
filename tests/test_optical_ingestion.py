#!/usr/bin/env python3
"""Unit tests for optical_ingestion.py"""

import unittest
import tempfile
from pathlib import Path
from PIL import Image
import sys

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

import optical_ingestion


class TestOpticalIngestion(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create a sample test image
        self.sample_img = self.temp_path / "test_frame.jpg"
        img = Image.new("RGB", (1920, 1080), color=(100, 150, 200))
        img.save(self.sample_img, format="JPEG")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_file_acquisition(self):
        acquired = optical_ingestion.acquire_image(str(self.sample_img))
        self.assertEqual(acquired, self.sample_img)
        self.assertTrue(acquired.exists())

    def test_file_uri_acquisition(self):
        acquired = optical_ingestion.acquire_image(f"file://{self.sample_img}")
        self.assertEqual(acquired, self.sample_img)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            optical_ingestion.acquire_image("/non/existent/path/image.jpg")

    def test_budgeted_image_scaling(self):
        dest_img = self.temp_path / "budgeted_512.jpg"
        out, meta = optical_ingestion.prepare_budgeted_image(self.sample_img, 512, dest_img)
        self.assertTrue(out.exists())
        self.assertEqual(meta["original_width"], 1920)
        self.assertEqual(meta["original_height"], 1080)
        self.assertFalse(meta["crop_applied"])
        with Image.open(out) as im:
            w, h = im.size
            self.assertEqual(w, 512)
            self.assertEqual(h, 288)

    def test_roi_crop_on_demand(self):
        dest_img = self.temp_path / "cropped_roi.jpg"
        # Crop normalized [ymin=200, xmin=300, ymax=600, xmax=700]
        crop_box = [200, 300, 600, 700]
        out, meta = optical_ingestion.prepare_budgeted_image(self.sample_img, 1024, dest_img, crop_bbox=crop_box)
        self.assertTrue(out.exists())
        self.assertTrue(meta["crop_applied"])
        self.assertIsNotNone(meta["crop_info"])
        # In 1920x1080:
        # xmin = 0.3 * 1920 = 576, xmax = 0.7 * 1920 = 1344 -> width = 768
        # ymin = 0.2 * 1080 = 216, ymax = 0.6 * 1080 = 648 -> height = 432
        # Max dim 768 < 1024, so no downsampling needed (100% native optical crop)
        with Image.open(out) as im:
            w, h = im.size
            self.assertEqual(w, 768)
            self.assertEqual(h, 432)

    def test_coordinate_remapping_from_crop(self):
        crop_box = [200, 300, 600, 700]
        _, meta = optical_ingestion.prepare_budgeted_image(self.sample_img, 1024, self.temp_path / "c.jpg", crop_bbox=crop_box)
        
        # Suppose model detects an object in the center of the cropped image: [400, 400, 600, 600]
        raw_vlm_output = '{"label": "medicine_bottle", "bbox_2d": [400, 400, 600, 600]}'
        boxes = optical_ingestion.parse_grounding_coordinates(
            raw_vlm_output,
            orig_w=meta["original_width"],
            orig_h=meta["original_height"],
            crop_info=meta["crop_info"]
        )
        self.assertEqual(len(boxes), 1)
        b = boxes[0]
        self.assertEqual(b["label"], "medicine_bottle")
        self.assertTrue(b["remapped_from_crop"])
        # Global ymin = 200 + (400/1000) * 400 = 360
        # Global xmin = 300 + (400/1000) * 400 = 460
        # Global ymax = 200 + (600/1000) * 400 = 440
        # Global xmax = 300 + (600/1000) * 400 = 540
        self.assertEqual(b["box_2d_norm"], [360, 460, 440, 540])
        # Center: [400, 500] -> center of view / center-center
        self.assertEqual(b["center_norm"], [400, 500])
        self.assertEqual(b["spatial_location"], "center of view")


if __name__ == "__main__":
    unittest.main()
