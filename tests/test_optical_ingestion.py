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
        out = optical_ingestion.prepare_budgeted_image(self.sample_img, 512, dest_img)
        self.assertTrue(out.exists())
        with Image.open(out) as im:
            w, h = im.size
            self.assertEqual(w, 512)
            self.assertEqual(h, 288)


if __name__ == "__main__":
    unittest.main()
