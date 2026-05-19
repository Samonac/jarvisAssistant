#!/usr/bin/env python3
import pyscreenshot as ImageGrab
import os

# Ensure the picture directory exists
if not os.path.exists("/picture/"):
    os.makedirs("/picture/")

# Take screenshot and save
screenshot = ImageGrab.grab()
screenshot.save("/picture/screenshot.png")