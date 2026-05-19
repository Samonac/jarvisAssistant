#!/usr/bin/env python3
import os
from PIL import Image
import pygetwindow as gw
import pyautogui
import datetime

# Ensure the /picture/ folder exists
if not os.path.exists("/picture/"):
    os.makedirs("/picture/")

# Take screenshot
screenshot = pyautogui.screenshot()

# Define the file name with timestamp
file_name = f'/picture/screenshot_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'

# Save the screenshot
screenshot.save(file_name)

print(f'Screenshot saved as {file_name}')