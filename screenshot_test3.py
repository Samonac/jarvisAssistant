import pyautogui
import os

os.makedirs('C:/picture/', exist_ok=True)
screenshot = pyautogui.screenshot()
screenshot.save('C:/picture/screenshot.png')
