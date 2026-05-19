import pyscreenshot as ImageGrab
import os

if not os.path.exists('picture/'):
    os.makedirs('picture/')

screenshot = ImageGrab.grab()
screenshot.save('picture/screenshot.png')