import time
import webbrowser
import numpy as np
import cv2
from PIL import ImageGrab

SCAN_DURATION = 10  # seconds
SCAN_INTERVAL = 0.5  # seconds between screenshots
opened_urls = set()

print(f"Scanning screen for QR codes for {SCAN_DURATION} seconds...")
end_time = time.time() + SCAN_DURATION

detector = cv2.QRCodeDetector()

while time.time() < end_time:
    screenshot = ImageGrab.grab()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    data, _, _ = detector.detectAndDecode(frame)

    if data:
        print(f"QR code detected: {data}")

        if data not in opened_urls:
            if data.startswith("http://") or data.startswith("https://"):
                print(f"Opening URL: {data}")
                webbrowser.open(data)
            else:
                print(f"QR content is not a URL, skipping browser open.")
            opened_urls.add(data)

    time.sleep(SCAN_INTERVAL)

print("Scan complete.")
if opened_urls:
    print(f"Opened {len(opened_urls)} unique URL(s): {', '.join(opened_urls)}")
else:
    print("No QR codes found.")
