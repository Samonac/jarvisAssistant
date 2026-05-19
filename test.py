import json
import random
import string
from datetime import datetime
from pathlib import Path

def generate_gmail():
    length = random.randint(6, 14)
    username = "".join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{username}@gmail.com"

def generate_password(length=12):
    characters = string.ascii_letters + string.digits + string.punctuation
    return "".join(random.choices(characters, k=length))

script_dir = Path(__file__).parent
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = script_dir / f"{timestamp}.json"

print("Starting script...")
email = generate_gmail()
password = generate_password()
print(f"Generated email: {email}")
print(f"Generated password: {password}")

data = {"email": email, "password": password}
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"JSON saved to: {output_path}")
print("Done.")