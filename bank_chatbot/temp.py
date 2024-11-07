from urllib.request import urlopen
import json

url = "https://random.dog/woof.json"


response = urlopen(url)

if response.getcode() == 200:
    data = json.loads(response.read())
    print(data)

    image_url: str = data['url']
    size = data['fileSizeBytes']
    print(image_url, size)

    if image_url.lower().endswith(".jpg") or image_url.lower().endswith("jpeg"):
        # Download the picture
        img_response = urlopen(image_url)
        if img_response.getcode() == 200:
            data = img_response.read()
            print('Image retrieved')