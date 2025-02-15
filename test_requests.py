import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
r = requests.get(url='https://p1-jj.byteimg.com/tos-cn-i-t2oaga2asx/gold-user-assets/2020/2/7/1701fbe941179b51~tplv-t2oaga2asx-jj-mark:3024:0:0:0:q75.awebp')
image = Image.open(BytesIO(r.content))

plt.imshow(image)
plt.show()
print(r.status_code)