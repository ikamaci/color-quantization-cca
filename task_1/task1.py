from PIL import Image
import matplotlib.pyplot as plt

im = Image.open("image01.jpeg")
plt.imshow(im)
points = plt.ginput(3, show_clicks=True)
print(points)