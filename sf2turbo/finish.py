from PIL import Image, ImageChops, ImageStat
import glob
import os

if __name__ == "__main__":
	for filename in glob.glob(os.path.join(".", "*.png")):
		if "_mask.png" in filename:
			img = Image.open(filename).convert("LA")
			img = Image.eval(img, lambda p: 255 * (int(p != 0)))
			img.save(filename)