import win32gui
import mss
import time
from PIL import Image
from ctypes import windll

# Make program aware of DPI scaling
user32 = windll.user32
user32.SetProcessDPIAware()

def window_exists(name):
	return win32gui.FindWindow(None, name) > 0

def get_window_size(name):
	win = win32gui.FindWindow(None, name)
	if win == 0: return None
	return win32gui.GetClientRect(win)

def get_window_bounds(name):
	win = win32gui.FindWindow(None, name)
	if win == 0: return None
	rect = win32gui.GetClientRect(win)
	origin = win32gui.ClientToScreen(win, (0,0))
	return (rect[0] + origin[0], rect[1] + origin[1], rect[2] + origin[0], rect[3] + origin[1])

def frame(name, filename = None, resize = None, **kwargs):
	with mss.mss() as sct:
		bounds = get_window_bounds(name)
		if bounds:
			img = sct.grab({
				'left': bounds[0],
				'top':  bounds[1],
				'width': bounds[2] - bounds[0],
				'height': bounds[3] - bounds[1]
			})

			img = Image.frombytes('RGB', img.size, img.rgb)
			if filename:				
				if resize:
					img = img.resize(resize, resample = Image.LANCZOS)
				img.save(filename, **kwargs)

			return img

def video(name):
	fps = 0;
	frames = 0;
	last_time = time.time()

	with mss.mss() as sct:
		while True:
			bounds = get_window_bounds(name)
			img = sct.grab({
				'left': bounds[0],
				'top':  bounds[1],
				'width': bounds[2] - bounds[0],
				'height': bounds[3] - bounds[1]
			})
