import glob, os

def match(f):
	return not (f[5] == '-' and f[11] == '.' and f[14] == '-')

for filename in glob.iglob("*.jpg"):
	if match(filename):
		t, frame_s, frame = os.path.splitext(filename)[0].split('-')
		new = "{:05d}-{:08.2f}-{:03d}.jpg".format(int(frame),float(t),int(frame_s))
		print(filename, "->", new)
		os.rename(filename, new)

