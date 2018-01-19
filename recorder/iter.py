
class Iterator(object):
	def __init__(self, info, data):
		self.info = info
		self.data = data
	def __next__(self):
		for sample in self.data:
			yield sample
	def __iter__(self):
		return next(self)

if __name__ == "__main__":
	it = Iterator("asdf", [1,2,3,4])
	next(it)
	for s in it:
		print(s)
