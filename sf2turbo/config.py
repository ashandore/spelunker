
def setup(game):
	pass

def map_keypresses(keypresses):
	keys = ['up', 'left', 'down', 'right', 'c', 'v', 'x', 'd', 'a', 's']
	return dict(zip(keys, list(keypresses)))
