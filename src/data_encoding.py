import numpy as np
import data_config
import torch

def encode(text):
		vector = np.zeros(data_config.ALL_CHAR_SET_LEN * data_config.MAX_CAPTCHA, dtype=float)
		for i, c in enumerate(text):
			if i != 0:
				start_idx = i * data_config.ALL_CHAR_SET_LEN
			else:
				start_idx = 0
			
			idx = start_idx + data_config.CAPTCHA_TO_INDEX_DICT[c]
			vector[idx] = 1.0
		return vector

def decode(vec):
		char_pos = vec.nonzero()[0]
		text=[]
		for i, pos in enumerate(char_pos):
			if pos >= 62:
				pos = pos - i * 62 
			captcha = data_config.INDEX_TO_CAPTCHA_DICT[pos]
			text.append(captcha)
		return "".join(text)


if __name__ == '__main__':
		# Verify that encoding a string into a neural network’s input and decoding the network’s output back into a string works correctly.
		print(data_config.CAPTCHA_TO_INDEX_DICT)
		vec_1 = encode("Q")
		vec_2 = encode("0B")
		vec_3 = encode("pO4B")
		vecs = np.array([vec_1,vec_2,vec_3])
		vecs = torch.tensor(vecs, dtype=torch.float)
		c0 = np.argmax(vecs[1, 0:data_config.ALL_CHAR_SET_LEN].numpy())
		c1 = np.argmax(vecs[1, data_config.ALL_CHAR_SET_LEN:2*data_config.ALL_CHAR_SET_LEN].numpy())
		print(data_config.INDEX_TO_CAPTCHA_DICT[c0])
		print(data_config.INDEX_TO_CAPTCHA_DICT[c1])
