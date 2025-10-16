import os
from os.path import dirname

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
			'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
						
ALL_CHAR_SET = NUMBER + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)

CAPTCHA_TO_INDEX_DICT = {char: index for index, char in enumerate(ALL_CHAR_SET)}
INDEX_TO_CAPTCHA_DICT = {index: char for index, char in enumerate(ALL_CHAR_SET)}

MAX_CAPTCHA = 4

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96

ROOT_PATH = dirname(dirname(dirname(__file__)))