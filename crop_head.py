from util import *
import os

SKIN_PATH = "data/skin/Skins"
OUTPUT_SKIN_PATH = "data/skin/Skins_head_88"

if __name__ == '__main__':
    os.makedirs(OUTPUT_SKIN_PATH, exist_ok=True)
    extract_head(SKIN_PATH, OUTPUT_SKIN_PATH)
    print("Done. Check on the output folder: " + OUTPUT_SKIN_PATH)