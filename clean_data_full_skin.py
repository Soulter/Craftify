from util import *

SKIN_PATH = "data/skin/Skins"

if __name__ == '__main__':
    print("Cleaning data...")
    data_clean(SKIN_PATH)
    print("Done. Check on the output folder: " + SKIN_PATH)