import sys
import cv2
import numpy as np

def getOne(infile, outfile, index):
    img = cv2.imread(infile)
    h, w, channels = img.shape
    sixteenth = w//16
    result = img[:, sixteenth*index:(index+1)*sixteenth]
    cv2.imwrite(outfile, result)

if __name__=="__main__":
    args = sys.argv[1:]
    infile = args[0]
    outfile = args[1]
    index = int(args[2])
    getOne(infile, outfile, index)