import sys
import cv2
import numpy as np

def transform(inname, outname, rowNum, colNum):
    img = cv2.imread(inname)
    h, w, channels = img.shape
    step = 16//(rowNum * colNum)
    sixteenth = w//16
    rowCounter = 0
    colCounter = 0
    total = None
    current = None
    for i in range(16):
        if(i%step==0):
            if(rowCounter == 0):
                current = img[:,i*sixteenth:(i+1)*sixteenth]
            else:
                current = cv2.hconcat([current, img[:,i*sixteenth:(i+1)*sixteenth]])
            if(rowCounter+1==colNum):
                if(colCounter == 0):
                    total = current
                else:
                    total = cv2.vconcat([total, current])
                rowCounter = 0
                colCounter += 1
            else:
                rowCounter += 1
    cv2.imwrite(outname, total)


if __name__ == "__main__":
    args = sys.argv[1:]
    infilename = args[0]
    outfilename = args[1]
    rowNum = int(args[2])
    colNum = int(args[3])
    transform(infilename, outfilename, rowNum, colNum)