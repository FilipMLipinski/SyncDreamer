import sys
import cv2

def transform2(inname, outname):
    img = cv2.imread(inname)
    h, w, channels = img.shape
    half = w//2
    left_part = img[:,:half]
    right_part = img[:,half:]
    result = cv2.vconcat([left_part, right_part])
    cv2.imwrite(outname, result)

def transform4(inname, outname):
    img = cv2.imread(inname)
    h, w, channels = img.shape
    quart = w//4
    part1 = img[:,:quart]
    part2 = img[:,quart:2*quart]
    part3 = img[:,quart*2:quart*3]
    part4 = img[:,quart*3:]
    result = cv2.vconcat([part1, part2, part3, part4])
    cv2.imwrite(outname, result)


if __name__ == "__main__":
    args = sys.argv[1:]
    infilename = args[0]
    outfilename = args[1]
    howmany = args[2]
    if(howmany=="2"):
        transform2(infilename, outfilename)
    elif(howmany=="4"):
        transform4(infilename, outfilename)
