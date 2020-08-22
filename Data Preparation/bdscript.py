import numpy as np
import cv2
import sys
import getopt


def main(argv):
    filename = outputDir = level = typeOfChange = False
    name = ""
    try:
        opts, args = getopt.getopt(argv, "hi:o:l:t:n:", ["input=", "output=","level=", "type=" , "name="])
    except getopt.GetoptError:
        print('bdscript.py -i <input> -o <output> -l <level> -t <type>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('bdscript.py -i <inputImage> -o <outputDirectory> -l <level> -t <type> -n <name>')
            print('input: Directory and name of image to change')
            print('output: Directory to save output in')
            print('level: Degree of how much dark/bright the output should be')
            print('type:"b" is for brighten, "d" for darken, "bd"/"db" for brighten and darken')
            print('name: name to be added to image name')
            sys.exit()
        elif opt in ("-i", "--input"):
            filename = arg
        elif opt in ("-o", "--output"):
            outputDir = arg
        elif opt in ("-l", "--level"):
            level = arg
        elif opt in ("-t", "--type"):
            typeOfChange = arg
        elif opt in ("-n" , "--name"):
        	name = arg
    if filename == False or outputDir == False or level == False or typeOfChange == False:
        print('Parameters are not valid.')
        sys.exit()
    im1 = cv2.imread(filename)
    # multiply each pixel by 0.9 (makes the image darker)
    x = np.zeros(im1.shape)
    x += float(level) * 255
    if typeOfChange == 'b':
        im2 = im1 + x
        cv2.imwrite(outputDir+'/Brighter-'+ str(name) + filename,im2)
    elif typeOfChange == 'd':
        im2 = im1 - x
        cv2.imwrite(outputDir+'/Darker-'+ str(name) + filename,im2)
    elif typeOfChange == 'bd' or typeOfChange == 'db':
        im2 = im1 + x
        im3 = im1 - x
        cv2.imwrite(outputDir+'/Brighter-'+ str(name) + filename  ,im2)
        cv2.imwrite(outputDir+'/Darker-' + str(name) + filename , im3)
        

if __name__ == "__main__":
   main(sys.argv[1:])
