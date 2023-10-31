import xml.etree.ElementTree as ET
import sys
import os

# Script to create an annotation file from a directory of XML files.
# Annotation line format: 
# ImageFilename width,height,depth xmin,xmax,ymin,ymax xmin,xmax,ymin,ymax xmin,xmax,ymin,ymax
#
# Successive bounding boxes for multiple bounding boxes within image.
#
# Usage: python3 xmlparser.py ./crowdhuman/train

def main():
    if len(sys.argv) == 1:
        print("No input directory.")
        exit(1)
    
    directory = sys.argv[1]
    
    annotation = open("annotation.txt", 'w')
    
    for filename in os.listdir(directory):
        name, ext = os.path.splitext(filename)
        if ext == '.xml':
            xml = os.path.join(directory, filename)
            if os.path.isfile(xml):
                annotation.write(parseXML(xml) + '\n')

# Parse individual xml file
def parseXML(xmlfile):
    tree = ET.parse(xmlfile)
    
    name, _ =  os.path.splitext(xmlfile)
    annotation_xml = name + '.jpg'
    
    for node in tree.findall(".//size"):
        width, height, depth = list(node)
        dims = ','.join([width.text, height.text, depth.text])
        
        annotation_xml = annotation_xml + " " + dims
    
    for node in tree.findall(".//object/bndbox"):
        xmin, xmax, ymin,ymax = list(node)
        bbox = ','.join([xmin.text, xmax.text, ymin.text, ymax.text])
        
        annotation_xml = annotation_xml + " " + bbox
            
    return annotation_xml

if __name__ == "__main__":
    main()