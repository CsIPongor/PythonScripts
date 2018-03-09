# -*- coding: utf-8 -*-
"""# -*- coding: utf-8 -*-
###############################################################################
This script is used to change the coordinate system used in the ics header file 
to use "video" instead of "cartesian". For more details on ics (Image Cytometry
 Standard) files: https://en.wikipedia.org/wiki/Image_Cytometry_Standard
 
The script will process ALL files in the directory of its residence with the 
*.ics extension and change occurences of "cartesian" to "video"  
###############################################################################
Created on Thu Feb 22 07:01:07 2018
@author: Nikon Center of Excellence - Institute of Experimental Medicine
@version: Python 3 (3.5)
"""

import os
import sys
import fnmatch
import fileinput



#path of image library
pathImageLib = os.path.dirname(sys.argv[0])


fileList=[]
for df in os.listdir(pathImageLib):
     
     if fnmatch.fnmatch(df, '*.ics'):
         fileList.append(df)
         with fileinput.FileInput(os.path.join(pathImageLib,df), inplace=True) as icsFile:
              for line in icsFile:
                  print(line.replace( 'cartesian','video').replace('\n', ''))
                  str(os.path.basename(__file__))


print('The following files have been processed:\n')
if len(fileList)!=0:
    for fileName in fileList: 
        print(fileName)
else:
    print('None') 
            
input('\n'+str(os.path.basename(__file__))+' has terminated!! \n Press ENTER to exit....')