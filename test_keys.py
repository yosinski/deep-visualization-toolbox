#! /usr/bin/env python

# Different platforms give different codes for keys. This can cause
# bindings to be messed up. To test the keymapping on your system, run
# this script.
#
# On Mac OS X 10.8, this produces:
# $ ./test_keys.py
# Click on the picture and then carefully push the following keys:
#     Press key     j:  got code 106 = j
#     Press key     k:  got code 107 = k
#     Press key     J:  got code 74 = J
#     Press key     K:  got code 75 = K
#     Press key     1:  got code 49 = 1
#     Press key     2:  got code 50 = 2
#     Press key  left:  got code 63234 = left
#     Press key right:  got code 63235 = right
#     Press key   esc:  got code 27 = esc

import sys
import cv2
import keys
img = cv2.imread('input_images/ILSVRC2012_val_00000610.jpg')    # load example image



def check_key(key_str):
    print '  Press key %5s: ' % key_str,
    sys.stdout.flush()
    key = cv2.waitKey(0)
    found = False
    for k,v in keys.Key.__dict__.iteritems():
        if '__' in k: continue     # Skip __module__, etc.
        num,st = v
        if num == key:
            print 'got code %d = %s' % (num,st)
            found = True
            break
    if not found:
        print 'code not found:', key


    
def main():
    print 'Click on the picture and then carefully push the following keys:'
    cv2.imshow('img',img)
    check_key('j')
    check_key('k')
    check_key('J')
    check_key('K')
    check_key('1')
    check_key('2')
    check_key('left')
    check_key('right')
    check_key('esc')



if __name__ == '__main__':
    main()
