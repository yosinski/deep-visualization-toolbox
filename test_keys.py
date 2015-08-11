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
from bindings import bindings
img = cv2.imread('input_images/ILSVRC2012_val_00000610.jpg')    # load example image



def check_key(key_str):
    print '  Press key %5s: ' % key_str,
    sys.stdout.flush()
    while True:
        keycode = cv2.waitKey(0)
        label, masked_vals = bindings.get_key_label_from_keycode(keycode, extra_info = True)
        if label and ('shift' in label or 'ctrl' in label):
            print '(ignoring modifier %s)' % label,
            sys.stdout.flush()
        else:
            break
    masked_vals_pp = ', '.join(['%d (%s)' % (mv, hex(mv)) for mv in masked_vals])
    if label == key_str:
        print '  %d (%s) matched %s' % (keycode, hex(keycode), label)
    elif label is not None:
        print '* %d (%s) failed, matched key %s (masked vals tried: %s)' % (keycode, hex(keycode), label, masked_vals_pp)
    else:
        print '* %d (%s) failed, no match found (masked vals tried: %s)' % (keycode, hex(keycode), masked_vals_pp)
    #print 'Got:', label
    #found = False
    #for k,v in keys.Key.__dict__.iteritems():
    #    if '__' in k: continue     # Skip __module__, etc.
    #    num,st = v
    #    if num == key:
    #        print 'got code %d = %s' % (num,st)
    #        found = True
    #        break
    #if not found:
    #    print 'code not found:', key


    
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
