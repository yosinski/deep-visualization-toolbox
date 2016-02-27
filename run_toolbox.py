#! /usr/bin/env python

import os
from live_vis import LiveVis
from bindings import bindings
try:
    import settings
except:
    print '\nError importing settings.py. Check the error message below for more information.'
    print "If you haven't already, you'll want to copy one of the settings_local.template-*.py files"
    print 'to settings_local.py and edit it to point to your caffe checkout. E.g. via:'
    print
    print '  $ cp models/caffenet-yos/settings_local.template-caffenet-yos.py settings_local.py'
    print '  $ < edit settings_local.py >\n'
    raise

if not os.path.exists(settings.caffevis_caffe_root):
    raise Exception('ERROR: Set caffevis_caffe_root in settings.py first.')



def main():
    lv = LiveVis(settings)

    help_keys, _ = bindings.get_key_help('help_mode')
    quit_keys, _ = bindings.get_key_help('quit')
    print '\n\nRunning toolbox. Push %s for help or %s to quit.\n\n' % (help_keys[0], quit_keys[0])
    lv.run_loop()


    
if __name__ == '__main__':
    main()
