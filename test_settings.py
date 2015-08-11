#! /usr/bin/env python

import os
import imp
import argparse



def try_import_module(mod_name, mod_file):
    if not os.path.exists(mod_file):
        raise Exception('ERROR: file "%s" does not exist.' % mod_file)

    try:
        mod = imp.load_source(mod_name, mod_file)
    except:
        print 'Importing file "%s" failed. Check and fix the below errors, then try again:\n' % mod_file
        raise

    return mod
    


def main():
    parser = argparse.ArgumentParser(description='Checks settings.py.template and settings.py and prints diagnostic information')
    parser.add_argument('--template_file', type = str, default = 'settings.py.template')
    parser.add_argument('--user_file',     type = str, default = 'settings.py')
    parser.add_argument('-v', '--verbose', action = 'store_true')
    args = parser.parse_args()
    
    print 'Note: this should be run from the directory containing settings.py.template and settings.py. If settings.py does not exist, create it first:'
    print '  $ cp settings.py.template settings.py'
    print '  $ < edit settings.py >\n'
    settings_template = try_import_module('settings_template', args.template_file)
    settings_user = try_import_module('settings_user', args.user_file)

    template_keys = set([key for key in dir(settings_template) if key[:2] != '__'])
    user_keys     = set([key for key in dir(settings_user) if key[:2] != '__'])
    key_superset  = sorted(list(template_keys.union(user_keys)))

    missing_present = False
    spurious_present = False
    
    print '%-10s %-32s %-30s %-30s' % ('', 'Setting', 'from ' + args.template_file, 'from ' + args.user_file)
    print '-' * 90
    for key in key_superset:
        intemp = hasattr(settings_template, key)
        inuser = hasattr(settings_user, key)
        valtemp = getattr(settings_template, key, None)
        valuser = getattr(settings_user, key, None)
        assert intemp or inuser
        if intemp and inuser:
            if valtemp == valuser:
                descrip = 'same'
            else:
                descrip = 'changed'
        elif intemp:
            descrip = 'MISSING'
            valuser = '--'
            missing_present = True
        elif inuser:
            descrip = 'SPURIOUS'
            valtemp = '--'
            spurious_present = True
        if descrip == 'same' and not args.verbose:
            continue
        valtempstr = repr(valtemp)
        valuserstr = repr(valuser)
        if len(valtempstr) > 30 or len(valuserstr) > 30:
            print '%-10s %-32s\n  %s\n  %s' % (descrip, key, valtemp, valuser)
        else:
            print '%-10s %-32s %-30s %-30s' % (descrip, key, valtemp, valuser)

    if missing_present or spurious_present:
        print '\nDiagnosis:'
        if missing_present:
            print '* There were some MISSING settings (defined in %s but not in %s). This could indicate a problem; consider copying these settings from the template file.' % (args.template_file, args.user_file)
        if spurious_present:
            print '* There were some SPURIOUS settings (defined in %s but not in %s). This could indicate a problem.' % (args.user_file, args.template_file)



if __name__ == '__main__':
    main()
