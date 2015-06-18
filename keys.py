# Define keys

#class KeyPatten(object):
#    '''Define a pattern that will be matched against a keycode.
#    
#    A KeyPattern is used to determine which key was pressed in
#    OpenCV. This process is complicated by the fact that different
#    platforms define different key codes for each key. Further, on
#    some platforms the value returned by OpenCV is different than that
#    returned by Python ord(). See the following link for more
#    information:
#    https://stackoverflow.com/questions/14494101/using-other-keys-for-the-waitkey-function-of-opencv/20577067#20577067
#    '''
#    def __init__(self, code, mask = None):
#        self.code = code
#        self.mask = mask
#        #self.mask = 0xffffffff    # 64 bits. All codes observed so far are < 2**64



# Larger masks (requiring a more specific pattern) are matched first
key_data = []
for letter in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
#for letter in 'abefghijklmnopqrstuvwxyzABEFGHIJKLMNOPQRSTUVWXYZ01456789':
    key_data.append((letter, ord(letter), 0xff))   # Match only lowest byte

key_data.extend([
    # Mac (note diff order vs Linux)
    ('up',         0xf700, 0xffff),
    ('down',       0xf701, 0xffff),
    ('left',       0xf702, 0xffff),
    ('right',      0xf703, 0xffff),

    # Ubuntu US/UK (note diff order vs Mac)
    ('left',       0xff51, 0xffff),
    ('up',         0xff52, 0xffff),
    ('right',      0xff53, 0xffff),
    ('down',       0xff54, 0xffff),

    # Ubuntu only; modified keys to not produce separate events on
    # Mac. These are included only so they be ignored without
    # producing error messages.
    ('leftshift',  0xffe1, 0xffff),
    ('rightshift', 0xffe2, 0xffff),
    ('leftctrl',   0xffe3, 0xffff),
    ('rightctrl',  0xffe4, 0xffff),
    ('esc',        27, 0xff),   # Mac
    ('enter',      13, 0xff),   # Mac
    ('enter',      10, 0xff),   # Ubuntu with UK keyboard
    ])

key_patterns = dict()
# Store key_patterns by mask in a dict of dicts
# Eventually, e.g.:
#   key_patterns[0xff][97] = 'a'
for key_datum in key_data:
    #print key_datum
    assert len(key_datum) in (2,3), 'Key information should be tuple of length 2 or 3 but it is %s' % repr(key_datum)
    if len(key_datum) == 3:
        label, key_code, mask = key_datum
    else:
        label, key_code = key_datum
        mask = 0xffffffff    # 64 bits. All codes observed so far are < 2**64
    if not mask in key_patterns:
        key_patterns[mask] = dict()
    if key_code in key_patterns[mask]:
        old_label = key_patterns[mask][code]
        if old_label != label:
            print 'Warning: key_patterns[%s][%s] old value %s being overwritten with %s' % (mask, key_code, old_label, label)
    if key_code != (key_code & mask):
        print 'Warning: key_code %s for key label %s will never trigger using mask %s' % (key_code, label, mask)
    key_patterns[mask][key_code] = label
    #if not label in key_patterns[mask]:
    #    key_patterns[mask][label] = set()
    #key_patterns[mask][label].add(code)



#class Key:
#    up=(63232, 'up')
#    right=(63235, 'right')
#    down=(63233, 'down')
#    left=(63234, 'left')
#    esc=(27, 'esc')
#    enter=(13, 'enter')
#    a =(ord('a'),'a')
#    b =(ord('b'),'b')
#    c =(ord('c'),'c')
#    d =(ord('d'),'d')
#    e =(ord('e'),'e')
#    f =(ord('f'),'f')
#    g =(ord('g'),'g')
#    h =(ord('h'),'h')
#    i =(ord('i'),'i')
#    j =(ord('j'),'j')
#    k =(ord('k'),'k')
#    l =(ord('l'),'l')
#    m =(ord('m'),'m')
#    n =(ord('n'),'n')
#    o =(ord('o'),'o')
#    p =(ord('p'),'p')
#    q =(ord('q'),'q')
#    r =(ord('r'),'r')
#    s =(ord('s'),'s')
#    t =(ord('t'),'t')
#    u =(ord('u'),'u')
#    v =(ord('v'),'v')
#    w =(ord('w'),'w')
#    x =(ord('x'),'x')
#    y =(ord('y'),'y')
#    z =(ord('z'),'z')
#    A =(ord('A'),'A')
#    B =(ord('B'),'B')
#    C =(ord('C'),'C')
#    D =(ord('D'),'D')
#    E =(ord('E'),'E')
#    F =(ord('F'),'F')
#    G =(ord('G'),'G')
#    H =(ord('H'),'H')
#    I =(ord('I'),'I')
#    J =(ord('J'),'J')
#    K =(ord('K'),'K')
#    L =(ord('L'),'L')
#    M =(ord('M'),'M')
#    N =(ord('N'),'N')
#    O =(ord('O'),'O')
#    P =(ord('P'),'P')
#    Q =(ord('Q'),'Q')
#    R =(ord('R'),'R')
#    S =(ord('S'),'S')
#    T =(ord('T'),'T')
#    U =(ord('U'),'U')
#    V =(ord('V'),'V')
#    W =(ord('W'),'W')
#    X =(ord('X'),'X')
#    Y =(ord('Y'),'Y')
#    Z =(ord('Z'),'Z')
#    n1=(ord('1'),'1')
#    n2=(ord('2'),'2')
#    n3=(ord('3'),'3')
#    n4=(ord('4'),'4')
#    n5=(ord('5'),'5')
#    n6=(ord('6'),'6')
#    n7=(ord('7'),'7')
#    n8=(ord('8'),'8')
#    n9=(ord('9'),'9')
#    n0=(ord('0'),'0')
