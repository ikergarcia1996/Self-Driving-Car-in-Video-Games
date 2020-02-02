# inputsHandler.py

# Authors: Iker García Ferrero and Eritz Yerga

from keyboard.game_control import ReleaseKey, PressKey


def noKey():
    ReleaseKey(0x11)
    ReleaseKey(0x1E)
    ReleaseKey(0x1F)
    ReleaseKey(0x20)


def W():
    PressKey(0x11)
    ReleaseKey(0x1E)
    ReleaseKey(0x1F)
    ReleaseKey(0x20)


def A():
    ReleaseKey(0x11)
    PressKey(0x1E)
    ReleaseKey(0x1F)
    ReleaseKey(0x20)


def S():
    ReleaseKey(0x11)
    ReleaseKey(0x1E)
    PressKey(0x1F)
    ReleaseKey(0x20)


def D():
    ReleaseKey(0x11)
    ReleaseKey(0x1E)
    ReleaseKey(0x1F)
    PressKey(0x20)


def WA():
    PressKey(0x11)
    PressKey(0x1E)
    ReleaseKey(0x1F)
    ReleaseKey(0x20)


def WD():
    PressKey(0x11)
    ReleaseKey(0x1E)
    ReleaseKey(0x1F)
    PressKey(0x20)


def SA():
    ReleaseKey(0x11)
    PressKey(0x1E)
    PressKey(0x1F)
    ReleaseKey(0x20)


def SD():
    ReleaseKey(0x11)
    ReleaseKey(0x1E)
    PressKey(0x1F)
    PressKey(0x20)


def select_key(key):
    if key == 0:
        noKey()
    elif key == 1:
        A()
    elif key == 2:
        D()
    elif key == 3:
        W()
    elif key == 4:
        S()
    elif key == 5:
        WA()
    elif key == 6:
        SA()
    elif key == 7:
        WD()
    elif key == 8:
        SD()
