# inputsHandler.py

# Authors: Iker García Ferrero and Eritz Yerga

from keyboard.game_control import ReleaseKey, PressKey


def noKey() -> None:
    """
    Release all keys
    """
    ReleaseKey(0x11)
    ReleaseKey(0x1E)
    ReleaseKey(0x1F)
    ReleaseKey(0x20)


def W() -> None:
    """
    Release all keys and push W
    """
    PressKey(0x11)
    ReleaseKey(0x1E)
    ReleaseKey(0x1F)
    ReleaseKey(0x20)


def A() -> None:
    """
    Release all keys and push A
    """
    ReleaseKey(0x11)
    PressKey(0x1E)
    ReleaseKey(0x1F)
    ReleaseKey(0x20)


def S() -> None:
    """
    Release all keys and push S
    """
    ReleaseKey(0x11)
    ReleaseKey(0x1E)
    PressKey(0x1F)
    ReleaseKey(0x20)


def D() -> None:
    """
    Release all keys and push D
    """
    ReleaseKey(0x11)
    ReleaseKey(0x1E)
    ReleaseKey(0x1F)
    PressKey(0x20)


def WA() -> None:
    """
    Release all keys and push W and A
    """
    PressKey(0x11)
    PressKey(0x1E)
    ReleaseKey(0x1F)
    ReleaseKey(0x20)


def WD() -> None:
    """
    Release all keys and push W and D
    """
    PressKey(0x11)
    ReleaseKey(0x1E)
    ReleaseKey(0x1F)
    PressKey(0x20)


def SA() -> None:
    """
    Release all keys and push S and A
    """
    ReleaseKey(0x11)
    PressKey(0x1E)
    PressKey(0x1F)
    ReleaseKey(0x20)


def SD() -> None:
    """
    Release all keys and push S and D
    """
    ReleaseKey(0x11)
    ReleaseKey(0x1E)
    PressKey(0x1F)
    PressKey(0x20)


def select_key(key: int) -> None:
    """
    Given a ket in integer format, send to windows the virtual ket push
    """
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
