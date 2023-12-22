# getkeys.py
# Citation: Box Of Hats (https://github.com/Box-Of-Hats)

try:
    import win32api as wapi
except ModuleNotFoundError:
    print("Module win32api not found. Please install it with pip install 'pip install pywin32'")
    exit()

keyList = []  # ["\b"]
for char in "WASDJL":  # "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    keyList.append(char)


def key_check() -> str:
    """
    Checks if any keys are pressed.
    """
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return "".join(set(keys))


def keys_to_id(keys: str) -> int:
    """
    Converts a  keys name to an integer.

    :param str keys: The keys name.
    :returns:
        int - The key id.
    """
    if keys == "A":
        return 1
    if keys == "D":
        return 2
    if keys == "W":
        return 3
    if keys == "S":
        return 4
    if keys == "AW" or keys == "WA":
        return 5
    if keys == "AS" or keys == "SA":
        return 6
    if keys == "DW" or keys == "WD":
        return 7
    if keys == "DS" or keys == "SD":
        return 8

    return 0


def id_to_key(key: int) -> str:
    """
    Converts a key id to a string.

    :param int key: The key id.
    :returns:
        str - The key name.
    """
    if key == 1:
        return "A"
    if key == 2:
        return "D"
    if key == 3:
        return "W"
    if key == 4:
        return "S"
    if key == 5:
        return "AW"
    if key == 6:
        return "AS"
    if key == 7:
        return "DW"
    if key == 8:
        return "DS"
    return "none"
