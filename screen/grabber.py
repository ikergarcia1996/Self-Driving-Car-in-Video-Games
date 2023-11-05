# Grabber.py https://gist.github.com/tzickel/5c2c51ddde7a8f5d87be730046612cd0
# Author: tzickel (https://gist.github.com/tzickel)
# A port of https://github.com/phoboslab/jsmpeg-vnc/blob/master/source/grabber.c to python
# License information (GPLv3) is here https://github.com/phoboslab/jsmpeg-vnc/blob/master/README.md

from ctypes import Structure, c_int, POINTER, WINFUNCTYPE, windll, WinError, sizeof
from ctypes.wintypes import (
    BOOL,
    HWND,
    RECT,
    HDC,
    HBITMAP,
    HGDIOBJ,
    DWORD,
    LONG,
    WORD,
    UINT,
    LPVOID,
)
import numpy as np

SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0


class BITMAPINFOHEADER(Structure):
    _fields_ = [
        ("biSize", DWORD),
        ("biWidth", LONG),
        ("biHeight", LONG),
        ("biPlanes", WORD),
        ("biBitCount", WORD),
        ("biCompression", DWORD),
        ("biSizeImage", DWORD),
        ("biXPelsPerMeter", LONG),
        ("biYPelsPerMeter", LONG),
        ("biClrUsed", DWORD),
        ("biClrImportant", DWORD),
    ]


def err_on_zero_or_null_check(result, func, args):
    if not result:
        raise WinError()
    return args


def quick_win_define(name, output, *args, **kwargs):
    dllname, fname = name.split(".")
    params = kwargs.get("params", None)
    if params:
        params = tuple([(x,) for x in params])
    func = (WINFUNCTYPE(output, *args))((fname, getattr(windll, dllname)), params)
    err = kwargs.get("err", err_on_zero_or_null_check)
    if err:
        func.errcheck = err
    return func


GetClientRect = quick_win_define(
    "user32.GetClientRect", BOOL, HWND, POINTER(RECT), params=(1, 2)
)
GetDC = quick_win_define("user32.GetDC", HDC, HWND)
CreateCompatibleDC = quick_win_define("gdi32.CreateCompatibleDC", HDC, HDC)
CreateCompatibleBitmap = quick_win_define(
    "gdi32.CreateCompatibleBitmap", HBITMAP, HDC, c_int, c_int
)
ReleaseDC = quick_win_define("user32.ReleaseDC", c_int, HWND, HDC)
DeleteDC = quick_win_define("gdi32.DeleteDC", BOOL, HDC)
DeleteObject = quick_win_define("gdi32.DeleteObject", BOOL, HGDIOBJ)
SelectObject = quick_win_define("gdi32.SelectObject", HGDIOBJ, HDC, HGDIOBJ)
BitBlt = quick_win_define(
    "gdi32.BitBlt", BOOL, HDC, c_int, c_int, c_int, c_int, HDC, c_int, c_int, DWORD
)
GetDIBits = quick_win_define(
    "gdi32.GetDIBits",
    c_int,
    HDC,
    HBITMAP,
    UINT,
    UINT,
    LPVOID,
    POINTER(BITMAPINFOHEADER),
    UINT,
)
GetDesktopWindow = quick_win_define("user32.GetDesktopWindow", HWND)


class Grabber(object):
    def __init__(self, window=None, with_alpha=False, bbox=None):
        window = window or GetDesktopWindow()
        self.window = window
        rect = GetClientRect(window)
        self.width = rect.right - rect.left
        self.height = rect.bottom - rect.top
        if bbox:
            bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            if not bbox[2] or not bbox[3]:
                bbox[2] = self.width - bbox[0]
                bbox[3] = self.height - bbox[1]
            self.x, self.y, self.width, self.height = bbox
        else:
            self.x = 0
            self.y = 0
        self.windowDC = GetDC(window)
        self.memoryDC = CreateCompatibleDC(self.windowDC)
        self.bitmap = CreateCompatibleBitmap(self.windowDC, self.width, self.height)
        self.bitmapInfo = BITMAPINFOHEADER()
        self.bitmapInfo.biSize = sizeof(BITMAPINFOHEADER)
        self.bitmapInfo.biPlanes = 1
        self.bitmapInfo.biBitCount = 32 if with_alpha else 24
        self.bitmapInfo.biWidth = self.width
        self.bitmapInfo.biHeight = -self.height
        self.bitmapInfo.biCompression = BI_RGB
        self.bitmapInfo.biSizeImage = 0
        self.channels = 4 if with_alpha else 3
        self.closed = False

    def __del__(self):
        self.close()

    def close(self):
        if self.closed:
            return
        ReleaseDC(self.window, self.windowDC)
        DeleteDC(self.memoryDC)
        DeleteObject(self.bitmap)
        self.closed = True

    def grab(self, output=None):
        if self.closed:
            raise ValueError("Grabber already closed")
        if output is None:
            output = np.empty((self.height, self.width, self.channels), dtype="uint8")
        else:
            if output.shape != (self.height, self.width, self.channels):
                raise ValueError("Invalid output dimentions")
        SelectObject(self.memoryDC, self.bitmap)
        BitBlt(
            self.memoryDC,
            0,
            0,
            self.width,
            self.height,
            self.windowDC,
            self.x,
            self.y,
            SRCCOPY,
        )
        GetDIBits(
            self.memoryDC,
            self.bitmap,
            0,
            self.height,
            output.ctypes.data,
            self.bitmapInfo,
            DIB_RGB_COLORS,
        )
        return output
