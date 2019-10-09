"""
.. module: uc480
   :platform: Windows, Linux
.. moduleauthor:: Daniel Dietze <daniel.dietze@berkeley.edu>

..
   This file is part of the uc480 python module.

   The uc480 python module is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The uc480 python module is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with the uc480 python module. If not, see <http://www.gnu.org/licenses/>.

   Copyright 2015 Daniel Dietze <daniel.dietze@berkeley.edu>.
"""
import time
import ctypes
import platform
_linux = (platform.system() == "Linux")
import numpy as np

from uc480_h import *

VERBOSE = False


# ##########################################################################################################
# helper functions
def ptr(x):
    return ctypes.pointer(x)


# ##########################################################################################################
# Error handling
class uc480Error(Exception):
    """uc480 exception class handling errors related to communication with the uc480 camera.
    """
    def __init__(self, error, mess, fname=""):
        """Constructor.

        :param int error: Error code.
        :param str mess: Text message associated with the error.
        :param str fname: Name of function that caused the error (optional).
        """
        self.error = error
        self.mess = mess
        self.fname = fname

    def __str__(self):
        if self.fname != "":
            return self.mess + ' in function ' + self.fname
        else:
            return self.mess


def assrt(retVal, fname=""):
    if not (retVal == IS_SUCCESS):
        raise uc480Error(retVal, "Error: uc480 function call failed! Error code = " + str(retVal), fname)
    return retVal


# ##########################################################################################################
# Camera Class
class uc480:
    """Main class for communication with one of Thorlabs' uc480 cameras.
    """
    def __init__(self):
        """Constructor.

        Takes no arguments but tries to automatically connect to the uc480 library and creates a list of all connected cameras.
        """
        # variables
        self._lib = None
        self._cam_list = []
        self._camID = None
        self._swidth = 0
        self._sheight = 0
        self._aoiwidth = 0
        self._aoiheight = 0
        self._hotPixelCorrectionOff = 0
        self._rgb = 0

        self._image = None
        self._imgID = None

        # library initialization
        # connect to uc480 DLL
        self.connect_to_library()

        # get list of cameras
        self.get_cameras()


    # wrapper around function calls to allow the user to call any library function
    def call(self, function, *args):
        """Wrapper around library function calls to allow the user to call any library function.

        :param str function: Name of the library function to be executed.
        :param mixed args: Arguments to pass to the function.
        :raises uc480Error: if function could not be properly executed.
        """
        if VERBOSE:
            print("calling %s.." % function)
        func = getattr(self._lib, function, None)
        if func is not None:
            if _linux and function in ["is_RenderBitmap", "is_GetDC", "is_ReleaseDC", "is_UpdateDisplay",
                                       "is_SetDisplayMode", "is_SetDisplayPos", "is_SetHwnd", "is_SetUpdateMode",
                                       "is_GetColorDepth", "is_SetOptimalCameraTiming", "is_DirectRenderer"]:
                print("WARNING: Function %s is not supported by this library version.." % function)
            else:
                assrt(func(*args), function)
        else:
            print("WARNING: Function %s does not exist in this library version.." % function)


    # use this version if the called function actually returns a value
    def query(self, function, *args):
        """Wrapper around library function calls to allow the user to call any library function AND query the response.

        :param str function: Name of the library function to be executed.
        :param mixed args: Arguments to pass to the function.
        :returns: Result of function call.
        :raises uc480Error: if function could not be properly executed.
        """
        if VERBOSE:
            print("querying %s.." % function)
        func = getattr(self._lib, function, None)
        if func is not None:
            if _linux and function in ["is_RenderBitmap", "is_GetDC", "is_ReleaseDC", "is_UpdateDisplay",
                                       "is_SetDisplayMode", "is_SetDisplayPos", "is_SetHwnd", "is_SetUpdateMode",
                                       "is_GetColorDepth", "is_SetOptimalCameraTiming", "is_DirectRenderer"]:
                print("WARNING: Function %s is not supported by this library version.." % function)
            else:
                return func(*args)
        else:
            print("WARNING: Function %s does not exist in this library version.." % function)
            return


    # connect to uc480 DLL library
    def connect_to_library(self, library=None):
        """Establish connection to uc480 library depending on operating system and version. If no library name is given (default), the function looks for

            - **uc480.dll** on Win32
            - **uc480_64.dll** on Win64
            - **libueye_api.so.3.82** on Linux32
            - **libueye_api64.so.3.82** on Linux64.

        :param str library: If not None, try to connect to the given library name.
        """
        print("Load uc480 library..")

        if library is None:
            if (platform.architecture()[0] == "32bit"):
                if _linux:
                    self._lib = ctypes.cdll.LoadLibrary("libueye_api.so.3.82")
                else:
                    self._lib = ctypes.cdll.LoadLibrary("uc480.dll")
            else:
                if _linux:
                    self._lib = ctypes.cdll.LoadLibrary("libueye_api64.so.3.82")
                else:
                    self._lib = ctypes.cdll.LoadLibrary("uc480_64.dll")
        else:
            self._lib = ctypes.cdll.LoadLibrary(library)

        # get version
        version = self.query("is_GetDLLVersion")
        build = version & 0xFFFF
        version = version >> 16
        minor = version & 0xFF
        version = version >> 8
        major = version & 0xFF
        print("API version %d.%d.%d" % (major, minor, build))


    # query number of connected cameras and retrieve a list with CameraIDs
    def get_cameras(self):
        """Queries the number of connected cameras and prints a list with the available CameraIDs.
        """
        nCams = ctypes.c_int()
        self.call("is_GetNumberOfCameras", ptr(nCams))
        nCams = nCams.value
        print("Found %d camera(s)" % nCams)
        if nCams > 0:
            self._cam_list = create_camera_list(nCams)
            self.call("is_GetCameraList", ptr(self._cam_list))

            for i in range(self._cam_list.dwCount):
                camera = self._cam_list.uci[i]
                print("Camera #%d: SerNo = %s, CameraID = %d, DeviceID = %d" % (i, camera.SerNo, camera.dwCameraID, camera.dwDeviceID))


    # query number of connected cameras and get back the CameraID of the camera with the Serial Number provided (if any)
    def get_cameras_id_by_SerNo(self, SerNo, return_useDevId=False):
        """Queries camera connected, and see if there is one with the Serial Number provided
        Serial Number should be provided as a string.
        """
        nCams = ctypes.c_int()
        self.call("is_GetNumberOfCameras", ptr(nCams))
        nCams = nCams.value

        if nCams > 0:
            self._cam_list = create_camera_list(nCams)
            self.call("is_GetCameraList", ptr(self._cam_list))

            for i in range(self._cam_list.dwCount):
                camera = self._cam_list.uci[i]
                if camera.SerNo.decode('ascii') == SerNo:
                    print("Found camera with given Serial Number! CameraId = %d DeviceID = %d" % (camera.dwCameraID,
                                                                                                  camera.dwDeviceID))
                    if return_useDevId:
                        return camera.dwDeviceID
                    return camera.dwCameraID

        print("No cameras found!")
        return 0


    # connect to camera with given Serial Number; If Serial Number is not found, connect to first available camera
    def connect_with_SerNo(self, SerNo):
        """Connect to the camera with the given Serial Number. If Serial Number is not found, connect to first available camera.
         When connected, sensor information is read out, image memory is reserved and some default parameters are submitted.
        Serial Number should be provided as a string.
        """
        dwCameraID = self.get_cameras_id_by_SerNo(SerNo, return_useDevId=False)
        self.connect(ID=dwCameraID, useDevID=False)


    # connect to camera with given cameraID; if cameraID = 0, connect to first available camera
    def connect(self, ID=0, useDevID=False):
        """Connect to the camera with the given cameraID. If cameraID is 0, connect to the first available camera. When connected, sensor information is read out, image memory is reserved and some default parameters are submitted.

        .. versionchanged:: 11-28-2016

            - Added `useDevID` to enable camera selection via cameraID or deviceID.

        :param int ID: ID of the camera to connect to. Set this to 0 to connect to the first available camera (default).
        :param bool useDevID: Set to True if camera should be identified by deviceID instead. By default (False), cameraID is used.
        """
        # connect to camera
        self._camID = HCAM(ID) if not useDevID else HCAM(ID | IS_USE_DEVICE_ID)
        self.call("is_InitCamera", ptr(self._camID), None)

        # get sensor info
        pInfo = SENSORINFO()
        self.call("is_GetSensorInfo", self._camID, ptr(pInfo))
        self._swidth = pInfo.nMaxWidth
        self._sheight = pInfo.nMaxHeight
        pParam = IS_SIZE_2D()
        self.call("is_AOI", self._camID, IS_AOI_IMAGE_GET_SIZE, ptr(pParam), ctypes.sizeof(pParam))
        self._aoiwidth = pParam.s32Width
        self._aoiheight = pParam.s32Height

        self._rgb = not (pInfo.nColorMode == IS_COLORMODE_MONOCHROME)
        if self._rgb:
            self.call("is_SetColorMode", self._camID, IS_CM_RGB8_PACKED)
            self._bitsperpixel = 24
        else:
            self.call("is_SetColorMode", self._camID, IS_CM_MONO8)
            self._bitsperpixel = 8
        print("Sensor: %d x %d pixels, RGB = %d, %d bits/px" % (self._swidth, self._sheight, self._rgb, self._bitsperpixel))

        dblRange = (ctypes.c_double * 3)()
        self.call("is_Exposure", self._camID, IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE, ptr(dblRange), ctypes.sizeof(dblRange))
        print("Valid exposure times: %fms to %fms in steps of %fms" % (dblRange[0], dblRange[1], dblRange[2]))
        self.expmin, self.expmax, self.expinc = dblRange

        # set default parameters
        self.call("is_ResetToDefault", self._camID)
        self.call("is_SetExternalTrigger", self._camID, IS_SET_TRIGGER_OFF)
        self.call("is_SetGainBoost", self._camID, IS_SET_GAINBOOST_OFF)
        self.call("is_SetHardwareGain", self._camID, 0, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER)
        self.call("is_Blacklevel", self._camID, IS_BLACKLEVEL_CMD_SET_MODE, ptr(ctypes.c_int(IS_AUTO_BLACKLEVEL_OFF)), ctypes.sizeof(ctypes.c_int))
        self.call("is_Exposure", self._camID, IS_EXPOSURE_CMD_SET_EXPOSURE, ptr(ctypes.c_double(self.expmin)), ctypes.sizeof(ctypes.c_double()))
        self.call("is_SetDisplayMode", self._camID, IS_SET_DM_DIB)

        self.create_buffer()


    # close connection and release memory!
    def disconnect(self):
        """Disconnect a currently connected camera.
        """
        self.call("is_ExitCamera", self._camID)
        self._image = None
        self._imgID = None


    def stop(self):
        """Same as `disconnect`.

        .. versionadded:: 01-07-2016
        """
        self.disconnect()


    # sensor info
    def get_sensor_size(self):
        """Returns the sensor size as tuple: (width, height).

        If not connected yet, it returns a zero tuple.

        .. versionadded:: 01-07-2016
        """
        return self._swith, self._sheight


    # set pixel_clock
    #WARNING:The use of the following functions will affect the frame rate:
    # set_clock(), set_AOI_size()
    # So call this again if you use the above functions
    def set_framerate(self, FPS):
        """Set the Framerate.

        :param int Framerate: New Framerate setting.
        """
        min_FPS, max_FPS, _ = self.get_framerate_limits()
        FPS = max(min_FPS, min(float(FPS), max_FPS))

        pParam = ctypes.c_double(FPS)
        result = ctypes.c_double()
        self.call("is_SetFrameRate", self._camID, pParam, ptr(result))
        # Suggested to be called after is_SetFrameRate
        pParam = ctypes.c_double()
        self.call("is_Exposure", self._camID, IS_EXPOSURE_CMD_GET_EXPOSURE, ptr(pParam), ctypes.sizeof(pParam))
        # Following also seems to help
        self.acquire()

        return result.value


    def get_framerate(self):
        """Get the Framerate.
        """
        pParam = ctypes.c_double()
        self.call("is_GetFramesPerSecond", self._camID, ptr(pParam))
        return pParam.value


    def get_framerate_limits(self):
        """Returns Framerate limits (*min, max, 1/framerate interval (i.e. time resolution in seconds) ).
        """
        pParam1 = ctypes.c_double()
        pParam2 = ctypes.c_double()
        pParam3 = ctypes.c_double()
        self.call("is_GetFrameTimeRange", self._camID, ptr(pParam1), ptr(pParam2), ptr(pParam3))
        return 1/pParam2.value, 1/pParam1.value, pParam3.value


    # set pixel_clock
    def set_clock(self, clock):
        """Set the hardware pixel clock.

        :param int clock: New pixel clock setting.
        """
        min_clock, max_clock, _ = self.get_clock_limits()
        clock = max(min_clock, min(int(clock), max_clock))

        pParam = ctypes.c_uint(clock)
        self.call("is_PixelClock", self._camID, IS_PIXELCLOCK_CMD_SET, ptr(pParam), ctypes.sizeof(pParam))


    def get_clock(self):
        """Get the hardware pixel clock.
        """

        pParam = ctypes.c_uint()
        self.call("is_PixelClock", self._camID, IS_PIXELCLOCK_CMD_GET, ptr(pParam), ctypes.sizeof(pParam))
        return pParam.value


    def get_clock_limits(self):
        """Returns clock limits (*min, max, increment*).
        """
        dblRange = (ctypes.c_uint * 3)()
        self.call("is_PixelClock", self._camID, IS_PIXELCLOCK_CMD_GET_RANGE, ptr(dblRange), ctypes.sizeof(dblRange))
        minClock, maxClock, intClok = dblRange
        return minClock, maxClock, intClok


    # set hardware gain (0..100)
    def set_gain(self, gain):
        """Set the hardware gain.

        :param int gain: New gain setting (0 - 100).
        """
        self.call("is_SetHardwareGain", self._camID, max(0, min(int(gain), 100)), IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER)


    # returns gain
    def get_gain(self):
        """Returns current gain setting.
        """
        pParam = self.query("is_SetHardwareGain", self._camID, IS_GET_MASTER_GAIN, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER)
        return pParam


    def get_gain_limits(self):
        """Returns gain limits (*min, max, increment*).
        """
        return 0, 100, 1


    # switch gain boost on/ off
    def set_gain_boost(self, onoff):
        """Switch gain boost on or off.
        """
        if onoff:
            self.call("is_SetGainBoost", self._camID, IS_SET_GAINBOOST_ON)
        else:
            self.call("is_SetGainBoost", self._camID, IS_SET_GAINBOOST_OFF)


    # returns AOI size minimum increment.
    def get_AOI_size_inc(self):
        """returns AOI size minimum increment.
        """
        pParam = IS_SIZE_2D()
        self.call("is_AOI", self._camID, IS_AOI_IMAGE_GET_SIZE_INC, ptr(pParam), ctypes.sizeof(pParam))
        return pParam.s32Width, pParam.s32Height


    # returns AOI max size
    def get_AOI_max_size(self):
        """returns AOI max size.
        """
        pParam = IS_SIZE_2D()
        self.call("is_AOI", self._camID, IS_AOI_IMAGE_GET_SIZE_MAX, ptr(pParam), ctypes.sizeof(pParam))
        return pParam.s32Width, pParam.s32Height


    # returns AOI min size
    def get_AOI_min_size(self):
        """returns AOI min size.
        """
        pParam = IS_SIZE_2D()
        self.call("is_AOI", self._camID, IS_AOI_IMAGE_GET_SIZE_MIN, ptr(pParam), ctypes.sizeof(pParam))
        return pParam.s32Width, pParam.s32Height


    # returns AOI size
    def get_AOI_size(self):
        """returns AOI size.
        """
        pParam = IS_SIZE_2D()
        self.call("is_AOI", self._camID, IS_AOI_IMAGE_GET_SIZE, ptr(pParam), ctypes.sizeof(pParam))
        self._aoiwidth = pParam.s32Width
        self._aoiheight = pParam.s32Height
        return self._aoiwidth, self._aoiheight


    # sets AOI size. This rounds the given parameters to the closest allowed values
    def set_AOI_size(self, width, height):
        """sets AOI size.
        """
        # Width and Height must be bigger than the minimum settable amount, and a multiple of the allowed increments
        minWidth, minHeight = self.get_AOI_min_size()
        maxWidth, maxHeight = self.get_AOI_max_size()
        incWidth, incHeight = self.get_AOI_size_inc()

        width = round(width / incWidth) * incWidth if (width % incWidth) is not 0 else width
        height = round(height / incHeight) * incHeight if (height % incHeight) is not 0 else height
        width = minWidth if width < minWidth else width
        height = minHeight if height < minHeight else height
        width = maxWidth if width > maxWidth else width
        height = maxHeight if height > maxHeight else height
        self._aoiwidth = width
        self._aoiheight = height

        pParam = IS_SIZE_2D(self._aoiwidth, self._aoiheight)
        self.call("is_AOI", self._camID, IS_AOI_IMAGE_SET_SIZE, ptr(pParam), ctypes.sizeof(pParam))

        _image = None
        self.create_buffer()


    # returns AOI position
    def get_AOI_max_position(self):
        """returns max AOI position.
        """
        pParam = IS_POINT_2D()
        self.call("is_AOI", self._camID, IS_AOI_IMAGE_GET_POS_MAX, ptr(pParam), ctypes.sizeof(pParam))
        return pParam.s32X, pParam.s32Y


    # returns AOI position
    def get_AOI_min_position(self):
        """returns min AOI position.
        """
        pParam = IS_POINT_2D()
        self.call("is_AOI", self._camID, IS_AOI_IMAGE_GET_POS_MIN, ptr(pParam), ctypes.sizeof(pParam))
        return pParam.s32X, pParam.s32Y


    # returns AOI position
    def get_AOI_position(self):
        """returns AOI position.
        """
        pParam = IS_POINT_2D()
        self.call("is_AOI", self._camID, IS_AOI_IMAGE_GET_POS, ptr(pParam), ctypes.sizeof(pParam))
        return pParam.s32X, pParam.s32Y


    # sets AOI position.
    def set_AOI_position(self, x, y):
        """sets AOI position.
        """
        pParam = IS_POINT_2D(x, y)
        self.call("is_AOI", self._camID, IS_AOI_IMAGE_SET_POS, ptr(pParam), ctypes.sizeof(pParam))


    # returns if AOI fast position is possible.
    def get_AOI_fast_position_possible(self):
        """returns if AOI fast position is possible.
        """
        pParam = ctypes.c_uint()
        self.call("is_AOI", self._camID, IS_AOI_IMAGE_SET_POS_FAST_SUPPORTED, ptr(pParam), ctypes.sizeof(pParam))
        return pParam.value


    def disable_hotPixelCorrection(self):
        self._hotPixelCorrectionOff = 1
        self.call("is_HotPixel", self._camID, IS_HOTPIXEL_DISABLE_CORRECTION)


    # sets AOI position in a fast way (check if supported first!).
    # For this to work, hot pixel correction should be disabled!
    def set_AOI_position_fast(self, x, y):
        """ sets AOI position in a fast way (check if supported first!).
        """
        pParam = IS_POINT_2D(x, y)
        if self._hotPixelCorrectionOff is 0:
            self.disable_hotPixelCorrection()

        self.call("is_AOI", self._camID, IS_AOI_IMAGE_SET_POS_FAST, ptr(pParam), ctypes.sizeof(pParam))


    # reset the AOI to the whole sensor size.
    def reset_AOI(self):
        """ reset the AOI to the whole sensor size.
        """
        self.set_AOI_position(0, 0)
        maxWidth, maxHeight = self.get_AOI_max_size()
        self.set_AOI_size(maxWidth, maxHeight)


    # set blacklevel compensation
    def set_blacklevel(self, blck):
        """Set blacklevel compensation on or off.
        """
        nMode = ctypes.c_int(blck)
        self.call("is_Blacklevel", self._camID, IS_BLACKLEVEL_CMD_SET_MODE, ptr(nMode), ctypes.sizeof(nMode))


    # sets exposure time in ms
    def set_exposure(self, exp):
        """Set exposure time in milliseconds.
        """
        pParam = ctypes.c_double(exp)
        self.call("is_Exposure", self._camID, IS_EXPOSURE_CMD_SET_EXPOSURE, ptr(pParam), ctypes.sizeof(pParam))
        # First image after this seems to have wrong exposure.
        # So acquire it here instead
        self.acquire()


    # returns exposure time in ms
    def get_exposure(self):
        """Returns current exposure time in milliseconds.
        """
        pParam = ctypes.c_double()
        self.call("is_Exposure", self._camID, IS_EXPOSURE_CMD_GET_EXPOSURE, ptr(pParam), ctypes.sizeof(pParam))
        return pParam.value


    def get_exposure_limits(self):
        """Returns the supported limits for the exposure time (*min, max, increment*).
        """
        dblRange = (ctypes.c_double * 3)()
        self.call("is_Exposure", self._camID, IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE, ptr(dblRange), ctypes.sizeof(dblRange))
        self.expmin, self.expmax, self.expinc = dblRange

        return self.expmin, self.expmax, self.expinc


    # create image buffers
    def create_buffer(self):
        """Create image buffer for raw data from camera.

        .. note:: This function is automatically invoked by :py:func:`connect`.
        """
        # allocate memory for raw data from camera
        if self._image:
            self.call("is_FreeImageMem", self._camID, self._image, self._imgID)
        self._image = ctypes.c_char_p()
        self._imgID = ctypes.c_int()
        self.call("is_AllocImageMem", self._camID, self._aoiwidth, self._aoiheight, self._bitsperpixel, ptr(self._image), ptr(self._imgID))
        self.call("is_SetImageMem", self._camID, self._image, self._imgID)


    # copy data from camera buffer to numpy frame buffer and return typecast to float
    def get_buffer(self):
        """Copy data from camera buffer to numpy array and return typecast to uint8.

        .. note:: This function is internally used by :py:func:`acquire`, :py:func:`acquireBinned`, and :py:func:`acquireMax` and there is normally no reason to directly call it.
        """
        # create usable numpy array for frame data
        if(self._bitsperpixel == 8):
            _framedata = np.zeros((self._aoiheight, self._aoiwidth), dtype=np.uint8)
        else:
            _framedata = np.zeros((self._aoiheight, self._aoiwidth, 3), dtype=np.uint8)

        self.call("is_CopyImageMem", self._camID, self._image, self._imgID, _framedata.ctypes.data_as(ctypes.c_char_p))
        return _framedata


    # captures N frames and returns the averaged image
    def acquire(self, N=1):
        """Synchronously captures some frames from the camera using the current settings and returns the averaged image.

        :param int N: Number of frames to acquire (> 1).
        :returns: Averaged image.
        """
        if VERBOSE:
            print("acquire %d frames" % N)
        if not self._image:
            if VERBOSE:
                print("  create buffer..")
            self.create_buffer()

        data = None
        for i in range(int(N)):
            if VERBOSE:
                print("  wait for data..")
            while self.query("is_FreezeVideo", self._camID, IS_WAIT) != IS_SUCCESS:
                time.sleep(0.1)
            if VERBOSE:
                print("  read data..")
            if data is None:
                data = self.get_buffer().astype(float)
            else:
                data = data + self.get_buffer()
        data = data / float(N)

        return data


    # captures N frames and returns the fully binned arrays
    # along x and y directions and the maximum intensity in the array
    def acquireBinned(self, N=1):
        """Record N frames from the camera using the current settings and return fully binned 1d arrays averaged over the N frames.

        :param int N: Number of images to acquire.
        :returns: - Averaged 1d array fully binned over the x-axis.
                  - Averaged 1d array fully binned over the y-axis.
                  - Maximum pixel intensity before binning, e.g. to detect over illumination.
        """
        data = self.acquire(N)
        return np.sum(data, axis=0), np.sum(data, axis=1), np.amax(data)


    # returns the column / row with the maximum intensity
    def acquireMax(self, N=1):
        """Record N frames from the camera using the current settings and return the column / row with the maximum intensity.

        :param int N: Number of images to acquire.
        :returns: - Column with maximum intensity (1d array).
                  - Row with maximum intensity (1d array).
        """
        data = self.acquire(N)
        return data[np.argmax(np.data, axis=0), :], data[:, np.argmax(np.data, axis=1)]


if __name__ == "__main__":

    import pylab as pl
    cam = uc480()
    cam.connect()
    img = cam.acquire(1)
    pl.plot(np.mean(img, axis=0))
    cam.disconnect()
    pl.show()
