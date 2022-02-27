import threading
import os
import numpy as np
import inspect
import cv2
import time
from utils.MvImport.MvCameraControl_class import *


def Async_raise(tid, excType):
    tid = ctypes.c_long(tid)
    if not inspect.isclass(excType):
        excType = type(excType)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(excType))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def Stop_thread(thread):
    Async_raise(thread.ident, SystemExit)


def To_hex_str(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return hexStr


def enum_devices(connect_num=None):
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tLayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tLayerType, deviceList)
    if ret != 0:
        print("enum devices fail! ret = " + To_hex_str(ret))
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print("find no device!")
        sys.exit()
    print("Find %d devices!" % deviceList.nDeviceNum)
    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            chUserDefinedName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName:
                if 0 == per:
                    break
                chUserDefinedName = chUserDefinedName + chr(per)
            print("device model name: %s" % chUserDefinedName)
            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            chUserDefinedName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName:
                if per == 0:
                    break
                chUserDefinedName = chUserDefinedName + chr(per)
            print("device model name: %s" % chUserDefinedName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)
    if connect_num is None:
        nConnectionNum = input("please input the number of the device to connect:")
    else:
        nConnectionNum = connect_num
    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print("input error!")
        sys.exit()

    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print("input error!")
        sys.exit()
    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()
    obj_cam_operation = CamOperation(cam, deviceList, nConnectionNum)
    return obj_cam_operation


class CamOperation:

    def __init__(self, obj_cam, st_device_list, image_return=None, save_folder=None, image_name=None, n_connect_num=0,
                 b_open_device=False, b_start_grabbing=False, h_thread_handle=None, b_thread_closed=False,
                 st_frame_info=None, b_exit=False, b_save_bmp=False, b_save_jpg=False, buf_save_image=None,
                 n_save_image_size=0, n_win_gui_id=0, frame_rate=0, exposure_time=0, gain=0):
        self.obj_cam = obj_cam
        self.st_device_list = st_device_list
        self.image_return = image_return
        self.save_folder = save_folder
        self.image_name = image_name
        self.n_connect_num = n_connect_num
        self.b_open_device = b_open_device
        self.b_start_grabbing = b_start_grabbing
        self.b_thread_closed = b_thread_closed
        self.st_frame_info = st_frame_info
        self.b_exit = b_exit
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_save_image = buf_save_image
        self.h_thread_handle = h_thread_handle
        self.n_win_gui_id = n_win_gui_id
        self.n_save_image_size = n_save_image_size
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.gain = gain

    def Open_device(self):
        if not self.b_open_device:
            # ch:选择设备并创建句柄 | en:Select device and create handle
            nConnectionNum = int(self.n_connect_num)
            stDeviceList = cast(self.st_device_list.pDeviceInfo[int(nConnectionNum)],
                                POINTER(MV_CC_DEVICE_INFO)).contents
            self.obj_cam = MvCamera()
            ret = self.obj_cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                self.obj_cam.MV_CC_DestroyHandle()
                print('show error', 'create handle fail! ret = ' + To_hex_str(ret))
                return ret

            ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                print('show error', 'open device fail! ret = ' + To_hex_str(ret))
                return ret
            print("open device successfully!")
            self.b_open_device = True
            self.b_thread_closed = False

            # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                    if ret != 0:
                        print("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    print("warning: set packet size fail! ret[0x%x]" % nPacketSize)

            stBool = c_bool(False)
            ret = self.obj_cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                print("get acquisition frame rate enable fail! ret[0x%x]" % ret)

            # ch:设置触发模式为off | en:Set trigger mode as off
            ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                print("set trigger mode fail! ret[0x%x]" % ret)
            return 0

    def Start_grabbing(self):
        if not self.b_start_grabbing and self.b_open_device:
            self.b_exit = False
            ret = self.obj_cam.MV_CC_StartGrabbing()
            if ret != 0:
                print('show error', 'start grabbing fail! ret = ' + To_hex_str(ret))
                return
            self.b_start_grabbing = True
            print("start grabbing successfully!")
            try:
                self.h_thread_handle = threading.Thread(target=CamOperation.__Work_thread, args=(self, None, None))
                self.h_thread_handle.start()
                self.b_thread_closed = True

            except:
                print('show error', 'error: unable to start thread')
                self.b_start_grabbing = False

    def Stop_grabbing(self):
        time.sleep(0.5)
        if self.b_start_grabbing and self.b_open_device:
            # 退出线程
            if self.b_thread_closed:
                Stop_thread(self.h_thread_handle)
                self.b_thread_closed = False
            ret = self.obj_cam.MV_CC_StopGrabbing()
            if ret != 0:
                print('show error', 'stop grabbing fail! ret = ' + To_hex_str(ret))
                return
            print("stop grabbing successfully!")
            self.b_start_grabbing = False
            self.b_exit = True

    def Close_device(self):
        if self.b_open_device:
            # 退出线程
            if self.b_thread_closed:
                Stop_thread(self.h_thread_handle)
                self.b_thread_closed = False
            ret = self.obj_cam.MV_CC_CloseDevice()
            if ret != 0:
                print('show error', 'close device fail! ret = ' + To_hex_str(ret))
                return

        # ch:销毁句柄 | Destroy handle
        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False
        self.b_exit = True
        print("close device successfully!")

    def __Work_thread(self, pData=0, nDataSize=0):
        stOutFrame = MV_FRAME_OUT()
        img_buff = None
        buf_cache = None
        while True:
            self.b_image_return = False
            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if 0 == ret:
                if buf_cache is None:
                    buf_cache = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                # 获取到图像的时间开始节点获取到图像的时间开始节点
                self.st_frame_info = stOutFrame.stFrameInfo
                cdll.msvcrt.memcpy(byref(buf_cache), stOutFrame.pBufAddr, self.st_frame_info.nFrameLen)
                # print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                #     self.st_frame_info.nWidth, self.st_frame_info.nHeight, self.st_frame_info.nFrameNum))
                self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
                if img_buff is None:
                    img_buff = (c_ubyte * self.n_save_image_size)()

                if self.b_save_jpg:
                    self.__Save_jpg(buf_cache)  # ch:保存Jpg图片 | en:Save Jpg
                    # print("save jpg successfully!")

                if self.b_save_bmp:
                    self.__Save_Bmp(buf_cache)  # ch:保存Bmp图片 | en:Save Bmp
                    # print("save bmp successfully!")
            else:
                print("no data, nRet = " + To_hex_str(ret))
                continue
            nRet = self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
            if self.b_exit:
                if img_buff is not None:
                    del img_buff
                if buf_cache is not None:
                    del buf_cache
                break

    def __Save_jpg(self, buf_cache):
        if buf_cache == 0:
            return
        self.buf_save_image = None
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Jpeg  # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth = self.st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stParam.nHeight = self.st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stParam.nDataLen = self.st_frame_info.nFrameLen
        stParam.pData = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer = cast(byref(self.buf_save_image), POINTER(c_ubyte))
        stParam.nBufferSize = self.n_save_image_size  # ch:存储节点的大小 | en:Buffer node size
        stParam.nJpgQuality = 80  # ch:jpg编码，仅在保存Jpg图像时有效。保存BMP时SDK内忽略该参数
        return_code = self.obj_cam.MV_CC_SaveImageEx2(stParam)

        if return_code != 0:
            print('show error', 'save jpg fail! ret = ' + To_hex_str(return_code))
            self.b_save_jpg = False
            return
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            npArr = np.fromstring(img_buff, np.uint8)
            self.image_return = cv2.imdecode(npArr, flags=1)
            writeStatus = cv2.imwrite(os.path.join(self.save_folder, self.image_name), self.image_return)
            print(writeStatus)
            self.b_save_jpg = False
            # print('show info', 'save jpg success!')
        except:
            self.b_save_jpg = False
            raise Exception("get one frame failed")
        if img_buff is not None:
            del img_buff
        if self.buf_save_image is not None:
            del self.buf_save_image

    def __Save_Bmp(self, buf_cache):
        if buf_cache == 0:
            return
        self.buf_save_image = None
        # file_path = str(self.st_frame_info.nFrameNum) + ".bmp"
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Bmp  # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth = self.st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stParam.nHeight = self.st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stParam.nDataLen = self.st_frame_info.nFrameLen
        stParam.pData = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer = cast(byref(self.buf_save_image), POINTER(c_ubyte))
        stParam.nBufferSize = self.n_save_image_size  # ch:存储节点的大小 | en:Buffer node size
        return_code = self.obj_cam.MV_CC_SaveImageEx2(stParam)
        if return_code != 0:
            print('show error', 'save bmp fail! ret = ' + To_hex_str(return_code))
            self.b_save_bmp = False
            return
        # file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            npArr = np.fromstring(img_buff, np.uint8)
            self.image_return = cv2.imdecode(npArr, flags=1)
            writeStatus = cv2.imwrite(os.path.join(self.save_folder, self.image_name), self.image_return)
            # print(writeStatus)
            self.b_save_bmp = False
            # print('show info', 'save bmp success!')
        except:
            self.b_save_bmp = False
            raise Exception("get one frame failed")
        if img_buff is not None:
            del img_buff
        if self.buf_save_image is not None:
            del self.buf_save_image

    @staticmethod
    def Color_numpy(data, nWidth, nHeight):
        data_ = np.frombuffer(data, count=int(nWidth * nHeight * 3), dtype=np.uint8, offset=0)
        data_r = data_[0:nWidth * nHeight * 3:3]
        data_g = data_[1:nWidth * nHeight * 3:3]
        data_b = data_[2:nWidth * nHeight * 3:3]

        data_r_arr = data_r.reshape(nHeight, nWidth)
        data_g_arr = data_g.reshape(nHeight, nWidth)
        data_b_arr = data_b.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 3], "uint8")

        numArray[:, :, 0] = data_r_arr
        numArray[:, :, 1] = data_g_arr
        numArray[:, :, 2] = data_b_arr
        return numArray

    def Get_parameter(self):
        if self.b_open_device:
            stFloatParam_FrameRate = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_FrameRate), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_exposureTime = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_exposureTime), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_gain = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_gain), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.obj_cam.MV_CC_GetFloatValue("AcquisitionFrameRate", stFloatParam_FrameRate)
            if ret != 0:
                print('show error',
                      'get acquisition frame rate fail! ret = ' + To_hex_str(ret))
            self.frame_rate = stFloatParam_FrameRate.fCurValue
            ret = self.obj_cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam_exposureTime)
            if ret != 0:
                print('show error', 'get exposure time fail! ret = ' + To_hex_str(ret))
            self.exposure_time = stFloatParam_exposureTime.fCurValue
            ret = self.obj_cam.MV_CC_GetFloatValue("Gain", stFloatParam_gain)
            if ret != 0:
                print('show error', 'get gain fail! ret = ' + To_hex_str(ret))
            self.gain = stFloatParam_gain.fCurValue
            print('show info', 'get parameter success!')

    def jpg_save(self, num, cameraCapture_folder, folderName, imageName):
        count = 0
        time_to_sleep = num * 0.015
        while count < num:
            time.sleep(time_to_sleep)
            self.b_save_jpg = True
            self.save_folder = os.path.join(os.getcwd(),
                                            cameraCapture_folder, folderName)
            self.image_name = imageName + '_' + str(count) + '.jpg'
            count += 1
        print('captured: ', num, ' images!')

    def bmp_save(self, num, cameraCapture_folder, folderName, imageName):
        count = 0
        time_to_sleep = num * 0.02
        while count < num:
            time.sleep(time_to_sleep)
            self.b_save_bmp = True
            self.save_folder = os.path.join(os.getcwd(),
                                            cameraCapture_folder, folderName)
            self.image_name = imageName + '_' + str(count) + '.bmp'
            count += 1
        print('captured: ', num, ' images!')

    def return_image(self):
        return self.image_return

    def set_parameter(self, frameRate, exposureTime, gain):
        self.frame_rate = frameRate
        self.obj_cam.frame_rate = self.frame_rate
        self.exposure_time = exposureTime
        self.obj_cam.exposure_time = self.exposure_time
        self.gain = gain
        if self.b_open_device:
            ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime", float(exposureTime))
            if ret != 0:
                print('show error', 'set exposure time fail! ret = ' + To_hex_str(ret))
            ret = self.obj_cam.MV_CC_SetFloatValue("Gain", float(gain))
            if ret != 0:
                print('show error', 'set gain fail! ret = ' + To_hex_str(ret))

            ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frameRate))
            if ret != 0:
                print('show error', 'set acquisition frame rate fail! ret = ' + To_hex_str(ret))

            print('set parameter success!')

    def get_parameter(self):
        if self.b_open_device:
            stFloatParam_FrameRate = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_FrameRate), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_exposureTime = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_exposureTime), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_gain = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_gain), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.obj_cam.MV_CC_GetFloatValue("AcquisitionFrameRate", stFloatParam_FrameRate)
            if ret != 0:
                print('show error', 'get acquisition frame rate fail! ret = ' + To_hex_str(ret))
            self.frame_rate = stFloatParam_FrameRate.fCurValue
            ret = self.obj_cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam_exposureTime)
            if ret != 0:
                print('show error', 'get exposure time fail! ret = ' + To_hex_str(ret))
            self.exposure_time = stFloatParam_exposureTime.fCurValue
            ret = self.obj_cam.MV_CC_GetFloatValue("Gain", stFloatParam_gain)
            if ret != 0:
                print('show error', 'get gain fail! ret = ' + To_hex_str(ret))
            self.gain = stFloatParam_gain.fCurValue
            print('show info', 'get parameter success!')
