import threading
from MvCameraControl_class import *


def Save_jpg(cam, stOutFrame, buf_cache):
    if buf_cache is None:
        return
    buf_save_image = None
    file_path = str(stOutFrame.stFrameInfo.nFrameNum) + ".jpg"
    n_save_image_size = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 3 + 2048
    if buf_save_image is None:
        buf_save_image = (c_ubyte * n_save_image_size)()

    stParam = MV_SAVE_IMAGE_PARAM_EX()
    stParam.enImageType = MV_Image_Jpeg  # ch:需要保存的图像类型 | en:Image format to save
    stParam.enPixelType = stOutFrame.stFrameInfo.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
    stParam.nWidth = stOutFrame.stFrameInfo.nWidth  # ch:相机对应的宽 | en:Width
    stParam.nHeight = stOutFrame.stFrameInfo.nHeight  # ch:相机对应的高 | en:Height
    stParam.nDataLen = stOutFrame.stFrameInfo.nFrameLen
    stParam.pData = cast(buf_cache, POINTER(c_ubyte))
    stParam.pImageBuffer = cast(byref(buf_save_image), POINTER(c_ubyte))
    stParam.nBufferSize = n_save_image_size  # ch:存储节点的大小 | en:Buffer node size
    stParam.nJpgQuality = 80  # ch:jpg编码，仅在保存Jpg图像时有效。保存BMP时SDK内忽略该参数
    return_code = cam.MV_CC_SaveImageEx2(stParam)

    if return_code != 0:
        return
    file_open = open(file_path.encode('ascii'), 'wb+')
    img_buff = (c_ubyte * stParam.nImageLen)()
    try:
        cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
        file_open.write(img_buff)
    except:
        b_save_jpg = False
        raise Exception("get one frame failed")


# 为线程定义一个函数
def work_thread(cam, pData=0, nDataSize=0):
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
    buf_cache = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
    cdll.msvcrt.memcpy(byref(buf_cache), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
    if stOutFrame.pBufAddr is not None:
        print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
            stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
        Save_jpg(cam, stOutFrame, buf_cache)
        nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
    else:
        print("no data")
    pass


def GetCam():
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tLayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tLayerType, deviceList)
    if ret != 0:
        print("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print("find no device!")
        sys.exit()

    print("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)

    nConnectionNum = input("please input the number of the device to connect:")

    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print("input error!")
        sys.exit()

    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print("input error!")
        sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()
    return deviceList, nConnectionNum, cam


def OpenCam(deviceList, nConnectionNum, cam):
    # ch:选择设备并创建句柄 | en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    stBool = c_bool(False)
    ret = cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
    if ret != 0:
        print("get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()
    return


def OpenStreaming(cam):
    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    try:
        print("press a key to stop grabbing.")
        hThreadHandle = threading.Thread(target=work_thread, args=(cam, None, None))
        hThreadHandle.start()
    except:
        print("error: unable to start thread")

    hThreadHandle.join()


def CloseStreaming(cam):
    # ch:停止取流 | en:Stop grab image
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("stop grabbing fail! ret[0x%x]" % ret)
        sys.exit()


def CloseCam(cam):
    # ch:关闭设备 | Close device
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("close deivce fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:销毁句柄 | Destroy handle
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("destroy handle fail! ret[0x%x]" % ret)
        sys.exit()
