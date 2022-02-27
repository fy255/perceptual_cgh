import time
from .pyEDSDK import *
from .CameraClass import *
import ctypes


class CanonController:
    def __init__(self, model=None):
        self.model = model
        self.isSDKLoaded = False
        self.b_haveCommand = False
        self.eventFired = False
        self.imageName = None
        self.imageFile = None
        self.save_folder = os.path.join(os.getcwd(),
                                        'citl')
        self.numImg = 1

    def Take_pic(self, numImg, cameraCapture_folder, folderName, imageName):
        self.numImg = numImg
        self.save_folder = os.path.join(os.getcwd(), cameraCapture_folder,
                                        folderName)
        self.imageName = imageName
        camera = IntPtr()
        isSDKLoaded = False
        capacity = EdsCapacity()
        capacity.NumberOfFreeClusters = 0x7FFFFFFF
        capacity.BytesPerSector = 0x1000
        capacity.Reset = 1
        saveTarget = EdsSaveTo.Host
        # Initialize SDK
        err = EdsInitializeSDK()
        if err == EDS_ERR_OK:
            isSDKLoaded = True

        # Get first camera
        if err == EDS_ERR_OK:
            err = self.getFirstCamera(camera)
        # Create Camera model
        EOS6DModel = CameraModel(camera=camera)
        self.model = EOS6DModel

        # Open session with camera
        err = EdsOpenSession(camera)

        # Set event handler
        handlePropertyEvent = EdsPropertyEventHandler(self.handlePropertyEvent)
        handleObjectEvent = EdsObjectEventHandler(self.handleObjectEvent)
        handleStateEvent = EdsStateEventHandler(self.handleStateEvent)
        ptr = ctypes.py_object(self)

        if err == EDS_ERR_OK:
            err = EdsSetPropertyEventHandler(camera, PropertyEvent_All, handlePropertyEvent,
                                             ptr)
        if err == EDS_ERR_OK:
            err = EdsSetObjectEventHandler(camera, ObjectEvent_All, handleObjectEvent, ptr)
        if err == EDS_ERR_OK:
            err = EdsSetCameraStateEventHandler(camera, StateEvent_All, handleStateEvent,
                                                ptr)

        err = EdsSetPropertyData(camera, PropID_SaveTo, 0, 4, saveTarget)

        err = EdsSetCapacity(camera, capacity)

        for i in range(self.numImg):
            self.imageFile = self.imageName + '_' + str(i + 1) + '.jpg'
            err = EdsSendCommand(self.model.Camera, EdsUInt32(CameraCommand_TakePicture), 0)
            while not self.eventFired:
                err = EdsGetEvent()
                time.sleep(0.3)
            self.eventFired = False
            err = EDS_ERR_OK
        # Close session with camera
        if err == EDS_ERR_OK:
            err = EdsCloseSession(self.model.Camera)

        # Release camera
        if self.model.Camera is not None:
            err = EdsRelease(self.model.Camera)

        # Terminate SDK
        # if self.isSDKLoaded:
        #     err = EdsTerminateSDK()
        err = EdsTerminateSDK()
        self.eventFired = False
        return err

    # def Take_pic(self):
    #     # / Take picture
    #     err = EdsSendCommand(self.model.Camera, EdsUInt32(CameraCommand_TakePicture), 0)
    #     # err = EdsGetEvent()
    #
    #     while not self.eventFired:
    #         err = EdsGetEvent()
    #         time.sleep(0.1)
    #
    #     self.eventFired = False

    @staticmethod
    def getFirstCamera(cameraRef):
        err = EDS_ERR_OK
        cameraList = IntPtr()
        err = EdsGetCameraList(cameraList)
        err = EdsGetChildAtIndex(cameraList, 0, cameraRef)
        return err

    def Close_device(self):
        err = EDS_ERR_OK
        # Close session with camera
        if err == EDS_ERR_OK:
            err = EdsCloseSession(self.model.Camera)

        # Release camera
        if self.model.Camera is not None:
            err = EdsRelease(self.model.Camera)

        # Terminate SDK
        if self.isSDKLoaded:
            EdsTerminateSDK()

    @staticmethod
    def handleStateEvent(inEvent, inParameter, inContext):
        return EDS_ERR_OK

    @staticmethod
    def handleObjectEvent(inEvent, inRef, content):
        controller = content
        err = EDS_ERR_OK
        if inEvent == ObjectEvent_DirItemRequestTransfer:
            downloadImage(controller, inRef, controller.imageFile)
            # Object must be released
            if inRef is not None:
                err = EdsRelease(inRef)
        controller.eventFired = True
        return err

    @staticmethod
    def handlePropertyEvent(event, property, inParam, context):
        return EDS_ERR_OK


def downloadImage(self, directoryItem, imageFile):
    err = EDS_ERR_OK
    stream = IntPtr()
    # Acquisition of the downloaded image information
    dirItemInfo = EdsDirectoryItemInfo()
    err = EdsGetDirectoryItemInfo(directoryItem, dirItemInfo)

    if err == EDS_ERR_OK:
        save_path = (os.path.join(os.getcwd(), self.save_folder) + "\\")
        dirPath = (save_path + imageFile).encode()
        err = EdsCreateFileStream(dirPath, EdsFileCreateDisposition.CreateAlways,
                                  EdsAccess.ReadWrite, stream)
        print(dirItemInfo.szFileName)
    if err == EDS_ERR_OK:
        err = EdsDownload(directoryItem, dirItemInfo.Size, stream)
    if err == EDS_ERR_OK:
        err = EdsDownloadComplete(directoryItem)
    if stream is not None:
        EdsRelease(stream)
        stream = None
    return err
