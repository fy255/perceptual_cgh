# Python wrapper for Canon camera
from enum import Enum
import os
from .pyEDSDKTypes import *

UnKnownCode = 0xffffffff


class CameraModel:
    def __init__(self, camera=None, AEMode=UnKnownCode, AFMode=UnKnownCode, DriveMode=UnKnownCode,
                 WhiteBalance=UnKnownCode, Av=UnKnownCode, Tv=UnKnownCode, Iso=UnKnownCode,
                 MeteringMode=UnKnownCode, ExposureCompensation=UnKnownCode, ImageQuality=UnKnownCode,
                 EvfMode=UnKnownCode, EvfOutputDevice=UnKnownCode, EvfDepthOfFieldPreview=UnKnownCode,
                 EvfAFMode=UnKnownCode, BatteryLabel=UnKnownCode, Zoom=UnKnownCode, FlashMode=UnKnownCode,
                 AvailableShot=0, canDownloadImage=True):

        self.SaveTo = None
        self.ImageCount = None
        self.CurrentStorage = None
        self.CurrentFolder = None
        self.ModelName = None
        self.AEModeDesc = None
        self.AutoPowerOffDesc = None
        self.MovieQualityDesc = None
        self.FlashModeDesc = None
        self.ZoomDesc = None
        self.EvfAFModeDesc = None
        self.ImageQualityDesc = None
        self.ExposureCompensationDesc = None
        self.MeteringModeDesc = None
        self.IsoDesc = None
        self.TvDesc = None
        self.AvDesc = None
        self.WhiteBalanceDesc = None
        self.DriveModeDesc = None
        self.isTypeDS = None
        self.isEvfEnable = None
        self.SizeJpegLarge = None
        self.ZoomRect = None
        self.VisibleRect = None
        self.FocusInfo = None
        self.Camera = camera
        self.AEMode = AEMode
        self.AFMode = AFMode
        self.DriveMode = DriveMode
        self.WhiteBalance = WhiteBalance
        self.Av = Av
        self.Tv = Tv
        self.Iso = Iso
        self.MeteringMode = MeteringMode
        self.ExposureCompensation = ExposureCompensation
        self.ImageQuality = ImageQuality
        self.AvailableShot = AvailableShot
        self.EvfMode = EvfMode
        self.EvfOutputDevice = EvfOutputDevice
        self.EvfDepthOfFieldPreview = EvfDepthOfFieldPreview
        self.EvfAFMode = EvfAFMode
        self.BatteryLabel = BatteryLabel
        self.Zoom = Zoom
        self.FlashMode = FlashMode
        self.canDownloadImage = canDownloadImage

    @property
    def isEvfEnable(self):
        return self._isEvfEnable

    # public EDSDKLib.EDSDK.EdsPropertyDesc DriveModeDesc  get se
    @isEvfEnable.setter
    def isEvfEnable(self, isEvfEnable):
        self._isEvfEnable = isEvfEnable

    @property
    def isTypeDS(self):
        return self._isTypeDS

    # public EDSDKLib.EDSDK.EdsPropertyDesc DriveModeDesc  get se
    @isTypeDS.setter
    def isTypeDS(self, isTypeDS):
        self._isTypeDS = isTypeDS

    @property
    def DriveModeDesc(self):
        return self._DriveModeDesc

    # public EDSDKLib.EDSDK.EdsPropertyDesc DriveModeDesc  get se
    @DriveModeDesc.setter
    def DriveModeDesc(self, DriveModeDesc):
        self._DriveModeDesc = DriveModeDesc

    @property
    def WhiteBalanceDesc(self):
        return self._WhiteBalanceDesc

    # public EDSDKLib.EDSDK.EdsPropertyDesc WhiteBalanceDesc  get se
    @WhiteBalanceDesc.setter
    def WhiteBalanceDesc(self, WhiteBalanceDesc):
        self._WhiteBalanceDesc = WhiteBalanceDesc

    @property
    def AvDesc(self):
        return self._AvDesc

    # public EDSDKLib.EDSDK.EdsPropertyDesc AvDesc  get se
    @AvDesc.setter
    def AvDesc(self, AvDesc):
        self._AvDesc = AvDesc

    @property
    def TvDesc(self):
        return self._TvDesc

    # public EDSDKLib.EDSDK.EdsPropertyDesc TvDesc  get se
    @TvDesc.setter
    def TvDesc(self, TvDesc):
        self._TvDesc = TvDesc

    @property
    def IsoDesc(self):
        return self._IsoDesc

    # public EDSDKLib.EDSDK.EdsPropertyDesc IsoDesc  get se
    @IsoDesc.setter
    def IsoDesc(self, IsoDesc):
        self._IsoDesc = IsoDesc

    @property
    def MeteringModeDesc(self):
        return self._MeteringModeDesc

    # public EDSDKLib.EDSDK.EdsPropertyDesc MeteringModeDesc  get se
    @MeteringModeDesc.setter
    def MeteringModeDesc(self, MeteringModeDesc):
        self._MeteringModeDesc = MeteringModeDesc

    @property
    def ExposureCompensationDesc(self):
        return self._ExposureCompensationDesc

    # public EDSDKLib.EDSDK.EdsPropertyDesc ExposureCompensationDesc  get se
    @ExposureCompensationDesc.setter
    def ExposureCompensationDesc(self, ExposureCompensationDesc):
        self._ExposureCompensationDesc = ExposureCompensationDesc

    @property
    def ImageQualityDesc(self):
        return self._ImageQualityDesc

    # public EDSDKLib.EDSDK.EdsPropertyDesc ImageQualityDesc  get se
    @ImageQualityDesc.setter
    def ImageQualityDesc(self, ImageQualityDesc):
        self._ImageQualityDesc = ImageQualityDesc

    @property
    def EvfAFModeDesc(self):
        return self._EvfAFModeDesc

    # public EDSDKLib.EDSDK.EdsPropertyDesc EvfAFModeDesc  get se
    @EvfAFModeDesc.setter
    def EvfAFModeDesc(self, EvfAFModeDesc):
        self._EvfAFModeDesc = EvfAFModeDesc

    @property
    def ZoomDesc(self):
        return self._ZoomDesc

    # public EDSDKLib.EDSDK.EdsPropertyDesc ZoomDesc  get se
    @ZoomDesc.setter
    def ZoomDesc(self, ZoomDesc):
        self._ZoomDesc = ZoomDesc

    @property
    def FlashModeDesc(self):
        return self._FlashModeDesc

    # public EDSDKLib.EDSDK.EdsPropertyDesc FlashModeDesc  get se
    @FlashModeDesc.setter
    def FlashModeDesc(self, FlashModeDesc):
        self._FlashModeDesc = FlashModeDesc

    class Status(Enum):
        NONE = 0,
        DOWNLOADING = 1,
        DELETEING = 2,
        CANCELING = 3

    # CHECK
    _ExecuteStatus = Status.NONE

    def SetPropertyUInt32(self, propertyID, value):
        if propertyID == PropID_AEMode:
            self.AEMode = value
            return
        if propertyID == PropID_AFMode:
            self.AFMode = value
            return
        if propertyID == PropID_DriveMode:
            self.DriveMode = value
            return
        if propertyID == PropID_Tv:
            self.Tv = value
            return
        if propertyID == PropID_Av:
            self.Av = value
            return
        if propertyID == PropID_ISOSpeed:
            self.Iso = value
            return
        if propertyID == PropID_MeteringMode:
            self.MeteringMode = value
            return
        if propertyID == PropID_ExposureCompensation:
            self.ExposureCompensation = value
            return
        if propertyID == PropID_ImageQuality:
            self.ImageQuality = value
            return
        if propertyID == PropID_Evf_Mode:
            self.EvfMode = value
            return
        if propertyID == PropID_Evf_OutputDevice:
            self.EvfOutputDevice = value
            return
        if propertyID == PropID_Evf_DepthOfFieldPreview:
            self.EvfDepthOfFieldPreview = value
            return
        if propertyID == PropID_Evf_AFMode:
            self.EvfAFMode = value
            return
        if propertyID == PropID_AvailableShots:
            self.AvailableShot = value
            return
        if propertyID == PropID_DC_Zoom:
            self.Zoom = value
            return
        if propertyID == PropID_DC_Strobe:
            self.FlashMode = value
            return
        if propertyID == PropID_SaveTo:
            self.SaveTo = value
            return

    def SetPropertyInt32(self, propertyID, value):
        if propertyID == PropID_WhiteBalance:
            self.AEMode = value
            return
        if propertyID == PropID_BatteryLevel:
            self.AFMode = value
            return

    def SetPropertyString(self, propertyID, value):
        if propertyID == PropID_ProductName:
            self.ModelName = value
            return
        if propertyID == PropID_CurrentFolder:
            self.CurrentFolder = value
            return
        if propertyID == PropID_CurrentStorage:
            self.CurrentStorage = value
            return

    def SetPropertyFocusInfo(self, propertyID, info):
        if propertyID == PropID_FocusInfo:
            self._FocusInfo = info
            return

    def SetPropertyDesc(self, propertyID, desc):
        if propertyID == PropID_AEModeSelect:
            self.AEModeDesc = desc
            return
        if propertyID == PropID_DriveMode:
            self.DriveModeDesc = desc
            return
        if propertyID == PropID_WhiteBalance:
            self.WhiteBalanceDesc = desc
            return
        if propertyID == PropID_Tv:
            self.TvDesc = desc
            return
        if propertyID == PropID_Av:
            self.AvDesc = desc
            return
        if propertyID == PropID_ISOSpeed:
            self.IsoDesc = desc
            return
        if propertyID == PropID_MeteringMode:
            self.MeteringModeDesc = desc
            return
        if propertyID == PropID_ExposureCompensation:
            self.ExposureCompensationDesc = desc
            return
        if propertyID == PropID_ImageQuality:
            self.ImageQualityDesc = desc
            return
        if propertyID == PropID_Evf_AFMode:
            self.EvfAFModeDesc = desc
            return
        if propertyID == PropID_DC_Zoom:
            self.ZoomDesc = desc
            return
        if propertyID == PropID_DC_Strobe:
            self.FlashModeDesc = desc
            return
        if propertyID == PropID_MovieParam:
            self.MovieQualityDesc = desc
            return
        if propertyID == PropID_AutoPowerOffSetting:
            self.AutoPowerOffDesc = desc
            return

    def SetPropertyRect(self, propertyID, info):
        if propertyID == PropID_Evf_ZoomRect:
            self.ZoomRect = info
            return
        if propertyID == PropID_Evf_VisibleRect:
            self.ZoomRect = info
            return


class CameraEvent:
    class Type(Enum):
        NONE = 0,
        ERROR = 1,
        DEVICE_BUSY = 2,
        DOWNLOAD_START = 3,
        DOWNLOAD_COMPLETE = 4,
        EVFDATA_CHANGED = 5,
        PROGRESS_REPORT = 6,
        PROPERTY_CHANGED = 7,
        PROPERTY_DESC_CHANGED = 8,
        DELETE_START = 9,
        DELETE_COMPLETE = 10,
        PROGRESS = 11,
        SHUT_DOWN = 12,

    InitType = Type.NONE

    def __init__(self, Type=InitType, arg=None):
        self._type = Type
        self._arg = arg
