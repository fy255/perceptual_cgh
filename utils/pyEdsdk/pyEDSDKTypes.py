# Python wrapper for EDSDK

from ctypes import *

EDS_MAX_NAME = 256
EDS_TRANSFER_BLOCK_SIZE = 512
# Basic Types
EdsInt = c_int
EdsBool = c_bool
EdsChar = c_char
EdsInt8 = c_int8
EdsInt16 = c_short
EdsInt32 = c_long
EdsInt64 = c_longlong
EdsFloat = c_float
EdsDouble = c_double
EdsUInt32 = c_uint32
EdsPropertyEvent = EdsUInt32
EdsObjectEvent = EdsUInt32
EdsStateEvent = EdsUInt32
EdsPropertyID = EdsUInt32
EdsCameraStatusCommand = EdsUInt32
EdsError = EdsUInt32
IntPtr = c_void_p
EdsString = create_string_buffer
UInt64 = c_ulonglong
EdsBaseRef = c_void_p

# class EdsString:
#     def __init__(self, init_or_size=EDS_MAX_NAME):
#         self.init_or_size = init_or_size
#

# region Data Types


class _EdsDataType_:  # enum
    Unknown = c_uint(0)
    Bool = c_uint(1)
    String = c_uint(2)
    Int8 = c_uint(3)
    UInt8 = c_uint(6)
    Int16 = c_uint(4)
    UInt16 = c_uint(7)
    Int32 = c_uint(8)
    UInt32 = c_uint(9)
    Int64 = c_uint(10)
    UInt64 = c_uint(11)
    Float = c_uint(12)
    Double = c_uint(13)
    ByteBlock = c_uint(14)
    Rational = c_uint(20)
    Point = c_uint(21)
    Rect = c_uint(22)
    Time = c_uint(23)

    Bool_Array = c_uint(30)
    Int8_Array = c_uint(31)
    Int16_Array = c_uint(32)
    Int32_Array = c_uint(33)
    UInt8_Array = c_uint(34)
    UInt16_Array = c_uint(35)
    UInt32_Array = c_uint(36)
    Rational_Array = c_uint(37)

    FocusInfo = c_uint(101)
    PictureStyleDesc = c_uint(102)


EdsDataType = _EdsDataType_

# endregion

# region Property IDs
# Camera Setting Properties#
PropID_Unknown = 0x0000ffff
PropID_ProductName = 0x00000002
PropID_BodyIDEx = 0x00000015
PropID_OwnerName = 0x00000004
PropID_MakerName = 0x00000005
PropID_DateTime = 0x00000006
PropID_FirmwareVersion = 0x00000007
PropID_BatteryLevel = 0x00000008
PropID_CFn = 0x00000009
PropID_SaveTo = 0x0000000b
PropID_CurrentStorage = 0x0000000c
PropID_CurrentFolder = 0x0000000d
PropID_BatteryQuality = 0x00000010

# Image Properties #
PropID_ImageQuality = 0x00000100
PropID_Orientation = 0x00000102
PropID_ICCProfile = 0x00000103
PropID_FocusInfo = 0x00000104
PropID_WhiteBalance = 0x00000106
PropID_ColorTemperature = 0x00000107
PropID_WhiteBalanceShift = 0x00000108
PropID_ColorSpace = 0x0000010d
PropID_PictureStyle = 0x00000114
PropID_PictureStyleDesc = 0x00000115
PropID_PictureStyleCaption = 0x00000200

# Capture Properties #
PropID_AEMode = 0x00000400
PropID_AEModeSelect = 0x00000436
PropID_DriveMode = 0x00000401
PropID_ISOSpeed = 0x00000402
PropID_MeteringMode = 0x00000403
PropID_AFMode = 0x00000404
PropID_Av = 0x00000405
PropID_Tv = 0x00000406
PropID_ExposureCompensation = 0x00000407
PropID_FocalLength = 0x00000409
PropID_AvailableShots = 0x0000040a
PropID_Bracket = 0x0000040b
PropID_WhiteBalanceBracket = 0x0000040c
PropID_LensName = 0x0000040d
PropID_AEBracket = 0x0000040e
PropID_FEBracket = 0x0000040f
PropID_ISOBracket = 0x00000410
PropID_NoiseReduction = 0x00000411
PropID_FlashOn = 0x00000412
PropID_RedEye = 0x00000413
PropID_FlashMode = 0x00000414
PropID_LensStatus = 0x00000416

PropID_Artist = 0x00000418
PropID_Copyright = 0x00000419

# EVF Properties #
PropID_Evf_OutputDevice = 0x00000500
PropID_Evf_Mode = 0x00000501
PropID_Evf_WhiteBalance = 0x00000502
PropID_Evf_ColorTemperature = 0x00000503
PropID_Evf_DepthOfFieldPreview = 0x00000504

# EVF IMAGE DATA Properties #
PropID_Evf_Zoom = 0x00000507
PropID_Evf_ZoomPosition = 0x00000508
PropID_Evf_ImagePosition = 0x0000050B
PropID_Evf_HistogramStatus = 0x0000050C
PropID_Evf_AFMode = 0x0000050E
PropID_Evf_HistogramY = 0x00000515
PropID_Evf_HistogramR = 0x00000516
PropID_Evf_HistogramG = 0x00000517
PropID_Evf_HistogramB = 0x00000518

PropID_Evf_CoordinateSystem = 0x00000540
PropID_Evf_ZoomRect = 0x00000541

PropID_Record = 0x00000510
# Image GPS Properties  #
PropID_GPSVersionID = 0x00000800
PropID_GPSLatitudeRef = 0x00000801
PropID_GPSLatitude = 0x00000802
PropID_GPSLongitudeRef = 0x00000803
PropID_GPSLongitude = 0x00000804
PropID_GPSAltitudeRef = 0x00000805
PropID_GPSAltitude = 0x00000806
PropID_GPSTimeStamp = 0x00000807
PropID_GPSSatellites = 0x00000808
PropID_GPSStatus = 0x00000809
PropID_GPSMapDatum = 0x00000812
PropID_GPSDateStamp = 0x0000081D

# DC Properties  #
PropID_DC_Zoom = 0x00000600
PropID_DC_Strobe = 0x00000601
PropID_LensBarrelStatus = 0x00000605

PropID_TempStatus = 0x01000415
PropID_Evf_RollingPitching = 0x01000544
PropID_FixedMovie = 0x01000422
PropID_MovieParam = 0x01000423

PropID_Evf_ClickWBCoeffs = 0x01000506
PropID_Evf_VisibleRect = 0x01000546
PropID_ManualWhiteBalanceData = 0x01000204

PropID_MirrorUpSetting = 0x01000438
PropID_MirrorLockUpState = 0x01000421

PropID_UTCTime = 0x01000016
PropID_TimeZone = 0x01000017
PropID_SummerTimeSetting = 0x01000018

PropID_AutoPowerOffSetting = 0x0100045e

# endregion


# region Camera commands

# Send Commands
CameraCommand_TakePicture = 0x00000000
CameraCommand_ExtendShutDownTimer = 0x00000001
CameraCommand_BulbStart = 0x00000002
CameraCommand_BulbEnd = 0x00000003
CameraCommand_DoEvfAf = 0x00000102
CameraCommand_DriveLensEvf = 0x00000103
CameraCommand_DoClickWBEvf = 0x00000104

CameraCommand_PressShutterButton = 0x00000004
CameraCommand_DrivePowerZoom = 0x0000010d
CameraCommand_SetRemoteShootingMode = 0x0000010f


# values for enumeration 'EdsEvfAf'

class _EdsEvfAf_:  # enum
    CameraCommand_EvfAf_OFF = c_uint(0)
    CameraCommand_EvfAf_ON = c_uint(1)


EdsEvfAf = _EdsEvfAf_


# values for enumeration 'EdsShutterButton'

class _EdsShutterButton_:  # enum
    CameraCommand_EvfAf_OFF = c_int(0x00000000)
    CameraCommand_EvfAf_ON = c_int(0x00000001)
    CameraCommand_ShutterButton_OFF = c_int(0x00000000)
    CameraCommand_ShutterButton_Halfway = c_int(0x00000001)
    CameraCommand_ShutterButton_Completely = c_int(0x00000003)
    CameraCommand_ShutterButton_Halfway_NonAF = c_int(0x00010001)
    CameraCommand_ShutterButton_Completely_NonAF = c_int(0x00010003)


EdsShutterButton = _EdsShutterButton_

# endregion

# region Camera status command

# Camera StatusCommands #
CameraState_UILock = 0x00000000
CameraState_UIUnLock = 0x00000001
CameraState_EnterDirectTransfer = 0x00000002
CameraState_ExitDirectTransfer = 0x00000003

# endregion


# region  Enumeration of property value

# Stream Seek Origins

# values for enumeration 'EdsShutterButton'
_EdsSeekOrigin_ = c_int  # enum
EdsSeekOrigin = _EdsSeekOrigin_
Cur = 0
Begin = 1
End = 2


class Command:
    NONE = c_int(0)
    DOWNLOAD = c_int(1)
    TAKE_PICTURE = c_int(2)
    SET_CAMERASETTING = c_int(3)
    PRESS_COMPLETELY = c_int(4)
    PRESS_HALFWAY = c_int(5)
    PRESS_OFF = c_int(6)
    START_EVF = c_int(7)
    END_EVF = c_int(8)
    GET_PROPERTY = c_int(9)
    GET_PROPERTYDESC = c_int(10)
    DOWNLOAD_EVF = c_int(11)
    SET_AE_MODE = c_int(12)
    SET_DRIVE_MODE = c_int(13)
    SET_WHITE_BALANCE = c_int(14)
    SET_METERING_MODE = c_int(15)
    SET_EXPOSURE_COMPENSATION = c_int(16)
    SET_IMAGEQUALITY = c_int(17)
    SET_AV = c_int(18)
    SET_TV = c_int(19)
    SET_ISO_SPEED = c_int(20)
    SET_EVF_AFMODE = c_int(21)
    SET_ZOOM = c_int(22)
    SET_AF_MODE = c_int(23)
    SET_FLASH_MODE = c_int(24)
    EVF_AF_ON = c_int(25)
    EVF_AF_OFF = c_int(26)
    FOCUS_NEAR1 = c_int(27)
    FOCUS_NEAR2 = c_int(28)
    FOCUS_NEAR3 = c_int(29)
    FOCUS_FAR1 = c_int(30)
    FOCUS_FAR2 = c_int(31)
    FOCUS_FAR3 = c_int(32)
    ZOOM_FIT = c_int(33)
    ZOOM_ZOOM = c_int(34)
    POSITION_UP = c_int(35)
    POSITION_LEFT = c_int(36)
    POSITION_RIGHT = c_int(37)
    POSITION_DOWN = c_int(38)
    REMOTESHOOTING_START = c_int(39)
    REMOTESHOOTING_STOP = c_int(40)
    SHUT_DOWN = c_int(41)
    CLOSING = c_int(42)
    SET_SAVE_TO = c_int(43)
    GET_IMAGE_COUNT = c_uint(44)

class CameraEvent:
    NONE = c_int(0)
    ERROR = c_int(1)
    DEVICE_BUSY = c_int(2)
    DOWNLOAD_START = c_int(3)
    DOWNLOAD_COMPLETE = c_int(4)
    EVFDATA_CHANGED = c_int(5)
    PROGRESS_REPORT = c_int(6)
    PROPERTY_CHANGED = c_int(7)
    PROPERTY_DESC_CHANGED = c_int(8)
    DELETE_START = c_int(9)
    DELETE_COMPLETE = c_int(10)
    PROGRESS = c_int(11)
    ANGLEINFO = c_int(12)
    MOUSE_CURSOR = c_int(13)
    SHUT_DOWN = c_int(14)


# File and Propaties Access
class _EdsAccess_:  # enum
    Read = c_int(0)
    Write = c_int(1)
    ReadWrite = c_int(2)
    Error = c_int(0xFFFFFFFF)


EdsAccess = _EdsAccess_


# File Create Disposition

class _EdsFileCreateDisposition_:  # enum
    CreateNew = c_int(0)
    CreateAlways = c_int(1)
    OpenExisting = c_int(2)
    OpenAlways = c_int(3)
    TruncateExsisting = c_int(4)


EdsFileCreateDisposition = _EdsFileCreateDisposition_


# Target Image Types

class _EdsTargetImageType_:  # enum
    Unknown = c_uint(0x00000000)
    Jpeg = c_int(0x00000001)
    TIFF = c_int(0x00000007)
    TIFF16 = c_int(0x00000008)
    RGB = c_int(0x00000009)
    RGB16 = c_int(0x0000000A)


EdsTargetImageType = _EdsTargetImageType_

# Image Source
_EEdsImageSource_ = c_int  # enum
EdsImageSource = _EEdsImageSource_

FullView = 0
Thumbnail = 1
Preview = 2


# Progress Option
class _EdsProgressOption_:
    NoReport = c_uint(0)
    Done = c_uint(1)
    Periodically = c_uint(2)


EdsProgressOption = _EdsProgressOption_

# file attribute
_EdsFileAttribute_ = c_int  # enum
EdsFileAttribute = _EdsFileAttribute_
Normal = 0x00000000
ReadOnly = 0x00000001
Hidden = 0x00000002
System = 0x00000004
Archive = 0x00000020


# Save To
# _EdsSaveTo_ = c_int  # enum
# EdsSaveTo = _EdsSaveTo_
#
# Camera = 1
# Host = 2
# Both = 3

class _EdsSaveTo_:
    Camera = c_uint(1)
    Host = c_uint(2)
    Both = c_uint(3)


EdsSaveTo = _EdsSaveTo_

# Storage Types
_EdsStorageType_ = c_int  # enum
EdsStorageType = _EdsStorageType_
Non = 0
CF = 1
SD = 2

# Transfer Option
_EdsTransferOption_ = c_int  # enum
EdsTransferOption = _EdsTransferOption_
ByDirectTransfer = 1
ByRelease = 2
ToDesktop = 0x00000100

# Drive Lens

EvfDriveLens_Near1 = 0x00000001
EvfDriveLens_Near2 = 0x00000002
EvfDriveLens_Near3 = 0x00000003
EvfDriveLens_Far1 = 0x00008001
EvfDriveLens_Far2 = 0x00008002
EvfDriveLens_Far3 = 0x00008003

# Depth of Field Preview

EvfDepthOfFieldPreview_OFF = 0x00000000
EvfDepthOfFieldPreview_ON = 0x00000001

# Image Format
ImageFormat_Unknown = 0x00000000
ImageFormat_Jpeg = 0x00000001
ImageFormat_CRW = 0x00000002
ImageFormat_RAW = 0x00000004

ImageFormat_CR2 = 0x00000006

ImageSize_Large = 0
ImageSize_Middle = 1
ImageSize_Small = 2
ImageSize_Middle1 = 5
ImageSize_Middle2 = 6
ImageSize_Unknown = 1

CompressQuality_Normal = 2
CompressQuality_Fine = 3
CompressQuality_Lossless = 4
CompressQuality_SuperFine = 5
CompressQuality_Unknown = 1

# Battery level
BatteryLevel_Empty = 1
BatteryLevel_Low = 30
BatteryLevel_Half = 50
BatteryLevel_Normal = 80
BatteryLevel_AC = 0xFFFFFFFF

# White Balance
WhiteBalance_Click = 1
WhiteBalance_Auto = 0
WhiteBalance_Daylight = 1
WhiteBalance_Cloudy = 2
WhiteBalance_Tungsten = 3
WhiteBalance_Fluorescent = 4
WhiteBalance_Strobe = 5
WhiteBalance_Shade = 8
WhiteBalance_ColorTemp = 9
WhiteBalance_Manual1 = 6
WhiteBalance_Manual2 = 15
WhiteBalance_Manual3 = 16
WhiteBalance_Manual4 = 18
WhiteBalance_Manual5 = 19
WhiteBalance_PCSet1 = 10
WhiteBalance_PCSet2 = 11
WhiteBalance_PCSet3 = 12
WhiteBalance_PCSet4 = 20
WhiteBalance_PCSet5 = 21
WhiteBalance_AwbWhite = 23

# Color Space

ColorSpace_sRGB = 1
ColorSpace_AdobeRGB = 2
ColorSpace_Unknown = 0xffffffff

# PictureStyle
PictureStyle_Standard = 0x0081
PictureStyle_Portrait = 0x0082
PictureStyle_Landscape = 0x0083
PictureStyle_Neutral = 0x0084
PictureStyle_Faithful = 0x0085
PictureStyle_Monochrome = 0x0086
PictureStyle_Auto = 0x0087
PictureStyle_FineDetail = 0x0088
PictureStyle_User1 = 0x0021
PictureStyle_User2 = 0x0022
PictureStyle_User3 = 0x0023
PictureStyle_PC1 = 0x0041
PictureStyle_PC2 = 0x0042
PictureStyle_PC3 = 0x0043

# AE Mode
AEMode_Program = 0
AEMode_Tv = 1
AEMode_Av = 2
AEMode_Mamual = 3
AEMode_Bulb = 4
AEMode_A_DEP = 5
AEMode_DEP = 6
AEMode_Custom = 7
AEMode_Lock = 8
AEMode_Green = 9
AEMode_NigntPortrait = 10
AEMode_Sports = 11
AEMode_Portrait = 12
AEMode_Landscape = 13
AEMode_Closeup = 14
AEMode_FlashOff = 15
AEMode_CreativeAuto = 19
AEMode_Movie = 20
AEMode_PhotoInMovie = 21
AEMode_SceneIntelligentAuto = 22
AEMode_SCN = 25
AEMode_HandheldNightScenes = 23
AEMode_Hdr_BacklightControl = 24
AEMode_Children = 26
AEMode_Food = 27
AEMode_CandlelightPortraits = 28
AEMode_CreativeFilter = 29
AEMode_RoughMonoChrome = 30
AEMode_SoftFocus = 31
AEMode_ToyCamera = 32
AEMode_Fisheye = 33
AEMode_WaterColor = 34
AEMode_Miniature = 35
AEMode_Hdr_Standard = 36
AEMode_Hdr_Vivid = 37
AEMode_Hdr_Bold = 38
AEMode_Hdr_Embossed = 39
AEMode_Movie_Fantasy = 40
AEMode_Movie_Old = 41
AEMode_Movie_Memory = 42
AEMode_Movie_DirectMono = 43
AEMode_Movie_Mini = 44
AEMode_Panning = 45
AEMode_GroupPhoto = 46

AEMode_SelfPortrait = 50
AEMode_PlusMovieAuto = 51
AEMode_SmoothSkin = 52
AEMode_Panorama = 53
AEMode_Silent = 54
AEMode_Flexible = 55
AEMode_OilPainting = 56
AEMode_Fireworks = 57
AEMode_StarPortrait = 58
AEMode_StarNightscape = 59
AEMode_StarTrails = 60
AEMode_StarTimelapseMovie = 61
AEMode_BackgroundBlur = 62
AEMode_Unknown = 0xffffffff

# Bracket
Bracket_AEB = 0x01
Bracket_ISOB = 0x02
Bracket_WBB = 0x04
Bracket_FEB = 0x08
Bracket_Unknown = 0xffffffff

# EVF Output Device[Flag]
EvfOutputDevice_TFT = 1
EvfOutputDevice_PC = 2

# EVF Zoom
EvfZoom_Fit = 1
EvfZoom_x5 = 5
EvfZoom_x10 = 10

_EdsEvfAFMode_ = c_int  # enum
EdsEvfAFMode = _EdsEvfAFMode_
Evf_AFMode_Quick = 0
Evf_AFMode_Live = 1
Evf_AFMode_LiveFace = 2
Evf_AFMode_LiveMulti = 3
Evf_AFMode_LiveZone = 4
Evf_AFMode_LiveCatchAF = 9
Evf_AFMode_LiveSpotAF = 10

# Strobo Mode
_EdsStroboMode_ = c_int  # enum
EdsStroboMode = _EdsStroboMode_
kEdsStroboModeInternal = 0
kEdsStroboModeExternalETTL = 1
kEdsStroboModeExternalATTL = 2
kEdsStroboModeExternalTTL = 3
kEdsStroboModeExternalAuto = 4
kEdsStroboModeExternalManual = 5
kEdsStroboModeManual = 6

_EdsETTL2Mode_ = c_int  # enum
EdsETTL2Mode = _EdsETTL2Mode_
kEdsETTL2ModeEvaluative = 0
kEdsETTL2ModeAverage = 1

# DC Strobe
_DcStrobe_ = c_int  # enum
DcStrobe = _DcStrobe_
DcStrobeAuto = 0
DcStrobeOn = 1
DcStrobeSlowsynchro = 2
DcStrobeOff = 3

# DC LensBarrel State
_DcLensBarrelState_ = c_int  # enum
DcLensBarrelState = _DcLensBarrelState_
DcLensBarrelStateInner = 0
DcLensBarrelStateOuter = 1

# DC Remote Shooting Mode
_DcRemoteShootingMode_ = c_int  # enum
DcRemoteShootingMode = _DcRemoteShootingMode_
DcRemoteShootingModeStop = 0
DcRemoteShootingModeStart = 1

_ImageQuality_ = c_int  # enum
ImageQuality = _ImageQuality_

# Jpeg Only #
EdsImageQuality_LJ = 0x0010ff0f  # Jpeg Large #
EdsImageQuality_M1J = 0x0510ff0f  # Jpeg Middle1 #
EdsImageQuality_M2J = 0x0610ff0f  # Jpeg Middle2 #
EdsImageQuality_SJ = 0x0210ff0f  # Jpeg Small #
EdsImageQuality_LJF = 0x0013ff0f  # Jpeg Large Fine #
EdsImageQuality_LJN = 0x0012ff0f  # Jpeg Large Normal #
EdsImageQuality_MJF = 0x0113ff0f  # Jpeg Middle Fine #
EdsImageQuality_MJN = 0x0112ff0f  # Jpeg Middle Normal #
EdsImageQuality_SJF = 0x0213ff0f  # Jpeg Small Fine #
EdsImageQuality_SJN = 0x0212ff0f  # Jpeg Small Normal #
EdsImageQuality_S1JF = 0x0E13ff0f  # Jpeg Small1 Fine #
EdsImageQuality_S1JN = 0x0E12ff0f  # Jpeg Small1 Normal #
EdsImageQuality_S2JF = 0x0F13ff0f  # Jpeg Small2 #
EdsImageQuality_S3JF = 0x1013ff0f  # Jpeg Small3 #

# RAW + Jpeg #
EdsImageQuality_LR = 0x0064ff0f  # RAW #
EdsImageQuality_LRLJF = 0x00640013  # RAW + Jpeg Large Fine #
EdsImageQuality_LRLJN = 0x00640012  # RAW + Jpeg Large Normal #
EdsImageQuality_LRMJF = 0x00640113  # RAW + Jpeg Middle Fine #
EdsImageQuality_LRMJN = 0x00640112  # RAW + Jpeg Middle Normal #
EdsImageQuality_LRSJF = 0x00640213  # RAW + Jpeg Small Fine #
EdsImageQuality_LRSJN = 0x00640212  # RAW + Jpeg Small Normal #
EdsImageQuality_LRS1JF = 0x00640E13  # RAW + Jpeg Small1 Fine #
EdsImageQuality_LRS1JN = 0x00640E12  # RAW + Jpeg Small1 Normal #
EdsImageQuality_LRS2JF = 0x00640F13  # RAW + Jpeg Small2 #
EdsImageQuality_LRS3JF = 0x00641013  # RAW + Jpeg Small3 #

EdsImageQuality_LRLJ = 0x00640010  # RAW + Jpeg Large #
EdsImageQuality_LRM1J = 0x00640510  # RAW + Jpeg Middle1 #
EdsImageQuality_LRM2J = 0x00640610  # RAW + Jpeg Middle2 #
EdsImageQuality_LRSJ = 0x00640210  # RAW + Jpeg Small #

# MRAW(SRAW1) + Jpeg #
EdsImageQuality_MR = 0x0164ff0f  # MRAW(SRAW1) #
EdsImageQuality_MRLJF = 0x01640013  # MRAW(SRAW1) + Jpeg Large Fine #
EdsImageQuality_MRLJN = 0x01640012  # MRAW(SRAW1) + Jpeg Large Normal #
EdsImageQuality_MRMJF = 0x01640113  # MRAW(SRAW1) + Jpeg Middle Fine #
EdsImageQuality_MRMJN = 0x01640112  # MRAW(SRAW1) + Jpeg Middle Normal #
EdsImageQuality_MRSJF = 0x01640213  # MRAW(SRAW1) + Jpeg Small Fine #
EdsImageQuality_MRSJN = 0x01640212  # MRAW(SRAW1) + Jpeg Small Normal #
EdsImageQuality_MRS1JF = 0x01640E13  # MRAW(SRAW1) + Jpeg Small1 Fine #
EdsImageQuality_MRS1JN = 0x01640E12  # MRAW(SRAW1) + Jpeg Small1 Normal #
EdsImageQuality_MRS2JF = 0x01640F13  # MRAW(SRAW1) + Jpeg Small2 #
EdsImageQuality_MRS3JF = 0x01641013  # MRAW(SRAW1) + Jpeg Small3 #

EdsImageQuality_MRLJ = 0x01640010  # MRAW(SRAW1) + Jpeg Large #
EdsImageQuality_MRM1J = 0x01640510  # MRAW(SRAW1) + Jpeg Middle1 #
EdsImageQuality_MRM2J = 0x01640610  # MRAW(SRAW1) + Jpeg Middle2 #
EdsImageQuality_MRSJ = 0x01640210  # MRAW(SRAW1) + Jpeg Small #

# SRAW(SRAW2) + Jpeg #
EdsImageQuality_SR = 0x0264ff0f  # SRAW(SRAW2) #
EdsImageQuality_SRLJF = 0x02640013  # SRAW(SRAW2) + Jpeg Large Fine #
EdsImageQuality_SRLJN = 0x02640012  # SRAW(SRAW2) + Jpeg Large Normal #
EdsImageQuality_SRMJF = 0x02640113  # SRAW(SRAW2) + Jpeg Middle Fine #
EdsImageQuality_SRMJN = 0x02640112  # SRAW(SRAW2) + Jpeg Middle Normal #
EdsImageQuality_SRSJF = 0x02640213  # SRAW(SRAW2) + Jpeg Small Fine #
EdsImageQuality_SRSJN = 0x02640212  # SRAW(SRAW2) + Jpeg Small Normal #
EdsImageQuality_SRS1JF = 0x02640E13  # SRAW(SRAW2) + Jpeg Small1 Fine #
EdsImageQuality_SRS1JN = 0x02640E12  # SRAW(SRAW2) + Jpeg Small1 Normal #
EdsImageQuality_SRS2JF = 0x02640F13  # SRAW(SRAW2) + Jpeg Small2 #
EdsImageQuality_SRS3JF = 0x02641013  # SRAW(SRAW2) + Jpeg Small3 #

EdsImageQuality_SRLJ = 0x02640010  # SRAW(SRAW2) + Jpeg Large #
EdsImageQuality_SRM1J = 0x02640510  # SRAW(SRAW2) + Jpeg Middle1 #
EdsImageQuality_SRM2J = 0x02640610  # SRAW(SRAW2) + Jpeg Middle2 #
EdsImageQuality_SRSJ = 0x02640210  # SRAW(SRAW2) + Jpeg Small #

# CRAW + Jpeg #
EdsImageQuality_CR = 0x0063ff0f  # CRAW #
EdsImageQuality_CRLJF = 0x00630013  # CRAW + Jpeg Large Fine #
EdsImageQuality_CRMJF = 0x00630113  # CRAW + Jpeg Middle Fine  #
EdsImageQuality_CRM1JF = 0x00630513  # CRAW + Jpeg Middle1 Fine  #
EdsImageQuality_CRM2JF = 0x00630613  # CRAW + Jpeg Middle2 Fine  #
EdsImageQuality_CRSJF = 0x00630213  # CRAW + Jpeg Small Fine  #
EdsImageQuality_CRS1JF = 0x00630E13  # CRAW + Jpeg Small1 Fine  #
EdsImageQuality_CRS2JF = 0x00630F13  # CRAW + Jpeg Small2 Fine  #
EdsImageQuality_CRS3JF = 0x00631013  # CRAW + Jpeg Small3 Fine  #
EdsImageQuality_CRLJN = 0x00630012  # CRAW + Jpeg Large Normal #
EdsImageQuality_CRMJN = 0x00630112  # CRAW + Jpeg Middle Normal #
EdsImageQuality_CRM1JN = 0x00630512  # CRAW + Jpeg Middle1 Normal #
EdsImageQuality_CRM2JN = 0x00630612  # CRAW + Jpeg Middle2 Normal #
EdsImageQuality_CRSJN = 0x00630212  # CRAW + Jpeg Small Normal #
EdsImageQuality_CRS1JN = 0x00630E12  # CRAW + Jpeg Small1 Normal #

EdsImageQuality_CRLJ = 0x00630010  # CRAW + Jpeg Large #
EdsImageQuality_CRM1J = 0x00630510  # CRAW + Jpeg Middle1 #
EdsImageQuality_CRM2J = 0x00630610  # CRAW + Jpeg Middle2 #
EdsImageQuality_CRSJ = 0x00630210  # CRAW + Jpeg Small #

EdsImageQuality_Unknown = 0xffffffff
# endregion


# region Event IDs
# Camera Events
# Property Event
PropertyEvent_All = 0x00000100
PropertyEvent_PropertyChanged = 0x00000101
PropertyEvent_PropertyDescChanged = 0x00000102
ObjectEvent_All = 0x00000200
ObjectEvent_VolumeInfoChanged = 0x00000201
ObjectEvent_VolumeUpdateItems = 0x00000202
ObjectEvent_FolderUpdateItems = 0x00000203
ObjectEvent_DirItemCreated = 0x00000204
ObjectEvent_DirItemRemoved = 0x00000205
ObjectEvent_DirItemInfoChanged = 0x00000206
ObjectEvent_DirItemContentChanged = 0x00000207
ObjectEvent_DirItemRequestTransfer = 0x00000208
ObjectEvent_DirItemRequestTransferDT = 0x00000209
ObjectEvent_DirItemCancelTransferDT = 0x0000020a
ObjectEvent_VolumeAdded = 0x0000020c
ObjectEvent_VolumeRemoved = 0x0000020d

# Notifies all state events
StateEvent_All = 0x00000300
StateEvent_Shutdown = 0x00000301
StateEvent_JobStatusChanged = 0x00000302
StateEvent_WillSoonShutDown = 0x00000303
StateEvent_ShutDownTimerUpdate = 0x00000304
StateEvent_CaptureError = 0x00000305
StateEvent_InternalError = 0x00000306
StateEvent_AfResult = 0x00000309


# endregion

# region Proto type defenition of EDSDK API
# TODO 859 LINE NUM


# TODO: STUCTS 2032 LINE NUM

# region Definition of base Structures


# Point
class _EdsPoint_(Structure):
    pass


_EdsPoint_._fields_ = [
    ('x', c_int),
    ('y', c_int),
]
EdsPoint = _EdsPoint_


# Rectangle
class _EdsRect_(Structure):
    pass


_EdsRect_._fields_ = [
    ('x', c_int),
    ('y', c_int),
    ('width', c_int),
    ('height', c_int),
]
EdsRect = _EdsRect_


# Size
class _EdsSize_(Structure):
    pass


_EdsSize_._fields_ = [
    ('width', c_int),
    ('height', c_int),
]
EdsSize = _EdsSize_


# Rational
class _EdsRational_(Structure):
    pass


_EdsRational_._fields_ = [
    ('Numerator', c_int),
    ('Denominator', c_uint),
]
EdsRational = _EdsRational_


# Time
class _EdsTime_(Structure):
    pass


_EdsTime_._fields_ = [
    ('Year', c_int),
    ('Month', c_int),
    ('Day', c_int),
    ('Hour', c_int),
    ('Minute', c_int),
    ('Second', c_int),
    ('Milliseconds', c_int),
]
EdsTime = _EdsTime_


# Device Info
class _EdsDeviceInfo_(Structure):
    pass


_EdsDeviceInfo_._fields_ = [
    # ('szPortName', POINTER(c_char) * EDS_MAX_NAME),
    # ('szDeviceDescription', POINTER(c_char) * EDS_MAX_NAME),
    ('szPortName', c_char * EDS_MAX_NAME),
    ('szDeviceDescription', c_char * EDS_MAX_NAME),
    # ('szPortName', c_char_p),
    # ('szDeviceDescription', c_char_p),
    ('DeviceSubType', c_uint),
    ('reserved', c_uint),
]
EdsDeviceInfo = _EdsDeviceInfo_


# Volume Info
class _EdsVolumeInfo_(Structure):
    pass


_EdsVolumeInfo_._fields_ = [
    ('StorageType', c_uint),
    ('Access', c_uint),
    ('MaxCapacity', c_ulong),
    ('FreeSpaceInBytes', c_ulong),
    #   [MarshalAs(UnmanagedType.ByValTStr, SizeConst=EDS_MAX_NAME)]
    ('szVolumeLabel', c_wchar_p),
]
EdsVolumeInfo = _EdsVolumeInfo_


# DirectoryItem Info
class _EdsDirectoryItemInfo_(Structure):
    pass


_EdsDirectoryItemInfo_._fields_ = [
    ('Size', c_ulonglong),
    ('isFolder', c_int),
    ('GroupID', c_uint),
    ('Option', c_uint),
    #   [MarshalAs(UnmanagedType.ByValTStr, SizeConst=EDS_MAX_NAME)]
    ('szFileName', c_char * EDS_MAX_NAME),
    ('format', c_uint),
    ('dateTime', c_uint),
]
EdsDirectoryItemInfo = _EdsDirectoryItemInfo_


# Image Info
class _EdsImageInfo_(Structure):
    pass


_EdsImageInfo_._fields_ = [
    ('Width', c_uint),
    ('Height', c_uint),
    ('NumOfComponents', c_uint),
    ('ComponentDepth', c_uint),

    #   [MarshalAs(UnmanagedType.ByValTStr, SizeConst=EDS_MAX_NAME)]
    ('EffectiveRect', EdsRect),
    ('reserved1', c_uint * 4),
    ('reserved2', c_uint * 4),
]
EdsImageInfo = _EdsImageInfo_


# SaveImage Setting
class _EdsSaveImageSetting_(Structure):
    pass


_EdsSaveImageSetting_._fields_ = [
    ('JPEGQuality', c_uint),
    ('iccProfileStream', c_void_p),
    ('reserved', c_uint * 4),
]
EdsSaveImageSetting = _EdsSaveImageSetting_


# Property Desc
class _EdsPropertyDesc_(Structure):
    pass


_EdsPropertyDesc_._fields_ = [
    ('Form', c_int),
    ('Access', c_uint),
    ('NumElements', c_int),
    #   [MarshalAs(UnmanagedType.ByValTStr, SizeConst=128)]
    ('PropDesc', c_int * 128),
]
EdsPropertyDesc = _EdsPropertyDesc_


# Picture Style Desc
class _EdsPictureStyleDesc_(Structure):
    pass


_EdsPictureStyleDesc_._fields_ = [
    ('contrast', c_int),
    ('sharpness', c_uint),
    ('saturation', c_int),
    ('colorTone', c_int),
    ('filterEffect', c_uint),
    ('toningEffect', c_uint),
    ('sharpFineness', c_uint),
    ('sharpThreshold', c_uint),
]
EdsPictureStyleDesc = _EdsPictureStyleDesc_


# Focus Info
class _EEdsFocusPoint_(Structure):
    pass


_EEdsFocusPoint_._fields_ = [
    ('valid', c_uint),
    ('selected', c_uint),
    ('justFocus', c_uint),
    ('rect', EdsRect),
    ('reserved', c_uint),
]
EdsFocusPoint = _EEdsFocusPoint_


class _EdsFocusInfo_(Structure):
    pass


_EdsFocusInfo_._fields_ = [
    ('imageRect', EdsRect),
    ('pointNumber', c_uint),
    # [MarshalAs(UnmanagedType.ByValArray, SizeConst=1053)]
    ('focusPoint', EdsFocusPoint * 1053),
    ('executeMode', c_uint),

]
EdsFocusInfo = _EdsFocusInfo_


# Capacity
class _EdsCapacity_(Structure):
    pass


_EdsCapacity_._fields_ = [
    ('NumberOfFreeClusters', c_int),
    ('BytesPerSector', c_int),
    ('Reset', c_int),
]
EdsCapacity = _EdsCapacity_


class _EVFDataSet_(Structure):
    pass


_EVFDataSet_._fields_ = [
    ('stream', c_void_p),
    ('zoom', c_int),
    ('zoomRect', EdsRect),
    ('visibleRect', EdsRect),
    ('imagePosition', EdsPoint),
    ('sizeJpegLarge', EdsSize),

]
EVFDataSet = _EVFDataSet_

# TODO: Done from 2252 LINE NUM
# region  Definition of error Codes

# -----------------------------------------------------------------------
# ED - SDK Error Code Masks
# ------------------------------------------------------------------------  #
EDS_ISSPECIFIC_MASK = 0x80000000
EDS_COMPONENTID_MASK = 0x7F000000
EDS_RESERVED_MASK = 0x00FF0000
EDS_ERRORID_MASK = 0x0000FFFF

# -----------------------------------------------------------------------
# ED - SDK Base Component IDs
# ------------------------------------------------------------------------  #
EDS_CMP_ID_CLIENT_COMPONENTID = 0x01000000
EDS_CMP_ID_LLSDK_COMPONENTID = 0x02000000
EDS_CMP_ID_HLSDK_COMPONENTID = 0x03000000

# -----------------------------------------------------------------------
# ED - SDK Function Success Code
# ------------------------------------------------------------------------ #
EDS_ERR_OK = 0x00000000

# -----------------------------------------------------------------------
# ED - SDK Generic Error IDs
# ------------------------------------------------------------------------ #
# Miscellaneous errors #
EDS_ERR_UNIMPLEMENTED = 0x00000001
EDS_ERR_INTERNAL_ERROR = 0x00000002
EDS_ERR_MEM_ALLOC_FAILED = 0x00000003
EDS_ERR_MEM_FREE_FAILED = 0x00000004
EDS_ERR_OPERATION_CANCELLED = 0x00000005
EDS_ERR_INCOMPATIBLE_VERSION = 0x00000006
EDS_ERR_NOT_SUPPORTED = 0x00000007
EDS_ERR_UNEXPECTED_EXCEPTION = 0x00000008
EDS_ERR_PROTECTION_VIOLATION = 0x00000009
EDS_ERR_MISSING_SUBCOMPONENT = 0x0000000A
EDS_ERR_SELECTION_UNAVAILABLE = 0x0000000B

# File errors #
EDS_ERR_FILE_IO_ERROR = 0x00000020
EDS_ERR_FILE_TOO_MANY_OPEN = 0x00000021
EDS_ERR_FILE_NOT_FOUND = 0x00000022
EDS_ERR_FILE_OPEN_ERROR = 0x00000023
EDS_ERR_FILE_CLOSE_ERROR = 0x00000024
EDS_ERR_FILE_SEEK_ERROR = 0x00000025
EDS_ERR_FILE_TELL_ERROR = 0x00000026
EDS_ERR_FILE_READ_ERROR = 0x00000027
EDS_ERR_FILE_WRITE_ERROR = 0x00000028
EDS_ERR_FILE_PERMISSION_ERROR = 0x00000029
EDS_ERR_FILE_DISK_FULL_ERROR = 0x0000002A
EDS_ERR_FILE_ALREADY_EXISTS = 0x0000002B
EDS_ERR_FILE_FORMAT_UNRECOGNIZED = 0x0000002C
EDS_ERR_FILE_DATA_CORRUPT = 0x0000002D
EDS_ERR_FILE_NAMING_NA = 0x0000002E

# Directory errors #
EDS_ERR_DIR_NOT_FOUND = 0x00000040
EDS_ERR_DIR_IO_ERROR = 0x00000041
EDS_ERR_DIR_ENTRY_NOT_FOUND = 0x00000042
EDS_ERR_DIR_ENTRY_EXISTS = 0x00000043
EDS_ERR_DIR_NOT_EMPTY = 0x00000044

# Property errors #
EDS_ERR_PROPERTIES_UNAVAILABLE = 0x00000050
EDS_ERR_PROPERTIES_MISMATCH = 0x00000051
EDS_ERR_PROPERTIES_NOT_LOADED = 0x00000053

# Function Parameter errors #
EDS_ERR_INVALID_PARAMETER = 0x00000060
EDS_ERR_INVALID_HANDLE = 0x00000061
EDS_ERR_INVALID_POINTER = 0x00000062
EDS_ERR_INVALID_INDEX = 0x00000063
EDS_ERR_INVALID_LENGTH = 0x00000064
EDS_ERR_INVALID_FN_POINTER = 0x00000065
EDS_ERR_INVALID_SORT_FN = 0x00000066

# Device errors #
EDS_ERR_DEVICE_NOT_FOUND = 0x00000080
EDS_ERR_DEVICE_BUSY = 0x00000081
EDS_ERR_DEVICE_INVALID = 0x00000082
EDS_ERR_DEVICE_EMERGENCY = 0x00000083
EDS_ERR_DEVICE_MEMORY_FULL = 0x00000084
EDS_ERR_DEVICE_INTERNAL_ERROR = 0x00000085
EDS_ERR_DEVICE_INVALID_PARAMETER = 0x00000086
EDS_ERR_DEVICE_NO_DISK = 0x00000087
EDS_ERR_DEVICE_DISK_ERROR = 0x00000088
EDS_ERR_DEVICE_CF_GATE_CHANGED = 0x00000089
EDS_ERR_DEVICE_DIAL_CHANGED = 0x0000008A
EDS_ERR_DEVICE_NOT_INSTALLED = 0x0000008B
EDS_ERR_DEVICE_STAY_AWAKE = 0x0000008C
EDS_ERR_DEVICE_NOT_RELEASED = 0x0000008D

# Stream errors #
EDS_ERR_STREAM_IO_ERROR = 0x000000A0
EDS_ERR_STREAM_NOT_OPEN = 0x000000A1
EDS_ERR_STREAM_ALREADY_OPEN = 0x000000A2
EDS_ERR_STREAM_OPEN_ERROR = 0x000000A3
EDS_ERR_STREAM_CLOSE_ERROR = 0x000000A4
EDS_ERR_STREAM_SEEK_ERROR = 0x000000A5
EDS_ERR_STREAM_TELL_ERROR = 0x000000A6
EDS_ERR_STREAM_READ_ERROR = 0x000000A7
EDS_ERR_STREAM_WRITE_ERROR = 0x000000A8
EDS_ERR_STREAM_PERMISSION_ERROR = 0x000000A9
EDS_ERR_STREAM_COULDNT_BEGIN_THREAD = 0x000000AA
EDS_ERR_STREAM_BAD_OPTIONS = 0x000000AB
EDS_ERR_STREAM_END_OF_STREAM = 0x000000AC

# Communications errors #
EDS_ERR_COMM_PORT_IS_IN_USE = 0x000000C0
EDS_ERR_COMM_DISCONNECTED = 0x000000C1
EDS_ERR_COMM_DEVICE_INCOMPATIBLE = 0x000000C2
EDS_ERR_COMM_BUFFER_FULL = 0x000000C3
EDS_ERR_COMM_USB_BUS_ERR = 0x000000C4

# Lock / Unlock #
EDS_ERR_USB_DEVICE_LOCK_ERROR = 0x000000D0
EDS_ERR_USB_DEVICE_UNLOCK_ERROR = 0x000000D1

# STI / WIA #
EDS_ERR_STI_UNKNOWN_ERROR = 0x000000E0
EDS_ERR_STI_INTERNAL_ERROR = 0x000000E1
EDS_ERR_STI_DEVICE_CREATE_ERROR = 0x000000E2
EDS_ERR_STI_DEVICE_RELEASE_ERROR = 0x000000E3
EDS_ERR_DEVICE_NOT_LAUNCHED = 0x000000E4

EDS_ERR_ENUM_NA = 0x000000F0
EDS_ERR_INVALID_FN_CALL = 0x000000F1
EDS_ERR_HANDLE_NOT_FOUND = 0x000000F2
EDS_ERR_INVALID_ID = 0x000000F3
EDS_ERR_WAIT_TIMEOUT_ERROR = 0x000000F4

# PTP #
EDS_ERR_SESSION_NOT_OPEN = 0x00002003
EDS_ERR_INVALID_TRANSACTIONID = 0x00002004
EDS_ERR_INCOMPLETE_TRANSFER = 0x00002007
EDS_ERR_INVALID_STRAGEID = 0x00002008
EDS_ERR_DEVICEPROP_NOT_SUPPORTED = 0x0000200A
EDS_ERR_INVALID_OBJECTFORMATCODE = 0x0000200B
EDS_ERR_SELF_TEST_FAILED = 0x00002011
EDS_ERR_PARTIAL_DELETION = 0x00002012
EDS_ERR_SPECIFICATION_BY_FORMAT_UNSUPPORTED = 0x00002014
EDS_ERR_NO_VALID_OBJECTINFO = 0x00002015
EDS_ERR_INVALID_CODE_FORMAT = 0x00002016
EDS_ERR_UNKNOWN_VENDER_CODE = 0x00002017
EDS_ERR_CAPTURE_ALREADY_TERMINATED = 0x00002018
EDS_ERR_INVALID_PARENTOBJECT = 0x0000201A
EDS_ERR_INVALID_DEVICEPROP_FORMAT = 0x0000201B
EDS_ERR_INVALID_DEVICEPROP_VALUE = 0x0000201C
EDS_ERR_SESSION_ALREADY_OPEN = 0x0000201E
EDS_ERR_TRANSACTION_CANCELLED = 0x0000201F
EDS_ERR_SPECIFICATION_OF_DESTINATION_UNSUPPORTED = 0x00002020
EDS_ERR_UNKNOWN_COMMAND = 0x0000A001
EDS_ERR_OPERATION_REFUSED = 0x0000A005
EDS_ERR_LENS_COVER_CLOSE = 0x0000A006
EDS_ERR_LOW_BATTERY = 0x0000A101
EDS_ERR_OBJECT_NOTREADY = 0x0000A102

# Capture Error  #
EDS_ERR_TAKE_PICTURE_AF_NG = 0x00008D01
EDS_ERR_TAKE_PICTURE_RESERVED = 0x00008D02
EDS_ERR_TAKE_PICTURE_MIRROR_UP_NG = 0x00008D03
EDS_ERR_TAKE_PICTURE_SENSOR_CLEANING_NG = 0x00008D04
EDS_ERR_TAKE_PICTURE_SILENCE_NG = 0x00008D05
EDS_ERR_TAKE_PICTURE_NO_CARD_NG = 0x00008D06
EDS_ERR_TAKE_PICTURE_CARD_NG = 0x00008D07
EDS_ERR_TAKE_PICTURE_CARD_PROTECT_NG = 0x00008D08

EDS_ERR_LAST_GENERIC_ERROR_PLUS_ONE = 0x000000F5

# endregion
