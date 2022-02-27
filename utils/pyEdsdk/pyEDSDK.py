# Python wrapper for EDSDK

# Camera operating function
import ctypes
import os
import platform
from .pyEDSDKTypes import *


if platform.uname()[0] == "Windows":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dllPath = os.path.join(dir_path, 'EDSDK.dll')
    dllImagePath = os.path.join(dir_path, 'EdsImage.dll')

    EDSDK = ctypes.WinDLL(dllPath)
    EDSDKImageDll = ctypes.WinDLL(dllImagePath)

    EdsPropertyEventHandler = WINFUNCTYPE(EdsError, EdsPropertyEvent, EdsPropertyID, EdsUInt32, py_object)
    EdsStateEventHandler = WINFUNCTYPE(c_uint, c_uint, IntPtr, IntPtr)
    EdsProgressCallback = WINFUNCTYPE(c_uint, c_void_p, IntPtr, c_bool)
    EdsObjectEventHandler = WINFUNCTYPE(EdsError, EdsObjectEvent, POINTER(c_void_p), py_object)

else:
    print("Only support windows operating system.")
    pass




# region Callback Functions

# public delegate uint EdsProgressCallback( uint inPercent, IntPtr inContext, ref bool outCancel);
# public delegate uint EdsCameraAddedHandler(IntPtr inContext);

# def EdsPropertyEventHandler(delegateFunc):
#     delegateFunc
#     pass
# public delegate uint EdsPropertyEventHandler(uint inEvent, uint inPropertyID, uint inParam, IntPtr inContext);


# public delegate uint EdsPropertyEventHandler(uint inEvent, uint inPropertyID, uint inParam, IntPtr inContext);
# public delegate uint EdsObjectEventHandler( uint inEvent, IntPtr inRef, IntPtr inContext);
# public delegate uint EdsStateEventHandler( uint inEvent, uint inParameter, IntPtr inContext);


# region Proto type definition of EDSDK API
'''
*----------------------------------
         Basic functions
----------------------------------*
'''


# *-----------------------------------------------------------------------------
#
#  Function:   EdsInitializeSDK
#
#  Description:
#      Initializes the libraries.
#      When using the EDSDK libraries, you must call this API once
#          before using EDSDK APIs.
#
#  Parameters:
#       In:    None
#      Out:    None
#
#  Returns:    Any of the sdk errors.
# ----------------------------------------------------------------------------- *

def EdsInitializeSDK():
    EDSDK.EdsInitializeSDK.restype = c_uint
    # C#原型:public extern static uint EdsInitializeSDK()
    return EDSDK.EdsInitializeSDK()


# *-----------------------------------------------------------------------------
#
#  Function:   EdsTerminateSDK
#
#  Description:
#      Terminates use of the libraries.
#      This function muse be called when ending the SDK.
#      Calling this function releases all resources allocated by the libraries.
#
#  Parameters:
#       In:    None
#      Out:    None
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*#

def EdsTerminateSDK():
    EDSDK.EdsTerminateSDK.restype = c_uint
    # C#原型:public extern static uint EdsInitializeSDK()
    return EDSDK.EdsTerminateSDK()


# *-----------------------------------------------------------------------------
#
#  Function:   EdsRelease
#
#  Description:
#      Decrements the reference counter to an object.
#      When the reference counter reaches 0, the object is released.
#
#  Parameters:
#       In:    inRef - The reference of the item.
#      Out:    None
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*#
# EdsUInt32 EDSAPI EdsRelease( EdsBaseRef  inRef )

def EdsRelease(cameraList):
    EDSDK.EdsRelease.argtype = IntPtr
    EDSDK.EdsRelease.restype = c_uint
    # C#原型:public extern static uint EdsRelease( IntPtr inRef )
    return EDSDK.EdsRelease(cameraList)


'''
#*----------------------------------
   Item-tree operating functions
----------------------------------*#
'''


# *-----------------------------------------------------------------------------
#
#  Function:   EdsGetChildCount
#
#  Description:
#      Gets the number of child objects of the designated object.
#      Example: Number of files in a directory
#
#  Parameters:
#       In:    inRef - The reference of the list.
#      Out:    outCount - Number of elements in this list.
#
#  Returns:    Any of the sdk errors.
# ----------------------------------------------------------------------------- *  #


def EdsGetChildCount(inRef, outCount):
    EDSDK.EdsGetChildCount.argtype = (c_void_p, c_int)
    EDSDK.EdsGetChildCount.restype = c_int
    # C#原型: public extern static uint EdsGetCameraList( out IntPtr  outCameraListRef)
    return EDSDK.EdsGetChildCount(inRef, byref(outCount))


# *-----------------------------------------------------------------------------
#
#  Function:   EdsGetChildAtIndex
#
#  Description:
#       Gets an indexed child object of the designated object.
#
#  Parameters:
#       In:    inRef - The reference of the item.
#              inIndex -  The index that is passed in, is zero based.
#      Out:    outRef - The pointer which receives reference of the
#                           specified index .
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*#

def EdsGetChildAtIndex(inRef, inIndex, outRef):
    EDSDK.EdsGetChildCount.argtype = (c_void_p, c_int, c_void_p)
    EDSDK.EdsGetChildCount.restype = c_int
    # C#原型: EdsGetChildAtIndex( IntPtr inRef, int inIndex, out IntPtr outRef)
    return EDSDK.EdsGetChildAtIndex(inRef, inIndex, byref(outRef))


'''
#*----------------------------------
    Property operating functions
----------------------------------*#
'''


# /*-----------------------------------------------------------------------------
#
#  Function:   EdsGetPropertySize
#
#  Description:
#      Gets the byte size and data type of a designated property
#          from a camera object or image object.
#
#  Parameters:
#       In:    inRef - The reference of the item.
#              inPropertyID - The ProprtyID
#              inParam - Additional information of property.
#                   We use this parameter in order to specify an index
#                   in case there are two or more values over the same ID.
#      Out:    outDataType - Pointer to the buffer that is to receive the property
#                        type data.
#              outSize - Pointer to the buffer that is to receive the property
#                        size.
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/


def EdsGetPropertySize(inRef, inPropertyID, inParam, outDataType, outSize):
    EDSDK.EdsGetPropertySize.argtype = (IntPtr, c_uint, c_int, c_uint, c_int)
    EDSDK.EdsGetPropertySize.restype = c_uint
    # C#原型: public extern static uint EdsGetPropertySize(IntPtr inRef, uint inPropertyID, int inParam,
    #              out EdsDataType outDataType, out int outSize)
    return EDSDK.EdsGetPropertySize(inRef, inPropertyID, inParam, byref(outDataType), byref(outSize))


# *-----------------------------------------------------------------------------
#
#  Function:   EdsGetPropertyData
#
#  Description:
#      Gets property information from the object designated in inRef.
#
#  Parameters:
#       In:    inRef - The reference of the item.
#              inPropertyID - The ProprtyID
#              inParam - Additional information of property.
#                   We use this parameter in order to specify an index
#                   in case there are two or more values over the same ID.
#              inPropertySize - The number of bytes of the prepared buffer
#                  for receive property-value.
#       Out:   outPropertyData - The buffer pointer to receive property-value.
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/

def EdsGetPropertyData(inRef, inPropertyID, inParam, inPropertySize, outPropertyData):
    EDSDK.EdsGetPropertyData.argtype = (c_void_p, c_uint, c_int, c_int, c_void_p)
    EDSDK.EdsGetPropertyData.restype = c_uint
    outProp = byref(outPropertyData)
    # C#原型: public extern static uint EdsGetPropertyData(IntPtr inRef, uint inPropertyID, int inParam,
    #              int inPropertySize, IntPtr outPropertyData)
    # if hasattr(outPropertyData, 'value'):
    #     if outPropertyData.value == EdsDataType.UInt32:
    #         EDSDK.EdsGetPropertyData.argtype = (IntPtr, c_uint, c_int, c_uint, c_int)
    #         outProp = outPropertyData

    # elif outPropertyData.value == EdsDataType.Int32:
    #     EDSDK.EdsGetPropertyData.argtype = (IntPtr, c_uint, c_int, c_int, c_int)
    #
    # elif outPropertyData.value == EdsDataType.String:
    #     EDSDK.EdsGetPropertyData.argtype = (IntPtr, c_uint, c_int, c_void_p, c_int)
    #
    # elif outPropertyData.value == EdsDataType.FocusInfo:
    #     EDSDK.EdsGetPropertyData.argtype = (IntPtr, c_uint, c_int, c_void_p, c_int)
    return EDSDK.EdsGetPropertyData(inRef, inPropertyID, inParam, inPropertySize, outProp)


# *-----------------------------------------------------------------------------
#
#  Function:   EdsSetPropertyData
#
#  Description:
#      Sets property data for the object designated in inRef.
#
#  Parameters:
#       In:    inRef - The reference of the item.
#              inPropertyID - The ProprtyID
#              inParam - Additional information of property.
#              inPropertySize - The number of bytes of the prepared buffer
#                  for set property-value.
#              inPropertyData - The buffer pointer to set property-value.
#      Out:    None
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/


def EdsSetPropertyData(inRef, inPropertyID, inParam, inPropertySize, inPropertyData):
    EDSDK.EdsSetPropertyData.argtype = (c_void_p, c_uint, c_int, c_int, c_void_p)
    EDSDK.EdsSetPropertyData.restype = c_uint
    # C#原型:public extern static uint EdsSetPropertyData( IntPtr inRef, uint inPropertyID,
    #              int inParam, int inPropertySize, [MarshalAs(UnmanagedType.AsAny), In] object inPropertyData);
    return EDSDK.EdsSetPropertyData(inRef, inPropertyID, inParam, inPropertySize, byref(inPropertyData))


# *-----------------------------------------------------------------------------
#
#  Function:   EdsGetPropertyDesc
#
#  Description:
#      Gets a list of property data that can be set for the object
#          designated in inRef, as well as maximum and minimum values.
#      This API is intended for only some shooting-related properties.
#
#  Parameters:
#       In:    inRef - The reference of the camera.
#              inPropertyID - The Property ID.
#       Out:   outPropertyDesc - Array of the value which can be set up.
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/


def EdsGetPropertyDesc(inRef, inPropertyID, outPropertyDesc):
    EDSDK.EdsGetPropertyDesc.argtype = (c_void_p, c_uint, c_void_p)
    EDSDK.EdsGetPropertyDesc.restype = c_uint
    # C#原型:public extern static uint EdsGetPropertyDesc( IntPtr inRef, uint inPropertyID,
    #              out EdsPropertyDesc outPropertyDesc);
    return EDSDK.EdsGetPropertyDesc(inRef, inPropertyID, byref(outPropertyDesc))


'''
#*--------------------------------------------
   Device-list and device operating functions
---------------------------------------------*#
'''


# *-----------------------------------------------------------------------------
#
#  Function:   EdsGetCameraList
#
#  Description:
#      Gets camera list objects.
#
#  Parameters:
#       In:    None
#      Out:    outCameraListRef - Pointer to the camera-list.
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*#

def EdsGetCameraList(outCameraListRef):
    EDSDK.EdsGetCameraList.restype = c_int
    # C#原型: public extern static uint EdsGetCameraList( out IntPtr  outCameraListRef)
    return EDSDK.EdsGetCameraList(byref(outCameraListRef))


'''
*----------------------------------
  Camera operating functions
----------------------------------*
'''


# *-----------------------------------------------------------------------------
##
#  Function:   EdsGetDeviceInfo
#
#  Description:
#      Gets device information, such as the device name.
#      Because device information of remote cameras is stored
#          on the host computer, you can use this API
#          before the camera object initiates communication
#          (that is, before a session is opened).
#
#  Parameters:
#       In:    inCameraRef - The reference of the camera.
#      Out:    outDeviceInfo - Information as device of camera.
#
#  Returns:    Any of the sdk errors.
# ----------------------------------------------------------------------------- *

def EdsGetDeviceInfo(camera, deviceInfo):
    EDSDK.EdsGetDeviceInfo.argtype = (c_void_p, c_void_p)
    EDSDK.EdsGetDeviceInfo.restype = c_uint
    # C#原型:public extern static uint EdsGetDeviceInfo( IntPtr  inCameraRef, out EdsDeviceInfo  outDeviceInfo)
    return EDSDK.EdsGetDeviceInfo(camera, byref(deviceInfo))


# *-----------------------------------------------------------------------------
#
#  Function:   EdsOpenSession
#
#  Description:
#      Establishes a logical connection with a remote camera.
#      Use this API after getting the camera's EdsCamera object.
#
#  Parameters:
#       In:    inCameraRef - The reference of the camera
#      Out:    None
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/
def EdsOpenSession(inCameraRef):
    EDSDK.EdsOpenSession.argtype = c_void_p
    EDSDK.EdsOpenSession.restype = c_uint
    # C#原型:ublic extern static uint EdsOpenSession( IntPtr inCameraRef)
    return EDSDK.EdsOpenSession(inCameraRef)


# *-----------------------------------------------------------------------------
#
#  Function:   EdsCloseSession
#
#  Description:
#       Closes a logical connection with a remote camera.
#
#  Parameters:
#       In:    inCameraRef - The reference of the camera
#      Out:    None
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/

def EdsCloseSession(inCameraRef):
    EDSDK.EdsCloseSession.argtype = c_void_p
    EDSDK.EdsCloseSession.restype = c_uint
    # C#原型:public extern static uint EdsCloseSession( IntPtr inCameraRef)
    return EDSDK.EdsCloseSession(inCameraRef)


# *-----------------------------------------------------------------------------
#
#  Function:   EdsSendCommand
#
#  Description:
#       Sends a command such as "Shoot" to a remote camera.
#
#  Parameters:
#       In:    inCameraRef - The reference of the camera which will receive the
#                      command.
#              inCommand - Specifies the command to be sent.
#              inParam -     Specifies additional command-specific information.
#      Out:    None
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/

def EdsSendCommand(inCameraRef, inCommand, inParam):
    EDSDK.EdsSendCommand.argtype = (c_void_p, c_uint, c_int)
    EDSDK.EdsSendCommand.restype = c_uint32
    # C#原型:EdsError EDSAPI EdsSendCommand( EdsCameraRef        inCameraRef,
    #                                 EdsCameraCommand    inCommand,
    #                                 EdsInt32            inParam );
    return EDSDK.EdsSendCommand(inCameraRef, inCommand, inParam)


# *-----------------------------------------------------------------------------
#
#  Function:   EdsSendStatusCommand
#
#  Description:
#       Sets the remote camera state or mode.
#
#  Parameters:
#       In:    inCameraRef - The reference of the camera which will receive the
#                      command.
#              inStatusCommand - Specifies the command to be sent.
#              inParam -     Specifies additional command-specific information.
#      Out:    None
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/


def EdsSendStatusCommand(inCameraRef, inCameraState, inParam):
    EDSDK.EdsSendStatusCommand.argtype = (c_void_p, c_uint, c_int)
    EDSDK.EdsSendStatusCommand.restype = c_uint
    # C#原型:public extern static uint EdsSendStatusCommand(IntPtr inCameraRef, uint inCameraState, int inParam)
    return EDSDK.EdsSendStatusCommand(inCameraRef, inCameraState, inParam)


# *-----------------------------------------------------------------------------
#
#  Function:   EdsSetCapacity
#
#  Description:
#      Sets the remaining HDD capacity on the host computer
#          (excluding the portion from image transfer),
#          as calculated by subtracting the portion from the previous time.
#      Set a reset flag initially and designate the cluster length
#          and number of free clusters.
#      Some type 2 protocol standard cameras can display the number of shots
#          left on the camera based on the available disk capacity
#          of the host computer.
#      For these cameras, after the storage destination is set to the computer,
#          use this API to notify the camera of the available disk capacity
#          of the host computer.
#
#  Parameters:
#       In:    inCameraRef - The reference of the camera which will receive the
#                      command.
#              inCapacity -  The remaining capacity of a transmission place.
#      Out:    None
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/

def EdsSetCapacity(inCameraRef, inCapacity):
    EDSDK.EdsSetCapacity.argtype = (c_void_p, c_void_p)
    EDSDK.EdsSetCapacity.restype = c_uint
    # C#原型:public extern static uint EdsSetCapacity( IntPtr inCameraRef, EdsCapacity inCapacity)
    return EDSDK.EdsSetCapacity(inCameraRef, inCapacity)


'''

#*------------------------------------
    Volume operating functions
-------------------------------------*/
'''


# *-----------------------------------------------------------------------------
#
#  Function:   EdsGetVolumeInfo
#
#  Description:
#      Gets volume information for a memory card in the camera.
#
#  Parameters:
#       In:    inVolumeRef - The reference of the volume.
#      Out:    outVolumeInfo - information of  the volume.
#
#  Returns:    Any of the sdk errors.
# ----------------------------------------------------------------------------- * /

def EdsGetVolumeInfo(inCameraRef, inCapacity):
    EDSDK.EdsGetVolumeInfo.argtype = (c_void_p, c_void_p)
    EDSDK.EdsGetVolumeInfo.restype = c_uint
    # C#原型:public extern static uint EdsGetVolumeInfo(IntPtr inCameraRef, out EdsVolumeInfo outVolumeInfo);
    return EDSDK.EdsGetVolumeInfo(inCameraRef, byref(inCapacity))


'''
*---------------------------------------
   Directory-item operating functions
---------------------------------------*
'''


# *-----------------------------------------------------------------------------
#
#  Function:   EdsGetDirectoryItemInfo
#
#  Description:
#      Gets information about the directory or file objects
#          on the memory card (volume) in a remote camera.
#
#  Parameters:
#       In:    inDirItemRef - The reference of the directory item.
#      Out:    outDirItemInfo - information of the directory item.
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/


def EdsGetDirectoryItemInfo(inDirItemRef, outDirItemInfo):
    EDSDK.EdsGetDirectoryItemInfo.argtype = (c_void_p, c_void_p)
    EDSDK.EdsGetDirectoryItemInfo.restype = c_uint
    # C#原型:EdsError EDSAPI EdsGetDirectoryItemInfo(
    #                                 EdsDirectoryItemRef    inDirItemRef,
    #                                 EdsDirectoryItemInfo*   outDirItemInfo)
    return EDSDK.EdsGetDirectoryItemInfo(inDirItemRef, byref(outDirItemInfo))


# *-----------------------------------------------------------------------------
#
#  Function:   EdsDownload
#
#  Description:
#       Downloads a file on a remote camera
#          (in the camera memory or on a memory card) to the host computer.
#      The downloaded file is sent directly to a file stream created in advance.
#      When dividing the file being retrieved, call this API repeatedly.
#      Also in this case, make the data block size a multiple of 512 (bytes),
#          excluding the final block.
#
#  Parameters:
#       In:    inDirItemRef - The reference of the directory item.
#              inReadSize   -
#
#      Out:    outStream    - The reference of the stream.
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/


def EdsDownload(inDirItemRef, inReadSize, outStream):
    EDSDK.EdsDownload.argtype = (c_void_p, c_uint64, c_void_p)
    EDSDK.EdsDownload.restype = c_uint
    # C#原型:public extern static uint EdsDownload(IntPtr inDirItemRef, UInt64 inReadSize, IntPtr outStream)
    return EDSDK.EdsDownload(inDirItemRef, inReadSize, outStream)


# *-----------------------------------------------------------------------------
#
#  Function:   EdsDownloadComplete
#
#  Description:
#       Must be called when downloading of directory items is complete.
#          Executing this API makes the camera
#              recognize that file transmission is complete.
#          This operation need not be executed when using EdsDownloadThumbnail.
#
#  Parameters:
#       In:    inDirItemRef - The reference of the directory item.
#
#      Out:    outStream    - None.
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/


def EdsDownloadComplete(inDirItemRef):
    EDSDK.EdsDownloadComplete.argtype = c_void_p
    EDSDK.EdsDownloadComplete.restype = c_uint
    # C#原型:public extern static uint EdsDownloadComplete ( IntPtr inDirItemRef);
    return EDSDK.EdsDownloadComplete(inDirItemRef)


'''
#*--------------------------------------------
        Stream operating functions
---------------------------------------------*/
'''


# *-----------------------------------------------------------------------------
#
#  Function:   EdsCreateFileStream
#
#  Description:
#      Creates a new file on a host computer (or opens an existing file)
#          and creates a file stream for access to the file.
#      If a new file is designated before executing this API,
#          the file is actually created following the timing of writing
#          by means of EdsWrite or the like with respect to an open stream.
#
#  Parameters:
#       In:    inFileName - Pointer to a null-terminated string that specifies
#                           the file name.
#              inCreateDisposition - Action to take on files that exist,
#                                and which action to take when files do not exist.
#              inDesiredAccess - Access to the stream (reading, writing, or both).
#      Out:    outStream - The reference of the stream.
#
#  Returns:    Any of the sdk errors.
# ----------------------------------------------------------------------------- * /


def EdsCreateFileStream(inFileName, inCreateDisposition, inDesiredAccess, outStream):
    EDSDK.EdsCreateFileStream.argtype = (c_void_p, c_uint, c_uint, c_void_p)
    EDSDK.EdsCreateFileStream.restype = c_uint
    # C#原型:public extern static uint EdsCreateFileStream( string inFileName,
    #                                                      EdsFileCreateDisposition inCreateDisposition,
    #                                                      EdsAccess inDesiredAccess, out IntPtr outStream);
    return EDSDK.EdsCreateFileStream(inFileName, inCreateDisposition, inDesiredAccess, byref(outStream))


# *-----------------------------------------------------------------------------
#
#  Function:   EdsCreateStreamEx
#
#  Description:
#      An extended version of EdsCreateStreamFromFile.
#      Use this function when working with Unicode file names.
#
#  Parameters:
#       In:    inFileName - Designate the file name.
#              inCreateDisposition - Action to take on files that exist,
#                                and which action to take when files do not exist.
#              inDesiredAccess - Access to the stream (reading, writing, or both).
#
#      Out:    outStream - The reference of the stream.
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/


def EdsCreateFileStreamEx(inFileName, inCreateDisposition, inDesiredAccess, outStream):
    EDSDK.EdsCreateFileStreamEx.argtype = (c_void_p, c_uint, c_uint, c_void_p)
    EDSDK.EdsCreateFileStreamEx.restype = c_uint
    # C#原型:public extern static uint EdsCreateStreamEx(
    #            string                       inFileName,
    #            EdsFileCreateDisposition     inCreateDisposition,
    #            EdsAccess                    inDesiredAccess,
    #            out IntPtr                   outStream
    #            );
    return EDSDK.EdsCreateFileStreamEx(inFileName, inCreateDisposition, inDesiredAccess, byref(outStream))


# *-----------------------------------------------------------------------------
#
#  Function:   EdsCreateMemoryStream
#
#  Description:
#      Creates a stream in the memory of a host computer.
#      In the case of writing in excess of the allocated buffer size,
#          the memory is automatically extended.
#
#  Parameters:
#       In:    inBufferSize - The number of bytes of the memory to allocate.
#      Out:    outStream - The reference of the stream.
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/


def EdsCreateMemoryStream(inBufferSize, outStream):
    EDSDK.EdsCreateMemoryStream.argtype = (c_ulonglong, c_void_p)
    EDSDK.EdsCreateMemoryStream.restype = c_uint
    # C#原型:public extern static uint EdsCreateMemoryStream(UInt64 inBufferSize, out IntPtr outStream);
    return EDSDK.EdsCreateMemoryStream(inBufferSize, byref(outStream))


# *-----------------------------------------------------------------------------
#
#  Function:   EdsGetPointer
#
#  Description:
#      Gets the pointer to the start address of memory managed by the memory stream.
#      As the EDSDK automatically resizes the buffer, the memory stream provides
#          you with the same access methods as for the file stream.
#      If access is attempted that is excessive with regard to the buffer size
#          for the stream, data before the required buffer size is allocated
#          is copied internally, and new writing occurs.
#      Thus, the buffer pointer might be switched on an unknown timing.
#      Caution in use is therefore advised.
#
#  Parameters:
#       In:    inStream - Designate the memory stream for the pointer to retrieve.
#      Out:    outPointer - If successful, returns the pointer to the buffer
#                  written in the memory stream.
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/

def EdsGetPointer(inStreamRef, outPointer):
    EDSDK.EdsGetPointer.argtype = (c_void_p, c_void_p)
    EDSDK.EdsGetPointer.restype = c_uint
    # C#原型:public extern static uint EdsGetPointer(IntPtr inStreamRef, out IntPtr outPointer);
    return EDSDK.EdsGetPointer(inStreamRef, byref(outPointer))

    # *-----------------------------------------------------------------------------


#
#  Function:   EdsGetLength
#
#  Description:
#      Gets the stream size.
#
#  Parameters:
#       In:    inStreamRef - The reference of the stream or image.
#      Out:    outLength - The length of the stream.
#
#  Returns:    Any of the sdk errors.
#
# -----------------------------------------------------------------------------*/


def EdsGetLength(inStreamRef, outLength):
    EDSDK.EdsGetLength.argtype = (c_void_p, c_ulonglong)
    EDSDK.EdsGetLength.restype = c_uint
    # C#原型:public extern static uint EdsGetLength(IntPtr inStreamRef, out UInt64 outLength);
    return EDSDK.EdsGetLength(inStreamRef, byref(outLength))


# *-----------------------------------------------------------------------------
#
#  Function:   EdsCopyData
#
#  Description:
#      Copies data from the copy source stream to the copy destination stream.
#      The read or write position of the data to copy is determined from
#          the current file read or write position of the respective stream.
#      After this API is executed, the read or write positions of the copy source
#          and copy destination streams are moved an amount corresponding to
#          inWriteSize in the positive direction.
#
#  Parameters:
#       In:    inStreamRef - The reference of the stream or image.
#              inWriteSize - The number of bytes to copy.
#      Out:    outStreamRef - The reference of the stream or image.
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/


def EdsCopyData(inStreamRef, inWriteSize, outStreamRef):
    EDSDK.EdsCopyData.argtype = (c_void_p, c_ulonglong, c_void_p)
    EDSDK.EdsCopyData.restype = c_uint
    # C#原型:public extern static uint EdsCopyData(IntPtr inStreamRef, UInt64 inWriteSize, IntPtr outStreamRef);
    return EDSDK.EdsCopyData(inStreamRef, inWriteSize, outStreamRef)


# *-----------------------------------------------------------------------------
#
#  Function:   EdsSetProgressCallback
#
#  Description:
#      Register a progress callback function.
#      An event is received as notification of progress during processing that
#          takes a relatively long time, such as downloading files from a
#          remote camera.
#      If you register the callback function, the EDSDK calls the callback
#          function during execution or on completion of the following APIs.
#      This timing can be used in updating on-screen progress bars, for example.
#
#  Parameters:
#       In:    inRef - The reference of the stream or image.
#              inProgressCallback - Pointer to a progress callback function.
#              inProgressOption - The option about progress is specified.
#                              Must be one of the following values.
#                         kEdsProgressOption_Done
#                             When processing is completed,a callback function
#                             is called only at once.
#                         kEdsProgressOption_Periodically
#                             A callback function is performed periodically.
#              inContext - Application information, passed in the argument
#                      when the callback function is called. Any information
#                      required for your program may be added.
#      Out:    None
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/



def EdsSetProgressCallback(inRef, inProgressFunc, inProgressOption, inContext):
    EDSDK.EdsSetProgressCallback.argtype = (c_void_p, EdsProgressCallback, EdsProgressOption, c_void_p)
    EDSDK.EdsSetProgressCallback.restype = c_uint
    # C#原型:public extern static uint EdsSetProgressCallback( IntPtr inRef, EdsProgressCallback inProgressFunc,
    #              EdsProgressOption inProgressOption, IntPtr inContext);
    return EDSDK.EdsSetProgressCallback(inRef, inProgressFunc, inProgressOption, inContext)


'''
# *--------------------------------------------
#               Image operating functions
# ---------------------------------------------*/
'''


# *-----------------------------------------------------------------------------
#
#  Function:   EdsCreateImageRef
#
#  Description:
#      Creates an image object from an image file.
#      Without modification, stream objects cannot be worked with as images.
#      Thus, when extracting images from image files,
#          you must use this API to create image objects.
#      The image object created this way can be used to get image information
#          (such as the height and width, number of color components, and
#           resolution), thumbnail image data, and the image data itself.
#
#  Parameters:
#       In:    inStreamRef - The reference of the stream.
#
#       Out:    outImageRef - The reference of the image.
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/


def EdsCreateImageRef(inStreamRef, outImageRef):
    EDSDK.EdsCreateImageRef.argtype = (c_void_p, c_void_p)
    EDSDK.EdsCreateImageRef.restype = c_uint
    # C#原型:public extern static uint EdsCreateImageRef( IntPtr inStreamRef,  out IntPtr outImageRef);
    return EDSDK.EdsCreateImageRef(inStreamRef, byref(outImageRef))

    # *-----------------------------------------------------------------------------
    #
    #  Function:   EdsGetImageInfo
    #
    #  Description:
    #      Gets image information from a designated image object.
    #      Here, image information means the image width and height,
    #          number of color components, resolution, and effective image area.
    #
    #  Parameters:
    #       In:    inStreamRef - Designate the object for which to get image information.
    #              inImageSource - Of the various image data items in the image file,
    #                  designate the type of image data representing the
    #                  information you want to get. Designate the image as
    #                  defined in Enum EdsImageSource.
    #
    #                      kEdsImageSrc_FullView
    #                                  The image itself (a full-sized image)
    #                      kEdsImageSrc_Thumbnail
    #                                  A thumbnail image
    #                      kEdsImageSrc_Preview
    #                                  A preview image
    #       Out:    outImageInfo - Stores the image data information designated
    #                      in inImageSource.
    #
    #  Returns:    Any of the sdk errors.
    # -----------------------------------------------------------------------------*/


def EdsGetImageInfo(inImageRef, inImageSource, outImageInfo):
    EDSDK.EdsGetImageInfo.argtype = (c_void_p, EdsImageSource, EdsImageInfo)
    EDSDK.EdsGetImageInfo.restype = c_uint
    # C#原型:public extern static uint EdsGetImageInfo( IntPtr inImageRef, EdsImageSource inImageSource,
    #               out EdsImageInfo outImageInfo );
    return EDSDK.EdsGetImageInfo(inImageRef, inImageSource, byref(outImageInfo))


# *-----------------------------------------------------------------------------
#
#  Function:   EdsGetImage
#
#  Description:
#      Gets designated image data from an image file, in the form of a
#          designated rectangle.
#      Returns uncompressed results for JPEGs and processed results
#          in the designated pixel order (RGB, Top-down BGR, and so on) for
#           RAW images.
#      Additionally, by designating the input/output rectangle,
#          it is possible to get reduced, enlarged, or partial images.
#      However, because images corresponding to the designated output rectangle
#          are always returned by the SDK, the SDK does not take the aspect
#          ratio into account.
#      To maintain the aspect ratio, you must keep the aspect ratio in mind
#          when designating the rectangle.
#
#  Parameters:
#      In:
#              inImageRef - Designate the image object for which to get
#                      the image data.
#              inImageSource - Designate the type of image data to get from
#                      the image file (thumbnail, preview, and so on).
#                      Designate values as defined in Enum EdsImageSource.
#              inImageType - Designate the output image type. Because
#                      the output format of EdGetImage may only be RGB, only
#                      kEdsTargetImageType_RGB or kEdsTargetImageType_RGB16
#                      can be designated.
#                      However, image types exceeding the resolution of
#                      inImageSource cannot be designated.
#              inSrcRect - Designate the coordinates and size of the rectangle
#                      to be retrieved (processed) from the source image.
#              inDstSize - Designate the rectangle size for output.
#
#      Out:
#              outStreamRef - Designate the memory or file stream for output of
#                      the image.
#  Returns:    Any of the sdk errors.
# ----------------------------------------------------------------------------- * /


def EdsGetImage(inImageRef, inImageSource, inImageType, inSrcRect, inDstSize, outStreamRef):
    EDSDK.EdsGetImage.argtype = (c_void_p, EdsImageSource, EdsTargetImageType, EdsRect, EdsSize, c_void_p)
    EDSDK.EdsGetImage.restype = c_uint
    # C#原型:public extern static uint EdsGetImage( IntPtr inImageRef, EdsImageSource inImageSource,
    #              EdsTargetImageType inImageType, EdsRect inSrcRect, EdsSize inDstSize, IntPtr outStreamRef );
    return EDSDK.EdsGetImage(inImageRef, inImageSource, inImageType, inSrcRect, inDstSize, outStreamRef)


'''
# ----------------------------------------------
#   Event handler registering functions
# ----------------------------------------------
'''

# *-----------------------------------------------------------------------------
#
#  Function:   EdsSetCameraAddedHandler
#
#  Description:
#      Registers a callback function for when a camera is detected.
#
#  Parameters:
#       In:    inCameraAddedHandler - Pointer to a callback function
#                          called when a camera is connected physically
#              inContext - Specifies an application-defined value to be sent to
#                          the callback function pointed to by CallBack parameter.
#      Out:    None
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*#
# public extern static uint EdsSetCameraAddedHandler(EdsCameraAddedHandler inCameraAddedHandler,IntPtr inContext);

# *-----------------------------------------------------------------------------
#
#  Function:   EdsSetPropertyEventHandler
#
#  Description:
#       Registers a callback function for receiving status
#          change notification events for property states on a camera.
#
#  Parameters:
#       In:    inCameraRef - Designate the camera object.
#              inEvent - Designate one or all events to be supplemented.
#              inPropertyEventHandler - Designate the pointer to the callback
#                      function for receiving property-related camera events.
#              inContext - Designate application information to be passed by
#                      means of the callback function. Any data needed for
#                      your application can be passed.
#      Out:    None
#
# Returns:    Any of the sdk errors. ----------------------------------------------------------------------------- *

# FIXME:

# @WINFUNCTYPE(None, c_uint, c_uint, c_uint, IntPtr)
# def EdsPropertyEventHandler(inEvent, inPropertyID, inParam, inContext):
#     pass
# function pointer:


def EdsSetPropertyEventHandler(inCameraRef, inEvent, inPropertyEventHandler, inContext):
    EDSDK.EdsSetPropertyEventHandler.argtype = (IntPtr, EdsPropertyEvent, EdsPropertyEventHandler, py_object)
    EDSDK.EdsSetPropertyEventHandler.restype = EdsError
    # C#原型:public extern static uint EdsSetPropertyEventHandler( IntPtr inCameraRef,  uint inEvnet,
    # EdsPropertyEventHandler  inPropertyEventHandler, IntPtr inContext );
    return EDSDK.EdsSetPropertyEventHandler(inCameraRef, inEvent, inPropertyEventHandler,
                                            inContext)


# *-----------------------------------------------------------------------------
#
#  Function:   EdsSetObjectEventHandler
#
#  Description:
#       Registers a callback function for receiving status
#          change notification events for objects on a remote camera.
#      Here, object means volumes representing memory cards, files and directories,
#          and shot images stored in memory, in particular.
#
#  Parameters:
#       In:    inCameraRef - Designate the camera object.
#              inEvent - Designate one or all events to be supplemented.
#                  To designate all events, use kEdsObjectEvent_All.
#              inObjectEventHandler - Designate the pointer to the callback function
#                  for receiving object-related camera events.
#              inContext - Passes inContext without modification,
#                  as designated as an EdsSetObjectEventHandler argument.
#      Out:    None
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*#
# EdsObjectEventHandler = WINFUNCTYPE(c_uint, c_uint, IntPtr, py_object)


def EdsSetObjectEventHandler(inCameraRef, inEvent, inObjectEventHandler, inContext):
    EDSDK.EdsSetObjectEventHandler.argtype = (IntPtr, EdsObjectEvent, EdsObjectEventHandler, IntPtr)
    EDSDK.EdsSetObjectEventHandler.restype = EdsError
    # C#原型:public extern static uint EdsSetObjectEventHandler( IntPtr inCameraRef,  uint inEvnet,
    # EdsObjectEventHandler  inObjectEventHandler, IntPtr inContext );
    return EDSDK.EdsSetObjectEventHandler(inCameraRef, inEvent, inObjectEventHandler,
                                          inContext)


# *-----------------------------------------------------------------------------
#
#  Function:  EdsSetCameraStateEventHandler
#
#  Description:
#      Registers a callback function for receiving status
#          change notification events for property states on a camera.
#
#  Parameters:
#       In:    inCameraRef - Designate the camera object.
#              inEvent - Designate one or all events to be supplemented.
#                  To designate all events, use kEdsStateEvent_All.
#              inStateEventHandler - Designate the pointer to the callback function
#                  for receiving events related to camera object states.
#              inContext - Designate application information to be passed
#                  by means of the callback function. Any data needed for
#                  your application can be passed.
#      Out:    None
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/


def EdsSetCameraStateEventHandler(inCameraRef, inEvent, inStateEventHandler, inContext):
    EDSDK.EdsSetCameraStateEventHandler.argtype = (IntPtr, EdsStateEvent, EdsStateEventHandler, IntPtr)
    EDSDK.EdsSetCameraStateEventHandler.restype = EdsError
    # C#原型:public extern static uint EdsSetCameraStateEventHandler( IntPtr inCameraRef,  uint inEvnet,
    # EdsStateEventHandler  inStateEventHandler, IntPtr inContext );
    return EDSDK.EdsSetCameraStateEventHandler(inCameraRef, inEvent, inStateEventHandler,
                                               inContext)


# *-----------------------------------------------------------------------------
#
#  Function:   EdsCreateEvfImageRef
#  Description:
#       Creates an object used to get the live view image data set.
#
#  Parameters:
#      In:     inStreamRef - The stream reference which opened to get EVF JPEG image.
#      Out:    outEvfImageRef - The EVFData reference.
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/


def EdsCreateEvfImageRef(inStreamRef, outEvfImageRef):
    EDSDK.EdsCreateEvfImageRef.argtype = (IntPtr, IntPtr)
    EDSDK.EdsCreateEvfImageRef.restype = c_uint
    # C#原型:public extern static uint EdsCreateEvfImageRef(IntPtr inStreamRef, out IntPtr outEvfImageRef);
    return EDSDK.EdsCreateEvfImageRef(inStreamRef, byref(outEvfImageRef))


# *-----------------------------------------------------------------------------
#
#  Function:   EdsDownloadEvfImage
#  Description:
#		Downloads the live view image data set for a camera currently in live view mode.
#		Live view can be started by using the property ID:kEdsPropertyID_Evf_OutputDevice and
#		data:EdsOutputDevice_PC to call EdsSetPropertyData.
#		In addition to image data, information such as zoom, focus position, and histogram data
#		is included in the image data set. Image data is saved in a stream maintained by EdsEvfImageRef.
#		EdsGetPropertyData can be used to get information such as the zoom, focus position, etc.
#		Although the information of the zoom and focus position can be obtained from EdsEvfImageRef,
#		settings are applied to EdsCameraRef.
#
#  Parameters:
#      In:     inCameraRef - The Camera reference.
#      In:     inEvfImageRef - The EVFData reference.
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/

def EdsDownloadEvfImage(inStreamRef, outEvfImageRef):
    EDSDK.EdsDownloadEvfImage.argtype = (IntPtr, IntPtr)
    EDSDK.EdsDownloadEvfImage.restype = c_uint
    # C#原型:public extern static uint EdsDownloadEvfImage(IntPtr inCameraRef, IntPtr outEvfImageRef);
    return EDSDK.EdsDownloadEvfImage(inStreamRef, outEvfImageRef)


# *-----------------------------------------------------------------------------
#
#  Function:   EdsGetEvent
#
#  Description:
#      This function acquires an event.
#      In console application, please call this function regularly to acquire
#      the event from a camera.
#
#  Parameters:
#       In:    None
#      Out:    None
#
#  Returns:    Any of the sdk errors.
# -----------------------------------------------------------------------------*/


def EdsGetEvent():
    EDSDK.EdsInitializeSDK.restype = c_uint
    # C#原型:public extern static uint EdsGetEvent()
    return EDSDK.EdsGetEvent()
