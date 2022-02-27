# Python wrapper for EDSDK

# Device Info
from ctypes import *
int8_t = c_int8
int16_t = c_int16
int32_t = c_int32
int64_t = c_int64
uint8_t = c_uint8
uint16_t = c_uint16
uint32_t = c_uint32
uint64_t = c_uint64
int_least8_t = c_byte
int_least16_t = c_short
int_least32_t = c_int
int_least64_t = c_long
uint_least8_t = c_ubyte
uint_least16_t = c_ushort
uint_least32_t = c_uint
uint_least64_t = c_ulong
int_fast8_t = c_byte
int_fast16_t = c_long
int_fast32_t = c_long
int_fast64_t = c_long
uint_fast8_t = c_ubyte
uint_fast16_t = c_ulong
uint_fast32_t = c_ulong
uint_fast64_t = c_ulong
intptr_t = c_long
uintptr_t = c_ulong
intmax_t = c_long
uintmax_t = c_ulong
string = c_char_p


# GigE设备信息    \~english GigE device info
class _EdsDeviceInfo_(Structure):
    pass

_EdsDeviceInfo_._fields_ = [
    ('szPortName', string),                     # < \~chinese IP配置选项         \~english Ip config option
    ('szDeviceDescription', string),                    # < \~chinese 当前IP地址配置     \~english IP configuration:bit31-static bit30-dhcp bit29-lla
    ('DeviceSubType', c_uint),                       # < \~chinese 当前主机IP地址     \~english Current host Ip
    ('reserved', c_uint),               # < \~chinese 当前子网掩码       \~english curtent subnet mask
]
EdsDeviceInfo = _EdsDeviceInfo_

