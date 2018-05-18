
import struct

GADGET_MESSAGE_INT_ID_MIN                             =    0
GADGET_MESSAGE_CONFIG_FILE                            =    1
GADGET_MESSAGE_CONFIG_SCRIPT                          =    2
GADGET_MESSAGE_PARAMETER_SCRIPT                       =    3
GADGET_MESSAGE_CLOSE                                  =    4
GADGET_MESSAGE_TEXT                                   =    5
GADGET_MESSAGE_INT_ID_MAX                             =  999
GADGET_MESSAGE_EXT_ID_MIN                             = 1000
GADGET_MESSAGE_ACQUISITION                            = 1001 # DEPRECATED
GADGET_MESSAGE_NEW_MEASUREMENT                        = 1002 # DEPRECATED
GADGET_MESSAGE_END_OF_SCAN                            = 1003 # DEPRECATED
GADGET_MESSAGE_IMAGE_CPLX_FLOAT                       = 1004 # DEPRECATED
GADGET_MESSAGE_IMAGE_REAL_FLOAT                       = 1005 # DEPRECATED
GADGET_MESSAGE_IMAGE_REAL_USHORT                      = 1006 # DEPRECATED
GADGET_MESSAGE_EMPTY                                  = 1007 # DEPRECATED
GADGET_MESSAGE_ISMRMRD_ACQUISITION                    = 1008
GADGET_MESSAGE_ISMRMRD_IMAGE_CPLX_FLOAT               = 1009 # DEPRECATED
GADGET_MESSAGE_ISMRMRD_IMAGE_REAL_FLOAT               = 1010 # DEPRECATED
GADGET_MESSAGE_ISMRMRD_IMAGE_REAL_USHORT              = 1011 # DEPRECATED
GADGET_MESSAGE_DICOM                                  = 1012 # DEPRECATED
GADGET_MESSAGE_CLOUD_JOB                              = 1013
GADGET_MESSAGE_GADGETCLOUD_JOB                        = 1014
GADGET_MESSAGE_ISMRMRD_IMAGEWITHATTRIB_CPLX_FLOAT     = 1015 # DEPRECATED
GADGET_MESSAGE_ISMRMRD_IMAGEWITHATTRIB_REAL_FLOAT     = 1016 # DEPRECATED
GADGET_MESSAGE_ISMRMRD_IMAGEWITHATTRIB_REAL_USHORT    = 1017 # DEPRECATED
GADGET_MESSAGE_DICOM_WITHNAME                         = 1018
GADGET_MESSAGE_DEPENDENCY_QUERY                       = 1019
GADGET_MESSAGE_ISMRMRD_IMAGE_REAL_SHORT               = 1020 # DEPRECATED
GADGET_MESSAGE_ISMRMRD_IMAGEWITHATTRIB_REAL_SHORT     = 1021 # DEPRECATED
GADGET_MESSAGE_ISMRMRD_IMAGE                          = 1022
GADGET_MESSAGE_RECONDATA                              = 1023
GADGET_MESSAGE_ISMRMRD_WAVEFORM                       = 1026
GADGET_MESSAGE_EXT_ID_MAX                             = 4096

GadgetMessageLength = struct.Struct('<I')
SIZEOF_GADGET_MESSAGE_LENGTH = len(GadgetMessageLength.pack(0))

GadgetMessageIdentifier = struct.Struct('<H')
SIZEOF_GADGET_MESSAGE_IDENTIFIER = len(GadgetMessageIdentifier.pack(0))

GadgetMessageConfigurationFile = struct.Struct('<1024s')
SIZEOF_GADGET_MESSAGE_CONFIGURATION_FILE = len(GadgetMessageConfigurationFile.pack(b''))

