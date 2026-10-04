"""Explicit env references or Windows Generic Credentials; no plaintext file."""
import ctypes,os,re,sys
from ctypes import wintypes
from uuid import uuid4

class CredentialError(ValueError):
    def __init__(self,message,win32_error=None):super().__init__(message);self.win32_error=win32_error

class CredentialVault:
    namespace='AutoAcousticsPro/LLM/'
    def _windows(self):
        if sys.platform!='win32':raise CredentialError('Windows 凭据存储不可用；可明确指定环境变量引用。')
        class FILETIME(ctypes.Structure):_fields_=[('low',wintypes.DWORD),('high',wintypes.DWORD)]
        class CREDENTIAL(ctypes.Structure):
            _fields_=[('Flags',wintypes.DWORD),('Type',wintypes.DWORD),('TargetName',wintypes.LPWSTR),('Comment',wintypes.LPWSTR),
                ('LastWritten',FILETIME),('CredentialBlobSize',wintypes.DWORD),('CredentialBlob',ctypes.POINTER(ctypes.c_ubyte)),
                ('Persist',wintypes.DWORD),('AttributeCount',wintypes.DWORD),('Attributes',ctypes.c_void_p),('TargetAlias',wintypes.LPWSTR),('UserName',wintypes.LPWSTR)]
        api=ctypes.WinDLL('Advapi32.dll',use_last_error=True)
        api.CredWriteW.argtypes=[ctypes.POINTER(CREDENTIAL),wintypes.DWORD];api.CredWriteW.restype=wintypes.BOOL
        api.CredReadW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.POINTER(ctypes.POINTER(CREDENTIAL))];api.CredReadW.restype=wintypes.BOOL
        api.CredDeleteW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD];api.CredDeleteW.restype=wintypes.BOOL
        api.CredFree.argtypes=[ctypes.c_void_p];api.CredFree.restype=None
        return api,CREDENTIAL
    def store(self,secret,name=None):
        if not isinstance(secret,str) or not secret:raise CredentialError('密钥不能为空。')
        encoded=secret.encode('utf-8')
        if len(encoded)>5120:raise CredentialError('密钥超过 Windows 凭据存储大小限制。')
        name=name or uuid4().hex
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',name):raise CredentialError('凭据名称须使用字母、数字、横线或下划线。')
        api,credential_type=self._windows();target=self.namespace+name
        blob=(ctypes.c_ubyte*len(encoded)).from_buffer_copy(encoded)
        credential=credential_type();credential.Type=1;credential.TargetName=target
        credential.CredentialBlobSize=len(encoded);credential.CredentialBlob=ctypes.cast(blob,ctypes.POINTER(ctypes.c_ubyte));credential.Persist=2;credential.UserName='AutoAcousticsPro'
        try:
            if not api.CredWriteW(ctypes.byref(credential),0):
                code=ctypes.get_last_error()
                raise CredentialError('当前受限 Windows 登录会话不可用凭据存储；请从正常桌面启动，或明确使用环境变量引用。' if code==1312 else 'Windows 未能安全保存所选凭据。',code)
            return 'wincred:'+target
        finally:ctypes.memset(blob,0,len(encoded))
    def get(self,reference):
        if not reference:return None
        if reference.startswith('env:'):
            name=reference[4:]
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}',name):raise CredentialError('环境变量引用名称无效。')
            value=os.environ.get(name)
            if not value:raise CredentialError('所选环境变量未提供密钥。')
            return value
        if not reference.startswith('wincred:'+self.namespace):raise CredentialError('凭据引用不属于本应用。')
        api,credential_type=self._windows();pointer=ctypes.POINTER(credential_type)()
        if not api.CredReadW(reference[8:],1,0,ctypes.byref(pointer)):raise CredentialError('所选 Windows 凭据不存在或不可读取。')
        try:
            credential=pointer.contents
            return ctypes.string_at(credential.CredentialBlob,credential.CredentialBlobSize).decode('utf-8')
        finally:
            if pointer.contents.CredentialBlobSize:ctypes.memset(pointer.contents.CredentialBlob,0,pointer.contents.CredentialBlobSize)
            api.CredFree(pointer)
    def delete(self,reference):
        if not reference:return False
        if reference.startswith('env:'):return False
        if not reference.startswith('wincred:'+self.namespace):raise CredentialError('凭据引用不属于本应用。')
        api,_=self._windows()
        if not api.CredDeleteW(reference[8:],1,0) and ctypes.get_last_error()!=1168:raise CredentialError('Windows 未能删除所选凭据。')
        return True
    def status(self,reference):
        if not reference:return 'not_configured'
        try:self.get(reference);return 'available'
        except CredentialError:return 'missing'
