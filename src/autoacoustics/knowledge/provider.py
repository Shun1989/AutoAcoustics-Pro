"""Minimal OpenAI-style HTTP protocol; no DB access and no prompt logging."""
from __future__ import annotations
from dataclasses import dataclass,asdict,field
import http.client,ipaddress,json,math,socket,ssl,threading,time
from urllib.parse import urlsplit
from typing import Mapping
from ..model import FrozenMap
from .credentials import CredentialVault,CredentialError

def loopback(host):
    if host=='localhost':return True
    try:return ipaddress.ip_address(host).is_loopback
    except ValueError:return False

def is_cancelled(cancel):
    if cancel is None:return False
    if hasattr(cancel,'cancelled'):return bool(cancel.cancelled)
    if hasattr(cancel,'is_set'):return bool(cancel.is_set())
    return bool(cancel()) if callable(cancel) else False

@dataclass(frozen=True)
class ProviderConfig:
    name:str
    base_url:str
    model:str
    timeout_s:float=30.
    credential_ref:str|None=None
    remote_send_confirmed:bool=False
    context_char_limit:int=12000
    output_token_limit:int|None=None
    def __post_init__(self):
        parsed=urlsplit(self.base_url)
        if not self.name.strip() or not self.model.strip():raise ValueError('服务名称和模型不能为空。')
        if not parsed.hostname or parsed.scheme not in ('https','http') or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('API 地址须为不带口令、查询参数或片段的 HTTPS 地址；本机 loopback 可用 HTTP。')
        if parsed.scheme=='http' and not loopback(parsed.hostname):raise ValueError('远端 API 必须使用 HTTPS。')
        if any(ord(char)<33 for char in self.base_url):raise ValueError('API 地址包含空白或控制字符。')
        if not math.isfinite(self.timeout_s) or not .1<=self.timeout_s<=300:raise ValueError('请求超时须为0.1–300秒。')
        if not 512<=self.context_char_limit<=12000:raise ValueError('输入字符上限须为512–12000。')
        if self.output_token_limit is not None and not 1<=self.output_token_limit<=32768:raise ValueError('输出 token 上限无效。')
        if self.credential_ref and not self.credential_ref.startswith(('wincred:AutoAcousticsPro/LLM/','env:')):raise ValueError('密钥须使用 Windows 凭据或明确环境变量引用。')
        # Force port validation here, before the user sends anything.
        try:parsed.port
        except ValueError as error:raise ValueError('API 端口无效。') from error
    @property
    def remote(self):return not loopback(urlsplit(self.base_url).hostname)
    def public_settings(self):return asdict(self)

@dataclass(frozen=True)
class ProviderReply:
    status:str
    text:str=''
    model:str=''
    message:str=''
    http_status:int|None=None
    usage:Mapping=field(default_factory=FrozenMap)
    def __post_init__(self):object.__setattr__(self,'usage',FrozenMap(self.usage))

class ProviderClient:
    def __init__(self,config,vault=None):self.config=config;self.vault=vault or CredentialVault();self._busy=threading.Lock()
    def test_connection(self,cancel=None):return self.complete([{'role':'user','content':'Reply briefly to confirm that this chat endpoint responds.'}],cancel=cancel)
    def complete(self,messages,cancel=None):
        if self.config.remote and not self.config.remote_send_confirmed:return ProviderReply('send_not_confirmed',message='请在发送片段预览中确认所选服务和实际发送操作。')
        if is_cancelled(cancel):return ProviderReply('cancelled',message='请求已取消。')
        if not isinstance(messages,(list,tuple)) or not messages:return ProviderReply('protocol_error',message='请求消息不能为空。')
        if any(not isinstance(item,dict) or item.get('role') not in ('system','user','assistant') or not isinstance(item.get('content'),str) for item in messages):return ProviderReply('protocol_error',message='只支持明确的文本聊天消息。')
        if sum(len(item['content']) for item in messages)>self.config.context_char_limit:return ProviderReply('context_error',message='请求内容超过已配置字符上限，请按资料边界缩减。')
        if not self._busy.acquire(blocking=False):return ProviderReply('busy',message='已有请求正在处理，请先取消或等待。')
        done=threading.Event();abort=threading.Event();state={'connection':None,'reply':None}
        def run():
            try:state['reply']=self._request(tuple(dict(item) for item in messages),abort,state)
            finally:self._busy.release();done.set()
        worker=threading.Thread(target=run,daemon=True,name='AutoAcoustics-HTTP');worker.start()
        deadline=time.monotonic()+self.config.timeout_s
        while not done.wait(.03):
            cancelled=is_cancelled(cancel)
            if cancelled or time.monotonic()>=deadline:
                abort.set();connection=state['connection']
                if connection is not None:
                    try:
                        if connection.sock is not None:connection.sock.shutdown(socket.SHUT_RDWR)
                        connection.close()
                    except OSError:pass
                return ProviderReply('cancelled' if cancelled else 'timeout',message='请求已取消。' if cancelled else 'API 请求超时。')
        if is_cancelled(cancel):return ProviderReply('cancelled',message='请求已取消。')
        return state['reply'] or ProviderReply('disconnected',message='API 请求没有可用响应。')
    def _request(self,messages,abort,state):
        connection=None;response=None
        try:
            parsed=urlsplit(self.config.base_url)
            cls=http.client.HTTPSConnection if parsed.scheme=='https' else http.client.HTTPConnection
            arguments={'timeout':self.config.timeout_s}
            if parsed.scheme=='https':arguments['context']=ssl.create_default_context()
            connection=cls(parsed.hostname,parsed.port,**arguments);state['connection']=connection
            headers={'Content-Type':'application/json','Accept':'application/json'}
            secret=self.vault.get(self.config.credential_ref)
            if secret:headers['Authorization']='Bearer '+secret
            payload={'model':self.config.model,'messages':messages,'stream':False}
            if self.config.output_token_limit is not None:payload['max_tokens']=self.config.output_token_limit
            body=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8')
            path=parsed.path.rstrip('/')
            endpoint=path if path.endswith('/chat/completions') else path+'/chat/completions'
            if abort.is_set():return ProviderReply('cancelled',message='请求已取消。')
            connection.request('POST',endpoint,body,headers);response=connection.getresponse()
            status=response.status
            if status!=200:
                name={401:'unauthorized',403:'unauthorized',429:'rate_limited'}.get(status,'service_error' if status>=500 else 'protocol_error')
                message={'unauthorized':'服务拒绝认证，请检查凭据引用与权限。','rate_limited':'服务限流，请稍后重试。','service_error':'服务端请求失败。','protocol_error':'端点返回不兼容状态或重定向；未转发凭据。'}[name]
                return ProviderReply(name,message=message,http_status=status)
            pieces=[];size=0
            while True:
                if abort.is_set():return ProviderReply('cancelled',message='请求已取消。')
                block=response.read(16384)
                if not block:break
                size+=len(block)
                if size>2*1024*1024:return ProviderReply('protocol_error',message='响应超过2 MiB上限。')
                pieces.append(block)
            data=json.loads(b''.join(pieces).decode('utf-8'))
            content=data['choices'][0]['message']['content'];model=data.get('model') or self.config.model
            if not isinstance(content,str) or not content.strip() or not isinstance(model,str):raise ValueError('non-text response')
            usage={key:value for key,value in data.get('usage',{}).items() if isinstance(value,(int,float)) and math.isfinite(value)} if isinstance(data.get('usage',{}),dict) else {}
            return ProviderReply('ok',content,model,http_status=200,usage=usage)
        except CredentialError:return ProviderReply('credential_missing',message='所选密钥引用不可用，请在连接设置中检查。')
        except (socket.timeout,TimeoutError):return ProviderReply('timeout',message='API 请求超时。')
        except (ValueError,KeyError,IndexError,TypeError,UnicodeError):return ProviderReply('protocol_error',message='响应不是兼容的文本聊天 JSON；未生成模型答案。')
        except (OSError,http.client.HTTPException):return ProviderReply('cancelled' if abort.is_set() else 'disconnected',message='请求已取消。' if abort.is_set() else '无法连接 API 或连接中断。')
        finally:
            if response is not None:response.close()
            if connection is not None:connection.close()

def save_provider(store,config):store.save_setting('provider',config.public_settings())
def load_provider(store):
    value=store.load_setting('provider')
    return ProviderConfig(**value) if value else None
