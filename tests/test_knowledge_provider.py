import importlib,json,threading,time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import pytest

@contextmanager
def server(status=200,delay=0,malformed=False):
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((self.path,body,self.headers.get('Authorization')))
            if delay:time.sleep(delay)
            self.send_response(status);self.send_header('Content-Type','application/json');self.end_headers()
            try:self.wfile.write(b'not-json' if malformed else json.dumps({'model':'fixture-model','choices':[{'message':{'content':'protocol fixture response'}}]}).encode())
            except (BrokenPipeError,ConnectionResetError,OSError):pass
        def log_message(self,*args):pass
    instance=ThreadingHTTPServer(('127.0.0.1',0),Handler);instance.daemon_threads=True
    thread=threading.Thread(target=instance.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{instance.server_port}/v1',requests
    finally:instance.shutdown();instance.server_close();thread.join(timeout=2)

def provider():return importlib.import_module('autoacoustics.knowledge.provider')

def test_local_http_request_actual_model_and_explicit_env_credential(monkeypatch):
    p=provider();monkeypatch.setenv('AAC_PROTOCOL_TEST_KEY','protocol-fixture-only')
    with server() as (url,requests):
        config=p.ProviderConfig('Protocol fixture',url,'asked-model',credential_ref='env:AAC_PROTOCOL_TEST_KEY')
        result=p.ProviderClient(config).complete([{'role':'user','content':'protocol test'}])
        assert result.status=='ok' and result.model=='fixture-model'
        assert requests[0][0]=='/v1/chat/completions'
        assert requests[0][1]['model']=='asked-model'
        assert requests[0][2]=='Bearer protocol-fixture-only'
        assert 'protocol-fixture-only' not in str(config.public_settings())

@pytest.mark.parametrize('status,expected',[(401,'unauthorized'),(429,'rate_limited'),(500,'service_error')])
def test_protocol_errors_do_not_make_fake_answers(status,expected):
    p=provider()
    with server(status=status) as (url,_):
        result=p.ProviderClient(p.ProviderConfig('Fixture',url,'fixture')).test_connection()
        assert result.status==expected and not result.text

def test_timeout_bad_json_cancel_and_duplicate_clicks():
    p=provider()
    with server(malformed=True) as (url,_):assert p.ProviderClient(p.ProviderConfig('Fixture',url,'fixture')).test_connection().status=='protocol_error'
    with server(delay=.3) as (url,_):assert p.ProviderClient(p.ProviderConfig('Fixture',url,'fixture',timeout_s=.1)).test_connection().status=='timeout'
    with server(delay=2) as (url,requests):
        client=p.ProviderClient(p.ProviderConfig('Fixture',url,'fixture',timeout_s=3));cancel=threading.Event();output=[]
        worker=threading.Thread(target=lambda:output.append(client.complete([{'role':'user','content':'test'}],cancel=cancel)));worker.start()
        deadline=time.monotonic()+2
        while not requests and time.monotonic()<deadline:time.sleep(.01)
        assert client.test_connection().status=='busy'
        start=time.monotonic();cancel.set();worker.join(timeout=1)
        assert not worker.is_alive() and time.monotonic()-start<1 and output[0].status=='cancelled'

def test_remote_https_and_consent_rules_are_actual_configuration_validation():
    p=provider()
    with pytest.raises(ValueError):p.ProviderConfig('Remote','http://example.com/v1','model')
    with pytest.raises(ValueError):p.ProviderConfig('Remote','https://user:password@example.com/v1','model')
    result=p.ProviderClient(p.ProviderConfig('Remote','https://example.com/v1','model')).test_connection()
    assert result.status=='send_not_confirmed'

def test_windows_credential_roundtrip_delete_and_environment_reference(monkeypatch):
    from autoacoustics.knowledge.credentials import CredentialVault,CredentialError
    import sys,uuid
    vault=CredentialVault();monkeypatch.setenv('AAC_PROTOCOL_TEST_SECRET','synthetic-test-value')
    assert vault.get('env:AAC_PROTOCOL_TEST_SECRET')=='synthetic-test-value'
    if sys.platform!='win32':pytest.skip('Windows Credential Manager integration')
    try:reference=vault.store('synthetic-credential-test',name='protocol-test-'+uuid.uuid4().hex)
    except CredentialError as error:
        if error.win32_error==1312:pytest.skip('工具受限登录会话 Win32 1312；正常桌面会话需另跑合成凭据 roundtrip。')
        raise
    try:assert vault.get(reference)=='synthetic-credential-test'
    finally:vault.delete(reference)
    assert vault.status(reference)=='missing'
