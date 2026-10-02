"""Local HTTP integration tests; no dataset server or GPU access."""
import io
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
import zipfile
import mp_download_utils as downloader


def payload():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as archive:
        archive.writestr('example.jpg', b'x' * 2000000)
    return buf.getvalue()


class DownloadTests(unittest.TestCase):
    def exercise(self, mode):
        body = payload()
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                byte_range = self.headers.get('Range')
                requests.append(byte_range)
                if len(requests) == 1 and mode in ('resume','ignore'):
                    self.send_response(200)
                    self.send_header('Content-Length',str(len(body)))
                    self.send_header('ETag','"fixed"')
                    self.end_headers()
                    self.wfile.write(body[:1100000])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                offset = int(byte_range.split('=')[1].split('-')[0]) if byte_range and mode == 'resume' else 0
                self.send_response(206 if offset else 200)
                self.send_header('Content-Length',str(len(body)-offset))
                self.send_header('ETag','"fixed"')
                if offset:
                    self.send_header('Content-Range',f'bytes {offset}-{len(body)-1}/{len(body)}')
                self.end_headers()
                if mode == 'crc' and len(requests) == 1:
                    bad = bytearray(body);bad[100]=ord('y');self.wfile.write(bad)
                else:
                    self.wfile.write(body[offset:])
        server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        original = dict(downloader.OPTIONS)
        try:
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory)/'data.zip'
                url = f'http://127.0.0.1:{server.server_port}/data.zip'
                downloader.OPTIONS.update(retries=0,timeout=5,retry_delay=0,verify_existing=True)
                if mode in ('resume','ignore'):
                    with self.assertRaises(downloader.DownloadError):
                        downloader.download_file(url,target)
                    self.assertFalse(target.exists())
                    self.assertGreater(Path(str(target)+'.part').stat().st_size,0)
                    downloader.OPTIONS['retries']=2
                else:
                    downloader.OPTIONS['retries']=2
                downloader.download_file(url,target)
                self.assertEqual(target.read_bytes(),body)
                self.assertFalse(Path(str(target)+'.part').exists())
                count=len(requests)
                downloader.download_file(url,target)
                self.assertEqual(len(requests),count)
                if mode in ('resume','ignore'):
                    self.assertEqual(requests[1],'bytes=1100000-')
                if mode == 'crc':
                    self.assertEqual(len(requests),2)
        finally:
            downloader.OPTIONS.update(original)
            server.shutdown();server.server_close();thread.join()

    def test_resume_across_invocations(self):
        self.exercise('resume')

    def test_range_ignored_restarts_without_appending(self):
        self.exercise('ignore')

    def test_bad_crc_retries_and_existing_skips(self):
        self.exercise('crc')


if __name__ == '__main__':
    unittest.main()
