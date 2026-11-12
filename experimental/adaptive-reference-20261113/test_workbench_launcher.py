import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from launch_workbench import probe_session,startup_lock


class LauncherChecks(unittest.TestCase):
    def test_startup_lock_excludes_other_handles_and_releases(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with startup_lock(root,timeout=.1):
                with self.assertRaises(TimeoutError):
                    with startup_lock(root,timeout=.1):pass
            with startup_lock(root,timeout=.1):pass

    def test_invalid_remote_or_wrong_instance_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            record=dict(url='https://example.com',pid=123,token='test')
            (root/'session.json').write_text(json.dumps(record))
            with patch('launch_workbench.urlopen') as fetch:
                self.assertIsNone(probe_session(root));fetch.assert_not_called()
            record['url']='http://127.0.0.1:12345'
            (root/'session.json').write_text(json.dumps(record))
            response=unittest.mock.MagicMock()
            response.__enter__.return_value.read.return_value=json.dumps(dict(
                app='reference-color-app',pid=123,instance=hashlib.sha256(str(root).encode()).hexdigest())).encode()
            with patch('launch_workbench.urlopen',return_value=response):
                self.assertIsNone(probe_session(root))


if __name__=='__main__':unittest.main()
