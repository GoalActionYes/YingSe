import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import launch_spatial_workbench as launcher
import hashlib


class Response:
    def __init__(self,value):self.value=value
    def __enter__(self):return io.StringIO(json.dumps(self.value))
    def __exit__(self,*args):pass


class SpatialLauncherTests(unittest.TestCase):
    def test_only_matching_service_is_reused(self):
        record=dict(url='http://127.0.0.1:12345',pid=80,token='test')
        health=dict(app='spatial-reference-workbench',pid=80,instance=hashlib.sha256(str(launcher.DATA.resolve()).encode()).hexdigest(),build='test-build')
        with patch.object(launcher,'saved_session',return_value=record):
            with patch.object(launcher,'urlopen',return_value=Response(health)):self.assertEqual(launcher.probe()['build'],'test-build')
            for key,value in [('app','unrelated'),('pid',81),('instance','wrong-root')]:
                with patch.object(launcher,'urlopen',return_value=Response(dict(health,**{key:value}))):self.assertIsNone(launcher.probe())

    def test_active_export_stop_error_is_readable(self):
        error=HTTPError('http://127.0.0.1:12345/api/stop',409,'Conflict',{},io.BytesIO(json.dumps({'error':'仍有导出任务'}).encode()))
        with patch.object(launcher,'urlopen',side_effect=error):
            with self.assertRaisesRegex(RuntimeError,'仍有导出任务'):launcher.stop_owned(dict(url='http://127.0.0.1:12345',token='test'))


if __name__=='__main__':unittest.main()
