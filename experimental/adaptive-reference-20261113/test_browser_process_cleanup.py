"""Real owned Node descendants must not survive verifier success or timeout."""
import ctypes
import json
import os
import shutil
import subprocess
import unittest

from verify_browser_workbench import run_owned_browser


@unittest.skipUnless(os.name == 'nt' and shutil.which('node'), 'Windows and existing Node required')
class BrowserProcessCleanupTests(unittest.TestCase):
    def assert_stopped(self, pid):
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
        if not handle:
            self.assertEqual(ctypes.get_last_error(), 87)  # no such process
            return
        try:
            self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
        finally:
            kernel.CloseHandle(handle)

    def command(self, hang):
        script = """
process.stdin.once('data',()=>{
  process.stdin.pause();
  const child=require('node:child_process').spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{stdio:'ignore'});
  child.unref();console.log(JSON.stringify({parent:process.pid,child:child.pid}));
  if(HANG)setInterval(()=>{},1000);
});
""".replace('HANG', 'true' if hang else 'false')
        return [shutil.which('node'), '-e', script]

    def test_timeout_kills_owned_descendants_and_preserves_stdout(self):
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            run_owned_browser(self.command(True), timeout=2)
        pids = json.loads(caught.exception.output)
        self.assert_stopped(pids['parent'])
        self.assert_stopped(pids['child'])

    def test_success_also_closes_any_owned_descendants(self):
        result = run_owned_browser(self.command(False), timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        pids = json.loads(result.stdout)
        self.assert_stopped(pids['parent'])
        self.assert_stopped(pids['child'])


if __name__ == '__main__':
    unittest.main()
