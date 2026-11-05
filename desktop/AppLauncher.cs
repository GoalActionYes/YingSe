using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;

[assembly: AssemblyTitle("映色 · 本机桌面演示版")]
[assembly: AssemblyDescription("Local desktop launcher for the reference-driven color grading prototype")]
[assembly: AssemblyVersion("0.2.1.0")]

internal static class AppLauncher
{
    [STAThread]
    private static int Main(string[] args)
    {
        bool created;
        using (var gate = new Mutex(true, @"Local\YingSeDesktopStartup", out created))
        {
            if (!created) return 0;
            try
            {
                string root = AppDomain.CurrentDomain.BaseDirectory;
                string script = Path.Combine(root, "StartDesktop.ps1");
                if (!File.Exists(script)) throw new Exception("请把映色.exe放回完整项目文件夹，与StartDesktop.ps1放在一起。此版不能单独复制EXE运行。");
                bool check = args.Length > 0 && args[0] == "--check";
                string userHome = Environment.GetEnvironmentVariable("USERPROFILE") ?? Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                string shell = Path.Combine(userHome, @".cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe");
                if (!File.Exists(shell)) shell = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), @"PowerShell\7\pwsh.exe");
                if (!File.Exists(shell)) throw new Exception("此本机演示版需要PowerShell 7，请使用原来的启动演示.cmd，或参考桌面版使用说明。");
                var start = new ProcessStartInfo {
                    FileName = shell,
                    Arguments = "-NoProfile -File \"" + script + "\"" + (check ? " -CheckOnly" : ""),
                    WorkingDirectory = root,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardError = true
                };
                using (var process = Process.Start(start))
                {
                    string shellError = process.StandardError.ReadToEnd();
                    process.WaitForExit();
                    if (process.ExitCode != 0)
                    {
                        string log = Path.Combine(root, @"desktop\last-start-error.txt");
                        string detail = !String.IsNullOrWhiteSpace(shellError) ? shellError : (File.Exists(log) ? File.ReadAllText(log) : "启动进程没有完成。");
                        throw new Exception("映色启动失败。可尝试项目中的“启动演示.cmd”。\n\n" + detail);
                    }
                }
                return 0;
            }
            catch (Exception error)
            {
                File.WriteAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"desktop\launcher-error.txt"), error.ToString());
                if (!(args.Length > 0 && args[0] == "--check"))
                    MessageBox.Show(error.Message, "映色 · 启动提示", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }
        }
    }
}
