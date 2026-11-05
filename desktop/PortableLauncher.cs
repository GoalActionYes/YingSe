using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Windows.Forms;

[assembly: AssemblyTitle("映色 · 小组体验版")]
[assembly: AssemblyVersion("0.2.2.0")]
internal static class PortableLauncher
{
    static string root = AppDomain.CurrentDomain.BaseDirectory;
    static string ready = Path.Combine(root, @"desktop\session.txt");
    static string ReadUrl()
    {
        if (!File.Exists(ready)) return null;
        string[] lines = File.ReadAllLines(ready);
        Uri uri;
        if (lines.Length < 2 || !Uri.TryCreate(lines[0], UriKind.Absolute, out uri) || uri.Host != "127.0.0.1" || uri.Scheme != "http") return null;
        try {
            var request = (HttpWebRequest)WebRequest.Create(lines[0] + "/desktop/health");
            request.Proxy = null; request.Timeout = 1000;
            using (var response = request.GetResponse())
            using (var reader = new StreamReader(response.GetResponseStream()))
                return reader.ReadToEnd() == lines[1] ? lines[0] : null;
        } catch { return null; }
    }
    static string EdgePath()
    {
        foreach (string folder in new[] { Environment.GetEnvironmentVariable("ProgramFiles(x86)"), Environment.GetEnvironmentVariable("ProgramFiles"), Environment.GetEnvironmentVariable("LOCALAPPDATA") }) {
            if (String.IsNullOrEmpty(folder)) continue;
            string candidate = Path.Combine(folder, @"Microsoft\Edge\Application\msedge.exe");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }
    [STAThread]
    static int Main(string[] args)
    {
        bool check = args.Length > 0 && args[0] == "--check";
        bool stop = args.Length > 0 && args[0] == "--stop";
        string key;
        using (var hash = SHA256.Create()) key = BitConverter.ToString(hash.ComputeHash(Encoding.UTF8.GetBytes(root.ToLowerInvariant()))).Replace("-", "").Substring(0,24);
        bool created;
        using (var gate = new Mutex(true, @"Local\YingSePortable_"+key, out created)) {
            if (!created) return 0;
            try {
                string python = Path.Combine(root, @"runtime\python.exe");
                string boot = Path.Combine(root, @"desktop\portable_boot.py");
                if (!File.Exists(python) || !File.Exists(boot)) throw new Exception("请先完整解压体验包，再运行映色.exe。不要在压缩包内运行，也不要单独移动EXE。");
                Directory.CreateDirectory(Path.Combine(root,"desktop"));
                string url = ReadUrl();
                if (stop) {
                    if (url != null) {
                        var req = (HttpWebRequest)WebRequest.Create(url+"/desktop/stop");
                        req.Proxy=null; req.Timeout=5000; req.Method="POST"; req.ContentLength=0;
                        req.Headers["X-Yingse-Token"] = File.ReadAllLines(ready)[1];
                        try { using (var response=req.GetResponse()) {} }
                        catch (WebException e) {
                            if (e.Response != null && ((HttpWebResponse)e.Response).StatusCode==HttpStatusCode.Conflict)
                                throw new Exception("视频仍在导出，请先等待完成或在页面取消，再关闭后台。");
                            throw;
                        }
                    }
                    MessageBox.Show("后台服务已关闭。可以关闭映色窗口。", "映色");
                    return 0;
                }
                if (url == null) {
                    var start = new ProcessStartInfo(python, "-I \""+boot+"\"") {
                        WorkingDirectory=root, UseShellExecute=false, CreateNoWindow=true
                    };
                    // Python -I and python312._pth isolate imports; bootstrap clears FFmpeg overrides.
                    // Do not enumerate the inherited environment: Windows may contain Path/PATH aliases.
                    using (var process=Process.Start(start)) {
                        for (int i=0; i<300; i++) {
                            Thread.Sleep(200);
                            url=ReadUrl();
                            if(url!=null) break;
                            if(process.HasExited) throw new Exception("服务启动失败，请将desktop中的启动日志发给组长。");
                        }
                    }
                    if(url==null) throw new Exception("启动超时。请确认已完整解压、目录可写，并检查desktop中的启动日志。");
                }
                File.WriteAllText(Path.Combine(root,@"desktop\last-check.txt"), DateTime.Now.ToString("s")+"\r\n"+url+"\r\nBundled Python ready\r\n", Encoding.UTF8);
                if(check) return 0;
                string edge=EdgePath();
                if(edge!=null) {
                    Process.Start(new ProcessStartInfo(edge, "--app="+url+"/?desktop=portable --user-data-dir=\""+Path.Combine(root,@"desktop\profile")+"\" --no-first-run --no-default-browser-check --disable-extensions --window-size=1440,960") { UseShellExecute=true });
                } else {
                    Process.Start(new ProcessStartInfo(url+"/?desktop=portable") { UseShellExecute=true });
                }
                return 0;
            } catch(Exception error) {
                try { File.WriteAllText(Path.Combine(root,@"desktop\launcher-error.txt"),error.ToString(),Encoding.UTF8); } catch {}
                if(!check) MessageBox.Show(error.Message,"映色 · 启动提示",MessageBoxButtons.OK,MessageBoxIcon.Error);
                return 1;
            }
        }
    }
}
