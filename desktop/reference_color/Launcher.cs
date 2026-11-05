using System;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Windows.Forms;
[assembly: System.Reflection.AssemblyTitle("映色参考调色")]
[assembly: System.Reflection.AssemblyVersion("1.0.0.0")]
internal static class ReferenceColorLauncher {
    [STAThread] static int Main(string[] args) {
        string root=AppDomain.CurrentDomain.BaseDirectory;
        bool stop=Array.IndexOf(args,"--stop")>=0, check=Array.IndexOf(args,"--check")>=0;
        string key;
        using(var hash=SHA256.Create()) key=BitConverter.ToString(hash.ComputeHash(Encoding.UTF8.GetBytes(root.ToLowerInvariant()))).Replace("-", "");
        using(var gate=new Mutex(false,"Local\\ReferenceColorPortable_"+key)) {
            bool held=false;
            try {
                try { held=gate.WaitOne(45000); } catch(AbandonedMutexException) { held=true; }
                if(!held) throw new Exception("启动器正忙，请稍后重试。");
                var start=new ProcessStartInfo(Path.Combine(root,@"runtime\python.exe"), "-I -B \""+Path.Combine(root,@"desktop\reference_color\boot.py")+"\" "+(stop?"--stop":"--check")) {UseShellExecute=false,CreateNoWindow=true,WorkingDirectory=root};
                using(var process=Process.Start(start)) {
                    if(!process.WaitForExit(45000)) throw new Exception("启动超时，请查看 data 中的日志。");
                    if(process.ExitCode!=0) throw new Exception("启动或关闭失败，请检查目录可写、完整解压，并查看 data/launcher-error.txt 和 server.log。导出期间请先完成或取消导出。");
                }
                if(stop || check) return 0;
                string url=File.ReadAllText(Path.Combine(root,@"data\launch-url.txt"));
                string edge=null;
                foreach(string folder in new[]{Environment.GetEnvironmentVariable("ProgramFiles(x86)"),Environment.GetEnvironmentVariable("ProgramFiles"),Environment.GetEnvironmentVariable("LOCALAPPDATA")}) {
                    if(String.IsNullOrEmpty(folder)) continue;
                    string candidate=Path.Combine(folder,@"Microsoft\Edge\Application\msedge.exe");
                    if(File.Exists(candidate)) {edge=candidate;break;}
                }
                if(edge!=null) Process.Start(new ProcessStartInfo(edge,"--app="+url+" --user-data-dir=\""+Path.Combine(root,@"data\browser-profile")+"\" --no-first-run --no-default-browser-check --disable-extensions --window-size=1440,960") {UseShellExecute=true});
                else Process.Start(new ProcessStartInfo(url){UseShellExecute=true});
                return 0;
            } catch(Exception exc) {
                try {Directory.CreateDirectory(Path.Combine(root,"data"));File.WriteAllText(Path.Combine(root,@"data\native-launcher-error.txt"),exc.ToString());} catch {}
                if(!check && !stop) MessageBox.Show(exc.Message,"映色参考调色",MessageBoxButtons.OK,MessageBoxIcon.Error);
                return 1;
            } finally {if(held) gate.ReleaseMutex();}
        }
    }
}
