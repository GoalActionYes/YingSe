"""Development entry points; product algorithms are imported from project source."""
from pathlib import Path
import argparse,json,os,runpy,shutil,subprocess,sys,unittest,webbrowser
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr,'reconfigure'):sys.stderr.reconfigure(encoding='utf-8')
def main():
    parser=argparse.ArgumentParser(description='YingSe source development tools')
    parser.add_argument('command',choices=('serve','test','test-ui','build','env'))
    parser.add_argument('--port',type=int,default=0)
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--ready-file',type=Path)
    args=parser.parse_args();os.chdir(ROOT)
    if args.command=='serve':
        from reference_color_app.server import make_server
        server=make_server(ROOT/'dev-data',port=args.port)
        url=f'http://127.0.0.1:{server.server_port}'
        if args.ready_file:
            args.ready_file.parent.mkdir(parents=True,exist_ok=True)
            args.ready_file.write_text(json.dumps({'url':url,'token':server.token,'pid':os.getpid()}),'utf-8')
        print(url,flush=True)
        if not args.no_browser:webbrowser.open(url)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:server.server_close()
        return 0
    if args.command=='test':
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/reference_color'),pattern='test_*.py')
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.command=='test-ui':
        node=shutil.which('node')
        if not node:
            print('Node.js not found. UI tests require Node.js; editing/running the app does not.');return 2
        return subprocess.run([node,'--test','tests/reference_color/test_ui.cjs']).returncode
    if args.command=='build':
        sys.argv=[str(ROOT/'desktop/reference_color/build.py')]
        runpy.run_path(sys.argv[0],run_name='__main__');return 0
    import numpy,PIL
    from reference_color_app.media import ffmpeg_path
    print(json.dumps({'python':sys.version,'numpy':numpy.__version__,'pillow':PIL.__version__,'ffmpeg':str(ffmpeg_path()),'project':str(ROOT)},ensure_ascii=False,indent=2));return 0
if __name__=='__main__':sys.exit(main())
