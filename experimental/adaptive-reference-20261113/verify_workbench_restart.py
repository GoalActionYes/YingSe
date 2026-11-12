"""Verify owned service restart/reuse and persisted model project, without UI input."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
from launch_workbench import DATA,ROOT,pid_alive,probe_session

before=probe_session(DATA)
assert before is not None
with urlopen(before['url']+'/api/projects/latest',timeout=10) as response:old=json.loads(response.read())
launcher=Path(__file__).with_name('launch_workbench.py')
subprocess.run([sys.executable,str(launcher),'--stop'],check=True,timeout=15)
deadline=time.monotonic()+10
while pid_alive(before['pid']) and time.monotonic()<deadline:time.sleep(.1)
assert not pid_alive(before['pid']),'Stopped service process still live; do not restart duplicates'
processes=[subprocess.Popen([sys.executable,str(launcher),'--check'],stdout=subprocess.PIPE,stderr=subprocess.PIPE) for _ in range(2)]
urls=[]
for process in processes:
    output,error=process.communicate(timeout=60)
    assert process.returncode==0,error
    urls.append(output.decode().strip())
assert urls[0]==urls[1]
after=probe_session(DATA);assert after is not None and after['pid']!=before['pid']
with urlopen(after['url']+'/api/projects/latest',timeout=10) as response:restored=json.loads(response.read())
assert restored['mapping_id']==old['mapping_id'] and restored['engine']=='model'
subprocess.run([sys.executable,str(launcher),'--check'],check=True,timeout=15)
reused=probe_session(DATA);assert reused['pid']==after['pid'] and reused['url']==after['url']
server_count=None
if os.name=='nt':
    literal="'"+str(DATA).replace("'","''")+"'"
    # Windows venv creates a redirector plus its base interpreter, both with
    # this command line. Count listening owners rather than Python wrappers.
    command="$taskPids=@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match '[\\\\/]workbench\\.py(?:\"|\\s|$)' -and $_.CommandLine.Contains("+literal+") } | Select-Object -ExpandProperty ProcessId); (@(Get-NetTCPConnection -State Listen | Where-Object { $taskPids -contains $_.OwningProcess } | Select-Object -ExpandProperty OwningProcess -Unique)).Count"
    result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True,timeout=15,
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    server_count=int(result.stdout.strip());assert server_count==1,server_count
report=dict(restart_verified=True,reuse_verified=True,project_mapping_preserved=True,engine=restored['engine'],
            concurrent_start_same_url=True,owned_listening_server_count=server_count,
            url=after['url'],scope='Owned service and HTTP project restore, not browser visual confirmation')
out=ROOT/'outputs/adaptive-reference-20261113/workbench-verification/restart.json'
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
