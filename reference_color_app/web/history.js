'use strict';
function createHistory(initial,limit=50){
  const copy=value=>JSON.parse(JSON.stringify(value));
  let entries=[copy(initial)],cursor=0,group=false,groupRecorded=false;
  const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
  const end=()=>{group=false;groupRecorded=false;};
  return {
    begin(){end();group=true;},end,
    record(value){
      if(same(entries[cursor],value))return;
      entries=entries.slice(0,cursor+1);
      if(group&&groupRecorded){
        entries[cursor]=copy(value);
        if(cursor>0&&same(entries[cursor],entries[cursor-1])){entries.pop();cursor--;groupRecorded=false;}
      }
      else{entries.push(copy(value));cursor++;groupRecorded=group;}
      while(entries.length>limit+1){entries.shift();cursor--;}
    },
    canUndo:()=>cursor>0,canRedo:()=>cursor<entries.length-1,
    undo(){end();if(cursor>0)cursor--;return copy(entries[cursor]);},
    redo(){end();if(cursor<entries.length-1)cursor++;return copy(entries[cursor]);},
    reset(value){entries=[copy(value)];cursor=0;end();}
  };
}
if(typeof module!=='undefined')module.exports={createHistory};
