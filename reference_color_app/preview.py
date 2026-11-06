"""Bounded decoded-frame reuse. Caller must verify owned media before lookup."""
from collections import OrderedDict
import threading


class PreviewFrames:
    def __init__(self,max_bytes=32*1024*1024,max_entries=2):
        self.max_bytes=max_bytes
        self.max_entries=max_entries
        self.bytes_used=0
        self._items=OrderedDict()
        self._lock=threading.RLock()

    def get(self,key,load,encode):
        with self._lock:
            if key in self._items:
                value,size=self._items.pop(key)
                self._items[key]=(value,size)
                return value
            rgb,metadata=load()
            before=encode(rgb)
            rgb.setflags(write=False)
            value=(rgb,metadata,before)
            size=rgb.nbytes+len(before)
            if size<=self.max_bytes:
                while self._items and (self.bytes_used+size>self.max_bytes or len(self._items)>=self.max_entries):
                    _,(_,old_size)=self._items.popitem(last=False)
                    self.bytes_used-=old_size
                if self.max_entries>0:
                    self._items[key]=(value,size)
                    self.bytes_used+=size
            return value
