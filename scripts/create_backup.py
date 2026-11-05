"""Create bounded-size release archives of this project with SHA256 manifests.

No credentials, home folders, environments, caches, or unrelated projects are read.
This script prepares files only; it never uploads or deletes project artifacts.
"""
import argparse,gzip,hashlib,io,json,os,tarfile
from datetime import datetime,timezone
from pathlib import Path


class SplitWriter(io.RawIOBase):
    def __init__(self,folder,name,limit):
        super().__init__();self.folder=folder;self.name=name;self.limit=limit
        self.stream=None;self.part_size=0;self.position=0;self.parts=[];self.digest=None
    def writable(self):return True
    def tell(self):return self.position
    def _finish(self):
        if self.stream:
            self.stream.flush();os.fsync(self.stream.fileno());self.stream.close()
            self.parts.append({'file':self.path.name,'bytes':self.part_size,'sha256':self.digest.hexdigest()})
            self.stream=None
    def write(self,data):
        remaining=memoryview(data);total=len(remaining)
        while remaining:
            if self.stream is None:
                self.path=self.folder/f'{self.name}.tar.gz.part{len(self.parts)+1:03d}'
                self.stream=self.path.open('xb');self.part_size=0;self.digest=hashlib.sha256()
            n=min(len(remaining),self.limit-self.part_size);block=remaining[:n]
            self.stream.write(block);self.digest.update(block);self.part_size+=n;self.position+=n
            remaining=remaining[n:]
            if self.part_size==self.limit:self._finish()
        return total
    def close(self):
        if not self.closed:self._finish()
        super().close()


class HashedReader:
    def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256();self.size=0
    def read(self,n=-1):
        block=self.stream.read(n);self.digest.update(block);self.size+=len(block);return block


def archive(root,out,name,paths,part_bytes):
    inventory=[];writer=SplitWriter(out,name,part_bytes)
    with writer,gzip.GzipFile(fileobj=writer,mode='wb',compresslevel=1,mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|',dereference=False) as tf:
        for path in paths:
            rel=path.relative_to(root).as_posix();info=tf.gettarinfo(str(path),arcname=rel)
            if info.issym():
                resolved=path.resolve();resolved.relative_to(root)
                tf.addfile(info);inventory.append({'path':rel,'type':'symlink','target':info.linkname})
            elif info.isfile():
                before=path.stat()
                with path.open('rb') as stream:
                    reader=HashedReader(stream);tf.addfile(info,reader)
                after=path.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or reader.size!=before.st_size:
                    raise RuntimeError(f'File changed during backup: {rel}')
                inventory.append({'path':rel,'bytes':reader.size,'sha256':reader.digest.hexdigest()})
            elif info.isdir():tf.addfile(info)
            else:raise RuntimeError(f'Unsupported artifact type: {rel}')
    report={'archive':name,'parts':writer.parts,'sources':inventory,'source_bytes':sum(x.get('bytes',0) for x in inventory)}
    (out/f'{name}.manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'archive':name,'source_files':len(inventory),'source_bytes':report['source_bytes'],'archive_bytes':sum(x['bytes'] for x in writer.parts),'parts':len(writer.parts)}),flush=True)
    return report


def walk(path):
    yield path
    if path.is_dir() and not path.is_symlink():
        for child in sorted(path.iterdir()):yield from walk(child)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/data/jaka-vla'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--part-mib',type=int,default=1500);args=p.parse_args()
    root=args.root.resolve();out=args.output.resolve()
    if not (root/'AGENT.md').exists() or not (root/'configs/robots/dual_zu7.json').exists():raise ValueError('Not the JAKA-VLA project')
    if not 1<=args.part_mib<=1800:raise ValueError('Each release asset must stay below 2 GiB')
    groups=[('datasets',[root/'data'])]
    groups += [(path.name,[path]) for path in sorted((root/'checkpoints').iterdir()) if path.is_dir()]
    groups += [('evidence',[root/'results',root/'logs']),('source-history',[root/'backup-source'])]
    for _,roots in groups:
        for base in roots:
            if not base.exists():raise ValueError(f'Missing backup source: {base}')
            if out==base.resolve() or base.resolve() in out.parents:raise ValueError('Output must be outside archived source directories')
    out.mkdir(parents=True,exist_ok=False)
    index={'schema_version':1,'project':'JAKA-VLA','created_at_utc':datetime.now(timezone.utc).isoformat(),'archives':[],
       'excluded':'Python environments, download/build caches, upstream repositories/assets (reproducible by pinned sources), credentials, unrelated chats/projects.'}
    for name,roots in groups:
        paths=[path for base in roots for path in walk(base)];report=archive(root,out,name,paths,args.part_mib*1024**2)
        index['archives'].append({'name':name,'manifest':f'{name}.manifest.json','parts':report['parts'],'source_files':len(report['sources']),'source_bytes':report['source_bytes']})
        (out/'backup-index.partial.json').write_text(json.dumps(index,indent=2)+'\n')
    (out/'backup-index.json').write_text(json.dumps(index,indent=2)+'\n')
    (out/'backup-index.partial.json').unlink()
    checks=[]
    for path in sorted(out.iterdir()):
        if not path.is_file():continue
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):digest.update(block)
        checks.append(f'{digest.hexdigest()}  {path.name}')
    (out/'SHA256SUMS').write_text('\n'.join(checks)+'\n')
    print('BACKUP_COMPLETE',out,flush=True)


if __name__=='__main__':main()
