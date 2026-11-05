"""Verify release parts and every source file; optionally restore into a NEW directory."""
import argparse,hashlib,io,json,os,tarfile
from pathlib import Path,PurePosixPath


class JoinedReader(io.RawIOBase):
    def __init__(self,paths):super().__init__();self.paths=iter(paths);self.stream=None
    def readable(self):return True
    def readinto(self,buffer):
        while True:
            if self.stream is None:
                try:self.stream=next(self.paths).open('rb')
                except StopIteration:return 0
            n=self.stream.readinto(buffer)
            if n:return n
            self.stream.close();self.stream=None
    def close(self):
        if self.stream:self.stream.close()
        super().close()


def checksum(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def verify(backup,target=None):
    if target is not None:target.mkdir(parents=True,exist_ok=False)
    # Validate metadata too, not only archive parts.
    for line in (backup/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name:raise ValueError('Unsafe manifest name')
        if checksum(backup/name)!=digest:raise ValueError(f'Backup checksum mismatch: {name}')
    index=json.loads((backup/'backup-index.json').read_text());total=0
    for item in index['archives']:
        manifest=json.loads((backup/item['manifest']).read_text())
        expected={x['path']:x for x in manifest['sources']};seen=set()
        parts=[]
        for part in item['parts']:
            path=backup/part['file']
            if path.stat().st_size!=part['bytes']:raise ValueError(f'Wrong part size: {path.name}')
            parts.append(path)
        with JoinedReader(parts) as raw,io.BufferedReader(raw) as stream,tarfile.open(fileobj=stream,mode='r|gz') as tf:
            for member in tf:
                rel=PurePosixPath(member.name)
                if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe archive path')
                dest=target.joinpath(*rel.parts) if target else None
                if dest is not None:dest.resolve().relative_to(target.resolve())
                if member.isdir():
                    if dest is not None:dest.mkdir(parents=True,exist_ok=True)
                    continue
                if member.name not in expected or member.name in seen:raise ValueError(f'Unexpected/duplicate member: {member.name}')
                seen.add(member.name);entry=expected[member.name]
                if dest is not None:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists() or dest.is_symlink():raise FileExistsError(dest)
                if member.issym():
                    if member.linkname!=entry.get('target'):raise ValueError('Symlink mismatch')
                    if PurePosixPath(member.linkname).is_absolute():raise ValueError('Absolute symlink is not portable')
                    if dest is not None:
                        (dest.parent/member.linkname).resolve().relative_to(target.resolve())
                        dest.symlink_to(member.linkname)
                elif member.isfile():
                    digest=hashlib.sha256();size=0;output=dest.open('xb') if dest is not None else None
                    try:
                        with tf.extractfile(member) as source:
                            for block in iter(lambda:source.read(8*1024**2),b''):
                                digest.update(block);size+=len(block)
                                if output:output.write(block)
                    finally:
                        if output:output.close()
                    if size!=entry['bytes'] or digest.hexdigest()!=entry['sha256']:raise ValueError(f'Source hash mismatch: {member.name}')
                    if dest is not None:dest.chmod(member.mode & 0o777)
                else:raise ValueError('Unsupported archive member')
        if seen!=set(expected):raise ValueError('Missing source files')
        total+=len(seen);print(f'VERIFIED {item["name"]}: {len(seen)} files/links',flush=True)
    print(f'BACKUP_VERIFIED: {total} files/links'+(f'; restored to {target}' if target else '; verification only'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--backup',type=Path,required=True);p.add_argument('--restore-to',type=Path)
    args=p.parse_args();verify(args.backup.resolve(),args.restore_to.resolve() if args.restore_to else None)


if __name__=='__main__':main()
