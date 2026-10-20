import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('fetch_assets',Path(__file__).resolve().parents[1]/'scripts/fetch-public-assets.py')
fetcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(fetcher)


class Response(io.BytesIO):
    status=200
    def __init__(self,data,url):super().__init__(data);self.url=url
    def geturl(self):return self.url


class AssetTests(unittest.TestCase):
    def manifest(self,data=b'public video bytes'):
        return {'schema_version':1,'origin':fetcher.ORIGIN,'assets':[{'path':'assets/test.mp4','url':fetcher.ORIGIN+'/assets/test.mp4','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}]}
    def test_preview_never_opens_network(self):
        with tempfile.TemporaryDirectory() as root:
            result=fetcher.fetch(self.manifest(),root,opener=lambda *a,**k:self.fail('Network used in preview'))
            self.assertEqual(result['missing'],1);self.assertEqual(list(Path(root).iterdir()),[])
    def test_explicit_download_checks_hash_and_reuses_verified_file(self):
        data=b'public video bytes'
        with tempfile.TemporaryDirectory() as root:
            result=fetcher.fetch(self.manifest(data),root,download=True,opener=lambda request,**kw:Response(data,request.full_url))
            self.assertEqual(result['downloaded'],1)
            result=fetcher.fetch(self.manifest(data),root,download=True,opener=lambda *a,**k:self.fail('Verified asset refetched'))
            self.assertEqual(result['downloaded'],0)
    def test_bad_bytes_and_redirects_leave_no_final_asset(self):
        for redirect in (False,True):
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ValueError):fetcher.fetch(self.manifest(),root,download=True,opener=lambda request,**kw:Response(b'bad' if not redirect else b'public video bytes','https://evil.invalid' if redirect else request.full_url))
                self.assertFalse((Path(root)/'assets/test.mp4').exists())
                self.assertEqual(list(Path(root).rglob('.asset-download-*')),[])
    def test_destination_created_at_publication_is_never_overwritten(self):
        from unittest.mock import patch
        original_link=fetcher.os.link
        def racing_link(source,destination):
            Path(destination).write_bytes(b'concurrent writer')
            return original_link(source,destination)
        with tempfile.TemporaryDirectory() as root,patch.object(fetcher.os,'link',racing_link):
            with self.assertRaises(FileExistsError):
                fetcher.fetch(self.manifest(),root,download=True,opener=lambda request,**kw:Response(b'public video bytes',request.full_url))
            self.assertEqual((Path(root)/'assets/test.mp4').read_bytes(),b'concurrent writer')
            self.assertEqual(list(Path(root).rglob('.asset-download-*')),[])
    def test_unsafe_paths_origins_and_sizes_rejected(self):
        for update in ({'path':'../secret'},{'path':'/tmp/secret'},{'path':'.git/config'},{'url':'http://roboswe.com/assets/test.mp4'},{'url':'https://roboswe.com@evil.invalid/assets/test.mp4'},{'bytes':-1}):
            manifest=self.manifest();manifest['assets'][0].update(update)
            with self.assertRaises(ValueError):fetcher.entries(manifest)
    def test_symlink_and_existing_mismatch_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root,tempfile.TemporaryDirectory() as outside:
            (Path(root)/'assets').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(ValueError):fetcher.fetch(self.manifest(),root)
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'assets/test.mp4';p.parent.mkdir();p.write_bytes(b'keep')
            with self.assertRaises(ValueError):fetcher.fetch(self.manifest(),root,download=True)
            self.assertEqual(p.read_bytes(),b'keep')


if __name__=='__main__':unittest.main()
