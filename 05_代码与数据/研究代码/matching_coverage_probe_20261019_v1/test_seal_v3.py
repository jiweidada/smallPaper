import json,pathlib,tempfile,unittest
from seal_v3 import sha,write_json,verify_partition,heldout_guard
class SealTests(unittest.TestCase):
 def test_tampered_prediction_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td);f=p/'packet';f.write_bytes(b'original')
   write_json(p/'seal.json',{'batches':1,'files':[{'relative':'packet','sha256':sha(f),'bytes':8}]})
   verify_partition(p);f.write_bytes(b'replaced')
   with self.assertRaises(AssertionError):verify_partition(p)
 def test_heldout_requires_release_before_loading(self):
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaises(FileNotFoundError):heldout_guard(td,1)
 def test_unknown_partition_file_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td);f=p/'packet';f.write_bytes(b'x')
   write_json(p/'seal.json',{'batches':1,'files':[{'relative':'packet','sha256':sha(f),'bytes':1}]})
   (p/'unexpected').write_bytes(b'x')
   with self.assertRaises(AssertionError):verify_partition(p)
if __name__=='__main__':unittest.main()
