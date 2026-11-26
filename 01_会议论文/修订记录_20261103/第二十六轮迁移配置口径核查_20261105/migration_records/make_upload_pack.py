# -*- coding: utf-8 -*-
"""
生成 Xshell/Xftp 上传用的压缩包（放在 _work/upload/ 下）。

产出：
  DEIM_migrated.zip       新版 DEIM 发行包 + 已完成的迁移（不含垃圾文件）
  pycocoeval.zip          pycocoeval 源码包（服务器上编译安装）
  visdrone_coco_json.zip  仅 VisDrone 的 3 个 COCO json（很小，图片走网盘/平台）
"""
import os
import sys
import time
import zipfile

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'upload')

DEIM_SRC = r'D:\BaiduNetdiskDownload\DEIM_1572862469'
PYCOCO_SRC = r'D:\BaiduNetdiskDownload\pycocoeval-1572862469'
DS_SRC = r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone'

EXCLUDE_DIRS = {'.vscode', '__pycache__', '.git', '.idea', '.pytest_cache', '.mypy_cache'}
EXCLUDE_FILES = {'.DS_Store', 'Thumbs.db'}


def add_tree(zf, root, arc_prefix='', log_every=2000):
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn in EXCLUDE_FILES or fn.endswith(('.pyc', '.pyo')):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            arc = os.path.join(arc_prefix, rel) if arc_prefix else rel
            try:
                zf.write(full, arc)
                n += 1
                if log_every and n % log_every == 0:
                    print('    ... %d 个文件' % n, flush=True)
            except OSError as e:
                print('    跳过 %s (%s)' % (rel, e))
    return n


def build(zip_name, fn, label):
    path = os.path.join(OUT_DIR, zip_name)
    t = time.time()
    print('-> %s' % zip_name, flush=True)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        total = fn(zf)
    size = os.path.getsize(path) / 1024 ** 2
    print('   %s: %d 个文件, %.1f MB, 用时 %.1fs\n' % (label, total, size, time.time() - t), flush=True)
    return size


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('输出目录: %s\n' % OUT_DIR)

    build('DEIM_migrated.zip',
          lambda zf: add_tree(zf, DEIM_SRC, 'DEIM_1572862469'),
          '新版 DEIM（含迁移）')

    build('pycocoeval.zip',
          lambda zf: add_tree(zf, PYCOCO_SRC, 'pycocoeval-1572862469'),
          'pycocoeval 源码')

    def jsons(zf):
        n = 0
        for sub in ['VisDrone2019-DET-train/annotations/train.json',
                    'VisDrone2019-DET-val/annotations/val.json',
                    'VisDrone2019-DET-test-dev/annotations/test.json']:
            p = os.path.join(DS_SRC, sub.replace('/', os.sep))
            if os.path.isfile(p):
                zf.write(p, os.path.join('dataset_visdrone', sub.replace('/', os.sep)))
                n += 1
            else:
                print('    缺失: %s' % sub)
        return n

    build('visdrone_coco_json.zip', jsons, 'VisDrone COCO json')

    print('=' * 66)
    print('上传清单')
    print('=' * 66)
    for f in sorted(os.listdir(OUT_DIR)):
        p = os.path.join(OUT_DIR, f)
        print('  %-26s %8.1f MB' % (f, os.path.getsize(p) / 1024 ** 2))
    print('  %-26s %8.1f MB   (手动：论文最佳权重，复评/测速/可视化用)'
          % ('best_stg2.pth', 215.7))
    print('  %-26s %8.1f MB   (可选：完整 VisDrone 图片，优先走网盘秒传)'
          % ('dataset_visdrone/ 图片', 2308.6))
    print()


if __name__ == '__main__':
    main()
