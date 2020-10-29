# -*- mode: python -*-

block_cipher = None


a = Analysis(['NLP01_MED_FujieLuo.py'],
             pathex=['C:\\Users\\Thinkpad\\Desktop\\nlp5'],
             binaries=[],
             datas=[],
             hiddenimports=['numpy.core._dtype_ctypes'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='NLP01_MED_FujieLuo',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False , icon='bitbug.ico')
