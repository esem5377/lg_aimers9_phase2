import os
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ZIP = os.path.join(ROOT, 'submit_v14.zip')

SUBMIT_REQUIREMENTS = "lightgbm\ncatboost\nxgboost\n"

files_to_add = [
    ('script.py', os.path.join(ROOT, 'script_v14.py')),
    ('model/lgb_fold0.pkl', os.path.join(ROOT, 'model_v14', 'lgb_fold0.pkl')),
    ('model/cat_fold0.pkl', os.path.join(ROOT, 'model_v14', 'cat_fold0.pkl')),
    ('model/xgb_fold0.pkl', os.path.join(ROOT, 'model_v14', 'xgb_fold0.pkl')),
    ('model/meta_info.pkl', os.path.join(ROOT, 'model_v14', 'meta_info.pkl')),
]

missing = [src for _, src in files_to_add if not os.path.exists(src)]
if missing:
    raise SystemExit(f"필수 파일 누락: {missing}")

if os.path.exists(OUT_ZIP):
    os.remove(OUT_ZIP)

with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for arcname, src in files_to_add:
        zf.write(src, arcname=arcname)
    zf.writestr('requirements.txt', SUBMIT_REQUIREMENTS)

print(f"생성 완료: {OUT_ZIP}")
with zipfile.ZipFile(OUT_ZIP) as zf:
    for info in zf.infolist():
        print(f"  {info.filename}  ({info.file_size} bytes)")
