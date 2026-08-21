"""제출용 submit.zip 생성. zip 엔트리 경로는 반드시 forward-slash여야
리눅스 평가 서버에서 압축 해제 시 model/ 디렉토리가 정상적으로 만들어진다.
"""
import os
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ZIP = os.path.join(ROOT, 'v18_seed_bagging.zip')

files_to_add = [
    ('script.py', os.path.join(ROOT, 'script.py')),
    ('requirements.txt', os.path.join(ROOT, 'requirements.txt')),
    ('model/catboost_seed42.cbm', os.path.join(ROOT, 'model', 'catboost_seed42.cbm')),
    ('model/catboost_seed7.cbm', os.path.join(ROOT, 'model', 'catboost_seed7.cbm')),
    ('model/catboost_seed123.cbm', os.path.join(ROOT, 'model', 'catboost_seed123.cbm')),
    ('model/feature_meta.json', os.path.join(ROOT, 'model', 'feature_meta.json')),
    ('model/trackman_context.pkl', os.path.join(ROOT, 'model', 'trackman_context.pkl')),
]

missing = [src for _, src in files_to_add if not os.path.exists(src)]
if missing:
    raise SystemExit(f"필수 파일 누락: {missing}")

if os.path.exists(OUT_ZIP):
    os.remove(OUT_ZIP)

with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for arcname, src in files_to_add:
        zf.write(src, arcname=arcname)

print(f"생성 완료: {OUT_ZIP}")
with zipfile.ZipFile(OUT_ZIP) as zf:
    total = 0
    for info in zf.infolist():
        print(f"  {info.filename}  ({info.file_size} bytes)")
        total += info.file_size
    print(f"  총 {total/1e6:.1f} MB (압축 전)")
