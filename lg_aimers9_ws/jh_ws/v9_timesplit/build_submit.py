"""제출용 submit.zip 생성. zip 엔트리 경로는 반드시 forward-slash여야
리눅스 평가 서버에서 압축 해제 시 model/ 디렉토리가 정상적으로 만들어진다
(과거 hackathon_clean에서 백슬래시 경로 버그로 FileNotFoundError 발생한 적 있음).
"""
import os
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ZIP = os.path.join(ROOT, 'submit.zip')

# 추론(script.py)에만 필요한 최소 패키지. optuna는 학습 전용이라 제외,
# scikit-learn/joblib은 평가 서버에 기본 설치돼 있어 제외.
SUBMIT_REQUIREMENTS = "lightgbm\ncatboost\nxgboost\n"

files_to_add = [
    ('script.py', os.path.join(ROOT, 'script.py')),
    ('model/lgb_fold0.pkl', os.path.join(ROOT, 'model', 'lgb_fold0.pkl')),
    ('model/cat_fold0.pkl', os.path.join(ROOT, 'model', 'cat_fold0.pkl')),
    ('model/xgb_fold0.pkl', os.path.join(ROOT, 'model', 'xgb_fold0.pkl')),
    ('model/meta_info.pkl', os.path.join(ROOT, 'model', 'meta_info.pkl')),
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
