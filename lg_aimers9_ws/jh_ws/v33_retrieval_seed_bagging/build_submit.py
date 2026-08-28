"""제출용 submit.zip 생성. zip 엔트리 경로는 반드시 forward-slash여야
리눅스 평가 서버에서 압축 해제 시 model/ 디렉토리가 정상적으로 만들어진다.
6시드 각각의 CatBoost/encoder/참조임베딩 + 공용 reference_labels 1개.
"""
import json
import os
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ZIP = os.path.join(ROOT, 'v33_retrieval_6seed.zip')

with open(os.path.join(ROOT, 'model', 'feature_meta.json'), encoding='utf-8') as f:
    SEEDS = json.load(f)['seeds']

files_to_add = [
    ('script.py', os.path.join(ROOT, 'script.py')),
    ('requirements.txt', os.path.join(ROOT, 'requirements.txt')),
    ('model/numeric_prep.pkl', os.path.join(ROOT, 'model', 'numeric_prep.pkl')),
    ('model/reference_labels.npy', os.path.join(ROOT, 'model', 'reference_labels.npy')),
    ('model/feature_meta.json', os.path.join(ROOT, 'model', 'feature_meta.json')),
    ('model/trackman_context.pkl', os.path.join(ROOT, 'model', 'trackman_context.pkl')),
]
for s in SEEDS:
    files_to_add += [
        (f'model/catboost_seed{s}.cbm', os.path.join(ROOT, 'model', f'catboost_seed{s}.cbm')),
        (f'model/retrieval_encoder_seed{s}.pt', os.path.join(ROOT, 'model', f'retrieval_encoder_seed{s}.pt')),
        (f'model/reference_embeddings_seed{s}.npy', os.path.join(ROOT, 'model', f'reference_embeddings_seed{s}.npy')),
    ]

missing = [src for _, src in files_to_add if not os.path.exists(src)]
if missing:
    raise SystemExit(f"필수 파일 누락: {missing}")

if os.path.exists(OUT_ZIP):
    os.remove(OUT_ZIP)

with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for arcname, src in files_to_add:
        zf.write(src, arcname=arcname)

print(f"생성 완료: {OUT_ZIP}  (seeds={SEEDS})")
with zipfile.ZipFile(OUT_ZIP) as zf:
    total = 0
    for info in zf.infolist():
        print(f"  {info.filename}  ({info.file_size/1e6:.2f} MB)")
        total += info.file_size
    print(f"  총 {total/1e6:.1f} MB (압축 전)")
