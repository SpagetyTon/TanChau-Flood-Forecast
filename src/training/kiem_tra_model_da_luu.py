"""Check the saved models: load models/, predict the test set again and compare with results/*/du_bao_test_*.csv.
Run from the repository root after data/processed and data/targets have been created:
    python src/training/kiem_tra_model_da_luu.py
Prints the maximum absolute difference for each algorithm and for the Blend.
"""
import sys, os, json, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import load_config, load_splits, chuan_bi_XY
cfg=load_config(); data=load_splits(cfg)
def run(mdir, target, thang, resf, clf=False):
    fl=json.load(open(f'{mdir}/feature_list.json'))['features']
    X,y,d=chuan_bi_XY(data['test'], fl, target, thang, cfg)
    imp=joblib.load(f'{mdir}/imputer.pkl'); sc=joblib.load(f'{mdir}/scaler.pkl')
    Xi=imp.transform(X); Xs=pd.DataFrame(sc.transform(Xi),columns=Xi.columns,index=Xi.index)
    w=json.load(open(f'{mdir}/blend_weights.json'))
    ref=pd.read_csv(resf)
    out=[]; blend=0
    for k in ['dt','rf','lgbm','catboost','svr']:
        p=f'{mdir}/{k}{"_clf" if clf else ""}.pkl'
        if not os.path.exists(p): continue
        m=joblib.load(p); A=Xs if k=='svr' else Xi
        pr=m.predict_proba(A)[:,1] if clf else m.predict(A)
        col=[c for c in ref.columns if c.endswith('_'+k)][0]
        out.append(f'{k}:{np.max(np.abs(pr-ref[col].values)):.1e}')
        blend=blend+w.get(k,0)*pr
    col=[c for c in ref.columns if c.endswith('_Blend')][0]
    out.append(f'Blend:{np.max(np.abs(blend-ref[col].values)):.1e}')
    ok=np.allclose(ref['thuc_te'].values,y.values)
    print(target, 'n=',len(y), 'y matches' if ok else 'y DIFFERS', ' '.join(out))
R='results'
for h in [1,2,3,5,7,14]: run(f'models/M1_mucnuoc/ngay{h:02d}',f'target_ngay{h}',None,f'{R}/M1/du_bao_test_target_ngay{h}.csv')
run('models/M2_soNgayVuotNguong','so_ngay_tren_nguong_10',cfg['mua_vu']['thang_mua_lu'],f'{R}/M2/du_bao_test_so_ngay_tren_nguong_10.csv')
for h in [7,15]: run(f'models/M4_han/han{h:02d}',f'han_{h}',cfg['mua_vu']['thang_mua_kho'],f'{R}/M4/du_bao_test_han_{h}.csv',clf=True)
run('models/M3_muaLu/ketthuc','ngay_den_ketthuc',[9,10,11,12],f'{R}/M3/du_bao_test_ngay_den_ketthuc.csv')
