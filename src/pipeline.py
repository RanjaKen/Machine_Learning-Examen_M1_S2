"""
Atlantic Haven Hotels — pipeline COMPLET (étapes 2 à 12).
Produit : results_comparison.csv, tuning, seuil, analyse erreurs, importances,
metrics.json, figures, submission.csv. Exécutable de bout en bout, sans fuite.
"""
import time, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score)

RS = 42; np.random.seed(RS)
M = {}  # dictionnaire de métriques -> metrics.json

# 1. Données ----------------------------------------------------------------
train = pd.read_csv("data/reservations_train.csv"); test = pd.read_csv("data/reservations_test.csv")
for c in ["date_reservation","date_arrivee"]:
    train[c]=pd.to_datetime(train[c]); test[c]=pd.to_datetime(test[c])
TARGET="reservation_annulee"; ID="reservation_id"

# 2. Feature engineering (leak-free) ---------------------------------------
def add_features(df):
    df=df.copy()
    df["res_mois"]=df["date_reservation"].dt.month
    df["res_jour_sem"]=df["date_reservation"].dt.dayofweek
    df["res_annee"]=df["date_reservation"].dt.year
    df["res_trimestre"]=df["date_reservation"].dt.quarter
    df["arr_mois"]=df["date_arrivee"].dt.month
    sais={12:"hiver",1:"hiver",2:"hiver",3:"printemps",4:"printemps",5:"printemps",
          6:"ete",7:"ete",8:"ete",9:"automne",10:"automne",11:"automne"}
    df["arr_saison"]=df["arr_mois"].map(sais)
    df["total_personnes"]=df["adultes"]+df["enfants"].fillna(0)
    df["pers_par_chambre"]=df["total_personnes"]/df["chambres"].replace(0,np.nan)
    df["a_enfants"]=(df["enfants"].fillna(0)>0).astype(int)
    df["prix_total_par_nuit"]=df["montant_total_eur"]/df["nuits"].replace(0,np.nan)
    df["montant_remise_eur"]=df["montant_total_eur"]*df["remise_pct"]/100.0
    df["remise_forte"]=(df["remise_pct"]>=10).astype(int)
    df["taux_annul_client"]=(df["annulations_passees"]/df["reservations_passees"].replace(0,np.nan)).fillna(0)
    df["client_a_historique"]=(df["reservations_passees"]>0).astype(int)
    df["reservation_directe"]=df["agent_id"].isna().astype(int)
    df["delai_long"]=(df["delai_reservation_jours"]>=60).astype(int)
    return df
train_fe=add_features(train); test_fe=add_features(test)

drop=[ID,TARGET,"date_reservation","date_arrivee","agent_id","hotel_id"]
num_cols=[c for c in train_fe.select_dtypes(include="number").columns if c not in drop]
cat_cols=[c for c in train_fe.columns if c not in drop and c not in num_cols
          and c not in ["date_reservation","date_arrivee"]]
base_num=[c for c in num_cols if c in train.columns]
base_cat=[c for c in cat_cols if c in train.columns]

def prep(numc,catc):
    return ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),numc),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                         ("ohe",OneHotEncoder(handle_unknown="ignore",min_frequency=10))]),catc)])

# 3. Validation temporelle : holdout chronologique -------------------------
y=train_fe[TARGET].values; n=len(train_fe); cut=int(n*0.80)
M["split"]={"n_train":cut,"n_valid":n-cut,
            "train_fin":str(train_fe['date_reservation'].iloc[cut-1].date()),
            "valid_debut":str(train_fe['date_reservation'].iloc[cut].date()),
            "valid_fin":str(train_fe['date_reservation'].iloc[-1].date())}
def split(numc,catc):
    X=train_fe[numc+catc]; return X.iloc[:cut],y[:cut],X.iloc[cut:],y[cut:]

def eval_model(model,numc,catc,name,thr=0.5):
    Xtr,ytr,Xva,yva=split(numc,catc)
    pipe=Pipeline([("prep",prep(numc,catc)),("clf",model)])
    t0=time.time(); pipe.fit(Xtr,ytr); dt=time.time()-t0
    p=pipe.predict_proba(Xva)[:,1]; pred=(p>=thr).astype(int)
    return dict(model=name,f1=f1_score(yva,pred),precision=precision_score(yva,pred),
                recall=recall_score(yva,pred),roc_auc=roc_auc_score(yva,p),
                train_time_s=round(dt,2)),pipe,p,yva

# 4+5+6. Comparaison (seuil 0.5) -------------------------------------------
rows=[]
rows.append(eval_model(LogisticRegression(max_iter=2000,class_weight="balanced",random_state=RS),
                       base_num,base_cat,"LogReg baseline (sans FE)")[0])
rows.append(eval_model(RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=RS,n_jobs=-1),
                       base_num,base_cat,"RandomForest (sans FE)")[0])
rows.append(eval_model(HistGradientBoostingClassifier(random_state=RS,class_weight="balanced"),
                       base_num,base_cat,"HistGB (sans FE)")[0])
rows.append(eval_model(LogisticRegression(max_iter=2000,class_weight="balanced",random_state=RS),
                       num_cols,cat_cols,"LogReg + FE")[0])
rows.append(eval_model(RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=RS,n_jobs=-1),
                       num_cols,cat_cols,"RandomForest + FE")[0])
rows.append(eval_model(HistGradientBoostingClassifier(random_state=RS,class_weight="balanced"),
                       num_cols,cat_cols,"HistGB + FE")[0])
comp=pd.DataFrame(rows)[["model","f1","precision","recall","roc_auc","train_time_s"]]
comp.round(4).to_csv("results_comparison.csv",index=False)
M["comparison_thr05"]=comp.round(4).to_dict(orient="records")
print("=== Comparaison @0.5 ==="); print(comp.round(4).to_string(index=False))

# 7. Optimisation hyperparamètres du meilleur modèle (LogReg) --------------
# Tuning via TimeSeriesSplit sur la PORTION train uniquement, scoring PR-AUC
# (average_precision : centré classe 1, indépendant du seuil).
Xtr,ytr,Xva,yva=split(num_cols,cat_cols)
tscv=TimeSeriesSplit(n_splits=4)
grid=GridSearchCV(
    Pipeline([("prep",prep(num_cols,cat_cols)),
              ("clf",LogisticRegression(max_iter=3000,random_state=RS))]),
    param_grid={"clf__C":[0.01,0.05,0.1,0.5,1.0,2.0],"clf__penalty":["l2"]},
    scoring="average_precision",cv=tscv,n_jobs=-1)
grid.fit(Xtr,ytr)
best_C=grid.best_params_["clf__C"]
M["tuning"]={"best_C":best_C,"cv_scoring":"average_precision","cv_best_score":round(grid.best_score_,4)}
print("\nMeilleur C:",best_C,"(PR-AUC CV=%.4f)"%grid.best_score_)

final_model=LogisticRegression(C=best_C,max_iter=3000,random_state=RS)
res_final,pipe_final,proba_val,yva=eval_model(final_model,num_cols,cat_cols,f"LogReg tuné (C={best_C})")
M["valid_model_thr05"]=res_final
print("Modèle tuné @0.5:",{k:round(v,4) if isinstance(v,float) else v for k,v in res_final.items()})

# 8. Optimisation du seuil sur la VALIDATION -------------------------------
ths=np.linspace(0.05,0.95,181)
f1s=[f1_score(yva,(proba_val>=t).astype(int)) for t in ths]
best_i=int(np.argmax(f1s)); best_thr=float(ths[best_i]); best_f1=float(f1s[best_i])
M["threshold"]={"best_threshold":round(best_thr,3),"f1_at_best":round(best_f1,4)}
pred_best=(proba_val>=best_thr).astype(int)
M["valid_final"]=dict(threshold=round(best_thr,3),
    f1=round(f1_score(yva,pred_best),4),precision=round(precision_score(yva,pred_best),4),
    recall=round(recall_score(yva,pred_best),4),roc_auc=round(roc_auc_score(yva,proba_val),4),
    confusion=confusion_matrix(yva,pred_best).tolist())
print("\nSeuil optimal:",round(best_thr,3),"-> F1=%.4f"%best_f1)
print("Matrice de confusion (val):",M["valid_final"]["confusion"])

# Figure seuil + confusion
fig,ax=plt.subplots(1,2,figsize=(13,5))
ax[0].plot(ths,f1s,color="#4C78A8"); ax[0].axvline(best_thr,ls="--",color="#E45756")
ax[0].axvline(0.5,ls=":",color="grey"); ax[0].set_xlabel("seuil"); ax[0].set_ylabel("F1 (classe 1)")
ax[0].set_title(f"F1 vs seuil — optimum={best_thr:.2f} (F1={best_f1:.3f})")
cm=confusion_matrix(yva,pred_best)
im=ax[1].imshow(cm,cmap="Blues")
for i in range(2):
    for j in range(2): ax[1].text(j,i,cm[i,j],ha="center",va="center",fontsize=14)
ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["prévu 0","prévu 1"])
ax[1].set_yticks([0,1]); ax[1].set_yticklabels(["réel 0","réel 1"])
ax[1].set_title("Matrice de confusion (validation, seuil optimal)")
plt.tight_layout(); plt.savefig("fig_seuil_confusion.png",dpi=110); plt.close()

# 9. Analyse des erreurs ----------------------------------------------------
val_idx=train_fe.index[cut:]
val_df=train_fe.loc[val_idx].copy()
val_df["proba"]=proba_val; val_df["pred"]=pred_best; val_df["reel"]=yva
val_df["cat_err"]=np.select(
    [(val_df.reel==1)&(val_df.pred==1),(val_df.reel==0)&(val_df.pred==0),
     (val_df.reel==0)&(val_df.pred==1),(val_df.reel==1)&(val_df.pred==0)],
    ["VP","VN","FP","FN"],default="NA")
prof=val_df.groupby("cat_err").agg(
    n=("proba","size"),delai=("delai_reservation_jours","mean"),
    prix=("montant_total_eur","mean"),remb=("tarif_remboursable",lambda s:(s=="oui").mean()),
    acompte_aucun=("type_acompte",lambda s:(s=="aucun").mean())).round(2)
M["error_profiles"]=prof.reset_index().to_dict(orient="records")
print("\n=== Profils par type d'erreur (validation) ==="); print(prof.to_string())
fp=val_df[val_df.cat_err=="FP"].nlargest(5,"proba")
fn=val_df[val_df.cat_err=="FN"].nsmallest(5,"proba")
cols_show=["reservation_id","tarif_remboursable","type_acompte","delai_reservation_jours",
           "canal_reservation","montant_total_eur","proba"]
M["examples_FP"]=fp[cols_show].round(2).to_dict(orient="records")
M["examples_FN"]=fn[cols_show].round(2).to_dict(orient="records")

# Performance par région (Q8)
by_reg=val_df.groupby("region_hotel").apply(
    lambda g: pd.Series({"n":len(g),"f1":f1_score(g.reel,g.pred) if g.reel.sum()>0 else np.nan})).round(3)
M["f1_by_region"]=by_reg.reset_index().to_dict(orient="records")

# 10. Interprétation : permutation importance ------------------------------
perm=permutation_importance(pipe_final,Xva,yva,scoring="f1",n_repeats=10,random_state=RS,n_jobs=-1)
# noms de features après OHE
feat_names=pipe_final.named_steps["prep"].get_feature_names_out()
# permutation_importance sur colonnes d'entrée (avant OHE) -> importance par colonne source
imp=pd.DataFrame({"feature":Xva.columns,"importance":perm.importances_mean}).sort_values(
    "importance",ascending=False)
M["perm_importance_top15"]=imp.head(15).round(5).to_dict(orient="records")
print("\n=== Top 15 permutation importance (F1) ===");print(imp.head(15).to_string(index=False))

# Coefficients (interprétation signée)
clf=pipe_final.named_steps["clf"]
coefs=pd.DataFrame({"feature":feat_names,"coef":clf.coef_[0]})
coefs["abs"]=coefs["coef"].abs()
top_pos=coefs.nlargest(10,"coef")[["feature","coef"]].round(3)
top_neg=coefs.nsmallest(10,"coef")[["feature","coef"]].round(3)
M["coef_top_positifs"]=top_pos.to_dict(orient="records")
M["coef_top_negatifs"]=top_neg.to_dict(orient="records")

fig,ax=plt.subplots(1,2,figsize=(14,6))
ii=imp.head(12).iloc[::-1]
ax[0].barh(ii["feature"],ii["importance"],color="#54A24B")
ax[0].set_title("Permutation importance (perte de F1)")
cc=pd.concat([top_pos.head(8),top_neg.head(8)]).sort_values("coef")
ax[1].barh(cc["feature"],cc["coef"],color=np.where(cc["coef"]>0,"#E45756","#4C78A8"))
ax[1].set_title("Coefficients LogReg (+ = ↑ annulation)")
plt.tight_layout(); plt.savefig("fig_importance.png",dpi=110); plt.close()

# 11. Modèle final : réentraînement sur TOUT le train ----------------------
Xall=train_fe[num_cols+cat_cols]; yall=y
pipe_all=Pipeline([("prep",prep(num_cols,cat_cols)),("clf",
    LogisticRegression(C=best_C,max_iter=3000,random_state=RS))])
pipe_all.fit(Xall,yall)
proba_test=pipe_all.predict_proba(test_fe[num_cols+cat_cols])[:,1]
pred_test=(proba_test>=best_thr).astype(int)

# 12. submission.csv --------------------------------------------------------
sub=pd.DataFrame({ID:test_fe[ID].values,
                  "probabilite_annulation":np.round(proba_test,6),
                  "reservation_annulee":pred_test})
sub.to_csv("submission.csv",index=False)
# Vérifications
sample=pd.read_csv("data/sample_submission.csv")
checks={"n_lignes":len(sub),"n_colonnes":sub.shape[1],
        "colonnes_ok":list(sub.columns)==["reservation_id","probabilite_annulation","reservation_annulee"],
        "ids_manquants":int((~test_fe[ID].isin(sub[ID])).sum()),
        "ordre_identique_au_test":bool((sub[ID].values==test_fe[ID].values).all()),
        "ordre_identique_au_sample":bool((sub[ID].values==sample[ID].values).all()),
        "proba_min":round(float(sub.probabilite_annulation.min()),4),
        "proba_max":round(float(sub.probabilite_annulation.max()),4),
        "pred_taux_annulation":round(float(sub.reservation_annulee.mean()),4),
        "valeurs_binaires_ok":sorted(sub.reservation_annulee.unique().tolist())==[0,1]}
M["submission_checks"]=checks
print("\n=== Vérifications submission ==="); print(json.dumps(checks,indent=2))

json.dump(M,open("metrics.json","w"),indent=2,ensure_ascii=False)
print("\nOK — metrics.json + submission.csv + figures écrits.")
