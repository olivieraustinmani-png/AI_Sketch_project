"""Télécharge un sous-ensemble Quick, Draw! et entraîne un CNN Keras."""
import argparse
import gzip
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

from predict import IMAGE_SIZE, ROOT, nearest_distances, numpy_open_set_predict, preprocess

LABELS = json.loads((ROOT/'labels.json').read_text(encoding='utf-8'))
UNKNOWN_CATEGORIES = ['apple', 'banana', 'fish', 'airplane', 'flower', 'cup']
BASE = 'https://storage.googleapis.com/quickdraw_dataset/full/simplified/{name}.ndjson'


def download(category, data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f'{category}.ndjson.gz'


def render(drawing):
    # Les coordonnées vectorielles QuickDraw sont tracées à l'échelle de l'image.
    image = Image.new('L', (256,256), 255); draw = ImageDraw.Draw(image)
    points = [p for stroke in drawing for p in zip(stroke[0], stroke[1])]
    if not points: return image
    lo_x=min(p[0] for p in points); hi_x=max(p[0] for p in points)
    lo_y=min(p[1] for p in points); hi_y=max(p[1] for p in points)
    scale=min(220/max(1,hi_x-lo_x), 220/max(1,hi_y-lo_y))
    for stroke in drawing:
        pts=[(18+(x-lo_x)*scale,18+(y-lo_y)*scale) for x,y in zip(stroke[0],stroke[1])]
        if len(pts)>1: draw.line(pts, fill=0, width=5, joint='curve')
        elif pts: draw.ellipse((pts[0][0]-2,pts[0][1]-2,pts[0][0]+2,pts[0][1]+2),fill=0)
    return image


def load_category(category, limit, data_dir):
    xs=[]
    path=download(category,data_dir)
    cached=[]
    if path.exists():
        with gzip.open(path,'rt',encoding='utf-8') as f:
            cached=[line.encode('utf-8') for line in f if line.strip()]
    if len(cached) < limit:
        print(f'Téléchargement progressif de {category} (objectif : {limit})…',flush=True)
        rows=[]; accepted=0; last_error=None
        for attempt in range(4):
            req=urllib.request.Request(BASE.format(name=category),headers={'Range':'bytes=0-5000000','Accept-Encoding':'gzip'})
            try:
                with urllib.request.urlopen(req,timeout=90) as response:
                    while accepted < limit:
                        line=response.readline()
                        if not line: break
                        try: row=json.loads(line)
                        except (json.JSONDecodeError,UnicodeDecodeError): continue
                        if not row.get('recognized',True) or row.get('countrycode') == '': continue
                        rows.append(line); accepted+=1
                break
            except (TimeoutError, OSError, urllib.error.URLError) as exc:
                last_error=exc
                if rows: break
                if attempt<3:
                    print(f'Nouvelle tentative pour {category} ({attempt+1}/3)…',flush=True); time.sleep(1)
        if last_error and not rows:
            if not cached: raise RuntimeError(f'Téléchargement interrompu pour {category}: {last_error}') from last_error
            print(f'Conservation des {len(cached)} exemples déjà en cache pour {category}.',flush=True)
        if rows:
            path.write_bytes(gzip.compress(b''.join(rows)))
        elif not cached:
            raise RuntimeError(f'Aucun exemple Quick, Draw! reçu pour {category}.')
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            try: row=json.loads(line)
            except json.JSONDecodeError: continue
            if not row.get('recognized', True) or row.get('countrycode') == '': continue
            xs.append(preprocess(render(row['drawing'])))
            if len(xs)>=limit: break
    return xs


def load_data(limit, data_dir, unknown_per_category):
    xs=[]; ys=[]; known_counts=[]
    for label_id, category in enumerate(LABELS[:5]):
        drawings=load_category(category,limit,data_dir)
        xs.extend(drawings); ys.extend([label_id]*len(drawings))
        known_counts.append(len(drawings))
        print(f'{category}: {len(drawings)} dessins')
    unknown=[]
    for category in UNKNOWN_CATEGORIES:
        drawings=load_category(category,unknown_per_category,data_dir)
        unknown.extend(drawings)
        print(f'{category} (hors classes): {len(drawings)} dessins')
    rng=np.random.default_rng(101)
    unknown_limit=min(limit,min(known_counts)) if known_counts else 0
    if len(unknown)>unknown_limit:
        chosen=rng.choice(len(unknown),size=unknown_limit,replace=False)
        unknown=[unknown[i] for i in chosen]
    xs.extend(unknown); ys.extend([5]*len(unknown))
    print(f'unknown: {len(unknown)} dessins hors classes')
    return np.asarray(xs,dtype=np.float32), np.asarray(ys,dtype=np.int64)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--per-class',type=int,default=1000)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data'/'quickdraw')
    parser.add_argument('--backend',choices=['auto','keras','sklearn'],default='auto')
    parser.add_argument('--unknown-per-category',type=int,default=100,
                        help='exemples maximum pour chaque catégorie hors classes (pomme incluse)')
    args=parser.parse_args()
    tf=None
    try:
        import tensorflow as tf
        backend='keras'
    except ImportError:
        backend='auto'
    if args.backend != 'auto': backend=args.backend
    if backend=='keras' and tf is None:
        try: import tensorflow as tf
        except ImportError as exc: raise SystemExit(f'TensorFlow indisponible: {exc}.')
    try:
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.svm import SVC
        import joblib
        sklearn_ready=True
    except ImportError:
        sklearn_ready=False
    if backend=='sklearn' and not sklearn_ready:
        raise SystemExit('scikit-learn indisponible. Installez requirements-training-fallback.txt.')
    if backend=='auto': backend='sklearn' if sklearn_ready else 'numpy'
    x,y=load_data(args.per_class,args.data_dir,args.unknown_per_category)
    rng=np.random.default_rng(42); train_ids=[]; val_ids=[]; test_ids=[]
    for class_id in range(len(LABELS)):
        ids=np.flatnonzero(y==class_id); rng.shuffle(ids)
        n_train=max(1,int(len(ids)*.70)); n_val=max(1,int(len(ids)*.15))
        train_ids.extend(ids[:n_train]); val_ids.extend(ids[n_train:n_train+n_val]); test_ids.extend(ids[n_train+n_val:])
    rng.shuffle(train_ids); rng.shuffle(val_ids); rng.shuffle(test_ids)
    x_train,y_train=x[train_ids],y[train_ids]; x_val,y_val=x[val_ids],y[val_ids]; x_test,y_test=x[test_ids],y[test_ids]
    if backend=='keras':
        model=tf.keras.Sequential([
            tf.keras.layers.Input((IMAGE_SIZE,IMAGE_SIZE,1)),
            tf.keras.layers.Conv2D(24,3,activation='relu'),tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(48,3,activation='relu'),tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dropout(.25),tf.keras.layers.Dense(len(LABELS),activation='softmax')])
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=12,batch_size=64,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)],verbose=2)
        scores=model.predict(x_test,verbose=0); pred=scores.argmax(axis=1)
        model_path=ROOT/'ai_sketch_model.keras'; model.save(model_path)
    elif backend=='sklearn':
        model=SVC(kernel='rbf',C=8,gamma='scale',probability=True,random_state=42)
        model.fit(x_train.reshape(len(x_train),-1),y_train)
        scores=model.predict_proba(x_test.reshape(len(x_test),-1)); pred=model.classes_[scores.argmax(axis=1)]
        model_path=ROOT/'ai_sketch_model.joblib'; joblib.dump(model,model_path)
    else:
        # Seuil ouvert calibré sur validation : connu si un exemple proche
        # existe dans la base, inconnu si le dessin est trop éloigné.
        known_mask=y_train < 5
        known_x=x_train[known_mask].reshape(np.sum(known_mask),-1)
        val_dist=nearest_distances(x_val,known_x).min(axis=1)
        val_is_known=y_val < 5
        # Seuil conservateur: les dessins valides de validation sont acceptés;
        # la classe inconnue est d'abord apprise via ses propres exemples.
        threshold=float(np.quantile(val_dist[val_is_known],0.98)) if np.any(val_is_known) else 0.35
        model={'train_x':x_train.reshape(len(x_train),-1).astype(np.float32),
               'train_y':y_train.astype(np.int64),
               'threshold':np.asarray(threshold,dtype=np.float32),'unknown_id':np.asarray(5,dtype=np.int64),
               'k':np.asarray(3,dtype=np.int64)}
        val_results=numpy_open_set_predict(model,x_val)
        val_pred=np.asarray([r[0] for r in val_results],dtype=np.int64)
        validation_accuracy=float(np.mean(val_pred==y_val))
        validation_unknown_recall=float(np.mean(val_pred[y_val==5]==5))
        results=numpy_open_set_predict(model,x_test)
        pred=np.asarray([r[0] for r in results],dtype=np.int64)
        model_path=ROOT/'ai_sketch_model.npz'
        np.savez_compressed(model_path,**model)
    accuracy=float(np.mean(y_test==pred))
    report={}
    for class_id,label in enumerate(LABELS):
        tp=int(np.sum((y_test==class_id)&(pred==class_id))); fp=int(np.sum((y_test!=class_id)&(pred==class_id))); fn=int(np.sum((y_test==class_id)&(pred!=class_id)))
        precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn)
        report[label]={'precision':precision,'recall':recall,'f1-score':2*precision*recall/max(1e-12,precision+recall),'support':int(np.sum(y_test==class_id))}
    metrics={'samples_total':int(len(y)), 'train':int(len(y_train)),'validation':int(len(y_val)),
        'test':int(len(y_test)), 'test_accuracy':accuracy,'backend':backend,
        'model_file':model_path.name,'per_class':report}
    if backend=='numpy':
        metrics['unknown_recall']=report.get('unknown',{}).get('recall',0.0)
        metrics['open_set_threshold']=float(model['threshold'])
        metrics['validation_accuracy']=validation_accuracy
        metrics['validation_unknown_recall']=validation_unknown_recall
    (ROOT/'training_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(f'Moteur: {backend}; précision test mesurée: {accuracy:.4f} ({len(y_test)} dessins)')

if __name__=='__main__': main()
