"""Prétraitement et prédiction partagés entre l'entraînement et l'application."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'ai_sketch_model.keras'
FALLBACK_MODEL_PATH = ROOT / 'ai_sketch_model.joblib'
LABELS_PATH = ROOT / 'labels.json'
IMAGE_SIZE = 64


def preprocess(image):
    """Convertit dessin ou image PIL en tenseur gris 64x64, trait clair sur fond noir."""
    if not isinstance(image, Image.Image):
        # QImage n'expose pas le protocole NumPy : lire son buffer avec le
        # stride réel (bytesPerLine) évite le tableau dtype=object retourné
        # par np.asarray(QImage).
        if hasattr(image, 'bits') and hasattr(image, 'bytesPerLine'):
            width, height = image.width(), image.height()
            stride = image.bytesPerLine()
            pointer = image.bits()
            pointer.setsize(image.sizeInBytes())
            raw = np.frombuffer(pointer, dtype=np.uint8).reshape(height, stride)
            fmt = image.format()
            if fmt in (image.Format_RGB32, image.Format_ARGB32, image.Format_ARGB32_Premultiplied):
                bgra = raw[:, :width*4].reshape(height, width, 4)
                rgb = bgra[:, :, [2, 1, 0]].copy()
                image = Image.fromarray(rgb, mode='RGB')
            elif fmt == image.Format_Grayscale8:
                gray = raw[:, :width].copy()
                image = Image.fromarray(gray, mode='L')
            else:
                converted = image.convertToFormat(image.Format_RGB888)
                stride = converted.bytesPerLine()
                pointer = converted.bits(); pointer.setsize(converted.sizeInBytes())
                raw = np.frombuffer(pointer, dtype=np.uint8).reshape(height, stride)
                rgb = raw[:, :width*3].reshape(height, width, 3).copy()
                image = Image.fromarray(rgb, mode='RGB')
        else:
            array = np.asarray(image)
            if array.dtype == object:
                raise TypeError('Format d’image non pris en charge pour la reconnaissance.')
            image = Image.fromarray(array)
    image = ImageOps.grayscale(image)
    # Dessins QuickDraw ont traits noirs sur fond blanc; l'app a la même convention.
    # Inverser fournit à la fois un fond sombre et des traits clairs pour le CNN.
    image = ImageOps.invert(image)
    bbox = image.point(lambda x: 255 if x > 24 else 0).getbbox()
    if bbox:
        image = image.crop(bbox)
        image.thumbnail((IMAGE_SIZE-8, IMAGE_SIZE-8), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), 0)
    if bbox:
        canvas.paste(image, ((IMAGE_SIZE-image.width)//2, (IMAGE_SIZE-image.height)//2))
    return np.asarray(canvas, dtype=np.float32)[..., None] / 255.0


def nearest_distances(batch, references, batch_size=8):
    vectors=np.asarray(batch,dtype=np.float32).reshape(len(batch),-1)
    references=np.asarray(references,dtype=np.float32).reshape(len(references),-1)
    ref_norm=np.sum(references*references,axis=1)[None,:]
    result=[]
    for start in range(0,len(vectors),batch_size):
        chunk=vectors[start:start+batch_size]
        d2=(np.sum(chunk*chunk,axis=1)[:,None]+ref_norm-2*chunk@references.T)
        result.append(np.sqrt(np.maximum(d2,0)/vectors.shape[1]))
    return np.concatenate(result,axis=0)


def numpy_open_set_predict(model, batch):
    """Nearest-neighbour prediction with a validation-calibrated OOD cutoff."""
    exemplars=np.asarray(model['train_x'],dtype=np.float32)
    exemplar_y=np.asarray(model['train_y'],dtype=np.int64)
    distances=nearest_distances(batch,exemplars)
    threshold=float(model['threshold'])
    results=[]
    for row, row_distances in enumerate(distances):
        k=int(np.asarray(model.get('k',7)).reshape(-1)[0])
        nearest=np.argsort(row_distances)[:min(k,len(row_distances))]
        nearest_distance=float(row_distances[nearest[0]])
        if nearest_distance > threshold:
            # Rejet explicite : ce score décrit l'écart au domaine, pas une
            # probabilité calibrée.
            score=nearest_distance/(nearest_distance+threshold+1e-8)
            results.append((int(model['unknown_id']),float(score),nearest_distance))
            continue
        weights=1.0/(row_distances[nearest]+0.02)
        votes=np.zeros(int(model['unknown_id'])+1,dtype=np.float64)
        for idx,weight in zip(nearest,weights): votes[exemplar_y[idx]]+=weight
        chosen=int(np.argmax(votes)); confidence=float(votes[chosen]/max(votes.sum(),1e-12))
        results.append((chosen,confidence,nearest_distance))
    return results


class ModelPredictor:
    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH):
        self.model = None; self.labels = []; self.backend = None
        try:
            self.labels = json.loads(Path(labels_path).read_text(encoding='utf-8'))
            if Path(model_path).is_file():
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path); self.backend = 'keras'
            elif FALLBACK_MODEL_PATH.is_file():
                import joblib
                self.model = joblib.load(FALLBACK_MODEL_PATH); self.backend = 'sklearn'
            elif (ROOT/'ai_sketch_model.npz').is_file():
                archive=np.load(ROOT/'ai_sketch_model.npz')
                self.model={key:archive[key] for key in archive.files}
                self.backend='numpy'
        except (ImportError, OSError, ValueError, json.JSONDecodeError, EOFError):
            pass
        self.available = self.model is not None and bool(self.labels)

    def predict(self, image):
        if not self.available: raise RuntimeError('Modèle absent ou TensorFlow indisponible.')
        x = preprocess(image)[None, ...]
        if self.backend == 'numpy':
            index,confidence,_=numpy_open_set_predict(self.model,x)[0]
            return self.labels[index],confidence
        elif self.backend == 'sklearn':
            scores = np.zeros(len(self.labels), dtype=np.float32)
            probabilities = self.model.predict_proba(x.reshape(1,-1))[0]
            for class_id, score in zip(self.model.classes_, probabilities): scores[int(class_id)] = score
        else:
            scores = np.asarray(self.model.predict(x, verbose=0))[0]
        index = int(np.argmax(scores))
        return self.labels[index], float(scores[index])
