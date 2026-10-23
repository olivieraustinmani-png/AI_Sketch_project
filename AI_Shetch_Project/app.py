"""AI Sketch — interface de dessin et classification."""
from pathlib import Path
import sys

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QFont
from PyQt5.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget)

from predict import ModelPredictor


class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(540, 410)
        self.image = QImage(900, 620, QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.last = None
        self.brush = 9
        self.setCursor(Qt.CrossCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.white)
        painter.drawImage(self.rect(), self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last = event.pos()
            self._line(event.pos(), event.pos())

    def mouseMoveEvent(self, event):
        if self.last is not None and event.buttons() & Qt.LeftButton:
            self._line(self.last, event.pos())
            self.last = event.pos()

    def mouseReleaseEvent(self, event):
        self.last = None

    def _line(self, start, end):
        p = QPainter(self.image)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor('#17243b'), self.brush, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        a = QPoint(round(start.x()*self.image.width()/self.width()), round(start.y()*self.image.height()/self.height()))
        b = QPoint(round(end.x()*self.image.width()/self.width()), round(end.y()*self.image.height()/self.height()))
        p.drawLine(a, b); p.end(); self.update()

    def clear(self):
        self.image.fill(Qt.white); self.update()

    def has_ink(self):
        for y in range(0, self.image.height(), 4):
            for x in range(0, self.image.width(), 4):
                if QColor(self.image.pixel(x,y)).value() < 245: return True
        return False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AI Sketch | Studio de dessin intelligent')
        self.resize(1180, 790); self.setMinimumSize(940, 680)
        self.predictor = ModelPredictor()
        root = QWidget(); root.setObjectName('root')
        outer = QHBoxLayout(root); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)

        side = QFrame(); side.setObjectName('sidebar'); side.setFixedWidth(235)
        sl = QVBoxLayout(side); sl.setContentsMargins(23,27,23,22); sl.setSpacing(12)
        brand = QLabel('✳  AI SKETCH'); brand.setObjectName('brand')
        sl.addWidget(brand)
        sl.addSpacing(30)
        sl.addWidget(self._side_label('VOTRE ESPACE'))
        nav = QLabel('✎   Atelier de dessin'); nav.setObjectName('navActive'); sl.addWidget(nav)
        sl.addSpacing(20); sl.addWidget(self._side_label('CATÉGORIES'))
        for icon, name in [('⌂','Maison'),('♣','Arbre'),('▰','Voiture'),('♡','Chat'),('☼','Soleil'),('？','Inconnu')]:
            item=QLabel(f'{icon}   {name}'); item.setObjectName('category'); sl.addWidget(item)
        sl.addStretch(1)
        note=QLabel('Dessinez librement,\nl’IA devine votre idée.'); note.setObjectName('sideNote'); sl.addWidget(note)
        version=QLabel('PROJET ANI-2083  •  v1.0'); version.setObjectName('version'); sl.addWidget(version)
        outer.addWidget(side)

        main = QWidget(); ml = QVBoxLayout(main); ml.setContentsMargins(34,25,34,26); ml.setSpacing(17)
        top=QHBoxLayout(); heading=QVBoxLayout(); title=QLabel('Atelier de dessin'); title.setObjectName('title')
        subtitle=QLabel('Donnez vie à une idée, un trait à la fois.'); subtitle.setObjectName('subtitle')
        heading.addWidget(title); heading.addWidget(subtitle); top.addLayout(heading); top.addStretch()
        badge=QLabel('●  MODE DESSIN'); badge.setObjectName('badge'); top.addWidget(badge,0,Qt.AlignVCenter); ml.addLayout(top)

        tools=QFrame(); tools.setObjectName('card'); tl=QHBoxLayout(tools); tl.setContentsMargins(18,11,18,11); tl.setSpacing(12)
        sizeTitle=QLabel('ÉPAISSEUR'); sizeTitle.setObjectName('toolLabel'); tl.addWidget(sizeTitle)
        slider=QSlider(Qt.Horizontal); slider.setRange(2,28); slider.setValue(9); slider.setFixedWidth(130)
        slider.valueChanged.connect(lambda value:setattr(self.canvas,'brush',value)); tl.addWidget(slider)
        self.size_value=QLabel('9 px'); self.size_value.setObjectName('sizeValue'); tl.addWidget(self.size_value)
        slider.valueChanged.connect(lambda value:self.size_value.setText(f'{value} px'))
        tl.addStretch()
        clear=QPushButton('⌫  Effacer'); clear.setObjectName('secondaryButton'); clear.clicked.connect(self._clear); tl.addWidget(clear)
        save=QPushButton('↓  Enregistrer'); save.setObjectName('secondaryButton'); save.clicked.connect(self._save); tl.addWidget(save)
        ml.addWidget(tools)

        board=QFrame(); board.setObjectName('canvasCard'); bl=QVBoxLayout(board); bl.setContentsMargins(13,13,13,13)
        self.canvas=Canvas(); self.canvas.setObjectName('canvas'); bl.addWidget(self.canvas,1); ml.addWidget(board,1)

        result=QFrame(); result.setObjectName('resultCard'); rl=QHBoxLayout(result); rl.setContentsMargins(18,14,18,14); rl.setSpacing(14)
        icon=QLabel('✦'); icon.setObjectName('resultIcon'); rl.addWidget(icon)
        texts=QVBoxLayout(); self.result_title=QLabel('Prêt à reconnaître votre dessin'); self.result_title.setObjectName('resultTitle')
        self.result_detail=QLabel('Les prédictions apparaîtront ici après analyse.'); self.result_detail.setObjectName('resultDetail')
        texts.addWidget(self.result_title); texts.addWidget(self.result_detail); rl.addLayout(texts,1)
        self.recognize=QPushButton('✦  Reconnaître'); self.recognize.setObjectName('primaryButton'); self.recognize.clicked.connect(self._recognize); rl.addWidget(self.recognize)
        ml.addWidget(result)
        foot=QLabel('AI Sketch choisit parmi cinq catégories. Le score est indicatif et ne garantit pas une prédiction exacte.')
        foot.setObjectName('footnote'); ml.addWidget(foot)
        outer.addWidget(main,1); self.setCentralWidget(root)
        self._style()
        if not self.predictor.available:
            self.result_title.setText('Reconnaissance à configurer')
            self.result_detail.setText('Le modèle IA sera disponible après son entraînement.')
        else:
            label_names={'house':'Maison','tree':'Arbre','car':'Voiture','cat':'Chat','sun':'Soleil','unknown':'Inconnu'}
            metrics_path=Path(__file__).resolve().parent/'training_metrics.json'
            try:
                import json
                metrics=json.loads(metrics_path.read_text(encoding='utf-8'))
                self.result_title.setText('Modèle prêt · prototype')
                self.result_detail.setText(f"Entraîné sur {metrics['samples_total']} dessins · précision test : {metrics['test_accuracy']:.0%} ({metrics['test']} exemples seulement)")
            except (OSError,ValueError,KeyError):
                self.result_title.setText('Modèle prêt à analyser')
                self.result_detail.setText('Les scores affichés sont indicatifs.')
            self.label_names=label_names

    @staticmethod
    def _side_label(text):
        label=QLabel(text); label.setObjectName('sideSection'); return label

    def _style(self):
        self.setStyleSheet('''
        QWidget#root { background:#f4f6fa; color:#17243b; font-family:'Segoe UI'; font-size:13px; }
        QFrame#sidebar { background:#17243b; border:0; }
        QLabel#brand { color:#fff; font-size:19px; font-weight:800; letter-spacing:1px; }
        QLabel#sideSection { color:#8390a5; font-size:10px; font-weight:700; letter-spacing:1.4px; padding:4px 2px; }
        QLabel#navActive { color:#fff; background:#2b3b56; border:1px solid #425573; border-radius:10px; padding:12px 10px; font-weight:600; }
        QLabel#category { color:#ccd4e1; padding:8px 9px; font-size:13px; }
        QLabel#sideNote { color:#b3bfd0; background:#21324e; border:1px solid #30435f; border-radius:12px; padding:13px; line-height:1.4; }
        QLabel#version { color:#73819b; font-size:9px; padding:10px 1px 0; }
        QLabel#title { color:#17243b; font-size:25px; font-weight:750; }
        QLabel#subtitle { color:#78859a; font-size:13px; }
        QLabel#badge { color:#526b99; background:#e9eef8; border:1px solid #d9e1f0; border-radius:12px; padding:8px 11px; font-size:10px; font-weight:700; }
        QFrame#card, QFrame#canvasCard, QFrame#resultCard { background:#fff; border:1px solid #e1e6ef; border-radius:15px; }
        QFrame#canvasCard { border-radius:18px; }
        QWidget#canvas { background:#fff; border:1px dashed #dce3ed; border-radius:12px; }
        QLabel#toolLabel { color:#748198; font-size:10px; font-weight:750; letter-spacing:1px; }
        QLabel#sizeValue { color:#394b68; min-width:35px; font-size:12px; }
        QSlider::groove:horizontal { height:5px; background:#e5eaf2; border-radius:2px; }
        QSlider::sub-page:horizontal { background:#5d78ed; border-radius:2px; }
        QSlider::handle:horizontal { background:#fff; border:2px solid #5d78ed; width:14px; margin:-6px 0; border-radius:8px; }
        QPushButton { min-height:19px; padding:9px 15px; border-radius:10px; font-weight:650; }
        QPushButton#secondaryButton { color:#40516e; background:#fff; border:1px solid #cfd7e4; }
        QPushButton#secondaryButton:hover { background:#f2f5fa; border-color:#9eabc0; }
        QPushButton#primaryButton { color:#fff; background:#526fe4; border:1px solid #405bc8; padding:11px 19px; }
        QPushButton#primaryButton:hover { background:#405fda; }
        QPushButton#primaryButton:pressed { background:#354fbd; }
        QLabel#resultIcon { color:#526fe4; background:#edf0ff; border-radius:20px; min-width:38px; max-width:38px; min-height:38px; max-height:38px; qproperty-alignment:AlignCenter; font-size:19px; }
        QLabel#resultTitle { color:#24344f; font-size:14px; font-weight:700; }
        QLabel#resultDetail { color:#8994a6; font-size:11px; }
        QLabel#footnote { color:#8792a3; font-size:10px; padding-left:3px; }
        ''')

    def _clear(self):
        self.canvas.clear(); self.result_title.setText('Canevas effacé'); self.result_detail.setText('Dessinez une nouvelle forme pour continuer.')

    def _save(self):
        path,_=QFileDialog.getSaveFileName(self,'Enregistrer le dessin',str(Path('saved_drawings')/'dessin.png'),'Images PNG (*.png)')
        if path:
            if not path.lower().endswith('.png'): path += '.png'
            Path(path).parent.mkdir(parents=True,exist_ok=True)
            if self.canvas.image.save(path,'PNG'):
                self.result_title.setText('Dessin enregistré'); self.result_detail.setText(path)
            else: QMessageBox.critical(self,'Erreur','Impossible d’enregistrer cette image.')

    def _recognize(self):
        if not self.canvas.has_ink():
            self.result_title.setText('Votre canevas est vide'); self.result_detail.setText('Tracez d’abord un dessin, puis relancez la reconnaissance.'); return
        if not self.predictor.available:
            self.result_title.setText('Modèle IA non entraîné'); self.result_detail.setText('Suivez les instructions du README pour entraîner le modèle.'); return
        try:
            label, confidence=self.predictor.predict(self.canvas.image)
            display_label=getattr(self,'label_names',{}).get(label,label)
            if label == 'unknown':
                self.result_title.setText('Résultat : Inconnu')
                self.result_detail.setText('Ce dessin ne correspond pas suffisamment aux classes enregistrées.')
            else:
                self.result_title.setText(f'Résultat : {display_label}  ·  {confidence:.1%}')
                self.result_detail.setText('Score indicatif. Une prédiction peut être erronée.')
        except Exception as exc:
            QMessageBox.critical(self,'Erreur de prédiction',str(exc))


if __name__=='__main__':
    app=QApplication(sys.argv); window=MainWindow(); window.show(); sys.exit(app.exec_())
