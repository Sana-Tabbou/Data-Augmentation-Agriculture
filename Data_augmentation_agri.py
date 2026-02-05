import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from tqdm import tqdm
import warnings
import timeit
import traceback

warnings.filterwarnings('ignore')

class DataAugmentationReelle:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        print(f"🔍 Recherche dans: {self.dataset_path}")
        self.images = self.load_images_with_fix()
        self.output_dir = Path("DataAugmentation_Results_V4_CORRIGE")
        self.output_dir.mkdir(exist_ok=True)
        print(f" Dossier de sortie: {self.output_dir.absolute()}")
        self.ROTATION_ANGLE = 15 
        self.CONTRAST_FACTOR = 1.4
        self.BRIGHTNESS_FACTOR = 1.3
        
    def load_images_with_fix(self):
        """Charge les chemins des images de manière robuste (inclut les sous-dossiers)."""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.tif', '.tiff', '.webp']
        images = []
        for ext in extensions:
            found = list(self.dataset_path.rglob(f"*{ext}"))
            images.extend(found)
        
        images = list(set(images))
        print(f" TOTAL: {len(images)} images trouvées")
        
        if images:
            print(" Exemples d'images:")
            for img in images[:5]:
                rel_path = img.relative_to(self.dataset_path)
                print(f"   - {rel_path} ({img.stat().st_size // 1024} KB)")
        else:
            print(" Aucune image trouvée. Vérifiez le chemin du dossier.")
            
        return images
    
    def read_image_with_fix(self, image_path):
        """Lecture robuste des images avec gestion des chemins problématiques"""
        try:
            with open(image_path, 'rb') as f:
                image_data = np.frombuffer(f.read(), np.uint8)
            
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image_rgb
            else:
                print(f"   Image vide: {image_path.name}")
                return None
                
        except Exception as e:
            print(f"   ERREUR LECTURE: {image_path.name} - {str(e)}")
            return None
    
    def save_comparison_grid(self, original, augmented_list, titles, filename):
        """Sauvegarde une grille de comparaison"""
        if not augmented_list:
            print(f"   Aucune image augmentée pour {filename}")
            return
            
        print(f" Sauvegarde: {filename}")
        
        n_cols = min(4, len(augmented_list) + 1)
        n_rows = (len(augmented_list) + 1 + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Image originale
        axes[0].imshow(original)
        axes[0].set_title('IMAGE ORIGINALE', fontweight='bold', color='red', fontsize=14)
        axes[0].axis('off')
        
        # Images augmentées
        for idx, (img, title) in enumerate(zip(augmented_list, titles)):
            if idx + 1 < len(axes):
                axes[idx + 1].imshow(img)
                axes[idx + 1].set_title(title, fontweight='bold', fontsize=12)
                axes[idx + 1].axis('off')
        
        for idx in range(len(augmented_list) + 1, len(axes)):
            if idx < len(axes):
                axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        if save_path.exists():
            file_size = save_path.stat().st_size // 1024
            print(f"   FICHIER CRÉÉ: {filename} ({file_size} KB)")
        else:
            print(f"   ÉCHEC CRÉATION: {filename}")
    
    def methodes_simples(self, image):
        """Méthodes Simples """
        aug_images, titles = [], []
        h, w = image.shape[:2]
        
        # 1. Rotation 
        matrix = cv2.getRotationMatrix2D((w/2, h/2), self.ROTATION_ANGLE, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), 
                                borderMode=cv2.BORDER_REFLECT_101,
                                borderValue=(120, 120, 120))
        aug_images.append(rotated)
        titles.append(f"ROTATION {self.ROTATION_ANGLE}°")
        
        # 2. Flip horizontal 
        flipped_h = cv2.flip(image, 1)
        aug_images.append(flipped_h)
        titles.append("FLIP HORIZONTAL")
        
        # 3. Flip diagonal 
        flipped_d = cv2.flip(image, -1)
        aug_images.append(flipped_d)
        titles.append("FLIP DIAGONAL")
        
        # 4. Luminosité via HSV 
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * self.BRIGHTNESS_FACTOR, 0, 255)
        bright = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        aug_images.append(bright)
        titles.append("SOLEIL FORT")
        
        # 5. Contraste CLAHE local 
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_channel)
        lab_clahe = cv2.merge([l_clahe, a, b])
        contrast = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        aug_images.append(contrast)
        titles.append("CONTRASTE CLAHE")
        
        # 6. Recadrage 
        crop_ratio = random.uniform(0.6, 0.8)
        crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
        y = random.randint(0, h - crop_h)
        x = random.randint(0, w - crop_w)
        cropped = image[y:y+crop_h, x:x+crop_w]
        cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
        aug_images.append(cropped)
        titles.append(f"ZOOM {int(crop_ratio*100)}%")
        
        return aug_images, titles
    
    def methodes_keras(self, image):
        """Méthodes Keras OPTIMISÉES """
        try:
            datagen = ImageDataGenerator(
                rotation_range=20,            
                width_shift_range=0.15,      
                height_shift_range=0.15,    
                shear_range=0.15,            
                zoom_range=0.25,             
                horizontal_flip=True,
                vertical_flip=False,         
                brightness_range=[0.8, 1.4], 
                fill_mode='reflect'
            )
            
            aug_images, titles = [], []
            img_expanded = np.expand_dims(image, 0)
            
            for i in range(6):
                batch = next(datagen.flow(img_expanded, batch_size=1))
                aug_img = batch[0].astype('uint8')
                aug_images.append(aug_img)
                titles.append(f"KERAS AUG {i+1}")
            
            return aug_images, titles
        except Exception as e:
            print(f"   Erreur Keras: {e}")
            return [], []
    
    def methodes_avancees(self, image):
        """Méthodes Albumentations OPTIMISÉES pour agriculture"""
        aug_images, titles = [], []
        
        try:
            # 1. Motion blur léger 
            transform = A.MotionBlur(blur_limit=(3, 7), allow_shifted=True, p=1)
            result = transform(image=image)
            if 'image' in result:
                aug_images.append(result['image'])
                titles.append("FLOU MOUVEMENT")
        except Exception as e:
            print(f"   MotionBlur échoué: {e}")
        
        try:
            # 2. Brightness/Contrast optimisé
            transform = A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                brightness_by_max=True,
                p=1
            )
            result = transform(image=image)
            if 'image' in result:
                aug_images.append(result['image'])
                titles.append("LUMIÈRE/CONTRASTE")
        except Exception as e:
            print(f"   BrightnessContrast échoué: {e}")
        
        try:
            # 3. Hue/Saturation 
            transform = A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=40,
                val_shift_limit=20,
                p=1
            )
            result = transform(image=image)
            if 'image' in result:
                aug_images.append(result['image'])
                titles.append("SATURATION VERT")
        except Exception as e:
            print(f"   HueSaturation échoué: {e}")
        
        try:
            # 4. Bruit gaussien léger
            transform = A.GaussNoise(var_limit=(10, 30), mean=0, p=1)
            result = transform(image=image)
            if 'image' in result:
                aug_images.append(result['image'])
                titles.append("BRUIT CAPTEUR")
        except Exception as e:
            print(f"   GaussNoise échoué: {e}")
        
        try:
            # 5. CLAHE amélioré
            transform = A.CLAHE(
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                p=1
            )
            result = transform(image=image)
            if 'image' in result:
                aug_images.append(result['image'])
                titles.append("CLAHE LOCAL")
        except Exception as e:
            print(f"   CLAHE échoué: {e}")
        
        try:
            transform = A.RandomFog(
                fog_coef_lower=0.1,
                fog_coef_upper=0.3,
                alpha_coef=0.08,
                p=1
            )
            result = transform(image=image)
            if 'image' in result:
                aug_images.append(result['image'])
                titles.append("BRUME LÉGÈRE")
        except Exception as e:
            print(f"   RandomFog échoué: {e}")
        
        return aug_images, titles
    
    def methodes_agricoles(self, image):
        """Méthodes Agricoles OPTIMISÉES """
        aug_images, titles = [], []
        
        try:
            h, w = image.shape[:2]
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # 1. Matin brumeux 
            morning = hsv.copy()
            morning[:,:,0] = np.clip(morning[:,:,0] - 10, 0, 179)  
            morning[:,:,1] = np.clip(morning[:,:,1] * 0.7, 0, 255)  
            morning[:,:,2] = np.clip(morning[:,:,2] * 0.9, 0, 255)  
            morning_rgb = cv2.cvtColor(morning.astype(np.uint8), cv2.COLOR_HSV2RGB)
            aug_images.append(morning_rgb)
            titles.append("MATIN BRUMEUX")
            
            # 2. Milieu de journée 
            midday = hsv.copy()
            midday[:,:,1] = np.clip(midday[:,:,1] * 1.3, 0, 255)  
            midday[:,:,2] = np.clip(midday[:,:,2] * 1.4, 0, 255)  
            midday_rgb = cv2.cvtColor(midday.astype(np.uint8), cv2.COLOR_HSV2RGB)
            aug_images.append(midday_rgb)
            titles.append("MIDI ENSOLEILLÉ")
            
            # 3. Sol sec 
            dry = hsv.copy()
            dry[:,:,0] = np.clip(dry[:,:,0] + 15, 0, 179)  
            dry[:,:,1] = np.clip(dry[:,:,1] * 0.6, 0, 255)  
            dry[:,:,2] = np.clip(dry[:,:,2] * 1.2, 0, 255)  
            dry_rgb = cv2.cvtColor(dry.astype(np.uint8), cv2.COLOR_HSV2RGB)
            aug_images.append(dry_rgb)
            titles.append("SOL SÉCHERESSE")
            
            # 4. Sol humide 
            wet = hsv.copy()
            wet[:,:,0] = np.clip(wet[:,:,0] - 5, 0, 179)  
            wet[:,:,1] = np.clip(wet[:,:,1] * 1.4, 0, 255)  
            wet[:,:,2] = np.clip(wet[:,:,2] * 0.85, 0, 255)  
            wet_rgb = cv2.cvtColor(wet.astype(np.uint8), cv2.COLOR_HSV2RGB)
            aug_images.append(wet_rgb)
            titles.append("SOL HUMIDE")
            
            # 5. Ombre nuageuse réaliste
            shadow = self.add_shadow_effect_realiste(image)
            aug_images.append(shadow)
            titles.append("OMBRE NUAGEUSE")
            
            # 6. Amélioration de contraste spécifique agriculture
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            
            # Égalisation d'histogramme adaptative sur le canal Y
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            y_eq = clahe.apply(y)
            
            # Augmentation légère de la saturation 
            cr_eq = cv2.add(cr, 10)
            
            ycrcb_eq = cv2.merge([y_eq, cr_eq, cb])
            agri_contrast = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)
            aug_images.append(agri_contrast)
            titles.append("CONTRASTE AGRICOLE")
            
        except Exception as e:
            print(f"   Erreur méthodes agricoles: {e}")
            traceback.print_exc()
            
        return aug_images, titles
    
    def add_shadow_effect_realiste(self, image):
        """Ajoute un effet d'ombre nuageuse """
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        
        # Position aléatoire de l'ombre
        center_x = random.randint(w//4, 3*w//4)
        center_y = random.randint(h//4, 3*h//4)
        radius = min(h, w) // random.randint(3, 5)
        
        # Créer un gradient d'ombre
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Fonction d'atténuation douce
        attenuation = np.clip(1.0 - (distance / radius)**2, 0.3, 1.0)
        
        # Appliquer l'ombre à chaque canal
        shadowed = image.astype(np.float32)
        for c in range(3):
            shadowed[:,:,c] = shadowed[:,:,c] * attenuation
        
        return np.clip(shadowed, 0, 255).astype(np.uint8)
    
    def add_cloud_shadow(self, image):
        """Ajoute une ombre de nuage """
        h, w = image.shape[:2]
        
        # Créer une texture de nuage avec du bruit
        cloud = np.random.randn(h//10, w//10).astype(np.float32)
        cloud = cv2.GaussianBlur(cloud, (5, 5), 2)
        cloud = cv2.resize(cloud, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Normaliser et créer un masque
        cloud = (cloud - cloud.min()) / (cloud.max() - cloud.min())
        cloud_mask = np.clip(cloud * 0.6 + 0.4, 0.4, 1.0)
        
        # Appliquer le masque
        shadowed = image.astype(np.float32)
        for c in range(3):
            shadowed[:,:,c] = shadowed[:,:,c] * cloud_mask
        
        return np.clip(shadowed, 0, 255).astype(np.uint8)

    def measure_time_performance(self, image: np.ndarray, augmentation_pipeline: A.Compose, 
                                 title: str, num_runs: int = 100):
        """Mesure le temps d'exécution d'une transformation"""
        if image is None:
            print(f"    ERREUR: L'image est vide pour '{title}'")
            return
            
        try:
            stmt = "augmentation_pipeline(image=image)['image']"
            setup = {
                'augmentation_pipeline': augmentation_pipeline,
                'image': image
            }
            
            time_taken = timeit.timeit(stmt, globals=setup, number=num_runs)
            avg_time_ms = (time_taken / num_runs) * 1000
            
            print(f"    '{title}': {avg_time_ms:.2f} ms/image")
        
        except Exception as e:
            print(f"    ERREUR temps pour '{title}': {e}")

    def traiter_images_reelles(self):
        """Traite les images avec toutes les méthodes """
        print("\n DÉMARRAGE TRAITEMENT IMAGES RÉELLES ")
        print("=" * 50)
        
        if not self.images:
            print(" Aucune image trouvée!")
            return 0
        
        images_traitees = 0
        
        # Limiter au nombre d'images disponibles
        num_images_a_traiter = min(5, len(self.images))
        
        for idx, img_path in enumerate(self.images[:num_images_a_traiter]):
            print(f"\n{'='*50}")
            print(f" TRAITEMENT IMAGE {idx+1}/{num_images_a_traiter}: {img_path.name}")
            print(f"{'='*50}")
            
            image = self.read_image_with_fix(img_path)
            if image is None:
                print(f"   Impossible de lire l'image")
                continue
            
            images_traitees += 1
            nom_base = img_path.stem
            
            print(f"   Dimensions: {image.shape[1]}x{image.shape[0]}")
            print(f"   Type: {image.dtype}, Range: [{image.min()}, {image.max()}]")
            
            # Vérifier la qualité de l'image
            if image.mean() < 10 or image.mean() > 245:
                print(f"   Image potentiellement surexposée/sous-exposée")
            
            try:
                print("\n   Méthodes simples ")
                aug_simple, titles_simple = self.methodes_simples(image)
                if aug_simple:
                    self.save_comparison_grid(image, aug_simple, titles_simple, 
                                            f"{idx+1}_Simples_{nom_base}.png")
                    print(f"   {len(aug_simple)} transformations simples sauvegardées")
                else:
                    print(f"   Aucune transformation simple générée")
            except Exception as e:
                print(f"   Erreur méthodes simples: {e}")
                traceback.print_exc()
            
            try:
                print("\n   Méthodes avancées")
                aug_avance, titles_avance = self.methodes_avancees(image)
                if aug_avance:
                    self.save_comparison_grid(image, aug_avance, titles_avance,
                                            f"{idx+1}_Avancees_{nom_base}.png")
                    print(f"   {len(aug_avance)} transformations avancées sauvegardées")
                else:
                    print(f"   Aucune transformation avancée générée")
            except Exception as e:
                print(f"   Erreur méthodes avancées: {e}")
                traceback.print_exc()
            
            try:
                print("\n   Méthodes agricoles ")
                aug_agri, titles_agri = self.methodes_agricoles(image)
                if aug_agri:
                    self.save_comparison_grid(image, aug_agri, titles_agri,
                                            f"{idx+1}_Agricoles_{nom_base}.png")
                    print(f"   {len(aug_agri)} transformations agricoles sauvegardées")
                else:
                    print(f"   Aucune transformation agricole générée")
            except Exception as e:
                print(f"   Erreur méthodes agricoles: {e}")
                traceback.print_exc()
            
            try:
                print("\n  Méthodes Keras...")
                aug_keras, titles_keras = self.methodes_keras(image)
                if aug_keras:
                    self.save_comparison_grid(image, aug_keras, titles_keras,
                                            f"{idx+1}_Keras_{nom_base}.png")
                    print(f"  {len(aug_keras)} transformations Keras sauvegardées")
            except Exception as e:
                print(f"   Erreur méthodes Keras: {e}")
                traceback.print_exc()
            
            # Mesure des performances
            print("\n   MESURE DES PERFORMANCES :")
            
            # Pipeline simple
            simple_pipeline = A.Compose([
                A.Rotate(limit=self.ROTATION_ANGLE, border_mode=cv2.BORDER_REFLECT_101, p=1),
                A.HorizontalFlip(p=0.5),
            ])
            self.measure_time_performance(image, simple_pipeline, "Pipeline simple", num_runs=50)
            
            # Pipeline complexe
            complex_pipeline = A.Compose([
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.CLAHE(p=0.3),
                A.RandomFog(p=0.2),
            ])
            self.measure_time_performance(image, complex_pipeline, "Pipeline complexe", num_runs=30)
            
            print(f"\n IMAGE {idx+1} TERMINÉE: {nom_base}")
        
        return images_traitees
    
    def generer_dataset_complet(self):
        """Génère un dataset complet avec toutes les images"""
        print(f"\n GÉNÉRATION DATASET COMPLET ")
        print("=" * 50)
        
        dataset_dir = self.output_dir / "Dataset_Complet"
        dataset_dir.mkdir(exist_ok=True)
        
        images_traitees = 0
        
        # Limiter pour tester
        images_a_traiter = self.images[:20] if len(self.images) > 20 else self.images
        
        for img_path in tqdm(images_a_traiter, desc="Génération dataset"):
            image = self.read_image_with_fix(img_path)
            if image is None:
                continue
                
            images_traitees += 1
            nom_base = img_path.stem
            
            # Sauvegarder original
            original_path = dataset_dir / f"{nom_base}_original.jpg"
            cv2.imwrite(str(original_path), 
                       cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Pipeline d'augmentation optimisé pour agriculture
            augmentation_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                A.CLAHE(p=0.3),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.2, p=0.1),
            ])
            
            # Générer 3 versions augmentées
            for i in range(3):
                try:
                    augmented = augmentation_pipeline(image=image)['image']
                    aug_path = dataset_dir / f"{nom_base}_aug_{i+1:02d}.jpg"
                    cv2.imwrite(str(aug_path), 
                               cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(f" Erreur augmentation {i+1}: {e}")
                    continue
        
        print(f" DATASET CRÉÉ: {images_traitees} images traitées")
        print(f" Dossier: {dataset_dir.absolute()}")
    
    def verifier_resultats(self):
        print("\n🔍 VÉRIFICATION DES RÉSULTATS...")
        
        patterns = [
            "*_Simples*.png",
            "*_Avancees*.png", 
            "*_Agricoles*.png",
            "*_Keras*.png"
        ]
        
        for pattern in patterns:
            fichiers = list(self.output_dir.glob(pattern))
            if fichiers:
                print(f"\n {len(fichiers)} fichiers {pattern} :")
                for f in fichiers[:3]:
                    taille = f.stat().st_size // 1024
                    print(f"    {f.name} ({taille} KB)")
            else:
                print(f"\n Aucun fichier {pattern} trouvé !")
        
        # Vérifier le dossier complet
        tous_fichiers = list(self.output_dir.glob("*.*"))
        print(f"\n TOTAL: {len(tous_fichiers)} fichiers dans {self.output_dir}")


def main():
    print("=" * 70)
    print(" DATA AUGMENTATION ")
    print("=" * 70)
    
    # Chemin exact 
    chemin_drone = r"C:\Users\sanat\Downloads\Images pour Sana\Vue aerienne Drone"
    
    print(f"\n Chemin spécifié: {chemin_drone}")
    
    # Vérifier si le chemin existe
    if not os.path.exists(chemin_drone):
        print(f"\n ERREUR: Le dossier n'existe pas!")
        print(f"   Vérifiez que le chemin est correct:")
        print(f"   {chemin_drone}")
        
        chemin_alternative = r"C:\Users\sanat\Downloads\Images pour Sana\Vue aérienne Drone"
        print(f"\n Essai avec chemin alternatif: {chemin_alternative}")
        
        if os.path.exists(chemin_alternative):
            print(f" Chemin alternatif trouvé!")
            chemin_drone = chemin_alternative
        else:
            chemin_drone = input(" Chemin: ").strip()
            
            if not chemin_drone:
                print(" ")
                chemin_drone = os.getcwd()
    
    print(f"\n Dossier confirmé: {chemin_drone}")
    
    # Initialiser le traitement
    traitement = DataAugmentationReelle(chemin_drone)
    
    if not traitement.images:
        print("\n AUCUNE IMAGE TROUVÉE DANS CE DOSSIER!")
        try:
            for item in Path(chemin_drone).iterdir():
                print(f"   - {item.name} ({'dossier' if item.is_dir() else 'fichier'})")
        except:
            print("   Impossible de lire le contenu du dossier")
        return
    
    print(f"\n PHASE 1: GÉNÉRATION DES VISUALISATIONS")
    print("-" * 50)
    
    images_traitees = traitement.traiter_images_reelles()
    
    if images_traitees == 0:
        print(" AUCUNE IMAGE N'A PU ÊTRE TRAITÉE!")
        return
    
    # Vérifier les résultats
    traitement.verifier_resultats()
    
    # Demander pour le dataset
    print("\n" + "=" * 50)
    reponse = input(" Voulez-vous générer le dataset complet corrigé? (o/n): ")
    
    if reponse.lower() in ['o', 'oui', 'y', 'yes']:
        print("\n GÉNÉRATION DU DATASET COMPLET...")
        traitement.generer_dataset_complet()
    
    # Résultats finaux
    print("\n" + "=" * 70)
    print(" TRAITEMENT TERMINÉ AVEC SUCCÈS!")
    print("=" * 70)
    
    result_path = traitement.output_dir.absolute()
    print(f" DOSSIER RÉSULTATS: {result_path}")
    
    # Ouvrir le dossier
    try:
        os.startfile(str(result_path))
        print("\n ")
    except:
        print(f"\n OUVREZ MANUELLEMENT: {result_path}")
    
    print("\n" + "=" * 70)
    print("=" * 70)


if __name__ == "__main__":
    main()