import numpy as np
from skimage import measure, morphology

class Morphologie:
    def __init__(self):
         self.image = None

    def choisir_operation_morphologique(self, image_segmentee):
        regions = measure.regionprops(measure.label(image_segmentee))
        
        if not regions:
            return "Ouverture"
        
        tailles = [region.area for region in regions]
        taille_moyenne = np.mean(tailles)
        discontinuites = np.std(tailles) 

        if taille_moyenne < 10:
            return "Ouverture"

        if discontinuites > 20:
            return "Fermeture"

        if taille_moyenne > 50:
            return "Dilatation"

        return "Erosion"

    def choisir_taille_disque(self, image_segmentee):
        regions = measure.regionprops(measure.label(image_segmentee))
        if not regions:
            return 1  
        
        tailles = [region.area for region in regions]
        taille_moyenne = np.mean(tailles)
        
        rayon = max(1, int(np.sqrt(taille_moyenne / np.pi)))
        return rayon

    def appliquer_operation(self, image_segmentee):
        operation = self.choisir_operation_morphologique(image_segmentee)
        print(f"Opération choisie : {operation}")
        
        rayon = self.choisir_taille_disque(image_segmentee)
        selem = morphology.disk(rayon)

        if operation == "Dilatation":
            return self.Dilatation(image_segmentee, selem)
        elif operation == "Erosion":
            return self.Erosion(image_segmentee, selem)
        elif operation == "Ouverture":
            return self.Ouverture(image_segmentee, selem)
        elif operation == "Fermeture":
            return self.Fermeture(image_segmentee, selem)

        return image_segmentee

    
    def pipeline_morphologique(self, image_segmentee):
        rayon = self.choisir_taille_disque(image_segmentee)
        selem = morphology.disk(rayon)

        # 1. Ouverture (nettoyage du bruit)
        image_ouverte = self.Ouverture(image_segmentee, selem)
        
        # 2. Fermeture (combler les trous)
        image_fermee = self.Fermeture(image_ouverte, selem)
        
        # 3. Érosion (affiner les coins)
        image_erodee = self.Erosion(image_fermee, selem)
        
        # 4. Dilatation (renforcer les coins)
        image_dilatee = self.Dilatation(image_erodee, selem)

        return image_dilatee

    def Dilatation(self, image, selem):
        return morphology.binary_dilation(image, selem)

    def Erosion(self, image, selem):
        return morphology.binary_erosion(image, selem)

    def Ouverture(self, image, selem):
        eroded = self.Erosion(image, selem)
        opened = self.Dilatation(eroded, selem)
        return opened

    def Fermeture(self, image, selem):
        dilated = self.Dilatation(image, selem)
        closed = self.Erosion(dilated, selem)
        return closed