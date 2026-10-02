import cv2
# Importe OpenCV.
# Conséquence : on peut utiliser la webcam, créer des fenêtres,
# lire des images, etc.


cap = cv2.VideoCapture(0)
# Demande à OpenCV d'ouvrir la caméra numéro 0.
# Conséquence : OpenCV essaie d'utiliser la webcam principale du PC.
# Si le PC possède plusieurs caméras, 0 correspond généralement à la webcam intégrée.


if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam")
# Vérifie si la caméra a réellement été ouverte.
# Conséquence : si Windows refuse l'accès ou si aucune caméra n'est disponible,
# le programme s'arrête immédiatement avec un message d'erreur.


print("Webcam détectée !")
# Affiche un message dans PowerShell.
# Conséquence : on sait que l'ouverture de la caméra a réussi.


while True:
    # Crée une boucle qui va tourner continuellement.
    # Conséquence : le programme peut récupérer image après image
    # pour produire le flux vidéo.


    ret, frame = cap.read()
    # Demande une nouvelle image à la webcam.
    #
    # ret   = True/False : indique si l'image a été récupérée correctement.
    # frame = l'image récupérée par la caméra.
    #
    # Conséquence : à chaque tour de boucle, on obtient une nouvelle frame.


    if not ret:
        print("Impossible de lire une image")
        break
    # Vérifie que la caméra nous a bien fourni une image.
    # Conséquence : si la caméra cesse de fonctionner, on sort de la boucle
    # au lieu de continuer avec une image inexistante.


    cv2.imshow("Cursor", frame)
    # Affiche la frame dans une fenêtre appelée "Cursor".
    # Conséquence : tu vois la vidéo de ta webcam.


    key = cv2.waitKey(1) & 0xFF
    # Attend environ 1 milliseconde une interaction clavier.
    # Conséquence : OpenCV peut traiter les événements de la fenêtre
    # et nous donner la touche éventuellement pressée.


    if key == 27:
        break
    # 27 correspond à la touche Échap.
    # Conséquence : appuyer sur Échap permet de sortir de la boucle.


    if cv2.getWindowProperty("Cursor", cv2.WND_PROP_VISIBLE) < 1:
        break
    # Vérifie si la fenêtre est toujours visible.
    # Si tu cliques sur le X de la fenêtre, elle devient invisible.
    # Conséquence : la boucle s'arrête également lorsque tu fermes
    # la fenêtre avec le bouton X.


cap.release()
# Libère la webcam.
# Conséquence : Windows récupère la caméra et un autre programme
# pourra l'utiliser.


cv2.destroyAllWindows()
# Ferme toutes les fenêtres OpenCV restantes.
# Conséquence : le programme se termine proprement.