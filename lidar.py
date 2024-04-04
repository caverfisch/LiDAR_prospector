import laspy
import numpy as np
import os

# définir la base du modulo
modulo = 15

current_directory = os.getcwd()

for las_file in os.listdir(current_directory) :
    print(las_file+" est en train d'être traité, patience...")
    if las_file.endswith('.las') :
        file_to_read = las_file
        output_file = "mod_"+file_to_read
       
        # charge le fichier .las et extrait les données non-classifiées et celles du sol
        las_file_content = laspy.read(file_to_read)
        ground = las_file_content.points[las_file_content.classification == 2]
        undefined = las_file_content.points[las_file_content.classification == 1]
       
        x_ground = ground.x
        y_ground = ground.y
        z_ground = ground.z
         
        x_undefined = undefined.x
        y_undefined = undefined.y
        z_undefined = undefined.z
       
        xyz_ground = np.column_stack((x_ground, y_ground, z_ground))
        xyz_undefined = np.column_stack((x_undefined, y_undefined, z_undefined))
       
        xyz = np.concatenate((xyz_ground, xyz_undefined), axis=0)
       
        # crée un nouveau 'header' pour le nouveau fichier .las
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.add_extra_dim(laspy.ExtraBytesParams(name="modulo_z", type=np.float64))
       
        header.offsets = np.min(xyz, axis=0)
        header.scales = np.array([0.1, 0.1, 0.1])
       
        # crée un nouveau fichier .las (nomé 'las') et définition de ses attributs
        las = laspy.LasData(header)
        las.x = xyz[:, 0]
        las.y = xyz[:, 1]
        las.z = xyz[:, 2]
       
        # calcule le modulo de base prédéfinie et ajoute un nouvel attribut avec le résultat
        modulo_z = np.array([(z) % modulo for z in xyz[:, 2]])
        las.modulo_z = modulo_z
       
        # enregistre le nouveau fichier
        las.write(output_file)
       
        print("nouveau fichier '"+output_file+"' créé avec le modulo {} de l'altitude en plus".format(modulo))
print('...terminé !')
