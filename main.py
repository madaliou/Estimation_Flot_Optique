import os

print('*********************************************')
print('*Bienvenue dans le programme du flot optique*')
print('*********************************************')

print('Choisissez un menu pour continuer : ')

options = ['1-Estimation du flot', '2-Visualisation Norme', '3-Segmentation Objet Plus Vite',
            '4-Segmentation Seuillage Differentes Vitesses',  '5-Segmentation Seuillage KMeans']

for i in range(len(options)): 
    print(options[i])

option = int(input("Entrez votre choix : "))

def switch(option):
    if option == 1:
        os.system('python Estimation_Flot.py')
        return
    elif option == 2:
        os.system('python Visualisation_norme.py')
        return
    elif option == 3:
        seuil = input("Entrez le seuil : ")
        os.system('python segmentation_objets_plus_rapide_seuillage.py {} {}'.format(seuil))
        return
    elif option == 4:
        seuil_1 = input("Entrez le seuil1 : ")
        seuil_2 = input("Entrez le seuil2 : ")
        os.system('python segmentation_seuillage_differentes_vitesses.py {} {}'.format(seuil_1, seuil_2))
        return
    elif option == 5:
        seuil_1 = input("Entrez le seuil1 : ")
        seuil_2 = input("Entrez le seuil2 : ")
        K = input("Entrez K : ")
        os.system('python segmentation_seuillage_KMeans.py {} {} {}'.format(seuil_1, seuil_2, K))
        return

switch(option)