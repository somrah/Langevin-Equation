9#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sofiane MRAH - M2 Biophysique - Sorbonne Universite/Institut Curie - Team SYKES 
Created on Thu Mar 12 13:54:16 2019
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys
# imporation bibliothèque de commande
import os
# importation bibliothèque de lecture csv
import csv
def mylistdir(directory):
    filelist = os.listdir(directory)
    return [x for x in filelist
    if not (x.startswith('.'))]
 
def adjustFigAspect(fig,aspect=1):
    xsize,ysize = fig.get_size_inches()
	minsize = min(xsize,ysize)
	xlim = .4*minsize/xsize
	ylim = .4*minsize/ysize
	
	if aspect < 1:
		xlim *= aspect
	else:
	    ylim /= aspect
	    fig.subplots_adjust(left=.5-xlim,
	    right=.5+xlim,
	    bottom=.5-ylim,
	    top=.5+ylim)

  
  
liste_matrice_position_x_area=[]
liste_matrice_position_y_area=[]
liste_matrice_position_x=[]
liste_matrice_position_y=[]
matrice_corrections=[]
liste_corr_x=[]
liste_corr_y=[]
liste_area=[]
liste_height=[]
liste_height_1=[]
liste_height_2=[]
liste_width=[]


# petit programme permettant de charger des csv de n'importe quelle taille
maxInt = sys.maxsize
decrement = True
while decrement:
 
	decrement = False
 	try:
 		csv.field_size_limit(maxInt)
 	except OverflowError:
 		maxInt = int(maxInt/10)
 		decrement = True

# répertoire de travail 
os.chdir("/Users/mrah/Desktop/Corrections_coor")
liste_corrections=np.sort(mylistdir("/Users/mrah/Desktop/Corrections_coor"))
print(liste_corrections)
print("Repertoire de travail:")
print(os.getcwd())

#print(liste_name_matrices)
#print(liste_corrections)
# Commenter !!!!!!


print("Debut du programme")



###############################################################################
############################## EXTRACTION DES DONNEES #########################
###############################################################################


# entrer le nom de csv 
for file in liste_corrections :
 
	Nom_du_fichier = file

 # entrer ici le nom de la colonne représentant la sortie du réseau
 #nom_colonne_sortie='DIMi(cm)'
 
 
 nombre_de_colonne=0
 nombre_de_ligne=0
 
 with open(Nom_du_fichier, newline='') as csvfile:
 	reader = csv.reader(csvfile)
 for ro in reader:
	 # Calcul du nombre de colonne
	 nombre_de_colonne=len(ro[0].split(';'))
	 # Calcul du nombre de lignes
	 nombre_de_ligne=nombre_de_ligne+1
 
 
 colonne_nb_calculs=0 
 colonne_dimension=0
 matrice_corrections=np.zeros((nombre_de_ligne,nombre_de_colonne))
 ii=0
 
 with open(Nom_du_fichier, newline='') as csvfile:
 	reader = csv.reader(csvfile)
 for ro in reader:
 	#print(ro[0].split(';')[5].split('\t')[0])
	 for j in range(0,nombre_de_colonne):

	 # si on utilise la méthode EOLE.py, on a une colonne NB_calculs qui ne doit etre convertie en 0

		 if ro[0].split(';')[j].split('\t')[0] == 'Nb_calculs' :
			 colonne_nb_calculs=j

			 # entrer ici le nom de la colonne représentant la sortie du réseau
			 #if ro[0].split(';')[j].split('\t')[0] == nom_colonne_sortie :
			 # colonne_dimension=j

		 try:
			 if type(float(ro[0].split(';')[j].split('\t')[0]))== str :
				 matrice_corrections[ii][j]=0

				 except:
				 continue

				 matrice_corrections[ii][j]= float(ro[0].split(';')[j].split('\t')[0])

				 ii=ii+1

				 # placement de la colonne choisie dans la derniere colonne de la matrice

				 liste_corr_x.append(matrice_corrections[1:,2])
				 liste_corr_y.append(matrice_corrections[1:,3])

os.chdir("/Users/mrah/Desktop/excel-cellules")
liste_name_matrices=np.sort(mylistdir("/Users/mrah/Desktop/excel-cellules"))
#print(liste_name_matrices)



for file in liste_name_matrices :
 
	Nom_du_fichier = file
	# entrer ici le nom de la colonne représentant la sortie du réseau
	#nom_colonne_sortie='DIMi(cm)'


	nombre_de_colonne=0
	nombre_de_ligne=0

	with open(Nom_du_fichier, newline='') as csvfile:
		reader = csv.reader(csvfile)
		for ro in reader:
			# Calcul du nombre de colonne
			nombre_de_colonne=len(ro[0].split(';'))
			# Calcul du nombre de lignes
			nombre_de_ligne=nombre_de_ligne+1


			colonne_nb_calculs=0 
			colonne_dimension=0
			matrice_donnees=np.zeros((nombre_de_ligne,nombre_de_colonne))
			ii=0

		with open(Nom_du_fichier, newline='') as csvfile:
			reader = csv.reader(csvfile)
			for ro in reader:
				#print(ro[0].split(';')[5].split('\t')[0])
			for j in range(0,nombre_de_colonne):

				# si on utilise la méthode EOLE.py, on a une colonne NB_calculs qui ne doit etre convertie en 0

			if ro[0].split(';')[j].split('\t')[0] == 'Nb_calculs' :
				colonne_nb_calculs=j

				# entrer ici le nom de la colonne représentant la sortie du réseau
				#if ro[0].split(';')[j].split('\t')[0] == nom_colonne_sortie :
				# colonne_dimension=j

				try:
					if type(float(ro[0].split(';')[j].split('\t')[0]))== str :
					matrice_donnees[ii][j]=0

				except:

					continue

					matrice_donnees[ii][j]= float(ro[0].split(';')[j].split('\t')[0])
	
					ii=ii+1

				# placement de la colonne choisie dans la derniere colonne de la matrice

				liste_matrice_position_x.append(matrice_donnees[1:,2])
				liste_matrice_position_y.append(matrice_donnees[1:,3])
				liste_matrice_position_x_area.append(matrice_donnees[1:,2])
				liste_matrice_position_y_area.append(matrice_donnees[1:,3])
				liste_area.append(matrice_donnees[1:,1])

				liste_height.append(matrice_donnees[1:,8])
				liste_width.append(matrice_donnees[1:,7])
 
  
###############################################################################
############################## CORRECTION DES DONNEES #####################
###############################################################################



for liste in range(0,len(liste_matrice_position_x)) :
	for nombre in range(0,len(liste_matrice_position_x[liste])):
		liste_matrice_position_x[liste][nombre] = liste_matrice_position_x[liste][nombre] + (liste_corr_x[liste][0] - liste_corr_x[liste][nombre]) 

for liste in range(0,len(liste_matrice_position_y)) :
	for nombre in range(0,len(liste_matrice_position_y[liste])):
		liste_matrice_position_y[liste][nombre] = liste_matrice_position_y[liste][nombre] + (liste_corr_y[liste][0] - liste_corr_y[liste][nombre]) 

 #if liste_corr_y[liste][0] - liste_corr_y[liste][nombre] > 10 :
 #print(liste_corr_y[liste][0] - liste_corr_y[liste][nombre])
print("")
print("liste_matrice_position") 
print("") 
#print(liste_matrice_position_x)
#print(liste_matrice_position_y)



liste_distance_par_matrice_temporaire=[]
liste_vitesse_par_matrice_temporaire=[]
liste_distance=[]
liste_vitesse=[]
liste_acceleration_par_matrice_temporaire=[]
liste_acceleration=[]
liste_position_a_la_vitesse=[]
liste_position_a_la_vitesse_temporaire=[]


###########################################################################
#################################### NORMALISATION DONNEES ################
###########################################################################
#
#listee_tempp=[]
#listee_tempp_total=[]
#
#for liste in liste_matrice_position_y :
# for i in range(0,len(liste)):
# listee_tempp.append( (liste[i]-min(liste)) / (max(liste)-min(liste)) * 450)
# 
# listee_tempp_total.append(listee_tempp)
# listee_tempp=[]
#liste_matrice_position_y=[]
#liste_matrice_position_y=listee_tempp_total
#
#print(liste_matrice_position_y)

##############################################################################
#################################### CALCUL VITESSE ##########################
##############################################################################


for matrix_x,matrix_y in zip(liste_matrice_position_x,liste_matrice_position_y) :
	for i in range(1,len(matrix_x)):
 
		 vitesse = matrix_y[i-1] - matrix_y[i]



		 liste_vitesse_par_matrice_temporaire.append(vitesse)
		 liste_position_a_la_vitesse_temporaire.append(int(matrix_y[i] - matrix_y[i-1])/2)
		 liste_vitesse.append(liste_vitesse_par_matrice_temporaire)
		 liste_position_a_la_vitesse.append(liste_position_a_la_vitesse_temporaire)
		 liste_vitesse_par_matrice_temporaire=[]
		 liste_position_a_la_vitesse_temporaire=[]
 
###############################################################################
################################ CALCUL ACCELERATION ######################
###############################################################################


for liste_vitesse_ in liste_vitesse :
	for i in range(1,len(liste_vitesse_)):
	    acceleration = liste_vitesse_[i] - liste_vitesse_[i-1]
		# if liste_vitesse_[i] >= liste_vitesse_[i-1]: 
		# acceleration=acceleration
		# else : 
		# acceleration=-acceleration
 
 
 		liste_acceleration_par_matrice_temporaire.append(acceleration)
 		liste_acceleration.append(liste_acceleration_par_matrice_temporaire)

 		liste_acceleration_par_matrice_temporaire=[]
###############################################################################
################################# GRAPHIQUES #################################
############################################################################### 
plt.figure(figsize=(15,10)) 
#
#for i in range(0,1):
# 
# plt.plot(np.sort(liste_matrice_position_y[i][1:]),liste_vitesse[i], label = liste_name_matrices[i])
#plt.legend()
#plt.show()
#plt.figure(figsize=(15,10)) 
#for i in range(1,6):
# 
# plt.plot(np.sort(liste_matrice_position_y[i][1:]),liste_vitesse[i], label = liste_name_matrices[i])
#plt.legend()
#plt.show()
#plt.figure(figsize=(15,10)) 
#
#for i in range(6,8):
# 
# plt.plot(np.sort(liste_matrice_position_y[i][1:]),liste_vitesse[i], label = liste_name_matrices[i])
#plt.legend()
#plt.show()
#plt.figure(figsize=(15,10)) 


for i in range(0,len(liste_name_matrices)):
 
	plt.plot(np.sort(liste_matrice_position_y[i][1:]),liste_vitesse[i], label = liste_name_matrices[i])

	
plt.axvline(x=46,color='k', linestyle='--')
plt.axvline(x=17,color='k', linestyle='--')
plt.legend()
plt.title("Vitesse v des cellules en fonction de la position x", fontsize=16)
plt.xlabel("position x (µm)", fontsize=16)
plt.ylabel("vitesse v (µm/δt)", fontsize=16)
#plt.savefig('vitesse.png')
plt.show()

###############################################################################
############################# RECHERCHE BORNES INTERVALLE #####################
###############################################################################


liste_tot=[]

for liste in liste_vitesse :
	liste_tot.append(max(liste))
	max_vitesse=int(max(liste_tot)) + 1
print(max(liste_tot))


liste_tot=[]

for liste in liste_matrice_position_y :
	liste_tot.append(max(liste))
	max_pos_y=int(max(liste_tot)) + 1
print(max(liste_tot))

liste_tot=[]

for liste in liste_matrice_position_y :
	liste_tot.append(min(liste))
	min_pos_y=int(min(liste_tot)) + 1
print(max(liste_tot))

liste_tot=[]


for liste in liste_vitesse :
	liste_tot.append(min(liste))
	min_vitesse=int(min(liste_tot)) + 1
print(max(liste_tot))


###############################################################################
############################# MATRICE DE LANGEVIN #############################
###############################################################################

#binning=10
#binning_res=450/binning
#
#matrice_langevin=np.zeros((int ((int(max_distance)+1)/binning_res),int((int(max_pos_y)+1)/binning_res)) )
#print(matrice_langevin.shape)

liste_temp=[]
liste_temp_gen=[]
for matrice in liste_matrice_position_y :
 
	for i in range(1,len(matrice)) :

		liste_temp.append((matrice[i-1]+matrice[i])/2)
		liste_temp_gen.append(liste_temp)
		liste_temp=[]

		liste_matrice_position_y=liste_temp_gen
 
 
liste_temp=[]
liste_temp_gen=[]
for matrice in liste_matrice_position_y :
 
	for i in range(1,len(matrice)) :
		
		liste_temp.append((matrice[i-1]+matrice[i])/2)
		liste_temp_gen.append(liste_temp)
		liste_temp=[]
  
liste_matrice_position_y=liste_temp_gen
liste_temp=[]
liste_temp_gen=[]

for matrice in liste_vitesse :
 
	for i in range(1,len(matrice)) :

		liste_temp.append((matrice[i-1]+matrice[i])/2)
		liste_temp_gen.append(liste_temp)
		liste_temp=[]
		
liste_vitesse=liste_temp_gen
#print(len(liste_vitesse[0]))
#print(len(liste_matrice_position_y[0]))
#print(len(liste_acceleration[0])) 

liste_vitesse_generale_triee=[]
liste_acceleration_generale_triee=[]

for i in range(0,len(liste_matrice_position_y)) :
	liste_vitesse_triee = [x for _,x in sorted(zip(liste_matrice_position_y[i],liste_vitesse[i]))]
	liste_vitesse_generale_triee.append(liste_vitesse_triee)
 
for i in range(0,len(liste_matrice_position_y)) :
	liste_acceleration_triee = [x for _,x in sorted(zip(liste_matrice_position_y[i],liste_acceleration[i]))]
	liste_acceleration_generale_triee.append(liste_acceleration_triee)
 
liste_position_y_generale_triee=[]
for i in range(0,len(liste_matrice_position_y)) :
	liste_position_y_generale_triee.append(np.sort(liste_matrice_position_y[i]))
 	#print(liste_position_y_generale_triee[i])
  
  
###############################################################################
############################# MATRICE DE LANGEVIN 3D STD ######################
############################################################################### 
 
binning = 18
matrice_langevin_std=np.zeros((binning,binning))
#print(matrice_langevin.shape)
intervalle_pos=np.linspace(min_pos_y,max_pos_y,binning)
intervalle_vitesse=np.linspace(min_vitesse,max_vitesse,binning) 


liste_temporaire = []
a=0
b=0


for i in range(0,len(intervalle_pos)-1) :
	for ii in range(0,len(intervalle_vitesse)-1) :
 		for bbb in range(0,len(liste_vitesse_generale_triee)) :
 			for bb in range(0,len(liste_vitesse_generale_triee[bbb])) :
 				if intervalle_vitesse[ii] <liste_vitesse_generale_triee[bbb][bb]<= intervalle_vitesse[ii+1] and 
					intervalle_pos[i]<liste_position_y_generale_triee[bbb][bb]<= intervalle_pos[i+1] :
 	liste_temporaire.append(liste_acceleration[bbb][bb])
 
	matrice_langevin_std[i][ii]=np.std(liste_temporaire) 
	liste_temporaire=[]
	
matrice_langevin_std[np.isnan(matrice_langevin_std)] = 0
matrice_langevin_std=np.transpose(matrice_langevin_std)


###############################################################################
############################# MATRICE DE LANGEVIN 3D MEAN #############################
############################################################################### 


binning = 18
matrice_langevin_mean=np.zeros((binning,binning))
#print(matrice_langevin.shape)
intervalle_pos=np.linspace(min_pos_y,max_pos_y,binning)
intervalle_vitesse=np.linspace(min_vitesse,max_vitesse,binning) 
liste_temporaire = []
a=0
b=0
for i in range(0,len(intervalle_pos)-1) :
	for ii in range(0,len(intervalle_vitesse)-1) :
 		for bbb in range(0,len(liste_vitesse_generale_triee)) :
 			for bb in range(0,len(liste_vitesse_generale_triee[bbb])) :
 				if intervalle_vitesse[ii] <liste_vitesse_generale_triee[bbb][bb]<= intervalle_vitesse[ii+1] and intervalle_pos[i]<liste_position_y_generale_triee[bbb][bb]<= intervalle_pos[i+1] :
					liste_temporaire.append(liste_acceleration[bbb][bb])

					matrice_langevin_mean[i][ii]=np.mean(liste_temporaire) 
liste_temporaire=[]

matrice_langevin_mean[np.isnan(matrice_langevin_mean)] = 0
matrice_langevin_mean=np.transpose(matrice_langevin_mean)
#print(matrice_langevin)



======================
3D surface (color map)
======================


plt.figure(figsize=(20,13)) 
X=intervalle_pos
Y=intervalle_vitesse
Z=np.true_divide(matrice_langevin_mean,matrice_langevin_std)
Z=np.absolute(Z)
Z[np.isnan(Z)] = 0
plt.imshow(Z,aspect=2.5, extent=[min_pos_y, max_pos_y, min_vitesse, max_vitesse], origin='lower',interpolation='bicubic',
 cmap='Reds')
plt.colorbar().set_label('Rapport F(x,v)/σ(x,v) ', fontsize=16)
plt.axis(aspect='image');
plt.axvline(x=46,color='r', linestyle='--')
plt.axvline(x=17,color='r', linestyle='--')
plt.title("Rapport de la composante déterministe F(x,v) sur la composante stochastique ",fontsize=16)
plt.xlabel("Position x (µm)",fontsize=16)
plt.ylabel("Vitesse v (µm/δt)",fontsize=16)
 
plt.ylim(-20,50) 
#plt.title(binning,fontsize=16)
#plt.savefig('ratio.png')
plt.show()


###############################################################################
############################# CALCUL FONCTION DE CORRELATION ##################
############################################################################### 


liste_bruit_temp=[]
liste_bruit_generale=[]
liste_mean=[]
liste_std=[]
liste_generale_std=[]
liste_generale_mean=[]
#for liste in liste_matrice_position_y[9:13] :
# for ele in liste :
31
 
a=0
b=0
for ee in range(1,5) :
	for i in range(0,len(intervalle_pos)-1) :
 		for ii in range(0,len(intervalle_vitesse)-1) :
 			for bbb in range(0,len(liste_vitesse[8+ee:9+ee])) :
 				for bb in range(0,len(liste_vitesse[8+ee:9+ee][bbb])) :
 					if intervalle_vitesse[ii] <liste_vitesse[8+ee:9+ee][bbb][bb]<= intervalle_vitesse[ii+1] and intervalle_pos[i]<liste_matrice_position_y[8+ee:9+ee][bbb][bb]<= intervalle_pos[i+1] :
 
 
						liste_mean.append(matrice_langevin_mean[i][ii])
						liste_std.append(matrice_langevin_std[i][ii])
						#liste_generale_std.append(liste_std)
						#liste_generale_mean.append(liste_mean)
						#liste_mean=[]
						#liste_std=[]
print("0") 
print(len(liste_mean))
liste_spe=[]
for i in range(0,binning) :
	for ii in range(0,binning) :
 		if matrice_langevin_std[i][ii] > 0 :
 			liste_spe.append(matrice_langevin_std[i][ii])
 
moye=np.mean(liste_spe)
#plt.show()
#plt.plot(liste_std)
#plt.plot(liste_mean)
#plt.show()
for i in range(0,len(liste_std)) :
	if liste_std[i]==0 :
 		liste_std[i]=moye
plt.figure(figsize=(15,10)) 
liste_bruit_temp=[]
liste_bruit_generale=[]
i=0


plt.legend()
plt.title("Fonction de corrélation < ∆ W(t+δt) ∆ W(t) > en fonction du temps", fontsize=16)
plt.ylabel("Fonction de corrélation < ∆ W(t+δt) ∆ W(t) >", fontsize=16)
plt.xlabel("Temps t (δt)", fontsize=16)


for liste in liste_acceleration_generale_triee[9:13] :
	for ele in liste : 
 
 		val=ele - liste_mean[i]
 		i=i+1
 		liste_bruit_temp.append(val)

 		liste_bruit_generale.append(liste_bruit_temp)
 		liste_bruit_temp=[]
 
		liste_bruit_finale=[]
 
for i in range(0,len(liste_bruit_generale[0])) :
 
	liste_bruit_finale.append(np.mean((liste_bruit_generale[3][i],liste_bruit_generale[2][i],liste_bruit_generale[1][i],liste_bruit_generale[0][i]))/
  	np.mean((liste_std[i],liste_std[i+108],liste_std[i+216],liste_std[i+324])))



for i in range(0,len(liste_bruit_finale)):
	liste_bruit_finale[i]=2*( (liste[i]-min(liste)) / (max(liste)-min(liste)))-1


plt.plot(liste_bruit_finale)
#print(len(liste_bruit_finale))
#print(liste_bruit_finale)
"Temps t (δt)"
plt.savefig('corre.png')
plt.show()
plt.figure(figsize=(15,10)) 
plt.legend()
plt.title("Position x en fonction du temps", fontsize=16)
plt.ylabel("Temps t (δt)", fontsize=16)
plt.xlabel("Position x (µm)", fontsize=16)
plt.axvline(x=46,color='k', linestyle='--')
plt.axvline(x=17,color='k', linestyle='--')
#print(liste_name_matrices)


azer=[]


for i in range(0,len(liste_corrections)) :
    liste_matrice_position_y[i].reverse()
	azer=np.arange(len(liste_matrice_position_y[i]))
	plt.plot(liste_matrice_position_y[i],azer,label = liste_name_matrices[i])
	plt.legend()


plt.savefig('pos.png')
plt.figure(figsize=(15,10)) 
plt.legend()
plt.title("Surface du noyau en fonction du temps", fontsize=16)
plt.ylabel("Surface du noyau (µm^2)", fontsize=16)
plt.xlabel("Temps t (δt)", fontsize=16)
#plt.axhline(y=361,color='k', linestyle='--')
#plt.axhline(y=102,color='k', linestyle='--')


for i in range(0,len(liste_corrections)) :
	liste_area[i]
 	plt.plot(liste_area[i],label = liste_name_matrices[i])
 	plt.legend()
 
plt.figure(figsize=(15,10)) 
plt.legend()
plt.title("Longueur du noyau en fonction du temps", fontsize=16)
plt.ylabel("Longueur du noyau (µm)", fontsize=16)
plt.xlabel("Temps t (δt)", fontsize=16)
#plt.axhline(y=361,color='k', linestyle='--')
#plt.axhline(y=102,color='k', linestyle='--')


for i in range(0,len(liste_corrections)) :
 	liste_height[i]
 	plt.plot(liste_height[i],label = liste_name_matrices[i])
 	plt.legend()
#plt.savefig('Longueur_temps.png')
plt.show()
#
#def tolerant_mean(arrs):
# lens = [len(i) for i in arrs]
# arr = np.ma.empty((np.max(lens),len(arrs)))
# arr.mask = True
# for idx, l in enumerate(arrs):
# arr[:len(l),idx] = l
# return arr.mean(axis = -1), arr.std(axis=-1)
#
# ax = plt.subplots()
# y, error = tolerant_mean(liste_area)
# ax.plot(np.arange(len(y))+1, y, color='green')
plt.show()
plt.figure(figsize=(15,10)) 
max_list_len=[]


for liste in liste_area :
 	max_list_len.append(len(liste))
	maxi_len=max(max_list_len) 
#
liste_area_modif=[]
lsite_area_tempo=[]
#
#for liste in range(0,len(liste_area)) :
# while len(liste_area[liste]) < maxi_len : 
# liste_area[liste]=np.append(liste_area[liste],0)
# i=i+1
# if len(liste_area[liste]) == maxi_len :
# 
# liste_area_modif.append(liste_area[liste])
34
# np.delete(liste_area, liste, 0)
#
#for liste in liste_area_modif :
# for i in range(0,len(liste)) :
# if liste[i] == 0 :
# 
# liste[i] = np.nan
#plt.figure(figsize=(15,10)) 
#t = np.arange(maxi_len)
#mu1 = np.nanmean(np.where(liste_area_modif !=0,liste_area_modif ,np.nan),0)
#sigma1 = np.nanstd(np.where(liste_area_modif !=0,liste_area_modif ,np.nan),0)
##mu2 = X2.mean(axis=1)
##sigma2 = X2.std(axis=1)
#
## plot it!
#
#fig, ax = plt.subplots(1,figsize=(15, 10))
##plt.savefig('s')
#ax.plot(t, mu1, lw=2, label=' moyenne du noyau', color='blue')
##ax.plot(t, mu2, lw=2, label='mean population 2', color='yellow')
#ax.fill_between(t, mu1+sigma1, mu1-sigma1, facecolor='blue', alpha=0.5)
##ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='yellow', alpha=0.5)
#ax.set_title('Longueur moyenne du noyau en fonction du temps')
#ax.legend(loc='upper left')
#ax.set_xlabel('Temps t (δt) ')
#ax.set_ylabel('Longueur moyenne du noyau (µm^2)')
#ax.grid()
##plt.savefig('s')
#
#plt.show()
#
#liste_height_1_modif=[]
#liste_height_tempo=[]
#
#for liste in range(0,len(liste_height)) :
35
# while len(liste_height_1[liste]) < maxi_len : 
# liste_height_1[liste]=np.append(liste_height[liste],0)
# i=i+1
# if len(liste_height_1[liste]) == maxi_len :
# 
# liste_height_1_modif.append(liste_height[liste])
# np.delete(liste_height, liste, 0)
#
#for liste in liste_height_1_modif :
# for i in range(0,len(liste)) :
# if liste[i] == 0 :
# 
# liste[i] = np.nan
#plt.figure(figsize=(15,10)) 
#t = np.arange(maxi_len)
inter=np.arange(0,50,1).tolist()
inter=[12,16,18,43,46,50,54,55]
i=0
a=0
liste_totale=[]
liste_temporaire=[]
for aa in range(0,len(inter)-1) :
 
 	for liste in liste_matrice_position_y :
    	for nb in liste :
        #print(nb)
        	if inter[aa]<nb<inter[aa+1] :
        	liste_temporaire.append(liste_vitesse[a][i])
 	#print(nb)
 	i=i+1
 	i=0 
 
 	a=a+1
 	a=0
 	liste_totale.append(liste_temporaire)
 	liste_temporaire=[]
	inter.pop()
print(inter)
liste_moyenne=[]
liste_std=[]


for i in liste_totale :

   	liste_moyenne.append(np.mean(i))
    liste_std.append(np.std(i))
mu1 = liste_moyenne
sigma1 = liste_std
#mu2 = X2.mean(axis=1)
#sigma2 = X2.std(axis=1)
# plot it


lower_bound = np.array(mu1) + np.array(sigma1)
upper_bound = np.array(mu1) - np.array(sigma1)
plt.axvline(x=46,color='k', linestyle='--')
plt.axvline(x=17,color='k', linestyle='--')
fig, ax = plt.subplots(1,figsize=(15, 10))


#plt.savefig('s')
ax.plot(inter,mu1, lw=2, label=' moyenne du noyau', color='blue')
#ax.plot(t, mu2, lw=2, label='mean population 2', color='yellow')
ax.fill_between(inter, lower_bound, upper_bound, facecolor='blue', alpha=0.5)
#ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='yellow', alpha=0.5)
ax.set_title("Vitesse moyenne du noyau en fonction de la position", fontsize=16)
ax.legend(loc='upper left')


ax.set_xlabel('Position x (µm) ', fontsize=16)
ax.set_ylabel('Vitesse moyenne du noyau (µm/δt)', fontsize=16)
ax.grid()
#plt.savefig('s')
ax.axvline(x=46,color='k', linestyle='--')
ax.axvline(x=17,color='k', linestyle='--')
plt.legend()
plt.show()
plt.figure(figsize=(15,10)) 
plt.legend()
plt.title("Surface du noyau en fonction de la position", fontsize=16)
plt.ylabel("Surface du noyau (µm^2)", fontsize=16)
plt.xlabel("Position x (µm)", fontsize=16)
plt.axvline(x=46,color='k', linestyle='--')
37
plt.axvline(x=17,color='k', linestyle='--')
for i in range(0,len(liste_area)):
	plt.plot(np.sort(liste_matrice_position_y_area[i]),liste_area[i],"s",label = liste_name_matrices[i])
    plt.legend()
    plt.show()
    plt.figure(figsize=(15,10)) 
    plt.legend()
    plt.title("Longeuur du noyau en fonction de la position", fontsize=16)

plt.ylabel("Longueur du noyau (µm)", fontsize=16)
plt.xlabel("Position x (µm)", fontsize=16)
plt.axvline(x=46,color='k', linestyle='--')
plt.axvline(x=17,color='k', linestyle='--')
for i in range(0,len(liste_area)):
	plt.plot(np.sort(liste_matrice_position_y_area[i]),liste_height[i],"s",label = liste_name_matrices[i])
 	plt.legend()
plt.show()
plt.figure(figsize=(15,10)) 
plt.legend()
plt.title("Largeur du noyau en fonction de la position", fontsize=16)
plt.ylabel("Longueur du noyau (µm)", fontsize=16)
plt.xlabel("Position x (µm)", fontsize=16)
plt.axvline(x=46,color='k', linestyle='--')
plt.axvline(x=17,color='k', linestyle='--')
for i in range(0,len(liste_area)):
	plt.plot(np.sort(liste_matrice_position_y_area[i]),liste_width[i],"s",label = liste_name_matrices[i])
 	plt.legend()
plt.show(
