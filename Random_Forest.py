
#import des librairies
import files_processing
from files_processing import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
import seaborn as sns

#lancement d'un chronomètre pour récupérer le temps que mets le code à afficher des résultats
start = time.process_time()

#initialisation de différentes variables
Predict = 0
Label2_true_Percentage = 0
Label2_false_Percentage = 0

#nb_iteration est le nombre maximum d'arbre voulu en sortie de la boucle ci-dessous
nb_iteration = 15

#a et b sont des valeurs arbitraire nous permettant de faire tourner la boucle a*b fois
#cela nous permettra ensuite de faire une moyenne de tous les résultats obtenus
a = 10
b = 10

Predict_list = []
Label2_true_Percentage_list = []
Label2_false_Percentage_list = []

#ces variables servent seulement pour faire de l'affichage
X_train = data.shape[0]*0.8
X_test = data.shape[0]*0.2
print("Taille de X_train :", int(X_train),"\nTaille de X_test :", int(X_test))
print("Nombre d'arbres maximum :", nb_iteration, "\nNombre d'itérations par arbre :", a*b)

#cette boucle nous permet de lancer notre modèle nb_iteration*a*b fois 
#
for z in range(0, nb_iteration) :
    for y in range(0,a) :
        for i in range(0, b) :

            random_forest_model = RandomForestClassifier(n_estimators = z + 1, random_state = 5)
            random_forest_model.fit(X_train_Exp, y_train_Exp.ravel())
            random_forest_model.get_params()
            y_pred = random_forest_model.predict(X_test_Exp)
            score = random_forest_model.score(X_test_Exp, y_test_Exp)
            Predict = Predict + score
    
    #Predict va nous permettre de faire une moyenne de tous les résultats 
    #obtenus par iteration d'arbre
    Predict = Predict/(a*b)
    Predict = round(Predict, 3)
    Predict_list.append(Predict)
    print('Nombre d arbre :', z + 1)
    print('Prédiction moyenne :', Predict)
    
    #création de la matrice de confusion
    cf_matrix = confusion_matrix(y_test_Exp, y_pred)
    
    print(cf_matrix)
    if (z + 1) == 1 :     
        print("Matrice de Confusion pour un arbre.")
    else :
        print("Matrice de Confusion pour " + str(z + 1) + " arbres.")
    
    #les 26 lignes suivantes vont nous permettre de créer un rapport de classification
    #pour avoir un affichage clair des résultats
    Percentage_values = np.zeros((3, 3))
    
    for i in range(0, 3) :
        for j in range(0, 3) :
            result = (cf_matrix[i][j]/sum(cf_matrix[i]))
            Percentage_values[i][j] = result
            
    Label2_true_Percentage = Percentage_values[1][1]
    Label2_false_Percentage = Percentage_values[1][0] + Percentage_values[1][2]
    print("Prédiction true du label 2 :" , round(Label2_true_Percentage, 2))
    print("Prédiction false du label 2 :" , round(Label2_false_Percentage, 2))

    Label2_true_Percentage_list.append(Label2_true_Percentage)
    Label2_false_Percentage_list.append(Label2_false_Percentage)

    if z == 0 :
        Percentage_values0 = Percentage_values
        cf_matrix0 = cf_matrix
    else :
        Percentage_values0 = (Percentage_values0 + Percentage_values)/ 2 
        cf_matrix0 = (cf_matrix0 + cf_matrix)/ 2
    
    if (z + 1) == 1 :     
        print("Matrice de Confusion pour un arbre.")
    else :
        print("Rapport de classification pour", str(z + 1), "arbres.")
    print(classification_report(y_test_Exp, y_pred))


#les 11 lignes suivantes vont nous permettre d'afficher une matrice de confusion
#claire du nombre maximum d'arbres que l'on a utilisé dans notre modèle
group_names = ['True Pred','Pred 2 \ninstead of 1','Pred 3 \ninstead of 1',
               'Pred 1 \ninstead of 2','True Pred','Pred 3 \ninstead of 2',
               'Pred 1 \ninstead of 3','Pred 2 \ninstead of 3','True Pred']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix0.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     Percentage_values0.flatten()]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(3,3)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', xticklabels=(1, 2, 3), yticklabels=(1, 2, 3))


nb_iteration_range = range(nb_iteration)
    
#figure affichant les prédictions 'recall' de notre modèle en fonction du nombre d'arbres
plt.figure()
plt.plot(nb_iteration_range, Predict_list, label = 'Prédiction totale du système')
plt.plot(nb_iteration_range, Label2_true_Percentage_list, label = 'Prédiction true Label 2')
plt.plot(nb_iteration_range, Label2_false_Percentage_list, label = 'Prédiction false Label 2')
plt.title('Valeurs des prédictions')
plt.legend()

#arrêt du chronomètre lancé en debut de code et affichage de la durée du code
end = time.process_time()
time = end - start
time = round(time, 2)
if round(time) > 60:
    time = round(time/60, 2)
    print("Temps écoulé en min :", time)
else :
    print("Temps écoulé en sec :", time)
