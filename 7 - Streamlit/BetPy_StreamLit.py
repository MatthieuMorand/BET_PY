###############################################################################
## Fichier : BetPy_StreamLit.py                                              ##
## Date de création : 08/09/2021                                             ##
## Date de dernière modification : 23/09/2021                                ##
## Version : 1.0                                                             ##
## Description :                                                             ##
##                                                                           ##
###############################################################################
## Auteur | Date     | Description                                           ##
## ------------------------------------------------------------------------- ##
## MMO    | 08/09/21 | Création du fichier.                                  ##
## MMO    | 09/09/21 | Écriture de la partie « Le Jeu De Données » + ajout   ##
##        |          | de deux parties.                                      ##
## MMO    | 11/09/21 | Écriture de la partie « Quelques visualisations ».    ##
## MMO    | 14/09/21 | Écriture de la partie « Machine Learning ».           ##
## MMO    | 15/09/21 | Fin de la partie « Machine Learning ».                ##
## MMO    | 20/09/21 | Début de la partie « Modélisation ».                  ##
## MMO    | 21/09/21 | Poursuite de la partie « Modélisation ».              ##
## MMO    | 22/09/21 | Poursuite de la partie « Modélisation ».              ##
## MMO    | 23/09/21 | Écriture de l'introduction et de la conclusion.       ##
##        |          | Fin de la partie « Modélisation ».                    ##
###############################################################################


###############################################################################
## IMPORTATION DES MODULES                                                   ##
###############################################################################

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import streamlit         as st

from datetime                import datetime as dt
from dateutil.relativedelta  import relativedelta
from os                      import chdir
from sklearn                 import preprocessing, metrics
from sklearn                 import svm, neighbors
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import accuracy_score, auc, classification_report, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from time                    import sleep


###############################################################################
## FONCTIONS SUPPLÉMENTAIRES                                                 ##
###############################################################################

def display_classification_report(tabular):
    # Reformatage
    tabular = tabular.split()
    
    # Affichage des informations
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write('_')
        st.write(tabular[4])
        st.write(tabular[9])
        st.write(tabular[14])
        st.write(tabular[17] + ' ' + tabular[18])
        st.write(tabular[23] + ' ' + tabular[24])
    with col2:
        st.write(tabular[0])
        st.write(float(tabular[5]))
        st.write(float(tabular[10]))
        st.write('_')
        st.write(float(tabular[19]))
        st.write(float(tabular[25]))
    with col3:
        st.write(tabular[1])
        st.write(float(tabular[6]))
        st.write(float(tabular[11]))
        st.write('_')
        st.write(float(tabular[20]))
        st.write(float(tabular[26]))
    with col4:
        st.write(tabular[2])
        st.write(float(tabular[7]))
        st.write(float(tabular[12]))
        st.write(float(tabular[15]))
        st.write(float(tabular[21]))
        st.write(float(tabular[27]))
    with col5:
        st.write(tabular[3])
        st.write(float(tabular[8]))
        st.write(float(tabular[13]))
        st.write(float(tabular[16]))
        st.write(float(tabular[22]))
        st.write(float(tabular[28]))

def display_matrix(tabular):
    # Affichage
    col1, col2, col3, _, _ = st.columns(5)
    with col1:
        st.write('_')
        st.write('0.0')
        st.write('1.0')
    with col2:
        st.write('0.0')
        st.write(tabular[0][0])
        st.write(tabular[0][1])
    with col3:
        st.write('1.0')
        st.write(tabular[1][0])
        st.write(tabular[1][1])

def display_ROC(algo, y_test, probs):
    # Calcul de la courbe ROC
    fpr, tpr, _ = roc_curve(y_test, probs[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    # Préparation du graphique
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='red', lw=2, label='Modèle ' + algo + ' (auc = %.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Aléatoire (auc = .5)')
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.05)
    ax.set_title('Courbe ROC ' + algo)
    ax.set_xlabel('Taux de faux positif')
    ax.set_ylabel('Taux de vrai positif')
    ax.legend(loc='lower right')
    
    # Affichage
    st.pyplot(fig)


###############################################################################
## INITIALISATION                                                            ##
###############################################################################

# https://github.com/MatthieuMorand/BET_PY/tree/main/7%20-%20Streamlit

chdir(r'D:\Users\User\Documents\GitHub\BET_PY_\7 - Streamlit')

path_data = '1 - Data\\'
path_imag = '0 - Images\\'


###############################################################################
## VARIABLES DE SESSION                                                      ##
###############################################################################

## Sauvegarde des variables pour la démo
if 'sel_features' not in st.session_state:
    st.session_state.sel_features = []               # Variables explicatives

## Sauvegarde des variables concernant le SVM
if 'sel_SVM_ratio' not in st.session_state:
    st.session_state.sel_SVM_ratio = .2              # Proportion des données de test
if 'sel_SVM_C' not in st.session_state:
    st.session_state.sel_SVM_C = []                  # SVM paramètre C
if 'sel_SVM_kernel' not in st.session_state:
    st.session_state.sel_SVM_kernel = []             # SVM paramètre kernel
if 'sel_SVM_gamma' not in st.session_state:
    st.session_state.sel_SVM_gamma = []              # SVM paramètre gamma

## Sauvegarde des variables concernant le KNN
if 'sel_KNN_ratio' not in st.session_state:
    st.session_state.sel_KNN_ratio = .2              # Proportion des données de test
if 'sel_KNN_neighbors_min' not in st.session_state:
    st.session_state.sel_KNN_neighbors_min = 1       # KNN nombre de voisins minimum
if 'sel_KNN_neighbors_max' not in st.session_state:
    st.session_state.sel_KNN_neighbors_max = 1       # KNN nombre de voisins maximum
if 'sel_KNN_metric' not in st.session_state:
    st.session_state.sel_KNN_metric = []             # KNN métrique à utiliser

## Sauvegarde des variables concernant la LR
if 'sel_LR_ratio' not in st.session_state:
    st.session_state.sel_LR_ratio = .2               # Proportion des données de test
if 'sel_LR_C' not in st.session_state:
    st.session_state.sel_LR_C = []                   # LR paramètre C
if 'sel_LR_penalty' not in st.session_state:
    st.session_state.sel_LR_penalty = []             # LR paramètre penalty
if 'sel_LR_solver' not in st.session_state:
    st.session_state.sel_LR_solver = []              # LR paramètre solver

## Sauvegarde des variables concernant la RF
if 'sel_RF_ratio' not in st.session_state:
    st.session_state.sel_RF_ratio = .2               # Proportion des données de test
if 'sel_RF_min_samples_min' not in st.session_state:
    st.session_state.sel_RF_min_samples_min = 1      # RF nombre d'échantillons au minimum
if 'sel_RF_min_samples_max' not in st.session_state:
    st.session_state.sel_RF_min_samples_max = 1      # RF nombre d'échantillons au maximum
if 'sel_RF_max_feat' not in st.session_state:
    st.session_state.sel_RF_max_feat = []            # RF paramètre max_features

## Couleurs des noms des colonnes lors des descriptions
if 'cols_color' not in st.session_state:
    st.session_state.cols_color = 'red'


###############################################################################
## MENU                                                                      ##
###############################################################################

title = 'Paris sportif'
menu_lst = ['Le projet',
            'Le Jeu De Données',
            'Quelques visualisations',
            'Préparation des données',
            'Modélisation',
            'Machine Learning',
            'Conclusion et perspective']

st.sidebar.header(title)
st.sidebar.write('Sélectionnez une partie :')
menu_sel = st.sidebar.radio('', menu_lst)

# Les auteurs
st.sidebar.subheader('Auteurs')
st.sidebar.write("""[Romain MICLO](https://www.linkedin.com/in/romain-miclo-b0538b59/)""")
st.sidebar.write("""[Matthieu MORAND](https://www.linkedin.com/in/matthieu-morand-001/)""")


###############################################################################
## PARTIE 0 : DÉFINITION DU PROJET                                           ##
###############################################################################

if menu_sel == menu_lst[0]:
    st.header(title)
    st.subheader(menu_sel, anchor='Le-projet')
    
    st.markdown("""Ce projet a été réalisés dans le cadre de notre formation en *data science* via l'organisme [Datascientest](https://datascientest.com).
                   L'objectif est de prédire l'issue de matchs de football, à partir du JDD disponible sur [kaggle](https://www.kaggle.com/ayotomiwasalau/club-football-event-data).
                """)
    st.markdown("""Ce *streamlit* présente notre démarche pour mener à bien ce projet, depuis l'exploration des données jusqu'à la création des variables explicatives.
                   Les meilleurs résultats que nous avons pu obtenir ne sont pas encore présent dans le *streamlit*, mais la partie **Machine Learning** vous permet
                   de tester vous même les variables que nous avons créées sur différents algorithmes.
                """)
    
    # TEST
    from os import getcwd
    st.write(getcwd())


###############################################################################
## PARTIE 1 : PRÉSENTATION DU JEU DE DONNÉES                                 ##
###############################################################################

if menu_sel == menu_lst[1]:
    #######################################
    ## PRÉSENTATION                      ##
    #######################################
    
    st.header(title)
    st.subheader(menu_sel, anchor='Le-Jeu-De-Données')
    
    st.write("""Le JDD pour ce projet (disponible sur [kaggle](https://www.kaggle.com/ayotomiwasalau/club-football-event-data))
                est constitué d'une grande collection de données sur les matchs, joueurs, clubs
                et arbitres des premières ligues de football anglaise, française, espagnole,
                allemande et italienne pour la saison 2017-2018. Ces données sont réparties sur les cinq
                tables présentées ci-dessous.""")
    
    st.image(path_imag + 'Table 00 - JDD de départ.png',
             caption='Les différentes tables du JDD avec leurs relations entre elles.')
    
    #######################################
    ## TABLE « CLUB »                    ##
    #######################################
    
    st.subheader('La table « club »')
    
    st.write("""Cette table recense 142 équipes de clubs ou nationnales avec leur nom et leur pays.
                Ses colonnes sont :""")
    st.markdown("- <font color=" + st.session_state.cols_color + ">*id*</font> : (*int*) clef primaire ;",              unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*name*</font> : (*varchar*) nom ;",                  unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*officialname*</font> : (*varchar*) nom officiel ;", unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*country*</font> : (*varchar*) pays.",               unsafe_allow_html=True)
    
    if st.checkbox('Afficher la table « club »'):
        df_club = pd.read_csv(path_data + 'club.csv', index_col='id')
        
        df_club
    
    #######################################
    ## TABLE « MATCH »                   ##
    #######################################
    
    st.subheader('La table « match »')
    
    st.markdown('''Cette table recense l'intégralité des résultats des matchs **non nuls** des premières
                   ligues italienne, française, espagnole, anglaise et allemande. Ses colonnes sont :''')
    st.markdown("- <font color=" + st.session_state.cols_color + ">*id*</font> : (*int*) clef primaire ;",                                                           unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*dateutc*</font> : (*varchar*) date du match ;",                                                  unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*competition*</font> : (*varchar*) première ligue concernée (italienne, française, ...) ;",       unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*season*</font> : (*int*) correspond à la saison 2017-2018 quelle que soit la valeur affichée ;", unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*venue*</font> : (*varchar*) nom du stade recevant le match ;",                                   unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*home_club*</font> : (*varchar*) nom du club jouant à domicile ;",                                unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*away_club*</font> : (*varchar*) nom du club jouant à l'extérieur ;",                             unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*winner*</font> : (*varchar*) nom du club vainqueur ;",                                           unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*goal_by_home_club*</font> : (*int*) nombre de buts validés par l'équipe jouant à domicile ;",    unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*goal_by_away_club*</font> : (*int*) nombre de buts validés par l'équipe jouant à l'extérieur ;", unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*referee_id*</font> : (*int*) clef étrangère vers la table « *referee* ».",                       unsafe_allow_html=True)
    
    if st.checkbox('Afficher la table « match »'):
        df_match = pd.read_csv(path_data + 'match.csv', index_col='id')
        
        df_match
    
    #######################################
    ## TABLE « MATCH_EVENT »             ##
    #######################################
    
    st.subheader('La table « match_event »')
    
    st.write("""Cette table recense, pour chaque match, l'ensemble des évènements ayant eu lieu au cours
                dudit match. Ses colonnes sont :""")
    st.markdown("- <font color=" + st.session_state.cols_color + ">*id*</font> : (*int*) clef primaire ;",                                                                                          unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*club_id*</font> : (*int*) clef étrangère vers la table « *club* » ;",                                                           unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*match_id*</font> : (*int*) clef étrangère vers la table « *match* » ;",                                                         unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*players_id*</font> : (*int*) clef étrangère vers la table « *player* » ;",                                                      unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*matchperiod*</font> : (*varchar*) période du match (« 1H » indique la 1re période, tandis que « 2H » indique la 2e période) ;", unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*eventsec*</font> : (*float*) nombre de seconde depuis le début de la période ;",                                                unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*eventname*</font> : (*varchar*) nom de l'évènement ;",                                                                          unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*action*</font> : (*varchar*) spécification sur l'évènement ;",                                                                  unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*modifier*</font> : (*varchar*) conséquence de l'action ;",                                                                      unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*x_begin*</font> : (*int*) position en abscisse du début de l'évènement (sur l'intervalle [0 ; 100]) ;",                         unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*y_begin*</font> : (*int*) position en ordonnée du début de l'évènement (sur l'intervalle [0 ; 100]) ;",                         unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*x_end*</font> : (*float*) position en abscisse de la fin de l'évènement (sur l'intervalle [0. ; 100.]) ;",                      unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*y_end*</font> : (*float*) position en ordonnée de la fin de l'évènement (sur l'intervalle [0. ; 100.]) ;",                      unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*is_succes*</font> : (*varchar*) indique si l'évènement fut un succès.",                                                         unsafe_allow_html=True)
    
    if st.checkbox('Afficher un extrait de la table « match_event »'):
        df_match_event = pd.read_csv(path_data + 'tiniest_match_event.csv', index_col='id')
        
        df_match_event
    
    #######################################
    ## TABLE « PLAYER »                  ##
    #######################################
    
    st.subheader('La table « player »')
    
    st.write("""Cette table recense les joueurs impliqués dans les différentes ligues, avec quelques
                informations intéressantes comme la date de naissance, le poste, le pied de préférence
                et la taille. Ses colonnes sont :""")
    st.markdown("- <font color=" + st.session_state.cols_color + ">*id*</font> : (*int*) clef primaire ;",                  unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*firtsname*</font> : (*varchar*) prénom ;",              unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*lastname*</font> : (*varchar*) nom ;",                  unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*birthdate*</font> : (*varchar*) date de naissance ;",   unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*country*</font> : (*varchar*) pays ;",                  unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*position*</font> : (*varchar*) poste sur le terrain ;", unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*foot*</font> : (*varchar*) pied de préférence ;",       unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*height*</font> : (*int*) taille (en cm).",              unsafe_allow_html=True)
    
    if st.checkbox('Afficher la table « player »'):
        df_player = pd.read_csv(path_data + 'player.csv', index_col='id')
        
        df_player
    
    #######################################
    ## TABLE « REFEREE »                 ##
    #######################################
    
    st.subheader('La table « referee »')
    
    st.write("""Cette table recense les arbitres de terrain ayant jugés les différents matchs.
                Ses colonnes sont :""")
    st.markdown("- <font color=" + st.session_state.cols_color + ">*id*</font> : (*int*) clef primaire ;",                unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*firtsname*</font> : (*varchar*) prénom ;",            unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*lastname*</font> : (*varchar*) nom ;",                unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*birthdate*</font> : (*varchar*) date de naissance ;", unsafe_allow_html=True)
    st.markdown("- <font color=" + st.session_state.cols_color + ">*country*</font> : (*varchar*) pays.",                 unsafe_allow_html=True)
    
    if st.checkbox('Afficher la table « referee »'):
        df_referee = pd.read_csv(path_data + 'referee.csv', index_col='id')
        
        df_referee


###############################################################################
## PARTIE 2 : QUELQUES VISUALISATIONS                                        ##
###############################################################################

if menu_sel == menu_lst[2]:
    #######################################
    ## PRÉSENTATION                      ##
    #######################################
    st.header(title)
    st.subheader(menu_sel, anchor='Quelques-visualisations')
    
    st.write("""Cette partie est là pour permettre une meilleure appréhension du JDD au travers de
                quelques visuels, avec un focus tout particulier sur les tables « match » et « match_event ».
                Ces dernières étant essentiels pour les parties suivantes.""")
    
    #######################################
    ## LES TYPES D'ÉVÈNEMENT             ##
    #######################################
    
    st.subheader("Les types d'évènements")
    st.write("""Cette section présente les différents évènements de la table
                « match_event ». Sélectionnez un évènement via le sélecteur pour connaître
                leurs actions associées.""")
    
    # Préparation des données
    dico_show = {'Arrêt': ['Arrêt réflexe', "Tentative d'arrêt"],
                 'Coup franc': ['Corner', 'Coup franc', 'Coup franc croisé', 'Penalty', 'Tir du gardien', 'Tir de coup franc', 'Touche'],
                 'Duel': ['Attague au sol', 'Défense au sol', 'Duel aérien', 'Perte de balle'],
                 'Faute': ['Carton en retard', 'Faute', 'Faute de main', 'Faute violente', 'Protestation', 'Temps perdu sur faute', 'Simulation'],
                 'Hors-jeu': ['Hors-jeu'],
                 'Interruption': ['Ballon en dehors du terrain', 'Coup de sifflet'],
                 'Passe': ['Lancer', 'Passe à la main', 'Passe croisée', 'Passe de la tête', 'Passe haute', 'Passe intelligente', 'Passe simple'],
                 'Pression adverse': ['Accélération', 'Contact', 'Dégagement'],
                 'Sortie du gardien': ['Sortie du gardien'],
                 'Tir': ['Tir']}
    
    # Sélection par l'utilisateur
    event_name = st.selectbox("Sélectionner un type d'évènement :", dico_show.keys())
    
    # Affichage
    for action in dico_show[event_name]:
        st.markdown('- ' + action)
    
    #######################################
    ## LA TABLE « MATCH_EVENT »          ##
    #######################################
    
    st.subheader("Observation des évènements d'un match")
    st.write("""Dans cette section, vous allez pouvoir suivre, évènement par évènement, les détails
                du match n° 985. Le sélecteur, ci-dessous, permet de faire défiler les évènements ou
                d'accéder directement à un évènement particulier en renseignant son identifiant.""")
    
    # Chargement des données
    df_match_sorted  = pd.read_csv(path_data + 'tiniest_match_event_sorted.csv', index_col='id')
    df_players_match = pd.read_csv(path_data + 'players_match_985.csv')
    dico_pos         = {'Goalkeeper': 'gardien',
                        'Defender':   'défenseur',
                        'Midfielder': 'milieu',
                        'Forward':    'attaquant'}
    
    # Préparation des informations
    dico_actions = {'Touch': '',
                    'Clearance': ' (dégagement)',
                    'Simple pass': ' (simple)',
                    'Ground attacking duel': ' (offensif au sol)',
                    'Free Kick': '',
                    'Ground loose ball duel': ' (perte de balle)',
                    'Shot': '',
                    'Air duel': ' (aérien)',
                    'Ground defending duel': ' (défensif au sol)',
                    'Hand pass': ' (manuelle)',
                    'Throw in': ' (touche)',
                    'High pass': ' (haute)',
                    'Launch': ' (lancer)',
                    'Smart pass': ' (intelligente)',
                    'Cross': ' (croisée)',
                    'Acceleration': ' (accélération)',
                    'Reflexes': ' (réflexe)',
                    'Head pass': ' (tête)',
                    'Free kick cross': ' (croisé)',
                    'Save attempt': ' (arrêt)',
                    'Free kick shot': ' (coup franc)',
                    'Corner': ' (corner)',
                    'Foul': '',
                    'Goal kick': ' (renvoi du gardien)',
                    'Hand foul': ' (faute de main)',
                    'Goalkeeper leaving line': ' (sortie du gardien)',
                    'Penalty': ' (penalty)',
                    'Ball out of the field': ' (sortie du ballon)',
                    'Violent Foul': ' (faute violente)',
                    'Protest': ' (protestation)',
                    'Out of game foul': ' (hors-jeu)',
                    'Simulation': ' (simulation)',
                    'Time lost foul': ' (temps perdu sur faute)',
                    'Whistle': ' (coup de sifflet)',
                    'Late card foul': ' (carton en retard)',
                    np.nan: '',
                    '': ''}
    dico_events = {'Duel': 'duel',
                   'Foul': 'faute',
                   'Free Kick': 'coup franc',
                   'Goalkeeper leaving line': 'sortie du gardien',
                   'Interruption': 'interruption',
                   'Offside': 'hors-jeu',
                   'Others on the ball': 'pression adverse',
                   'Pass': 'passe',
                   'Save attempt': 'arrêt',
                   'Shot': 'tir'}
    dico_success = {'t': 'oui',
                    'f': 'non',
                    np.nan: ''}
    
    # Sélection par l'utilisateur de l'action à afficher
    event_nbr = st.number_input('Renseigner un identifiant :', min_value=0, max_value=len(df_match_sorted)-1)
    
    # Récupération des informations
    if event_nbr > 0: # évènement précédent pour affichage du vecteur
        event_lst       = df_match_sorted.iloc[event_nbr-1]
        # Calcul de la trajectoire de l'évènement précédent
        if event_lst[0] == 6:
            event_vect_prev = [100 - event_lst[9],           100 - event_lst[8],
                               event_lst[9] - event_lst[11], event_lst[8] - event_lst[10]]
        else:
            event_vect_prev = [event_lst[9],                 event_lst[8],
                               event_lst[11] - event_lst[9], event_lst[10] - event_lst[8]]
    
    event_lst  = df_match_sorted.iloc[event_nbr]
    player_lst = df_players_match[df_players_match.id == event_lst[2]].iloc[0]
    time       = '(' + str(int(event_lst[4] / 60.)) + ' min ' + str(int(event_lst[4] % 60.)) + ' sec)'
    
    # Calcul de la trajectoire de l'évènement courant
    if event_lst[0] == 6:
        event_vect = [100 - event_lst[9],           100 - event_lst[8],
                      event_lst[9] - event_lst[11], event_lst[8] - event_lst[10]]
        color_vect = 'red'
    else:
        event_vect = [event_lst[9],                 event_lst[8],
                      event_lst[11] - event_lst[9], event_lst[10] - event_lst[8]]
        color_vect = 'blue'
    
    ## Affichage du terrain
    fig, ax = plt.subplots(figsize=(3, 5), facecolor='green')
    ax.set_facecolor('green')
    ax.set_facecolor('green')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Ligne médiane du terrain
    ax.plot([0, 100], [50, 50], 'w--')
    ax.plot([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], 'w-', linewidth=3.5)
    ax.scatter(50, 50, s=10, color='w')
    ax.scatter(50, 50, s=3000, color='w', facecolors='none')
    
    if event_nbr > 0:
        ax.quiver(event_vect_prev[0], event_vect_prev[1], event_vect_prev[2], event_vect_prev[3],
                  color='grey', units='xy', scale=1)
    
    ax.quiver(event_vect[0], event_vect[1], event_vect[2], event_vect[3],
              color=color_vect, units='xy', scale=1)
    
    col1, col2 = st.columns(2) # partitionnement de la page en 2
    with col1:
        st.pyplot(fig)
    
    ## Affichage des informations textuelles
    with col2:
        st.write('Période :', '1re mi-temps' if event_lst[3] == '1H' else '2e mi-temps', time)
        st.write('Club :', 'Getafe' if event_lst[0] == 6 else 'Girona')
        st.write('Joueur :', player_lst[1] + ' ' + player_lst[2] + ' (' + str(relativedelta(dt.strptime('2017-12-17', '%Y-%m-%d'), dt.strptime(player_lst[3], '%Y-%m-%d')).years) + ' ans)')
        st.write('Poste :', dico_pos[player_lst[5]])
        st.write('Évènement :', dico_events[event_lst[5]] + dico_actions[event_lst[6]])
        st.write('Succès :', dico_success[event_lst[-1]])


###############################################################################
## PARTIE 3 : PRÉPARATION DES DONNÉES                                        ##
###############################################################################

if menu_sel == menu_lst[3]:
    st.header(title)
    st.subheader(menu_sel, anchor='Préparation-des-données')
    
    st.write("""Dans cette partie, nous allons voir quels sont les traitements que nous avons effectués
                sur le JDD afin de le « nettoyer » et de la préparer pour la suite.""")
    
    st.code("""
    club        = pd.read_csv('club.csv', index_col='id')
    match       = pd.read_csv('match.csv', index_col='id')
    match_event = pd.read_csv('match_event.csv').sort_values(by=['match_id', 'matchperiod', 'eventsec'])
    player      = pd.read_csv('player.csv', index_col='id')
    referee     = pd.read_csv('referee.csv', index_col='id')
    """, language='python')
    
    st.write("""Dès l'import des données, la table « match_event » est triée par ordre croissant sur
             les colonnes « match_id », « matchperiod » et « eventsec ».""")
    
    #######################################
    ## GESTION DES NA                    ##
    #######################################
    st.subheader('Gestion des NA')
    st.write("""Seules quelques colonnes des tables « player » et « match_event » contiennent des NA
                (voir graphique ci-dessous). Néanmoins, ces colonnes n'étant pas utilisées par la suite,
                toutes les NA ont été conservées tels quels.""")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Préparation des données sur les NA
        pourcents_NA = np.array([.11])
        labels_NA    = ['foot']
        
        # Prédiction du graphique
        fig, ax = plt.subplots()
        ax.bar(range(len(pourcents_NA)), pourcents_NA)
        ax.text(-.05, .05, str(pourcents_NA[0]) + ' %', color='r', size=20)
        ax.set_xticks(range(len(pourcents_NA)))
        ax.set_xticklabels(labels_NA, rotation=45)
        ax.set_xlabel('Colonnes')
        ax.set_ylabel('Pourcentage de NA (%)')
        ax.set_title('Table « player »')
        
        # Affichage
        st.pyplot(fig)
    
    with col2:
        # Préparation des données sur les NA
        pourcents_NA = np.array([.27, 60.34, .02, .02, 9.02])
        labels_NA    = ['action', 'modifier', 'x_end', 'y_end', 'is_success']
        
        # Prédiction du graphique
        fig, ax = plt.subplots()
        ax.bar(range(len(pourcents_NA)), pourcents_NA)
        for x, v in enumerate(pourcents_NA):
            ax.text(x-.2, v+.5, str(v) + ' %', color='r', size=10)
        ax.set_xticks(range(len(pourcents_NA)))
        ax.set_xticklabels(labels_NA, rotation=45)
        ax.set_xlabel('Colonnes')
        ax.set_ylabel('Pourcentage de NA (%)')
        ax.set_title('Table « match_event »')
        
        # Affichage
        st.pyplot(fig)
    
    #######################################
    ## NETTOYAGE DES DONNÉES             ##
    #######################################
    st.subheader('Nettoyage du JDD')
    st.write("""Cette section présente le nettoyage effectué sur les tables.""")
    
    st.write("""Certains nom de club sont mal écrit du fait d'un mauvais encodage. On retrouve
                des exemples de cela dans les tables « club » et « match ». Ces noms ont donc été
                réécrit dans les tables, et colonnes, concernées (voir code ci-dessous).""")
    st.code("""
    # Correction des noms de certains clubs
    dico_correct                              = dict()
    dico_correct['Angers SCO']                = 'Angers'
    dico_correct['Atl\\\\u00e9tico Madrid']     = 'Atlético Madrid'
    dico_correct['Bayern M\\\\u00fcnchen']      = 'Bayern München'
    dico_correct['Deportivo Alav\\\\u00e9s']    = 'Deportivo Alavés'
    dico_correct['Deportivo La Coru\\\\u00f1a'] = 'Deportivo La Coruña'
    dico_correct['K\\\\u00f6ln']                = 'Köln'
    dico_correct['Legan\\\\u00e9s']             = 'Leganés'
    dico_correct['Saint']                     = 'Saint-étienne'
    dico_correct['Saint-\\\\u00c9tienne']       = 'Saint-étienne'
    dico_correct['\\\\u00c9tienne']             = 'Saint-étienne'

    club.name                                   = club.name.replace(dico_correct)
    match[['home_club', 'away_club', 'winner']] = match[['home_club', 'away_club', 'winner']].replace(dico_correct)
    """, language='python')
    
    st.write("""Sur certaines lignes de la table « match », les adverses de Saint-étienne sont mal
                renseignée. Après recherche des opposants corrects, le code, ci-dessous, a permis de
                corrigé la table impactée.""")
    st.code("""
    # Correction des lignes concernant les matchs à domicile de Saint-étienne
    dico_correct                        = dict()
    dico_correct['2017-08-05 18:00:00'] = 'Nice'
    dico_correct['2017-08-19 18:00:00'] = 'Amiens SC'
    dico_correct['2017-10-14 18:00:00'] = 'Metz'
    dico_correct['2017-10-20 18:45:00'] = 'Montpellier'
    dico_correct['2017-11-05 20:00:00'] = 'Olympique Lyonnais'
    dico_correct['2017-12-15 19:45:00'] = 'Monaco'
    dico_correct['2018-01-14 14:00:00'] = 'Toulouse'
    dico_correct['2018-01-27 19:00:00'] = 'Caen'
    dico_correct['2018-03-18 16:00:00'] = 'Guingamp'
    dico_correct['2018-04-22 15:00:00'] = 'Troyes'
    dico_correct['2018-05-06 13:00:00'] = 'Bordeaux'
    dico_correct['2018-05-19 19:00:00'] = 'Lille'
    
    cond = (match.home_club == 'Saint-étienne') & (match.dateutc.isin(dico_correct.keys()))
    match.loc[cond, 'away_club'] = match.loc[cond, 'dateutc'].apply(lambda x: dico_correct[x])
    """, language='python')
    
    #######################################
    ## PRÉPARATION AVANCÉES              ##
    #######################################
    st.subheader('Préparation avancées')
    st.write("""Cette section présente les transformations apportées sur certaines tables, afin que
                celles-ci puissent être plus facilement exploitées par la suite.""")
    
    st.write("""Le format de la colonne « dateutc », portant la date du match, est sous forme de chaîne
                de caractères. Nous avons donc retypé celle-ci.""")
    st.code("""
    # Re-typage
    match.dateutc = match.dateutc.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    """, language='python')
    
    st.write("""Afin de pouvoir être exploitées dans les parties suivantes, la table « match »
                doit être triée par ordre croissant sur la colonne « dateutc ».""")
    st.code("""
    # Tri
    match = match.sort_values(by=['dateutc'])
    """, language='python')
    
    st.write("""Le poste de chaque joueur n'étant pas présent dans la table « match_event », cette
                donné est récupéré depuis la table « player ».""")
    st.code("""
    # Récupération du poste de chaque joueur dans la table « match_event » depuis la table « player »
    match_event = match_event.join(player['position'], on='players_id')
    """, language='python')


###############################################################################
## PARTIE 4 : MODÉLISATION                                                   ##
###############################################################################

if menu_sel == menu_lst[4]:
    st.header(title)
    
    #######################################
    ## MODÉLISATION                      ##
    #######################################
    st.subheader(menu_sel, anchor='Modélisation')
    
    st.markdown("""Dans cette partie, nous allons vous présenter le processus utilisé pour créer les variables explicatives.""")
    st.markdown("""Le schéma, ci-dessous, représente le JDD après nettoyage. De celui-ci, nous ne nous servirons que des tables « match »
                et « match_event » pour créer les *features*.""")
    
    st.image(path_imag + 'Table 01 - JDD après nettoyage.png',
             caption='Les différentes tables du JDD après nettoyage.')
    
    st.markdown("""
                Le schéma, ci-dessous, illustre les grandes étapes pour la création des *features* :
                <ol>
                    <li>
                        la création de la table « match_results » à partir de la table « match »
                        (elle va contenir les résultats, les séries de victoires, les buts marqués, ...) ;
                    </li>
                    <li>
                        la création de la table « match_infos » à partir de la table « match_event »
                        (elle contiendra des statistiques sur les matchs tels que les nombres de tirs , fautes, ...) ;
                    </li>
                    <li>
                        l'augmentation de la table « match_infos » à partir d'elle même
                        (les statistiques seront agrégé ensemble afin d'avoir des tendances sur les 1, 3, 5 derniers matchs non-nuls) ;
                    </li>
                    <li>
                        la création de la « match_all » qui est le regroupement des tables « match_results » et « match_infos » ;
                    </li>
                    <li>
                        la création de la « data » qui reprend des éléments de la table « match_all »
                        afin d'en faire des variables explicatives.
                    </li>
                </ol>
                """, unsafe_allow_html=True)
    
    st.image(path_imag + 'Schéma global du process.png',
             caption="Schéma global du processus d'extraction des variables explicatives.")
    
    #######################################
    ## MATCH_RESULTS                     ##
    #######################################
    st.subheader('Création de la table « match_results »')
    
    st.write("""Dans cette section, nous présentons quels sont les colonnes présentes dans la table « match_results ».
                La description de ces colonnes, ainsi que le code utilisé pour les obtenir sont disponibles ci-dessous.""")
    
    # Schéma pour la création de la table
    st.image(path_imag + 'Table 02 - match vers match_results.png',
             caption='Passage de la table « match » à la table « match_results ».')
    
    # Présentation détaillée des colonnes de la tables « match_results »
    if st.checkbox('Afficher la présentation détaillée des colonnes de la table « match_results »'):
        st.markdown('**Colonnes de la table « match_results » :**')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- <font color=" + st.session_state.cols_color + ">Index                </font> : clef primaire ;",                                                                          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Club                 </font> : nom du club ;",                                                                            unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Période_jour         </font> : chiffre indiquant la période de la journée où le match fut joué ;",                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Match_id             </font> : clef étrangère vers la table « match » ;",                                                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Série_vic        </font> : nombre de victoire sur la saison ;",                                                       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_pour    </font> : nombre de buts marqués sur la saison ;",                                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_contre  </font> : nombre de buts encaissés sur la saison ;",                                                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_diff    </font> : différence entre les buts marqués et ceux encaissés ;",                                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Série_vic_5      </font> : nombre de victoire sur les 5 derniers matchs non nuls ;",                                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_pour_5  </font> : nombre de buts marqués sur les 5 derniers matchs non nuls ;",                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_contre_5</font> : nombre de buts encaissés sur les 5 derniers match non nuls ;",                             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_diff_5  </font> : différence entre les buts marqué et ceux encaissés sur les 5 derniers matchs non nuls ;",  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Série_vic_3      </font> : nombre de victoire sur les 3 derniers matchs non nuls ;",                                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_pour_3  </font> : nombre de buts marqués sur les 3 derniers matchs non nuls ;",                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_contre_3</font> : nombre de buts encaissés sur les 3 derniers match non nuls ;",                             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_diff_3  </font> : différence entre les buts marqués et ceux encaissés sur les 3 derniers matchs non nuls ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Série_vic_1      </font> : nombre de victoire sur le dernier match non nul ;",                                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_pour_1  </font> : nombre de buts marqués sur le dernier match non nul ;",                                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_contre_1</font> : nombre de buts encaissés sur le dernier match non nul ;",                                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_diff_1  </font> : différence entre les buts marqués et ceux encaissés sur le dernier match non nul ;",       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Nb_jours         </font> : nombre de jours depuis le dernier match non nul ;",                                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Nb_jours         </font> : nombre de jours depuis le dernier match non nul pour le club adverse ;",                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">[CLB_Buts]           </font> : nombre de buts marqués à la fin du match (variable cible) ;",                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">[ADV_Buts]           </font> : nombre de buts encaissés à la fin du match (variable cible) ;",                            unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">[Résultat]           </font> : indique la victoire ou la défaite (variable cible).",                                      unsafe_allow_html=True)
        
        with col2:
            st.markdown("- <font color=" + st.session_state.cols_color + ">Date                 </font> : date du match ;",                                                                                               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Club_ADV             </font> : nom du club adverse ;",                                                                                         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Arbitre_id           </font> : clef étrangère vers la table « referee » ;",                                                                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Dom_Ext              </font> : indique si le club joue à domicile ou à l'extérieur ;",                                                         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Série_vic        </font> : nombre de victoire sur la saison pour le club adverse ;",                                                       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_pour    </font> : nombre de buts marqués sur la saison pour le club adverse ;",                                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_contre  </font> : nombre de buts encaissés sur la saison pour le club adverse ;",                                                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_diff    </font> : différence entre les buts marqués et ceux encaissés pour le club adverse ;",                                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Série_vic_5      </font> : nombre de victoire sur les 5 derniers matchs non nuls pour le club adverse ;",                                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_pour_5  </font> : nombre de buts marqués sur les 5 derniers matchs non nuls pour le club adverse ;",                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_contre_5</font> : nombre de buts encaissés sur les 5 derniers match non nuls pour le club adverse ;",                             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_diff_5  </font> : différence entre les buts marqué et ceux encaissés sur les 5 derniers matchs non nuls pour le club adverse ;",  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Série_vic_3      </font> : nombre de victoire sur les 3 derniers matchs non nuls pour le club adverse ;",                                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_pour_3  </font> : nombre de buts marqués sur les 3 derniers matchs non nuls pour le club adverse ;",                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_contre_3</font> : nombre de buts encaissés sur les 3 derniers match non nuls pour le club adverse ;",                             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_diff_3  </font> : différence entre les buts marqués et ceux encaissés sur les 3 derniers matchs non nuls pour le club adverse ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Série_vic_1      </font> : nombre de victoire sur le dernier match non nul pour le club adverse ;",                                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_pour_1  </font> : nombre de buts marqués sur le dernier match non nul pour le club adverse ;",                                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_contre_1</font> : nombre de buts encaissés sur le dernier match non nul pour le club adverse ;",                                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_diff_1  </font> : différence entre les buts marqués et ceux encaissés sur le dernier match non nul pour le club adverse ;",       unsafe_allow_html=True)
    
    # Code utilisé pour la création de la table
    st.write("""Le code, ci-dessous, permet la création de la table « match_results ».""")
    st.code("""
    # Création de la table « match_results »
    match_results = Collect_results(match)
    """, language='python')
    
    if st.checkbox('Afficher le code de la fonction « Collect_results »'):
        st.code("""
        import numpy  as np
        import pandas as pd
        
        def Collect_results(df):
            dico_dates         = {} # Dictionnaire pour la conservation de la date du dernier match de chaque équipe
            dico_streak        = {} # Dictionnaire pour la conservation des victoires et défaites de chaque équipe
            dico_goals_for     = {} # Dictionnaire pour la consertation des buts marqués de chaque équipe
            dico_goals_against = {} # Dictionnaire pour la consertation des buts encaissés de chaque équipe
            
            # Période de la journée
            #  - 0 : [10 h - 12 h[
            #  - 1 : [12 h - 14 h[
            #  - 2 : [14 h - 16 h[
            #  - 3 : [16 h - 18 h[
            #  - 4 : [18 h - 20 h[
            #  - 5 : [20 h - 22 h[
            #  - 6 : [22 h - 24 h[
            periods = {}
            for i in range(10, 21, 2):
                val = i / 2 - 5
                periods[i    ] = val
                periods[i + 1] = val
            
            ###############################################
            ## Composition de la table « match_results » ##
            ###############################################
            
            dico_res = {}
            
            #############################
            ## Variables contextuelles ##
            #############################
            
            dico_res['Date']                  = [] # - Date et heure du match
            dico_res['Club']                  = [] # - Nom du club
            dico_res['Club_ADV']              = [] # - Nom du club adverse
            dico_res['Période_jour']          = [] # - Période de la journée dans laquelle se déroule le match
            dico_res['Arbitre_id']            = [] # - Identifiant de l'arbitre du match
            dico_res['Match_id']              = [] # - Identifiant du match
            
            ###########################
            ## Variable sur le match ##
            ###########################
            
            dico_res['Dom_Ext']               = [] # - Booléen (1. ou 0.) indiquant si le club joue à domicile ou à l'extérieur
            
            ###########################
            ## Variables sur le club ##
            ###########################
            
            # Dynamique du club depuis le début de la saison
            dico_res['CLB_Série_vic']         = [] # - Nombre de matchs remportés par le club
            dico_res['CLB_Tot_buts_pour']     = [] # - Nombre de buts marqués
            dico_res['CLB_Tot_buts_contre']   = [] # - Nombre de buts encaissés
            dico_res['CLB_Tot_buts_diff']     = [] # - Différence de buts
            
            # Dynamique du club sur les 5 derniers matchs
            dico_res['CLB_Série_vic_5']       = [] # - Nombre de matchs remportés par le club
            dico_res['CLB_Tot_buts_pour_5']   = [] # - Nombre de buts marqués
            dico_res['CLB_Tot_buts_contre_5'] = [] # - Nombre de buts encaissés
            dico_res['CLB_Tot_buts_diff_5']   = [] # - Différence de buts
            
            # Dynamique du club sur les 3 derniers matchs
            dico_res['CLB_Série_vic_3']       = [] # - Nombre de matchs remportés par le club
            dico_res['CLB_Tot_buts_pour_3']   = [] # - Nombre de buts marqués
            dico_res['CLB_Tot_buts_contre_3'] = [] # - Nombre de buts encaissés
            dico_res['CLB_Tot_buts_diff_3']   = [] # - Différence de buts
            
            # Dynamique du club sur le dernier match
            dico_res['CLB_Série_vic_1']       = [] # - Nombre de matchs remportés par le club
            dico_res['CLB_Tot_buts_pour_1']   = [] # - Nombre de buts marqués
            dico_res['CLB_Tot_buts_contre_1'] = [] # - Nombre de buts encaissés
            dico_res['CLB_Tot_buts_diff_1']   = [] # - Différence de buts
            
            # Nombre de jours depuis le derniers match gagné ou perdu
            dico_res['CLB_Nb_jours']          = [] # - Nombre de jours depuis le dernier match gagné ou perdu
            
            ###################################
            ## Variables sur le club adverse ##
            ###################################
            
            # Dynamique du club adverse depuis le début de la saison
            dico_res['ADV_Série_vic']         = [] # - Nombre de matchs remportés par le club
            dico_res['ADV_Tot_buts_pour']     = [] # - Nombre de buts marqués
            dico_res['ADV_Tot_buts_contre']   = [] # - Nombre de buts encaissés
            dico_res['ADV_Tot_buts_diff']     = [] # - Différence de buts
            
            # Dynamique du club adverse sur les 5 derniers matchs
            dico_res['ADV_Série_vic_5']       = [] # - Nombre de matchs remportés par le club
            dico_res['ADV_Tot_buts_pour_5']   = [] # - Nombre de buts marqués
            dico_res['ADV_Tot_buts_contre_5'] = [] # - Nombre de buts encaissés
            dico_res['ADV_Tot_buts_diff_5']   = [] # - Différence de buts
            
            # Dynamique du club adverse sur les 3 derniers matchs
            dico_res['ADV_Série_vic_3']       = [] # - Nombre de matchs remportés par le club
            dico_res['ADV_Tot_buts_pour_3']   = [] # - Nombre de buts marqués
            dico_res['ADV_Tot_buts_contre_3'] = [] # - Nombre de buts encaissés
            dico_res['ADV_Tot_buts_diff_3']   = [] # - Différence de buts
            
            # Dynamique du club adverse sur le dernier match
            dico_res['ADV_Série_vic_1']       = [] # - Nombre de matchs remportés par le club
            dico_res['ADV_Tot_buts_pour_1']   = [] # - Nombre de buts marqués
            dico_res['ADV_Tot_buts_contre_1'] = [] # - Nombre de buts encaissés
            dico_res['ADV_Tot_buts_diff_1']   = [] # - Différence de buts
            
            # Nombre de jours depuis le derniers match gagné ou perdu
            dico_res['ADV_Nb_jours']          = [] # - Nombre de jours depuis le dernier match gagné ou perdu
            
            #########################
            ## Variables à prédire ##
            #########################
            
            dico_res['[Résultat]']            = [] # - Valeur (1. ou 0.) indiquant si le club à gagné ou perdu le match
            dico_res['[CLB_Buts]']            = [] # - Nombre de buts marqués par l'équipe jouant à domicile
            dico_res['[ADV_Buts]']            = [] # - Nombre de buts marqués par l'équipe jouant à l'extérieur
            
            ###############################################
            ## Remplissage de la table « match_results » ##
            ###############################################
            
            for index, row in df.iterrows():
                # Calcul de la valeur correspondant au résultat du match
                vic_home = 1. if row.goal_by_home_club > row.goal_by_away_club else 0.
                
                # Gestion du nombre de jours sans victoires ni défaites
                if row.home_club not in dico_dates.keys():
                    home_days = None
                else:
                    home_days = (row.dateutc.date() - dico_dates[row.home_club]).days
                
                if row.away_club not in dico_dates.keys():
                    away_days = None
                else:
                    away_days = (row.dateutc.date() - dico_dates[row.away_club]).days
                
                dico_dates[row.home_club] = row.dateutc.date() # Insertion de la date du match courant
                dico_dates[row.away_club] = row.dateutc.date() # Insertion de la date du match courant
                
                # Gestion de la série de victoires
                if row.home_club not in dico_streak.keys():
                    dico_streak[row.home_club] = [vic_home]
                else:
                    dico_streak[row.home_club].append(vic_home)
                
                if row.away_club not in dico_streak.keys():
                    dico_streak[row.away_club] = [1. - vic_home]
                else:
                    dico_streak[row.away_club].append(1. - vic_home)
                
                # Gestion des buts pour
                if row.home_club not in dico_goals_for.keys():
                    dico_goals_for[row.home_club] = [row.goal_by_home_club]
                else:
                    dico_goals_for[row.home_club].append(row.goal_by_home_club)
                
                if row.away_club not in dico_goals_for.keys():
                    dico_goals_for[row.away_club] = [row.goal_by_away_club]
                else:
                    dico_goals_for[row.away_club].append(row.goal_by_away_club)
                
                # Gestion des buts contre
                if row.home_club not in dico_goals_against.keys():
                    dico_goals_against[row.home_club] = [row.goal_by_away_club]
                else:
                    dico_goals_against[row.home_club].append(row.goal_by_away_club)
                
                if row.away_club not in dico_goals_against.keys():
                    dico_goals_against[row.away_club] = [row.goal_by_home_club]
                else:
                    dico_goals_against[row.away_club].append(row.goal_by_home_club)
                
                # Insertion des éléments communs aux 2 clubs
                for _ in range(2):
                    dico_res['Date'].append(row.dateutc)
                    dico_res['Période_jour'].append(periods[row.dateutc.hour])
                    dico_res['Arbitre_id'].append(row.referee_id)
                    dico_res['Match_id'].append(index)
                
                # Calcul des éléments à insérer
                A_Serie_vic         = sum(dico_streak[row.home_club][:-1])
                A_Tot_buts_pour     = sum(dico_goals_for[row.home_club][:-1])
                A_Tot_buts_contre   = sum(dico_goals_against[row.home_club][:-1])
                A_Série_vic_5       = sum(dico_streak[row.home_club][-min(6, len(dico_streak[row.home_club])):-1])
                A_Tot_buts_pour_5   = sum(dico_goals_for[row.home_club][-min(6, len(dico_streak[row.home_club])):-1])
                A_Tot_buts_contre_5 = sum(dico_goals_against[row.home_club][-min(6, len(dico_streak[row.home_club])):-1])
                A_Série_vic_3       = sum(dico_streak[row.home_club][-min(4, len(dico_streak[row.home_club])):-1])
                A_Tot_buts_pour_3   = sum(dico_goals_for[row.home_club][-min(4, len(dico_streak[row.home_club])):-1])
                A_Tot_buts_contre_3 = sum(dico_goals_against[row.home_club][-min(4, len(dico_streak[row.home_club])):-1])
                A_Série_vic_1       = sum(dico_streak[row.home_club][-min(2, len(dico_streak[row.home_club])):-1])
                A_Tot_buts_pour_1   = sum(dico_goals_for[row.home_club][-min(2, len(dico_streak[row.home_club])):-1])
                A_Tot_buts_contre_1 = sum(dico_goals_against[row.home_club][-min(2, len(dico_streak[row.home_club])):-1])
                
                B_Serie_vic         = sum(dico_streak[row.away_club][:-1])
                B_Tot_buts_pour     = sum(dico_goals_for[row.away_club][:-1])
                B_Tot_buts_contre   = sum(dico_goals_against[row.away_club][:-1])
                B_Série_vic_5       = sum(dico_streak[row.away_club][-min(6, len(dico_streak[row.away_club])):-1])
                B_Tot_buts_pour_5   = sum(dico_goals_for[row.away_club][-min(6, len(dico_streak[row.away_club])):-1])
                B_Tot_buts_contre_5 = sum(dico_goals_against[row.away_club][-min(6, len(dico_streak[row.away_club])):-1])
                B_Série_vic_3       = sum(dico_streak[row.away_club][-min(4, len(dico_streak[row.away_club])):-1])
                B_Tot_buts_pour_3   = sum(dico_goals_for[row.away_club][-min(4, len(dico_streak[row.away_club])):-1])
                B_Tot_buts_contre_3 = sum(dico_goals_against[row.away_club][-min(4, len(dico_streak[row.away_club])):-1])
                B_Série_vic_1       = sum(dico_streak[row.away_club][-min(2, len(dico_streak[row.away_club])):-1])
                B_Tot_buts_pour_1   = sum(dico_goals_for[row.away_club][-min(2, len(dico_streak[row.away_club])):-1])
                B_Tot_buts_contre_1 = sum(dico_goals_against[row.away_club][-min(2, len(dico_streak[row.away_club])):-1])
                
                # Insertion des éléments concernant le premier club (celui jouant à domicile)
                dico_res['Club'].append(row.home_club)
                dico_res['Club_ADV'].append(row.away_club)
                dico_res['Dom_Ext'].append(1.)
                
                dico_res['CLB_Série_vic'].append(A_Serie_vic)
                dico_res['CLB_Tot_buts_pour'].append(A_Tot_buts_pour)
                dico_res['CLB_Tot_buts_contre'].append(A_Tot_buts_contre)
                dico_res['CLB_Tot_buts_diff'].append(A_Tot_buts_pour - A_Tot_buts_contre)
                dico_res['CLB_Série_vic_5'].append(A_Série_vic_5)
                dico_res['CLB_Tot_buts_pour_5'].append(A_Tot_buts_pour_5)
                dico_res['CLB_Tot_buts_contre_5'].append(A_Tot_buts_contre_5)
                dico_res['CLB_Tot_buts_diff_5'].append(A_Tot_buts_pour_5 - A_Tot_buts_contre_5)
                dico_res['CLB_Série_vic_3'].append(A_Série_vic_3)
                dico_res['CLB_Tot_buts_pour_3'].append(A_Tot_buts_pour_3)
                dico_res['CLB_Tot_buts_contre_3'].append(A_Tot_buts_contre_3)
                dico_res['CLB_Tot_buts_diff_3'].append(A_Tot_buts_pour_3 - A_Tot_buts_contre_3)
                dico_res['CLB_Série_vic_1'].append(A_Série_vic_1)
                dico_res['CLB_Tot_buts_pour_1'].append(A_Tot_buts_pour_1)
                dico_res['CLB_Tot_buts_contre_1'].append(A_Tot_buts_contre_1)
                dico_res['CLB_Tot_buts_diff_1'].append(A_Tot_buts_pour_1 - A_Tot_buts_contre_1)
                dico_res['CLB_Nb_jours'].append(home_days)
                
                dico_res['ADV_Série_vic'].append(B_Serie_vic)
                dico_res['ADV_Tot_buts_pour'].append(B_Tot_buts_pour)
                dico_res['ADV_Tot_buts_contre'].append(B_Tot_buts_contre)
                dico_res['ADV_Tot_buts_diff'].append(B_Tot_buts_pour - B_Tot_buts_contre)
                dico_res['ADV_Série_vic_5'].append(B_Série_vic_5)
                dico_res['ADV_Tot_buts_pour_5'].append(B_Tot_buts_pour_5)
                dico_res['ADV_Tot_buts_contre_5'].append(B_Tot_buts_contre_5)
                dico_res['ADV_Tot_buts_diff_5'].append(B_Tot_buts_pour_5 - B_Tot_buts_contre_5)
                dico_res['ADV_Série_vic_3'].append(B_Série_vic_3)
                dico_res['ADV_Tot_buts_pour_3'].append(B_Tot_buts_pour_3)
                dico_res['ADV_Tot_buts_contre_3'].append(B_Tot_buts_contre_3)
                dico_res['ADV_Tot_buts_diff_3'].append(B_Tot_buts_pour_3 - B_Tot_buts_contre_3)
                dico_res['ADV_Série_vic_1'].append(B_Série_vic_1)
                dico_res['ADV_Tot_buts_pour_1'].append(B_Tot_buts_pour_1)
                dico_res['ADV_Tot_buts_contre_1'].append(B_Tot_buts_contre_1)
                dico_res['ADV_Tot_buts_diff_1'].append(B_Tot_buts_pour_1 - B_Tot_buts_contre_1)
                dico_res['ADV_Nb_jours'].append(away_days)
                
                dico_res['[Résultat]'].append(vic_home)
                dico_res['[CLB_Buts]'].append(row.goal_by_home_club)
                dico_res['[ADV_Buts]'].append(row.goal_by_away_club)
                
                # Insertion des éléments concernant le second club (celui jouant à l'extérieur)
                dico_res['Club'].append(row.away_club)
                dico_res['Club_ADV'].append(row.home_club)
                dico_res['Dom_Ext'].append(0.)
                
                dico_res['CLB_Série_vic'].append(B_Serie_vic)
                dico_res['CLB_Tot_buts_pour'].append(B_Tot_buts_pour)
                dico_res['CLB_Tot_buts_contre'].append(B_Tot_buts_contre)
                dico_res['CLB_Tot_buts_diff'].append(B_Tot_buts_pour - B_Tot_buts_contre)
                dico_res['CLB_Série_vic_5'].append(B_Série_vic_5)
                dico_res['CLB_Tot_buts_pour_5'].append(B_Tot_buts_pour_5)
                dico_res['CLB_Tot_buts_contre_5'].append(B_Tot_buts_contre_5)
                dico_res['CLB_Tot_buts_diff_5'].append(B_Tot_buts_pour_5 - B_Tot_buts_contre_5)
                dico_res['CLB_Série_vic_3'].append(B_Série_vic_3)
                dico_res['CLB_Tot_buts_pour_3'].append(B_Tot_buts_pour_3)
                dico_res['CLB_Tot_buts_contre_3'].append(B_Tot_buts_contre_3)
                dico_res['CLB_Tot_buts_diff_3'].append(B_Tot_buts_pour_3 - B_Tot_buts_contre_3)
                dico_res['CLB_Série_vic_1'].append(B_Série_vic_1)
                dico_res['CLB_Tot_buts_pour_1'].append(B_Tot_buts_pour_1)
                dico_res['CLB_Tot_buts_contre_1'].append(B_Tot_buts_contre_1)
                dico_res['CLB_Tot_buts_diff_1'].append(B_Tot_buts_pour_1 - B_Tot_buts_contre_1)
                dico_res['CLB_Nb_jours'].append(away_days)
                
                dico_res['ADV_Série_vic'].append(A_Serie_vic)
                dico_res['ADV_Tot_buts_pour'].append(A_Tot_buts_pour)
                dico_res['ADV_Tot_buts_contre'].append(A_Tot_buts_contre)
                dico_res['ADV_Tot_buts_diff'].append(A_Tot_buts_pour - A_Tot_buts_contre)
                dico_res['ADV_Série_vic_5'].append(A_Série_vic_5)
                dico_res['ADV_Tot_buts_pour_5'].append(A_Tot_buts_pour_5)
                dico_res['ADV_Tot_buts_contre_5'].append(A_Tot_buts_contre_5)
                dico_res['ADV_Tot_buts_diff_5'].append(A_Tot_buts_pour_5 - A_Tot_buts_contre_5)
                dico_res['ADV_Série_vic_3'].append(A_Série_vic_3)
                dico_res['ADV_Tot_buts_pour_3'].append(A_Tot_buts_pour_3)
                dico_res['ADV_Tot_buts_contre_3'].append(A_Tot_buts_contre_3)
                dico_res['ADV_Tot_buts_diff_3'].append(A_Tot_buts_pour_3 - A_Tot_buts_contre_3)
                dico_res['ADV_Série_vic_1'].append(A_Série_vic_1)
                dico_res['ADV_Tot_buts_pour_1'].append(A_Tot_buts_pour_1)
                dico_res['ADV_Tot_buts_contre_1'].append(A_Tot_buts_contre_1)
                dico_res['ADV_Tot_buts_diff_1'].append(A_Tot_buts_pour_1 - A_Tot_buts_contre_1)
                dico_res['ADV_Nb_jours'].append(home_days)
                
                dico_res['[Résultat]'].append(1. - vic_home)
                dico_res['[CLB_Buts]'].append(row.goal_by_away_club)
                dico_res['[ADV_Buts]'].append(row.goal_by_home_club)
                
            return pd.DataFrame(data=dico_res)
        """, language='python')
    
    # Affichage d'un extrait de la table « match_results »
    df_match_results_samples = pd.read_csv(path_data + 'match_results_samples.csv', index_col=0)
    
    if st.checkbox('Afficher un extrait de la table « match_results »'):
        df_match_results_samples
    
    #######################################
    ## MATCH_INFOS                       ##
    #######################################
    
    st.subheader('Création de la table « match_infos »')
    st.write("""Dans cette section, nous présentons quels sont les colonnes présentes dans la table « match_infos ».
                La description de ces colonnes, ainsi que le code utilisé pour les obtenir sont disponibles ci-dessous.""")
    
    # Schéma pour la création de la table
    st.image(path_imag + 'Table 03 - match_events vers match_infos.png',
             caption='Passage de la table « match_event » à la table « match_infos ».')
    
    # Présentation détaillée des colonnes de la tables « match_infos »
    if st.checkbox('Afficher la présentation détaillée des colonnes de la table « match_infos »'):
        st.markdown('**Colonnes de la table « match_infos » :**')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- <font color=" + st.session_state.cols_color + ">Index                   </font> : clef primaire ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Équipe_id               </font> : clef étrangère vers la table « club » ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Équipe_2H)        </font> : nombre de tirs proches du but adverse pendant la 2<sup>e</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Milieux_2H)       </font> : nombre de tirs des milieux proches du but adverse pendant la 2<sup>e</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Défenseurs_2H)    </font> : nombre de tirs des défenseurs proches du but adverse pendant la 2<sup>e</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Attaquants_2H)    </font> : nombre de tirs des attaquants proches du but adverse pendant la 2<sup>e</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Passes_Équipe_2H)      </font> : nombre de passes intelligentes pendant la 2<sup>e</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Équipe_2H)      </font> : nombre de fautes en défense pendant la 2<sup>e</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Milieux_2H)     </font> : nombre de fautes des milieux en défense pendant la 2<sup>e</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Gardiens_2H)    </font> : nombre de fautes des gardiens pendant la 2<sup>e</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Défenseurs_2H)  </font> : nombre de fautes des défenseurs en défense pendant la 2<sup>e</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Attaquants_2H)  </font> : nombre de fautes des attaquants en défense pendant la 2<sup>e</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Arrêts_Gardiens_2H)    </font> : nombre d'arrêts des gardiens pendant la 2<sup>e</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Équipe_2H)    </font> : nombre d'évènements impliquant l'équipe pendant la 2<sup>e</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Milieux_2H)   </font> : nombre d'évènements impliquant les milieux pendant la 2<sup>e</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Défenseurs_2H)</font> : nombre d'évènements impliquant les défenseurs pendant la 2<sup>e</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Attaquants_2H)</font> : nombre d'évènements impliquant les attaquants pendant la 2<sup>e</sup> mi-temps ;", unsafe_allow_html=True)
        with col2:
            st.markdown("<font color='black'>.</font>", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Match_id                </font> : clef étrangère vers la table « match » ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Équipe_1H)        </font> : nombre de tirs proches du but adverse pendant la 1<sup>re</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Milieux_1H)       </font> : nombre de tirs des milieux proches du but adverse pendant la 1<sup>re</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Défenseurs_1H)    </font> : nombre de tirs des défenseurs proches du but adverse pendant la 1<sup>re</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Attaquants_1H)    </font> : nombre de tirs des attaquants proches du but adverse pendant la 1<sup>re</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Passes_Équipe_1H)      </font> : nombre de passes intelligentes pendant la 1<sup>re</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Équipe_1H)      </font> : nombre de fautes en défense pendant la 1<sup>re</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Milieux_1H)     </font> : nombre de fautes des milieux en défense pendant la 1<sup>re</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Gardiens_1H)    </font> : nombre de fautes des gardiens pendant la 1<sup>re</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Défenseurs_1H)  </font> : nombre de fautes des défenseurs en défense pendant la 1<sup>re</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Attaquants_1H)  </font> : nombre de fautes des attaquants en défense pendant la 1<sup>re</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Arrêts_Gardiens_1H)    </font> : nombre d'arrêts des gardiens pendant la 1<sup>re</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Équipe_1H)    </font> : nombre d'évènements impliquant l'équipe pendant la 1<sup>re</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Milieux_1H)   </font> : nombre d'évènements impliquant les milieux pendant la 1<sup>re</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Défenseurs_1H)</font> : nombre d'évènements impliquant les défenseurs pendant la 1<sup>re</sup>> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Attaquants_1H)</font> : nombre d'évènements impliquant les attaquants pendant la 1<sup>re</sup> mi-temps.", unsafe_allow_html=True)
    
    # Code utilisé pour la création de la table
    st.write("""Le code, ci-dessous, permet la création de la table « match_infos ».""")
    st.code("""
    match_infos = None
    
    # Boucle sur l'ensemble des matchs présent dans la table « match_results »
    for match_id in match_results.Match_id.unique():
        # Récupération des événements du match courant
        current_match = match_event[match_event.match_id == match_id].sort_values(by=['matchperiod', 'eventsec'])
        
        # Collecte des informations sur le match courant
        meta_infos, infos = Collect_infos(current_match)
        
        # Agrégation des informations collectées ci-avant
        if match_infos is None:
            n_index, match_infos = Aggregate_infos(match_id, meta_infos, infos)
        else:
            n_index, stats = Aggregate_infos(match_id, meta_infos, infos, n_index)
            match_infos    = match_infos.append(stats)

    # Ré-ordination des colonnes
    match_infos = match_infos.reindex(sorted(match_infos.columns, reverse=True), axis=1)
    """, language='python')
    
    if st.checkbox('Afficher le code de la fonction « Collect_infos »'):
        st.code("""
        import numpy  as np
        import pandas as pd
        
        def Collect_infos(df):
            # Dictionnaire pour le stockage des informations sur les joueurs de la rencontre
            #  - CLEF   : identifiant du joueur ;
            #  - VALEUR : [identifiant de l'équipe, position].
            dico_meta = {}
            # Dictionnaire pour le stockage d'informations sur le match concernant chaque joueur.
            # Les types d'informations collectées sont résumés dans le tableau ci-dessous. De plus,
            # ces informations sont récupérées pour chaque période du match.
            # Liste des clefs :
            #  - (type) + '_' + (identifiant du joueur) + '_' + (période du match)
            # avec
            #  - type dans [Activité, Arrêts, Fautes, Passes, Tirs] ;
            #  - période du match dans [1H, 2H] (1re mi-temps, 2e mi-temps).
            dico_infos = {}
            
            #=========================================================================#
            # Activité   | Calcul du nombre d'actions pour chaque joueur.             #
            # ----------------------------------------------------------------------- #
            # Arrêts     | Calcul du nombre d'arrêts.                                 #
            # ----------------------------------------------------------------------- #
            # Fautes     | Calcul du nombre de fautes dans les x mètres.              #
            # ----------------------------------------------------------------------- #
            # Passes     | Calcul du nombre de passes intelligentes.                  #
            # ----------------------------------------------------------------------- #
            # Tirs       | Calcul du nombre de tirs dans les x derniers mètres.       #
            #=========================================================================#
            
            # Initialisation
            for player in df.players_id.unique():
                # Récupération de l'équipe et du poste de chaque joueur
                dico_meta[player] = list(df[df.players_id == player][['club_id', 'position']].iloc[0])
                
                # Initialisation des informations des joueurs
                #   - Activité : nombre d'actions réalisés ;
                #   - Arrêts   : nombre d'arrêts réalisés ;
                #   - Fautes   : nombre de fautes commises en défense ;
                #   - Passes   : nombre de passes intelligentes réalisés ;
                #   - Tirs     : nombre de tirs réalisés proche du but adverse.
                for half in ['1H', '2H']:
                    for t in ['Activité', 'Arrêts', 'Fautes', 'Passes', 'Tirs']:
                        dico_infos['_'.join([t, str(player), half])] = 0.
            
            # Début du calcul
            for half in ['1H', '2H']:
                # Début de la collecte des informations
                for i, row in df[df['matchperiod'] == half].iterrows():
                    player = str(row['players_id'])
                    
                    # Activité : incrément de l'activité du joueur (nombre d'actions)
                    dico_infos['_'.join(['Activité', player, half])] += 1.
                    
                    # Arrêts : incrément du nombre d'arrêts
                    if row.eventname == 'Save attempt' and row.is_success != 'f':
                        dico_infos['_'.join(['Arrêts', player, half])] += 1.
                    
                    # Fautes : incrément du nombre de fautes commises dans sa première moitié de terrain (=> fautes commises en défense)
                    elif row.eventname == 'Foul':
                        if row.x_begin <= 25:
                            dico_infos['_'.join(['Fautes', player, half])] += 1.
                    
                    # Passes : incrément du nombre de passes intelligentes
                    elif row.eventname == 'Pass' and row.action == 'Smart pass':
                        dico_infos['_'.join(['Passes', player, half])] += 1.
                    
                    # Tirs :incrément du nombre de tirs dans la première moitié du terrain adverse (celle proche du but)
                    elif row.eventname == 'Shot':
                        if row.x_begin >= 75:
                            dico_infos['_'.join(['Tirs', player, half])] += 1.
            
            # Envoie des informations
            return dico_meta, pd.DataFrame(data=dico_infos, index=[0])
        """, language='python')
    
    # Affichage d'un exemple de sortie de la fonction « Collect_infos »
    if st.checkbox('Afficher un exemple de sortie de la fonction « Collect_infos() »'):
        col1, col2 = st.columns(2)
        
        # Partie MÉTA INFOS
        with col1:
            st.markdown('**meta_infos**')
            dico_meta_data = {311:  [59, 'Forward'],
                              1314: [59, 'Midfielder'],
                              389:  [59, 'Midfielder'],
                              403:  [59, 'Defender'],
                              216:  [59, 'Defender'],
                              257:  [59, 'Midfielder'],
                              1242: [6, 'Forward'],
                              1460: [59, 'Midfielder'],
                              71:   [59, 'Defender'],
                              2162: [6, 'Defender'],
                              3234: [6, 'Defender'],
                              161:  [59, 'Midfielder'],
                              1839: [6, 'Midfielder'],
                              442:  [6, 'Midfielder'],
                              282:  [6, 'Midfielder'],
                              287:  [6, 'Forward'],
                              378:  [6, 'Forward'],
                              359:  [6, 'Defender'],
                              1739: [59, 'Goalkeeper'],
                              2065: [59, 'Defender'],
                              150:  [6, 'Goalkeeper'],
                              1673: [6, 'Defender'],
                              304:  [59, 'Defender'],
                              3217: [6, 'Midfielder'],
                              2215: [6, 'Midfielder'],
                              3235: [59, 'Forward'],
                              575:  [59, 'Defender']}
            
            dico_meta_data
        
        # Partie INFOS
        with col2:
            st.markdown('**infos**')
            df_collect_infos_samples = pd.read_csv(path_data + 'collect_infos_samples.csv', index_col=0)
            
            df_collect_infos_samples
    
    if st.checkbox('Afficher le code de la fonction « Aggregate_infos() »'):
        st.code("""
        import numpy  as np
        import pandas as pd

        import re
        
        def Aggregate_infos(match_id, meta_infos, infos, n_index=0):
            results1,              results2              = dict(), dict()
            results1['Équipe_id'], results2['Équipe_id'] = None, None
            results1['Match_id'],  results2['Match_id']  = None, None
            
            team1,               team2               = dict(), dict()
            team1['Équipe'],     team2['Équipe']     = list(set([arr[0] for arr in meta_infos.values()]))
            team1['Tous'],       team2['Tous']       = list(), list()
            team1['Attaquants'], team2['Attaquants'] = list(), list()
            team1['Milieux'],    team2['Milieux']    = list(), list()
            team1['Défenseurs'], team2['Défenseurs'] = list(), list()
            team1['Gardiens'],   team2['Gardiens']   = list(), list()
            
            # Traitement des méta-informations
            for player in meta_infos:
                # Récupération des informations sur le joueur courant (équipe + poste)
                team, pos = meta_infos[player]
                
                # Regroupement des informations
                ptr_team = team1 if team == team1['Équipe'] else team2
                if pos == 'Forward':
                    ptr_team['Tous'].append(player)
                    ptr_team['Attaquants'].append(player)
                elif pos == 'Midfielder':
                    ptr_team['Tous'].append(player)
                    ptr_team['Milieux'].append(player)
                elif pos == 'Defender':
                    ptr_team['Tous'].append(player)
                    ptr_team['Défenseurs'].append(player)
                elif pos == 'Goalkeeper':
                    ptr_team['Tous'].append(player)
                    ptr_team['Gardiens'].append(player)
            
            # Identifiants équipes + match
            results1['Équipe_id'], results2['Équipe_id'] = team1['Équipe'], team2['Équipe']
            results1['Match_id'],  results2['Match_id']  = match_id,        match_id
            
            ############################
            ## L'ACTIVITÉ             ##
            ############################

            for ptr_res, ptr_team in zip([results1, results2], [team1, team2]):
                for half in ['1H', '2H']:
                    # Calcul du nombre total d'activité pour l'équipe courante (sauf gardiens)
                    pattern = '^Activité_' + ''.join(map(lambda x: '(' + str(x) + ')?', ptr_team['Attaquants'] + ptr_team['Milieux'] + ptr_team['Défenseurs'])) + '_' + half + '$'
                    ptr_res['(Activité_Équipe_' + half + ')'] = infos.filter(regex=pattern, axis=1).sum().sum()

                    # Boucle sur les différents postes (sauf gardiens)
                    for t in ['Attaquants', 'Milieux', 'Défenseurs']:
                        pattern = '^Activité_' + ''.join(map(lambda x: '(' + str(x) + ')?', ptr_team[t])) + '_' + half + '$'
                        ptr_res['(Activité_' + t + '_' + half + ')'] = infos.filter(regex=pattern, axis=1).sum().sum()
            
            ############################
            ## LES ARRÊTS             ##
            ############################

            for ptr_res, ptr_team in zip([results1, results2], [team1, team2]):
                for half in ['1H', '2H']:
                    # Calcul du nombre d'arrêt fait par les gardiens
                    pattern = '^Arrêts_' + ''.join(map(lambda x: '(' + str(x) + ')?', ptr_team['Gardiens'])) + '_' + half + '$'
                    ptr_res['(Arrêts_Gardiens_' + half + ')'] = infos.filter(regex=pattern, axis=1).sum().sum()
            
            ############################
            ## LES FAUTES             ##
            ############################

            for ptr_res, ptr_team in zip([results2, results1], [team1, team2]): # Inversion des dictionnaires de sorties car on s'intéresse au fautes commises par l'équipe adversaire
                for half in ['1H', '2H']:
                    # Calcul du nombre total de fautes, dans sa partie de terrain, pour l'équipe courante
                    pattern = '^Fautes_' + ''.join(map(lambda x: '(' + str(x) + ')?', ptr_team['Tous'])) + '_' + half + '$'
                    ptr_res['(Fautes_Équipe_' + half + ')'] = infos.filter(regex=pattern, axis=1).sum().sum()

                    # Boucle sur les différents postes
                    for t in ['Attaquants', 'Milieux', 'Défenseurs', 'Gardiens']:
                        pattern = '^Fautes_' + ''.join(map(lambda x: '(' + str(x) + ')?', ptr_team[t])) + '_' + half + '$'
                        ptr_res['(Fautes_' + t + '_' + half + ')'] = infos.filter(regex=pattern, axis=1).sum().sum()
            
            ############################
            ## LES PASSES INTEL       ##
            ############################

            for ptr_res, ptr_team in zip([results1, results2], [team1, team2]):
                for half in ['1H', '2H']:
                    # Calcul du nombre total de passes intelligentes pour l'équipe courante
                    pattern = '^Passes_' + ''.join(map(lambda x: '(' + str(x) + ')?', ptr_team['Tous'])) + '_' + half + '$'
                    ptr_res['(Passes_Équipe_' + half + ')'] = infos.filter(regex=pattern, axis=1).sum().sum()
            
            ############################
            ## LES TIRS PROCHES       ##
            ############################

            for ptr_res, ptr_team in zip([results1, results2], [team1, team2]):
                for half in ['1H', '2H']:
                    # Calcul du nombre total de tirs, à proximité du but adverse, pour l'équipe courante
                    pattern = '^Tirs_' + ''.join(map(lambda x: '(' + str(x) + ')?', ptr_team['Tous'])) + '_' + half + '$'
                    ptr_res['(Tirs_Équipe_' + half + ')'] = infos.filter(regex=pattern, axis=1).sum().sum()

                    # Boucle sur les différents postes (sauf gardiens)
                    for t in ['Attaquants', 'Milieux', 'Défenseurs']:
                        pattern = '^Tirs_' + ''.join(map(lambda x: '(' + str(x) + ')?', ptr_team[t])) + '_' + half + '$'
                        ptr_res['(Tirs_' + t + '_' + half + ')'] = infos.filter(regex=pattern, axis=1).sum().sum()
            
            ############################
            ## RETOURS                ##
            ############################
            
            # Envoie du prochain numéro d'index + les 2 lignes de résultats dans un DataFrame
            return n_index + 2, pd.concat([pd.DataFrame(data=results1, index=[n_index]), pd.DataFrame(data=results2, index=[n_index + 1])], axis=0)
        """, language='python')
    
    # Affichage d'un extrait de la table « match_results »
    if st.checkbox('Afficher un extrait de la table « match_infos »'):
        # Chargement des données
        df_match_infos_samples = pd.read_csv(path_data + 'match_infos_985.csv', index_col=0)
        
        df_match_infos_samples
    
    # /!\ Rappel important
    st.warning("""Les informations calculés correspondes au statistiques du match duquel il faut prédire l'issue.
                  Il ne sera donc pas possible de s'en servir comme variables explicatives pour ce match. Par contre,
                  elles peuvent être utilisées pour les matchs ultérieurs. C'est l'objet de la partir augmentation
                  de cette table (voir la section suivante).""")
    
    #######################################
    ## AUGMENTATION DE MATCH_INFOS       ##
    #######################################
    
    st.subheader('Augmentation de la table « match_infos »')
    
    st.write("""Dans cette section, nous présentons quels sont les colonnes présentes dans la table « match_infos » une fois augmenté.
                Cette augmentation se déroule en deux temps. Il s'agit, tout d'abord, de récupérer depuis la table « match_results » la date des matchs.
                La jointure entre les tables « match_results » et « match_infos » se fait via les colonnes « Match_id » et « Club ».
                Cette dernière colonne n'existant pas dans la table « match_infos », elle est à récupérer depuis la table « club » (colonne « name »).""")
    
    # Schéma pour l'augmentation de la table « match_infos » (phase 1)
    st.image(path_imag + 'Table 04 - augmentation match_infos phase 1.png',
             caption='Augmentation de la table « match_infos » (phase 1).')
    
    st.write("""Ci-dessous, le code permettant la réalisation de cette première phase.""")
    
    st.code("""
    # Récupération des noms des clubs dans la table « match_infos » depuis la table « club »
    match_infos.insert(2, 'Club', match_infos.join(club.name, on='Équipe_id').name)
    
    # Récupération de la date des match dans la table « match_infos » depuis la table « match_results »
    match_infos.insert(0, 'Date', pd.merge(match_results[['Match_id', 'Club', 'Date']], match_infos[['Match_id', 'Club']], on=['Match_id', 'Club']).Date)
    """, language='python')
    
    st.write("""Vient ensuite la seconde phase qui permet d'agréger les statistiques, calculés précédemment sur les 1, 3 et 5 derniers matchs,
                à partir de chaque match (opération disponible uniquement car nous venons de récupérer les dates de chacun d'entres eux).""")
    st.write("""Le détail de chaque colonne de la table, ainsi que le code utilisé pour les obtenir sont disponibles ci-dessous.""")
    
    # Schéma pour l'augmentation de la table « match_infos » (phase 2)
    st.image(path_imag + 'Table 05 - augmentation match_infos phase 2.png',
             caption='Augmentation de la table « match_infos » (phase 2).')
    
    # Présentation détaillée des colonnes de la tables « match_infos » augmentée
    if st.checkbox('Afficher la présentation détaillée des colonnes de la table « match_infos » augmentée'):
        st.markdown('**Colonnes de la table « match_infos » augmentée :**')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- <font color=" + st.session_state.cols_color + ">Index                   </font> : clef primaire ;",                                                                           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Équipe_id               </font> : clef étrangère vers la table « club » ;",                                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Club                    </font> : nom du club ;",                                                                             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Équipe_1H)        </font> : nombre de tirs proches du but adverse pendant la 1<sup>re</sup> mi-temps ;",                unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Milieux_1H)       </font> : nombre de tirs des milieux proches du but adverse pendant la 1<sup>re</sup> mi-temps ;",    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Défenseurs_1H)    </font> : nombre de tirs des défenseurs proches du but adverse pendant la 1<sup>re</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Attaquants_1H)    </font> : nombre de tirs des attaquants proches du but adverse pendant la 1<sup>re</sup> mi-temps ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Passes_Équipe_1H)      </font> : nombre de passes intelligentes pendant la 1<sup>re</sup> mi-temps ;",                       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Équipe_1H)      </font> : nombre de fautes en défense pendant la 1<sup>re</sup> mi-temps ;",                          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Milieux_1H)     </font> : nombre de fautes des milieux en défense pendant la 1<sup>re</sup> mi-temps ;",              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Gardiens_1H)    </font> : nombre de fautes des gardiens pendant la 1<sup>re</sup> mi-temps ;",                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Défenseurs_1H)  </font> : nombre de fautes des défenseurs en défense pendant la 1<sup>re</sup> mi-temps ;",           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Attaquants_1H)  </font> : nombre de fautes des attaquants en défense pendant la 1<sup>re</sup> mi-temps ;",           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Arrêts_Gardiens_1H)    </font> : nombre d'arrêts des gardiens pendant la 1<sup>re</sup> mi-temps ;",                         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Équipe_1H)    </font> : nombre d'évènements impliquant l'équipe pendant la 1<sup>re</sup> mi-temps ;",              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Milieux_1H)   </font> : nombre d'évènements impliquant les milieux pendant la 1<sup>re</sup> mi-temps ;",           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Défenseurs_1H)</font> : nombre d'évènements impliquant les défenseurs pendant la 1<sup>re</sup> mi-temps ;",        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Attaquants_1H)</font> : nombre d'évènements impliquant les attaquants pendant la 1<sup>re</sup> mi-temps ;",        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_att_1H_3        </font> : somme des activités des attaquants sur la 1<sup>re</sup> mi-temps des 3 derniers matchs ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_att_2H_5        </font> : somme des activités des attaquants sur la 2<sup>e</sup> mi-temps des 5 derniers matchs ;",  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_att_2H_1        </font> : somme des activités des attaquants sur la 2<sup>e</sup> mi-temps du dernier match ;",       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_att_3H_3        </font> : somme des activités des attaquants sur l'ensemble des 3 derniers matchs ;",                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_def_1H_5        </font> : somme des activités des défenseurs sur la 1<sup>re</sup> mi-temps des 5 derniers matchs ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_def_1H_1        </font> : somme des activités des défenseurs sur la 1<sup>re</sup> mi-temps du dernier match ;",      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_def_2H_3        </font> : somme des activités des défenseurs sur la 2<sup>e</sup> mi-temps des 3 derniers matchs ;",  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_def_3H_5        </font> : somme des activités des défenseurs sur l'ensemble des 5 derniers matchs ;",                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_def_3H_1        </font> : somme des activités des défenseurs sur l'ensemble du dernier match ;",                      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Gar_Tot_1H_3            </font> : somme des arrêts des gardiens sur la 1<sup>re</sup> mi-temps des 3 derniers matchs ;",      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Gar_Tot_2H_5            </font> : somme des arrêts des gardiens sur la 2<sup>e</sup> mi-temps des 5 derniers matchs ;",       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Gar_Tot_2H_1            </font> : somme des arrêts des gardiens sur la 2<sup>e</sup> mi-temps du dernier match ;",            unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Gar_Tot_3H_3            </font> : somme des arrêts des gardiens sur l'ensemble des 3 derniers matchs ;",                      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Fte_Tot_1H_5            </font> : somme des fautes sur la 1<sup>re</sup> mi-temps des 5 derniers matchs ;",                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Fte_Tot_1H_1            </font> : somme des fautes sur la 1<sup>re</sup> mi-temps du dernier match ;",                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Fte_Tot_2H_3            </font> : somme des fautes sur la 2<sup>e</sup> mi-temps des 3 derniers matchs ;",                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Fte_Tot_3H_5            </font> : somme des fautes sur l'ensemble des 5 derniers matchs ;",                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Fte_Tot_3H_1            </font> : somme des fautes sur l'ensemble du dernier match ;",                                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Pass_Tot_1H_3           </font> : somme des passes intelligentes sur la 1<sup>re</sup> mi-temps des 3 derniers matchs ;",     unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Pass_Tot_2H_5           </font> : somme des passes intelligentes sur la 2<sup>e</sup> mi-temps des 5 derniers matchs ;",      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Pass_Tot_2H_1           </font> : somme des passes intelligentes sur la 2<sup>e</sup> mi-temps du dernier match ;",           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Pass_Tot_3H_3           </font> : somme des passes intelligentes sur l'ensemble des 3 derniers matchs ;",                     unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Tirs_Tot_1H_5           </font> : somme des tirs proche du but sur la 1<sup>re</sup> mi-temps des 5 derniers matchs ;",       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Tirs_Tot_1H_1           </font> : somme des tirs proche du but sur la 1<sup>re</sup> mi-temps du dernier match ;",            unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Tirs_Tot_2H_3           </font> : somme des tirs proche du but sur la 2<sup>e</sup> mi-temps des 3 derniers matchs ;",        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Tirs_Tot_3H_5           </font> : somme des tirs proche du but sur l'ensemble des 5 derniers matchs ;",                       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Tirs_Tot_3H_1           </font> : somme des tirs proche du but sur l'ensemble du dernier match ;",                            unsafe_allow_html=True)
        with col2:
            st.markdown("- <font color=" + st.session_state.cols_color + ">Date                    </font> : date du match ;",                                                                           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Match_id                </font> : clef étrangère vers la table « match » ;",                                                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Équipe_2H)        </font> : nombre de tirs proches du but adverse pendant la 2<sup>e</sup> mi-temps ;",                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Milieux_2H)       </font> : nombre de tirs des milieux proches du but adverse pendant la 2<sup>e</sup> mi-temps ;",     unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Défenseurs_2H)    </font> : nombre de tirs des défenseurs proches du but adverse pendant la 2<sup>e</sup> mi-temps ;",  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Attaquants_2H)    </font> : nombre de tirs des attaquants proches du but adverse pendant la 2<sup>e</sup> mi-temps ;",  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Passes_Équipe_2H)      </font> : nombre de passes intelligentes pendant la 2<sup>e</sup> mi-temps ;",                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Équipe_2H)      </font> : nombre de fautes en défense pendant la 2<sup>e</sup> mi-temps ;",                           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Milieux_2H)     </font> : nombre de fautes des milieux en défense pendant la 2<sup>e</sup> mi-temps ;",               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Gardiens_2H)    </font> : nombre de fautes des gardiens pendant la 2<sup>e</sup> mi-temps ;",                         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Défenseurs_2H)  </font> : nombre de fautes des défenseurs en défense pendant la 2<sup>e</sup> mi-temps ;",            unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Attaquants_2H)  </font> : nombre de fautes des attaquants en défense pendant la 2<sup>e</sup> mi-temps ;",            unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Arrêts_Gardiens_2H)    </font> : nombre d'arrêts des gardiens pendant la 2<sup>e</sup> mi-temps ;",                          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Équipe_2H)    </font> : nombre d'évènements impliquant l'équipe pendant la 2<sup>e</sup> mi-temps ;",               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Milieux_2H)   </font> : nombre d'évènements impliquant les milieux pendant la 2<sup>e</sup> mi-temps ;",            unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Défenseurs_2H)</font> : nombre d'évènements impliquant les défenseurs pendant la 2<sup>e</sup> mi-temps ;",         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Attaquants_2H)</font> : nombre d'évènements impliquant les attaquants pendant la 2<sup>e</sup> mi-temps ;",         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_att_1H_5        </font> : somme des activités des attaquants sur la 1<sup>re</sup> mi-temps des 5 derniers matchs ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_att_1H_1        </font> : somme des activités des attaquants sur la 1<sup>re</sup> mi-temps du dernier match ;",      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_att_2H_3        </font> : somme des activités des attaquants sur la 2<sup>e</sup> mi-temps des 3 derniers matchs ;",  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_att_3H_5        </font> : somme des activités des attaquants sur l'ensemble des 5 derniers matchs ;",                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_att_3H_1        </font> : somme des activités des attaquants sur l'ensemble du dernier match ;",                      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_def_1H_3        </font> : somme des activités des défenseurs sur la 1<sup>re</sup> mi-temps des 3 derniers matchs ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_def_2H_5        </font> : somme des activités des défenseurs sur la 2<sup>e</sup> mi-temps des 5 derniers matchs ;",  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_def_2H_1        </font> : somme des activités des défenseurs sur la 2<sup>e</sup> mi-temps du dernier match ;",       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Act_Tot_def_3H_3        </font> : somme des activités des défenseurs sur l'ensemble des 3 derniers matchs ;",                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Gar_Tot_1H_5            </font> : somme des arrêts des gardiens sur la 1<sup>re</sup> mi-temps des 5 derniers matchs ;",      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Gar_Tot_1H_1            </font> : somme des arrêts des gardiens sur la 1<sup>re</sup> mi-temps du dernier match ;",           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Gar_Tot_2H_3            </font> : somme des arrêts des gardiens sur la 2<sup>e</sup> mi-temps des 3 derniers matchs ;",       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Gar_Tot_3H_5            </font> : somme des arrêts des gardiens sur l'ensemble des 5 derniers matchs ;",                      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Gar_Tot_3H_1            </font> : somme des arrêts des gardiens sur l'ensemble du dernier match ;",                           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Fte_Tot_1H_3            </font> : somme des fautes sur la 1<sup>re</sup> mi-temps des 3 derniers matchs ;",                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Fte_Tot_2H_5            </font> : somme des fautes sur la 2<sup>e</sup> mi-temps des 5 derniers matchs ;",                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Fte_Tot_2H_1            </font> : somme des fautes sur la 2<sup>e</sup> mi-temps du dernier match ;",                         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Fte_Tot_3H_3            </font> : somme des fautes sur l'ensemble des 5 derniers matchs ;",                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Pass_Tot_1H_5           </font> : somme des passes intelligentes sur la 1<sup>re</sup> mi-temps des 5 derniers matchs ;",     unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Pass_Tot_1H_1           </font> : somme des passes intelligentes sur la 1<sup>re</sup> mi-temps du dernier match ;",          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Pass_Tot_2H_3           </font> : somme des passes intelligentes sur la 2<sup>e</sup> mi-temps des 3 derniers matchs ;",      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Pass_Tot_3H_5           </font> : somme des passes intelligentes sur l'ensemble des 5 derniers matchs ;",                     unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Pass_Tot_3H_1           </font> : somme des passes intelligentes sur l'ensemble du dernier match ;",                          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Tirs_Tot_1H_3           </font> : somme des tirs proche du but sur la 1<sup>re</sup> mi-temps des 3 derniers matchs ;",       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Tirs_Tot_2H_5           </font> : somme des tirs proche du but sur la 2<sup>e</sup> mi-temps des 5 derniers matchs ;",        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Tirs_Tot_2H_1           </font> : somme des tirs proche du but sur la 2<sup>e</sup> mi-temps du dernier match ;",             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Tirs_Tot_3H_3           </font> : somme des tirs proche du but sur l'ensemble des 3 derniers matchs.",                        unsafe_allow_html=True)
    
    st.code("""
    # Tri sur le champs 'Date'
    match_infos = match_infos.sort_values('Date').reset_index(drop=True)
    
    # Traitement des informations présentes dans « match_infos » + création de variables explicatives
    match_infos = pd.concat([match_infos, Exploit_infos(match_infos)], axis=1)
    """, language='python')
    
    # Affichage du code de la fonction « Exploit_infos »
    if st.checkbox('Afficher le code de la fonction « Exploit_infos() »'):
        st.code("""
        def Exploit_infos(df):
            dico_shot_team_1H = {} # Dictionnaires pour le nombre de tirs au but dans la zone adverse
            dico_shot_team_2H = {}
            dico_pass_team_1H = {} # Dictionnaires pour le nombre de passes intelligentes
            dico_pass_team_2H = {}
            dico_foul_team_1H = {} # Dictionnaires pour le nombre de fautes commises par les équipes
            dico_foul_team_2H = {}
            dico_save_1H      = {} # Dictionnaires pour le nombre d'arrêt des gardiens
            dico_save_2H      = {}
            dico_act_def_1H   = {} # Dictionnaires pour le calcul des activités des attaquants
            dico_act_def_2H   = {}
            dico_act_for_1H   = {} # Dictionnaires pour le calcul des activités des défenseurs
            dico_act_for_2H   = {}
            
            # Les nouvelles variables explicatives
            dico_res = {}
            
            ############################
            ## LES  ACTIVITÉS         ##
            ############################
            
            # Activité des attaquants sur la 1re mi-temps pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Act_Tot_att_1H_5'] = [] # - Activité des attaquants sur la 1re mi-temps sur les 5 derniers matchs
            dico_res['Act_Tot_att_1H_3'] = [] # - Activité des attaquants sur la 1re mi-temps sur les 3 derniers matchs
            dico_res['Act_Tot_att_1H_1'] = [] # - Activité des attaquants sur la 1re mi-temps sur le dernier match
            
            # Activité des attaquants sur la 2e mi-temps pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Act_Tot_att_2H_5'] = [] # - Activité des attaquants sur la 2e mi-temps sur les 5 derniers matchs
            dico_res['Act_Tot_att_2H_3'] = [] # - Activité des attaquants sur la 2e mi-temps sur les 3 derniers matchs
            dico_res['Act_Tot_att_2H_1'] = [] # - Activité des attaquants sur la 2e mi-temps sur le dernier match
            
            # Activité des attaquants sur l'ensemble du match pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Act_Tot_att_3H_5'] = [] # - Activité des attaquants sur l'ensemble du match sur les 5 derniers matchs
            dico_res['Act_Tot_att_3H_3'] = [] # - Activité des attaquants sur l'ensemble du match sur les 3 derniers matchs
            dico_res['Act_Tot_att_3H_1'] = [] # - Activité des attaquants sur l'ensemble du match sur le dernier match
            
            # Activité des défenseurs sur la 1re mi-temps pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Act_Tot_def_1H_5'] = [] # - Activité des défenseurs sur la 1re mi-temps sur les 5 derniers matchs
            dico_res['Act_Tot_def_1H_3'] = [] # - Activité des défenseurs sur la 1re mi-temps sur les 3 derniers matchs
            dico_res['Act_Tot_def_1H_1'] = [] # - Activité des défenseurs sur la 1re mi-temps sur le dernier match
            
            # Activité des défenseurs sur la 2e mi-temps pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Act_Tot_def_2H_5'] = [] # - Activité des défenseurs sur la 2e mi-temps sur les 5 derniers matchs
            dico_res['Act_Tot_def_2H_3'] = [] # - Activité des défenseurs sur la 2e mi-temps sur les 3 derniers matchs
            dico_res['Act_Tot_def_2H_1'] = [] # - Activité des défenseurs sur la 2e mi-temps sur le dernier match
            
            # Activité des défenseurs sur l'ensemble du match pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Act_Tot_def_3H_5'] = [] # - Activité des défenseurs sur l'ensemble du match sur les 5 derniers matchs
            dico_res['Act_Tot_def_3H_3'] = [] # - Activité des défenseurs sur l'ensemble du match sur les 3 derniers matchs
            dico_res['Act_Tot_def_3H_1'] = [] # - Activité des défenseurs sur l'ensemble du match sur le dernier match
            
            ############################
            ## LES ARRÊTS             ##
            ############################
            
            # Activité des défenseurs sur la 1re mi-temps pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Gar_Tot_1H_5'] = [] # - Nombre d'arrêt du gardien sur la 1re mi-temps sur les 5 derniers matchs
            dico_res['Gar_Tot_1H_3'] = [] # - Nombre d'arrêt du gardien sur la 1re mi-temps sur les 3 derniers matchs
            dico_res['Gar_Tot_1H_1'] = [] # - Nombre d'arrêt du gardien sur la 1re mi-temps sur le dernier match
            
            # Activité des défenseurs sur la 2e mi-temps pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Gar_Tot_2H_5'] = [] # - Nombre d'arrêt du gardien sur la 2e mi-temps sur les 5 derniers matchs
            dico_res['Gar_Tot_2H_3'] = [] # - Nombre d'arrêt du gardien sur la 2e mi-temps sur les 3 derniers matchs
            dico_res['Gar_Tot_2H_1'] = [] # - Nombre d'arrêt du gardien sur la 2e mi-temps sur le dernier match
            
            # Activité des défenseurs sur l'ensemble du match pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Gar_Tot_3H_5'] = [] # - Nombre d'arrêt du gardien sur l'ensemble du match sur les 5 derniers matchs
            dico_res['Gar_Tot_3H_3'] = [] # - Nombre d'arrêt du gardien sur l'ensemble du match sur les 3 derniers matchs
            dico_res['Gar_Tot_3H_1'] = [] # - Nombre d'arrêt du gardien sur l'ensemble du match sur le dernier match
            
            ############################
            ## LES FAUTES             ##
            ############################
            
            # Nombre de fautes des joueurs adverses proche de leur but sur la 1re mi-temps pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Fte_Tot_1H_5'] = [] # - Nombre de fautes sur la 1re mi-temps sur les 5 derniers matchs
            dico_res['Fte_Tot_1H_3'] = [] # - Nombre de fautes sur la 1re mi-temps sur les 3 derniers matchs
            dico_res['Fte_Tot_1H_1'] = [] # - Nombre de fautes sur la 1re mi-temps sur le dernier match
            
            # Nombre de fautes des joueurs adverses proche de leur but sur la 2e mi-temps pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Fte_Tot_2H_5'] = [] # - Nombre de fautes sur la 2e mi-temps sur les 5 derniers matchs
            dico_res['Fte_Tot_2H_3'] = [] # - Nombre de fautes sur la 2e mi-temps sur les 3 derniers matchs
            dico_res['Fte_Tot_2H_1'] = [] # - Nombre de fautes sur la 2e mi-temps sur le dernier match
            
            # Nombre de fautes des joueurs adverses proche de leur but sur l'ensemble du match pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Fte_Tot_3H_5'] = [] # - Nombre de fautes sur l'ensemble du match sur les 5 derniers matchs
            dico_res['Fte_Tot_3H_3'] = [] # - Nombre de fautes sur l'ensemble du match sur les 3 derniers matchs
            dico_res['Fte_Tot_3H_1'] = [] # - Nombre de fautes sur l'ensemble du match sur le dernier match
            
            ############################
            ## LES PASSES INTEL       ##
            ############################
            
            # Nombre de passes intelligentes faites par l'équipe sur la 1re mi-temps pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Pass_Tot_1H_5'] = [] # - Nombre de passes intelligentes sur la 1re mi-temps sur les 5 derniers matchs
            dico_res['Pass_Tot_1H_3'] = [] # - Nombre de passes intelligentes sur la 1re mi-temps sur les 3 derniers matchs
            dico_res['Pass_Tot_1H_1'] = [] # - Nombre de passes intelligentes sur la 1re mi-temps sur le dernier match
            
            # Nombre de passes intelligentes faites par l'équipe sur la 2e mi-temps pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Pass_Tot_2H_5'] = [] # - Nombre de passes intelligentes sur la 2e mi-temps sur les 5 derniers matchs
            dico_res['Pass_Tot_2H_3'] = [] # - Nombre de passes intelligentes sur la 2e mi-temps sur les 3 derniers matchs
            dico_res['Pass_Tot_2H_1'] = [] # - Nombre de passes intelligentes sur la 2e mi-temps sur le dernier match
            
            # Nombre de passes intelligentes faites par l'équipe sur l'ensemble du match pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Pass_Tot_3H_5'] = [] # - Nombre de passes intelligentes sur l'ensemble du match sur les 5 derniers matchs
            dico_res['Pass_Tot_3H_3'] = [] # - Nombre de passes intelligentes sur l'ensemble du match sur les 3 derniers matchs
            dico_res['Pass_Tot_3H_1'] = [] # - Nombre de passes intelligentes sur l'ensemble du match sur le dernier match
            
            ############################
            ## LES TIRS PROCHES       ##
            ############################
            
            # Tirs des joueurs depuis la première moitié de terrain adverse sur la 1re mi-temps pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Tirs_Tot_1H_5'] = [] # - Tirs des joueurs sur la 1re mi-temps sur les 5 derniers matchs
            dico_res['Tirs_Tot_1H_3'] = [] # - Tirs des joueurs sur la 1re mi-temps sur les 3 derniers matchs
            dico_res['Tirs_Tot_1H_1'] = [] # - Tirs des joueurs sur la 1re mi-temps sur le dernier match
            
            # Tirs des joueurs depuis la première moitié de terrain adverse sur la 2e mi-temps pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Tirs_Tot_2H_5'] = [] # - Tirs des joueurs sur la 2e mi-temps sur les 5 derniers matchs
            dico_res['Tirs_Tot_2H_3'] = [] # - Tirs des joueurs sur la 2e mi-temps sur les 3 derniers matchs
            dico_res['Tirs_Tot_2H_1'] = [] # - Tirs des joueurs sur la 2e mi-temps sur le dernier match
            
            # Tirs des joueurs depuis la première moitié de terrain adverse sur l'ensemble du match pour les 1, 3 et 5 derniers matchs non nuls
            dico_res['Tirs_Tot_3H_5'] = [] # - Tirs des joueurs sur l'ensemble du match sur les 5 derniers matchs
            dico_res['Tirs_Tot_3H_3'] = [] # - Tirs des joueurs sur l'ensemble du match sur les 3 derniers matchs
            dico_res['Tirs_Tot_3H_1'] = [] # - Tirs des joueurs sur l'ensemble du match sur le dernier match
            
            # Boucle sur l'ensemble des lignes
            for _, row in df.iterrows():
                team = row['Équipe_id']
                
                # Récupération des informations nécessaire aux calculs
                for ptr_dico, col_name in zip([dico_shot_team_1H, dico_pass_team_1H, dico_foul_team_1H, dico_save_1H, dico_act_def_1H, dico_act_for_1H,
                                               dico_shot_team_2H, dico_pass_team_2H, dico_foul_team_2H, dico_save_2H, dico_act_def_2H, dico_act_for_2H],
                                              ['(Tirs_Équipe_1H)', '(Passes_Équipe_1H)', '(Fautes_Équipe_1H)', '(Arrêts_Gardiens_1H)', '(Activité_Défenseurs_1H)', '(Activité_Attaquants_1H)',
                                               '(Tirs_Équipe_2H)', '(Passes_Équipe_2H)', '(Fautes_Équipe_2H)', '(Arrêts_Gardiens_2H)', '(Activité_Défenseurs_2H)', '(Activité_Attaquants_2H)']):
                    if team not in ptr_dico.keys():
                        ptr_dico[team] = [row[col_name]]
                    else:
                        ptr_dico[team].append(row[col_name])
                
                # Calcul des nouvelles variables explicatives
                
                ############################
                ## L'ACTIVITÉ             ##
                ############################
                
                dico_res['Act_Tot_att_1H_5'].append(np.nan if len(dico_act_for_1H[team]) <= 5 else sum(dico_act_for_1H[team][-6:-1]))
                dico_res['Act_Tot_att_1H_3'].append(np.nan if len(dico_act_for_1H[team]) <= 3 else sum(dico_act_for_1H[team][-4:-1]))
                dico_res['Act_Tot_att_1H_1'].append(np.nan if len(dico_act_for_1H[team]) <= 1 else     dico_act_for_1H[team][-2])
                dico_res['Act_Tot_att_2H_5'].append(np.nan if len(dico_act_for_2H[team]) <= 5 else sum(dico_act_for_2H[team][-6:-1]))
                dico_res['Act_Tot_att_2H_3'].append(np.nan if len(dico_act_for_2H[team]) <= 3 else sum(dico_act_for_2H[team][-4:-1]))
                dico_res['Act_Tot_att_2H_1'].append(np.nan if len(dico_act_for_2H[team]) <= 1 else     dico_act_for_2H[team][-2])
                
                dico_res['Act_Tot_def_1H_5'].append(np.nan if len(dico_act_def_1H[team]) <= 5 else sum(dico_act_def_1H[team][-6:-1]))
                dico_res['Act_Tot_def_1H_3'].append(np.nan if len(dico_act_def_1H[team]) <= 3 else sum(dico_act_def_1H[team][-4:-1]))
                dico_res['Act_Tot_def_1H_1'].append(np.nan if len(dico_act_def_1H[team]) <= 1 else     dico_act_def_1H[team][-2])
                dico_res['Act_Tot_def_2H_5'].append(np.nan if len(dico_act_def_2H[team]) <= 5 else sum(dico_act_def_2H[team][-6:-1]))
                dico_res['Act_Tot_def_2H_3'].append(np.nan if len(dico_act_def_2H[team]) <= 3 else sum(dico_act_def_2H[team][-4:-1]))
                dico_res['Act_Tot_def_2H_1'].append(np.nan if len(dico_act_def_2H[team]) <= 1 else     dico_act_def_2H[team][-2])
                
                dico_res['Act_Tot_att_3H_5'].append(np.nan if dico_res['Act_Tot_att_1H_5'][-1] == np.nan else dico_res['Act_Tot_att_1H_5'][-1] + dico_res['Act_Tot_att_2H_5'][-1])
                dico_res['Act_Tot_att_3H_3'].append(np.nan if dico_res['Act_Tot_att_1H_3'][-1] == np.nan else dico_res['Act_Tot_att_1H_3'][-1] + dico_res['Act_Tot_att_2H_3'][-1])
                dico_res['Act_Tot_att_3H_1'].append(np.nan if dico_res['Act_Tot_att_1H_1'][-1] == np.nan else dico_res['Act_Tot_att_1H_1'][-1] + dico_res['Act_Tot_att_2H_1'][-1])
                
                dico_res['Act_Tot_def_3H_5'].append(np.nan if dico_res['Act_Tot_def_1H_5'][-1] == np.nan else dico_res['Act_Tot_def_1H_5'][-1] + dico_res['Act_Tot_def_2H_5'][-1])
                dico_res['Act_Tot_def_3H_3'].append(np.nan if dico_res['Act_Tot_def_1H_3'][-1] == np.nan else dico_res['Act_Tot_def_1H_3'][-1] + dico_res['Act_Tot_def_2H_3'][-1])
                dico_res['Act_Tot_def_3H_1'].append(np.nan if dico_res['Act_Tot_def_1H_1'][-1] == np.nan else dico_res['Act_Tot_def_1H_1'][-1] + dico_res['Act_Tot_def_2H_1'][-1])
                
                ############################
                ## LES ARRÊTS             ##
                ############################
                
                dico_res['Gar_Tot_1H_5'].append(np.nan if len(dico_save_1H[team]) <= 5 else sum(dico_save_1H[team][-6:-1]))
                dico_res['Gar_Tot_1H_3'].append(np.nan if len(dico_save_1H[team]) <= 3 else sum(dico_save_1H[team][-4:-1]))
                dico_res['Gar_Tot_1H_1'].append(np.nan if len(dico_save_1H[team]) <= 1 else     dico_save_1H[team][-2])
                dico_res['Gar_Tot_2H_5'].append(np.nan if len(dico_save_2H[team]) <= 5 else sum(dico_save_2H[team][-6:-1]))
                dico_res['Gar_Tot_2H_3'].append(np.nan if len(dico_save_2H[team]) <= 3 else sum(dico_save_2H[team][-4:-1]))
                dico_res['Gar_Tot_2H_1'].append(np.nan if len(dico_save_2H[team]) <= 1 else     dico_save_2H[team][-2])
                
                dico_res['Gar_Tot_3H_5'].append(np.nan if dico_res['Gar_Tot_1H_5'][-1] == np.nan else dico_res['Gar_Tot_1H_5'][-1] + dico_res['Gar_Tot_2H_5'][-1])
                dico_res['Gar_Tot_3H_3'].append(np.nan if dico_res['Gar_Tot_1H_3'][-1] == np.nan else dico_res['Gar_Tot_1H_3'][-1] + dico_res['Gar_Tot_2H_3'][-1])
                dico_res['Gar_Tot_3H_1'].append(np.nan if dico_res['Gar_Tot_1H_1'][-1] == np.nan else dico_res['Gar_Tot_1H_1'][-1] + dico_res['Gar_Tot_2H_1'][-1])
                
                ############################
                ## LES FAUTES             ##
                ############################
                
                dico_res['Fte_Tot_1H_5'].append(np.nan if len(dico_foul_team_1H[team]) <= 5 else sum(dico_foul_team_1H[team][-6:-1]))
                dico_res['Fte_Tot_1H_3'].append(np.nan if len(dico_foul_team_1H[team]) <= 3 else sum(dico_foul_team_1H[team][-4:-1]))
                dico_res['Fte_Tot_1H_1'].append(np.nan if len(dico_foul_team_1H[team]) <= 1 else     dico_foul_team_1H[team][-2])
                dico_res['Fte_Tot_2H_5'].append(np.nan if len(dico_foul_team_2H[team]) <= 5 else sum(dico_foul_team_2H[team][-6:-1]))
                dico_res['Fte_Tot_2H_3'].append(np.nan if len(dico_foul_team_2H[team]) <= 3 else sum(dico_foul_team_2H[team][-4:-1]))
                dico_res['Fte_Tot_2H_1'].append(np.nan if len(dico_foul_team_2H[team]) <= 1 else     dico_foul_team_2H[team][-2])
                
                dico_res['Fte_Tot_3H_5'].append(np.nan if dico_res['Fte_Tot_1H_5'][-1] == np.nan else dico_res['Fte_Tot_1H_5'][-1] + dico_res['Fte_Tot_2H_5'][-1])
                dico_res['Fte_Tot_3H_3'].append(np.nan if dico_res['Fte_Tot_1H_3'][-1] == np.nan else dico_res['Fte_Tot_1H_3'][-1] + dico_res['Fte_Tot_2H_3'][-1])
                dico_res['Fte_Tot_3H_1'].append(np.nan if dico_res['Fte_Tot_1H_1'][-1] == np.nan else dico_res['Fte_Tot_1H_1'][-1] + dico_res['Fte_Tot_2H_1'][-1])
                
                ############################
                ## LES PASSES INTEL       ##
                ############################
                
                dico_res['Pass_Tot_1H_5'].append(np.nan if len(dico_pass_team_1H[team]) <= 5 else sum(dico_pass_team_1H[team][-6:-1]))
                dico_res['Pass_Tot_1H_3'].append(np.nan if len(dico_pass_team_1H[team]) <= 3 else sum(dico_pass_team_1H[team][-4:-1]))
                dico_res['Pass_Tot_1H_1'].append(np.nan if len(dico_pass_team_1H[team]) <= 1 else     dico_pass_team_1H[team][-2])
                dico_res['Pass_Tot_2H_5'].append(np.nan if len(dico_pass_team_2H[team]) <= 5 else sum(dico_pass_team_2H[team][-6:-1]))
                dico_res['Pass_Tot_2H_3'].append(np.nan if len(dico_pass_team_2H[team]) <= 3 else sum(dico_pass_team_2H[team][-4:-1]))
                dico_res['Pass_Tot_2H_1'].append(np.nan if len(dico_pass_team_2H[team]) <= 1 else     dico_pass_team_2H[team][-2])
                
                dico_res['Pass_Tot_3H_5'].append(np.nan if dico_res['Pass_Tot_1H_5'][-1] == np.nan else dico_res['Pass_Tot_1H_5'][-1] + dico_res['Pass_Tot_2H_5'][-1])
                dico_res['Pass_Tot_3H_3'].append(np.nan if dico_res['Pass_Tot_1H_3'][-1] == np.nan else dico_res['Pass_Tot_1H_3'][-1] + dico_res['Pass_Tot_2H_3'][-1])
                dico_res['Pass_Tot_3H_1'].append(np.nan if dico_res['Pass_Tot_1H_1'][-1] == np.nan else dico_res['Pass_Tot_1H_1'][-1] + dico_res['Pass_Tot_2H_1'][-1])
                
                ############################
                ## LES TIRS PROCHES       ##
                ############################
                
                dico_res['Tirs_Tot_1H_5'].append(np.nan if len(dico_shot_team_1H[team]) <= 5 else sum(dico_shot_team_1H[team][-6:-1]))
                dico_res['Tirs_Tot_1H_3'].append(np.nan if len(dico_shot_team_1H[team]) <= 3 else sum(dico_shot_team_1H[team][-4:-1]))
                dico_res['Tirs_Tot_1H_1'].append(np.nan if len(dico_shot_team_1H[team]) <= 1 else     dico_shot_team_1H[team][-2])
                dico_res['Tirs_Tot_2H_5'].append(np.nan if len(dico_shot_team_2H[team]) <= 5 else sum(dico_shot_team_2H[team][-6:-1]))
                dico_res['Tirs_Tot_2H_3'].append(np.nan if len(dico_shot_team_2H[team]) <= 3 else sum(dico_shot_team_2H[team][-4:-1]))
                dico_res['Tirs_Tot_2H_1'].append(np.nan if len(dico_shot_team_2H[team]) <= 1 else     dico_shot_team_2H[team][-2])
                
                dico_res['Tirs_Tot_3H_5'].append(np.nan if dico_res['Tirs_Tot_1H_5'][-1] == np.nan else dico_res['Tirs_Tot_1H_5'][-1] + dico_res['Tirs_Tot_2H_5'][-1])
                dico_res['Tirs_Tot_3H_3'].append(np.nan if dico_res['Tirs_Tot_1H_3'][-1] == np.nan else dico_res['Tirs_Tot_1H_3'][-1] + dico_res['Tirs_Tot_2H_3'][-1])
                dico_res['Tirs_Tot_3H_1'].append(np.nan if dico_res['Tirs_Tot_1H_1'][-1] == np.nan else dico_res['Tirs_Tot_1H_1'][-1] + dico_res['Tirs_Tot_2H_1'][-1])
                
            return pd.DataFrame(data=dico_res)
        """, language='python')
    
    # Affichage d'un échantillon de la table « match_infos » augmentée
    if st.checkbox('Afficher un extrait de la table « match_infos » augmentée (focus sur le club de Toulouse)'):
        # Chargement des données
        df_match_infos_aug = pd.read_csv(path_data + 'match_infos_aug_TOULOUSE.csv', index_col=0)
        
        df_match_infos_aug
    
    #######################################
    ##  MATCH_ALL                        ##
    #######################################
    
    st.subheader('Création de la table « match_all »')
    
    st.write("""La table « match_all » est la fusion des tables « match_results » et « match_infos ».
                Le détail de ces colonnes et le code utilisé pour les obtenir sont diponibles ci-dessous.""")
    
    # Schéma pour la création de la table « match_all »
    st.image(path_imag + 'Table 06 - match_all.png',
             caption='Fusion des tables « match_results » et « match_infos » pour donner la table « match_all ».')
    
    # Présentation détaillée des colonnes de la tables « match_all »
    if st.checkbox('Afficher la présentation détaillée des colonnes de la table « match_all »'):
        st.markdown('**Colonnes de la table « match_all » :**')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- <font color=" + st.session_state.cols_color + ">Index                   </font> : clef primaire ;",                                                                                                unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Club                    </font> : nom du club ;",                                                                                                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Période_jour            </font> : chiffre indiquant la période de la journée où le match fut joué ;",                                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Match_id                </font> : clef étrangère vers la table « match » ;",                                                                       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Série_vic           </font> : nombre de victoire sur la saison ;",                                                                             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_contre     </font> : nombre de buts encaissés sur la saison ;",                                                                       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Série_vic_5         </font> : nombre de victoire sur les 5 derniers matchs non nuls ;",                                                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_contre_5   </font> : nombre de buts encaissés sur les 5 derniers match non nuls ;",                                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Série_vic_3         </font> : nombre de victoire sur les 3 derniers matchs non nuls ;",                                                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_contre_3   </font> : nombre de buts encaissés sur les 3 derniers match non nuls ;",                                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Série_vic_1         </font> : nombre de victoire sur le dernier match non nul ;",                                                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_contre_1   </font> : nombre de buts encaissés sur le dernier match non nul ;",                                                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Nb_jours            </font> : nombre de jours depuis le dernier match non nul ;",                                                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_pour       </font> : nombre de buts marqués sur la saison pour le club adverse ;",                                                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_diff       </font> : différence entre les buts marqués et ceux encaissés pour le club adverse ;",                                     unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_pour_5     </font> : nombre de buts marqués sur les 5 derniers matchs non nuls pour le club adverse ;",                               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_diff_5     </font> : différence entre les buts marqué et ceux encaissés sur les 5 derniers matchs non nuls pour le club adverse ;",   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_pour_3     </font> : nombre de buts marqués sur les 3 derniers matchs non nuls pour le club adverse ;",                               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_diff_3     </font> : différence entre les buts marqués et ceux encaissés sur les 3 derniers matchs non nuls pour le club adverse ;",  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_pour_1     </font> : nombre de buts marqués sur le dernier match non nul pour le club adverse ;",                                     unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_diff_1     </font> : différence entre les buts marqués et ceux encaissés sur le dernier match non nul pour le club adverse ;",        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">[Résultat]              </font> : indique la victoire ou la défaite (variable cible) ;",                                                           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">[ADV_Buts]              </font> : nombre de buts encaissés à la fin du match (variable cible) ;",                                                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Équipe_1H)        </font> : nombre de tirs proches du but adverse pendant la 1<sup>re</sup> mi-temps ;",                                     unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Milieux_1H)       </font> : nombre de tirs des milieux proches du but adverse pendant la 1<sup>re</sup> mi-temps ;",                         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Défenseurs_1H)    </font> : nombre de tirs des défenseurs proches du but adverse pendant la 1<sup>re</sup> mi-temps ;",                      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Attaquants_1H)    </font> : nombre de tirs des attaquants proches du but adverse pendant la 1<sup>re</sup> mi-temps ;",                      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Passes_Équipe_1H)      </font> : nombre de passes intelligentes pendant la 1<sup>re</sup> mi-temps ;",                                            unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Équipe_1H)      </font> : nombre de fautes en défense pendant la 1<sup>re</sup> mi-temps ;",                                               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Milieux_1H)     </font> : nombre de fautes des milieux en défense pendant la 1<sup>re</sup> mi-temps ;",                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Gardiens_1H)    </font> : nombre de fautes des gardiens pendant la 1<sup>re</sup> mi-temps ;",                                             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Défenseurs_1H)  </font> : nombre de fautes des défenseurs en défense pendant la 1<sup>re</sup> mi-temps ;",                                unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Attaquants_1H)  </font> : nombre de fautes des attaquants en défense pendant la 1<sup>re</sup> mi-temps ;",                                unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Arrêts_Gardiens_1H)    </font> : nombre d'arrêts des gardiens pendant la 1<sup>re</sup> mi-temps ;",                                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Équipe_1H)    </font> : nombre d'évènements impliquant l'équipe pendant la 1<sup>re</sup> mi-temps ;",                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Milieux_1H)   </font> : nombre d'évènements impliquant les milieux pendant la 1<sup>re</sup> mi-temps ;",                                unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Défenseurs_1H)</font> : nombre d'évènements impliquant les défenseurs pendant la 1<sup>re</sup> mi-temps ;",                             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Attaquants_1H)</font> : nombre d'évènements impliquant les attaquants pendant la 1<sup>re</sup> mi-temps ;",                             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_att_1H_3    </font> : somme des activités des attaquants sur la 1<sup>re</sup> mi-temps des 3 derniers matchs pour le club ;",         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_att_2H_5    </font> : somme des activités des attaquants sur la 2<sup>e</sup> mi-temps des 5 derniers matchs pour le club ;",          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_att_2H_1    </font> : somme des activités des attaquants sur la 2<sup>e</sup> mi-temps du dernier match pour le club ;",               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_att_3H_3    </font> : somme des activités des attaquants sur l'ensemble des 3 derniers matchs pour le club ;",                         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_def_1H_5    </font> : somme des activités des défenseurs sur la 1<sup>re</sup> mi-temps des 5 derniers matchs pour le club ;",         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_def_1H_1    </font> : somme des activités des défenseurs sur la 1<sup>re</sup> mi-temps du dernier match pour le club ;",              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_def_2H_3    </font> : somme des activités des défenseurs sur la 2<sup>e</sup> mi-temps des 3 derniers matchs pour le club ;",          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_def_3H_5    </font> : somme des activités des défenseurs sur l'ensemble des 5 derniers matchs pour le club ;",                         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_def_3H_1    </font> : somme des activités des défenseurs sur l'ensemble du dernier match pour le club ;",                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Gar_Tot_1H_3        </font> : somme des arrêts des gardiens sur la 1<sup>re</sup> mi-temps des 3 derniers matchs pour le club ;",              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Gar_Tot_2H_5        </font> : somme des arrêts des gardiens sur la 2<sup>e</sup> mi-temps des 5 derniers matchs pour le club ;",               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Gar_Tot_2H_1        </font> : somme des arrêts des gardiens sur la 2<sup>e</sup> mi-temps du dernier match pour le club ;",                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Gar_Tot_3H_3        </font> : somme des arrêts des gardiens sur l'ensemble des 3 derniers matchs pour le club ;",                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Fte_Tot_1H_5        </font> : somme des fautes sur la 1<sup>re</sup> mi-temps des 5 derniers matchs pour le club ;",                           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Fte_Tot_1H_1        </font> : somme des fautes sur la 1<sup>re</sup> mi-temps du dernier match pour le club ;",                                unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Fte_Tot_2H_3        </font> : somme des fautes sur la 2<sup>e</sup> mi-temps des 3 derniers matchs pour le club ;",                            unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Fte_Tot_3H_5        </font> : somme des fautes sur l'ensemble des 5 derniers matchs pour le club ;",                                           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Fte_Tot_3H_1        </font> : somme des fautes sur l'ensemble du dernier match pour le club ;",                                                unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Pass_Tot_1H_3       </font> : somme des passes intelligentes sur la 1<sup>re</sup> mi-temps des 3 derniers matchs pour le club ;",             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Pass_Tot_2H_5       </font> : somme des passes intelligentes sur la 2<sup>e</sup> mi-temps des 5 derniers matchs pour le club ;",              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Pass_Tot_2H_1       </font> : somme des passes intelligentes sur la 2<sup>e</sup> mi-temps du dernier match pour le club ;",                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Pass_Tot_3H_3       </font> : somme des passes intelligentes sur l'ensemble des 3 derniers matchs pour le club ;",                             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tirs_Tot_1H_5       </font> : somme des tirs proche du but sur la 1<sup>re</sup> mi-temps des 5 derniers matchs pour le club ;",               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tirs_Tot_1H_1       </font> : somme des tirs proche du but sur la 1<sup>re</sup> mi-temps du dernier match pour le club ;",                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tirs_Tot_2H_3       </font> : somme des tirs proche du but sur la 2<sup>e</sup> mi-temps des 3 derniers matchs pour le club ;",                unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tirs_Tot_3H_5       </font> : somme des tirs proche du but sur l'ensemble des 5 derniers matchs pour le club ;",                               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tirs_Tot_3H_1       </font> : somme des tirs proche du but sur l'ensemble du dernier match pour le club ;",                                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_att_1H_3    </font> : somme des activités des attaquants sur la 1<sup>re</sup> mi-temps des 3 derniers matchs pour le club adverse ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_att_2H_5    </font> : somme des activités des attaquants sur la 2<sup>e</sup> mi-temps des 5 derniers matchs pour le club adverse ;",  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_att_2H_1    </font> : somme des activités des attaquants sur la 2<sup>e</sup> mi-temps du dernier match pour le club adverse ;",       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_att_3H_3    </font> : somme des activités des attaquants sur l'ensemble des 3 derniers matchs pour le club adverse ;",                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_def_1H_5    </font> : somme des activités des défenseurs sur la 1<sup>re</sup> mi-temps des 5 derniers matchs pour le club adverse ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_def_1H_1    </font> : somme des activités des défenseurs sur la 1<sup>re</sup> mi-temps du dernier match pour le club adverse ;",      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_def_2H_3    </font> : somme des activités des défenseurs sur la 2<sup>e</sup> mi-temps des 3 derniers matchs pour le club adverse ;",  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_def_3H_5    </font> : somme des activités des défenseurs sur l'ensemble des 5 derniers matchs pour le club adverse ;",                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_def_3H_1    </font> : somme des activités des défenseurs sur l'ensemble du dernier match pour le club adverse ;",                      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Gar_Tot_1H_3        </font> : somme des arrêts des gardiens sur la 1<sup>re</sup> mi-temps des 3 derniers matchs pour le club adverse ;",      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Gar_Tot_2H_5        </font> : somme des arrêts des gardiens sur la 2<sup>e</sup> mi-temps des 5 derniers matchs pour le club adverse ;",       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Gar_Tot_2H_1        </font> : somme des arrêts des gardiens sur la 2<sup>e</sup> mi-temps du dernier match pour le club adverse ;",            unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Gar_Tot_3H_3        </font> : somme des arrêts des gardiens sur l'ensemble des 3 derniers matchs pour le club adverse ;",                      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Fte_Tot_1H_5        </font> : somme des fautes sur la 1<sup>re</sup> mi-temps des 5 derniers matchs pour le club adverse ;",                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Fte_Tot_1H_1        </font> : somme des fautes sur la 1<sup>re</sup> mi-temps du dernier match pour le club adverse ;",                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Fte_Tot_2H_3        </font> : somme des fautes sur la 2<sup>e</sup> mi-temps des 3 derniers matchs pour le club adverse ;",                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Fte_Tot_3H_5        </font> : somme des fautes sur l'ensemble des 5 derniers matchs pour le club adverse ;",                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Fte_Tot_3H_1        </font> : somme des fautes sur l'ensemble du dernier match pour le club adverse ;",                                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Pass_Tot_1H_3       </font> : somme des passes intelligentes sur la 1<sup>re</sup> mi-temps des 3 derniers matchs pour le club adverse ;",     unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Pass_Tot_2H_5       </font> : somme des passes intelligentes sur la 2<sup>e</sup> mi-temps des 5 derniers matchs pour le club adverse ;",      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Pass_Tot_2H_1       </font> : somme des passes intelligentes sur la 2<sup>e</sup> mi-temps du dernier match pour le club adverse ;",           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Pass_Tot_3H_3       </font> : somme des passes intelligentes sur l'ensemble des 3 derniers matchs pour le club adverse ;",                     unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tirs_Tot_1H_5       </font> : somme des tirs proche du but sur la 1<sup>re</sup> mi-temps des 5 derniers matchs pour le club adverse ;",       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tirs_Tot_1H_1       </font> : somme des tirs proche du but sur la 1<sup>re</sup> mi-temps du dernier match pour le club adverse ;",            unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tirs_Tot_2H_3       </font> : somme des tirs proche du but sur la 2<sup>e</sup> mi-temps des 3 derniers matchs pour le club adverse ;",        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tirs_Tot_3H_5       </font> : somme des tirs proche du but sur l'ensemble des 5 derniers matchs pour le club adverse ;",                       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tirs_Tot_3H_1       </font> : somme des tirs proche du but sur l'ensemble du dernier match pour le club adverse.",                             unsafe_allow_html=True)
        with col2:
            st.markdown("- <font color=" + st.session_state.cols_color + ">Date                    </font> : date du match ;",                                                                                                unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Club_ADV                </font> : nom du club adverse ;",                                                                                          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Arbitre_id              </font> : clef étrangère vers la table « referee » ;",                                                                     unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Dom_Ext                 </font> : indique si le club joue à domicile ou à l'extérieur ;",                                                          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_pour       </font> : nombre de buts marqués sur la saison ;",                                                                         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_diff       </font> : différence entre les buts marqués et ceux encaissés ;",                                                          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_pour_5     </font> : nombre de buts marqués sur les 5 derniers matchs non nuls ;",                                                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_diff_5     </font> : différence entre les buts marqué et ceux encaissés sur les 5 derniers matchs non nuls ;",                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_pour_3     </font> : nombre de buts marqués sur les 3 derniers matchs non nuls ;",                                                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_diff_3     </font> : différence entre les buts marqués et ceux encaissés sur les 3 derniers matchs non nuls ;",                       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_pour_1     </font> : nombre de buts marqués sur le dernier match non nul ;",                                                          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tot_buts_diff_1     </font> : différence entre les buts marqués et ceux encaissés sur le dernier match non nul ;",                             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Série_vic           </font> : nombre de victoire sur la saison pour le club adverse ;",                                                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_contre     </font> : nombre de buts encaissés sur la saison pour le club adverse ;",                                                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Série_vic_5         </font> : nombre de victoire sur les 5 derniers matchs non nuls pour le club adverse ;",                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_contre_5   </font> : nombre de buts encaissés sur les 5 derniers match non nuls pour le club adverse ;",                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Série_vic_3         </font> : nombre de victoire sur les 3 derniers matchs non nuls pour le club adverse ;",                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_contre_3   </font> : nombre de buts encaissés sur les 3 derniers match non nuls pour le club adverse ;",                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Série_vic_1         </font> : nombre de victoire sur le dernier match non nul pour le club adverse ;",                                         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tot_buts_contre_1   </font> : nombre de buts encaissés sur le dernier match non nul pour le club adverse ;",                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Nb_jours            </font> : nombre de jours depuis le dernier match non nul pour le club adverse ;",                                         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">[CLB_Buts]              </font> : nombre de buts marqués à la fin du match (variable cible) ;",                                                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Équipe_2H)        </font> : nombre de tirs proches du but adverse pendant la 2<sup>e</sup> mi-temps ;",                                      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Milieux_2H)       </font> : nombre de tirs des milieux proches du but adverse pendant la 2<sup>e</sup> mi-temps ;",                          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Défenseurs_2H)    </font> : nombre de tirs des défenseurs proches du but adverse pendant la 2<sup>e</sup> mi-temps ;",                       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Tirs_Attaquants_2H)    </font> : nombre de tirs des attaquants proches du but adverse pendant la 2<sup>e</sup> mi-temps ;",                       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Passes_Équipe_2H)      </font> : nombre de passes intelligentes pendant la 2<sup>e</sup> mi-temps ;",                                             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Équipe_2H)      </font> : nombre de fautes en défense pendant la 2<sup>e</sup> mi-temps ;",                                                unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Milieux_2H)     </font> : nombre de fautes des milieux en défense pendant la 2<sup>e</sup> mi-temps ;",                                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Gardiens_2H)    </font> : nombre de fautes des gardiens pendant la 2<sup>e</sup> mi-temps ;",                                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Défenseurs_2H)  </font> : nombre de fautes des défenseurs en défense pendant la 2<sup>e</sup> mi-temps ;",                                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Fautes_Attaquants_2H)  </font> : nombre de fautes des attaquants en défense pendant la 2<sup>e</sup> mi-temps ;",                                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Arrêts_Gardiens_2H)    </font> : nombre d'arrêts des gardiens pendant la 2<sup>e</sup> mi-temps ;",                                               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Équipe_2H)    </font> : nombre d'évènements impliquant l'équipe pendant la 2<sup>e</sup> mi-temps ;",                                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Milieux_2H)   </font> : nombre d'évènements impliquant les milieux pendant la 2<sup>e</sup> mi-temps ;",                                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Défenseurs_2H)</font> : nombre d'évènements impliquant les défenseurs pendant la 2<sup>e</sup> mi-temps ;",                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">(Activité_Attaquants_2H)</font> : nombre d'évènements impliquant les attaquants pendant la 2<sup>e</sup> mi-temps ;",                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_att_1H_5    </font> : somme des activités des attaquants sur la 1<sup>re</sup> mi-temps des 5 derniers matchs pour le club ;",         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_att_1H_1    </font> : somme des activités des attaquants sur la 1<sup>re</sup> mi-temps du dernier match pour le club ;",              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_att_2H_3    </font> : somme des activités des attaquants sur la 2<sup>e</sup> mi-temps des 3 derniers matchs pour le club ;",          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_att_3H_5    </font> : somme des activités des attaquants sur l'ensemble des 5 derniers matchs pour le club ;",                         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_att_3H_1    </font> : somme des activités des attaquants sur l'ensemble du dernier match pour le club ;",                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_def_1H_3    </font> : somme des activités des défenseurs sur la 1<sup>re</sup> mi-temps des 3 derniers matchs pour le club ;",         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_def_2H_5    </font> : somme des activités des défenseurs sur la 2<sup>e</sup> mi-temps des 5 derniers matchs pour le club ;",          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_def_2H_1    </font> : somme des activités des défenseurs sur la 2<sup>e</sup> mi-temps du dernier match pour le club ;",               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Act_Tot_def_3H_3    </font> : somme des activités des défenseurs sur l'ensemble des 3 derniers matchs pour le club ;",                         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Gar_Tot_1H_5        </font> : somme des arrêts des gardiens sur la 1<sup>re</sup> mi-temps des 5 derniers matchs pour le club ;",              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Gar_Tot_1H_1        </font> : somme des arrêts des gardiens sur la 1<sup>re</sup> mi-temps du dernier match pour le club ;",                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Gar_Tot_2H_3        </font> : somme des arrêts des gardiens sur la 2<sup>e</sup> mi-temps des 3 derniers matchs pour le club ;",               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Gar_Tot_3H_5        </font> : somme des arrêts des gardiens sur l'ensemble des 5 derniers matchs pour le club ;",                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Gar_Tot_3H_1        </font> : somme des arrêts des gardiens sur l'ensemble du dernier match pour le club ;",                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Fte_Tot_1H_3        </font> : somme des fautes sur la 1<sup>re</sup> mi-temps des 3 derniers matchs pour le club ;",                           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Fte_Tot_2H_5        </font> : somme des fautes sur la 2<sup>e</sup> mi-temps des 5 derniers matchs pour le club ;",                            unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Fte_Tot_2H_1        </font> : somme des fautes sur la 2<sup>e</sup> mi-temps du dernier match pour le club ;",                                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Fte_Tot_3H_3        </font> : somme des fautes sur l'ensemble des 5 derniers matchs pour le club ;",                                           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Pass_Tot_1H_5       </font> : somme des passes intelligentes sur la 1<sup>re</sup> mi-temps des 5 derniers matchs pour le club ;",             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Pass_Tot_1H_1       </font> : somme des passes intelligentes sur la 1<sup>re</sup> mi-temps du dernier match pour le club ;",                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Pass_Tot_2H_3       </font> : somme des passes intelligentes sur la 2<sup>e</sup> mi-temps des 3 derniers matchs pour le club ;",              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Pass_Tot_3H_5       </font> : somme des passes intelligentes sur l'ensemble des 5 derniers matchs pour le club ;",                             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Pass_Tot_3H_1       </font> : somme des passes intelligentes sur l'ensemble du dernier match pour le club ;",                                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tirs_Tot_1H_3       </font> : somme des tirs proche du but sur la 1<sup>re</sup> mi-temps des 3 derniers matchs pour le club ;",               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tirs_Tot_2H_5       </font> : somme des tirs proche du but sur la 2<sup>e</sup> mi-temps des 5 derniers matchs pour le club ;",                unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tirs_Tot_2H_1       </font> : somme des tirs proche du but sur la 2<sup>e</sup> mi-temps du dernier match pour le club ;",                     unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Tirs_Tot_3H_3       </font> : somme des tirs proche du but sur l'ensemble des 3 derniers matchs pour le club ;",                               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_att_1H_5    </font> : somme des activités des attaquants sur la 1<sup>re</sup> mi-temps des 5 derniers matchs pour le club adverse ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_att_1H_1    </font> : somme des activités des attaquants sur la 1<sup>re</sup> mi-temps du dernier match pour le club adverse ;",      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_att_2H_3    </font> : somme des activités des attaquants sur la 2<sup>e</sup> mi-temps des 3 derniers matchs pour le club adverse ;",  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_att_3H_5    </font> : somme des activités des attaquants sur l'ensemble des 5 derniers matchs pour le club adverse ;",                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_att_3H_1    </font> : somme des activités des attaquants sur l'ensemble du dernier match pour le club adverse ;",                      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_def_1H_3    </font> : somme des activités des défenseurs sur la 1<sup>re</sup> mi-temps des 3 derniers matchs pour le club adverse ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_def_2H_5    </font> : somme des activités des défenseurs sur la 2<sup>e</sup> mi-temps des 5 derniers matchs pour le club adverse ;",  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_def_2H_1    </font> : somme des activités des défenseurs sur la 2<sup>e</sup> mi-temps du dernier match pour le club adverse ;",       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Act_Tot_def_3H_3    </font> : somme des activités des défenseurs sur l'ensemble des 3 derniers matchs pour le club adverse ;",                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Gar_Tot_1H_5        </font> : somme des arrêts des gardiens sur la 1<sup>re</sup> mi-temps des 5 derniers matchs pour le club adverse ;",      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Gar_Tot_1H_1        </font> : somme des arrêts des gardiens sur la 1<sup>re</sup> mi-temps du dernier match pour le club adverse ;",           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Gar_Tot_2H_3        </font> : somme des arrêts des gardiens sur la 2<sup>e</sup> mi-temps des 3 derniers matchs pour le club adverse ;",       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Gar_Tot_3H_5        </font> : somme des arrêts des gardiens sur l'ensemble des 5 derniers matchs pour le club adverse ;",                      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Gar_Tot_3H_1        </font> : somme des arrêts des gardiens sur l'ensemble du dernier match pour le club adverse ;",                           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Fte_Tot_1H_3        </font> : somme des arrêts des gardiens sur l'ensemble du dernier match pour le club adverse ;",                           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Fte_Tot_2H_5        </font> : somme des fautes sur la 2<sup>e</sup> mi-temps des 5 derniers matchs pour le club adverse ;",                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Fte_Tot_2H_1        </font> : somme des fautes sur la 2<sup>e</sup> mi-temps du dernier match pour le club adverse ;",                         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Fte_Tot_3H_3        </font> : somme des fautes sur l'ensemble des 5 derniers matchs pour le club adverse ;",                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Pass_Tot_1H_5       </font> : somme des passes intelligentes sur la 1<sup>re</sup> mi-temps des 5 derniers matchs pour le club adverse ;",     unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Pass_Tot_1H_1       </font> : somme des passes intelligentes sur la 1<sup>re</sup> mi-temps du dernier match pour le club adverse ;",          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Pass_Tot_2H_3       </font> : somme des passes intelligentes sur la 2<sup>e</sup> mi-temps des 3 derniers matchs pour le club adverse ;",      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Pass_Tot_3H_5       </font> : somme des passes intelligentes sur l'ensemble des 5 derniers matchs pour le club adverse ;",                     unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Pass_Tot_3H_1       </font> : somme des passes intelligentes sur l'ensemble du dernier match pour le club adverse ;",                          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tirs_Tot_1H_3       </font> : somme des tirs proche du but sur la 1<sup>re</sup> mi-temps des 3 derniers matchs pour le club adverse ;",       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tirs_Tot_2H_5       </font> : somme des tirs proche du but sur la 2<sup>e</sup> mi-temps des 5 derniers matchs pour le club adverse ;",        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tirs_Tot_2H_1       </font> : somme des tirs proche du but sur la 2<sup>e</sup> mi-temps du dernier match pour le club adverse ;",             unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Tirs_Tot_3H_3       </font> : somme des tirs proche du but sur l'ensemble des 3 derniers matchs pour le club adverse ;",                       unsafe_allow_html=True)
    
    st.code("""
    # Jointure entre les tables « match_results » et « match_infos »
    match_all = pd.merge(match_results, match_infos.drop(['Date', 'Équipe_id'], axis=1), on=['Match_id', 'Club'])
    
    # Auto-jointure pour la récupération des variables explicatives concernant les clubs adverses
    match_all = pd.merge(match_all, match_all.filter(regex='(Date|Club$|^Act_.*|^Gar_.*|^Fte_.*|^Pass_.*|^Tirs_.*)', axis=1).rename(columns={'Club': 'Club_ADV'}), how='inner', on=['Date', 'Club_ADV'])
    
    # Fonction pour le renommage des colonnes après l'auto-jointure
    def renameCols(name):
        if name[-2:] == '_x':
            return 'CLB_' + name[:-2]
        elif name[-2:] == '_y':
            return 'ADV_' + name[:-2]
        else:
            return name
    
    # Renommage des colonnes
    match_all.columns = match_all.columns.map(renameCols)
    """, language='python')
    
    # Affichage d'un extrait de la table « match_all »
    if st.checkbox('Afficher un extrait de la table « match_all »'):
        # Chargement des données
        df_match_all = pd.read_csv(path_data + 'match_all_samples.csv', index_col=0)
        
        df_match_all
    
    #######################################
    ##  DATA                             ##
    #######################################
    
    st.subheader('Création de la table « data »')
    
    st.write("""Les colonnes de la table « data » sont, soit des reprises telles quelles de colonnes de la table « match_all »
                (comme par exemple la colonne « Dom_Ext »). Soit la différence entre 2 colonnes de la table « match_all »
                qui viennent formées un écart pour une même statistique, entre la donnée du club et celle de l'adversaire.""")
    
    st.write("""Par exemple, la colonne « Écart_Série_vic » est obtenue par la différence entre les colonnes « CLB_Série_vic » et « ADV_Série_vic ».""")
    
    st.write("""Le détail de ces colonnes et le code utilisé pour les obtenir sont diponibles ci-dessous.""")
    
    # Schéma pour la création de la table « data »
    st.image(path_imag + 'Table 07 - data.png',
             caption='Création de la table « data ».')
    
    # Présentation détaillée des colonnes de la tables « data »
    if st.checkbox('Afficher la présentation détaillée des colonnes de la table « data »'):
        st.markdown('**Colonnes de la table « data » :**')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- <font color=" + st.session_state.cols_color + ">Index               </font> : clef primaire ;",                                                                                          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">CLB_Nb_jours        </font> : nombre de jours depuis le dernier match non nul ;",                                                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Série_vic     </font> : écart entre les nombres de victoires du club et de l'avdersaire ;",                                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Série_vic_3   </font> : écart entre les nombres de victoires du club et de l'adversaire sur les 3 derniers matchs ;",              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Buts_diff     </font> : écart entre les différences de victoires-défaites du club et de l'adversaire ;",                           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Buts_diff_3   </font> : écart entre les différences de victoires-défaites du club et de l'adversaire sur les 3 derniers matchs ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Buts_pour     </font> : écart entre les nombres de buts marqués du club et de l'adversaire ;",                                     unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Buts_pour_3   </font> : écart entre les nombres de buts marqués du club et de l'adversaire sur les 3 derniers matchs ;",           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Buts_contre   </font> : écart entre les nombres de buts encaissés du club et de l'adversaire ;",                                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Buts_contre_3 </font> : écart entre les nombres de buts encaissés du club et de l'adversaire sur les 3 derniers matchs ;",         unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Activité_att_5</font> : écart entre les activités des attaquants du club et de l'adversaire sur les 5 derniers matchs ;",          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Activité_att_3</font> : écart entre les activités des attaquants du club et de l'adversaire sur les 3 derniers matchs ;",          unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Activité_att_1</font> : écart entre les activités des attaquants du club et de l'adversaire sur le dernier match ;",               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Arrêts_5      </font> : écart entre les nombres d'arrêts du club et de l'adversaire sur les 5 derniers matchs ;",                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Arrêts_3      </font> : écart entre les nombres d'arrêts du club et de l'adversaire sur les 3 derniers matchs ;",                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Arrêts_1      </font> : écart entre les nombres d'arrêts du club et de l'adversaire sur le dernier match ;",                       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Passes_5      </font> : écart entre les nombres de passes intelligentes du club et de l'adversaire sur les 5 derniers matchs ;",   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Passes_3      </font> : écart entre les nombres de passes intelligentes du club et de l'adversaire sur les 3 derniers matchs ;",   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Passes_1      </font> : écart entre les nombres de passes intelligentes du club et de l'adversaire sur le dernier match ;",        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">[CLB_Buts]          </font> : nombre de buts marqués à la fin du match (variable cible) ;",                                              unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">[ADV_Buts]          </font> : nombre de buts encaissés à la fin du match (variable cible) ;",                                            unsafe_allow_html=True)
        with col2:
            st.markdown("- <font color=" + st.session_state.cols_color + ">Dom_Ext             </font> : indique si le club joue à domicile ou à l'extérieur ;",                                                            unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">ADV_Nb_jours        </font> : nombre de jours depuis le dernier match non nul pour le club adverse ;",                                           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Série_vic_5   </font> : écart entre les nombres de victoires du club et de l'adversaire sur les 5 derniers matchs ;",                      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Série_vic_1   </font> : écart entre les nombres de victoires du club et de l'adversaire sur le dernier match ;",                           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Buts_diff_5   </font> : écart entre la différence de victoires-défaites du club et de l'adversaire sur les 5 derniers matchs ;",           unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Buts_diff_1   </font> : écart entre la différence de victoires-défaites du club et de l'adversaire sur le dernier match ;",                unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Buts_pour_5   </font> : écart entre les nombres de buts marqués du club et de l'adversaire sur les 5 derniers matchs ;",                   unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Buts_pour_1   </font> : écart entre les nombres de buts marqués du club et de l'adversaire sur le dernier match ;",                        unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Buts_contre_5 </font> : écart entre les nombres de buts encaissés du club et de l'adversaire sur les 5 derniers matchs ;",                 unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Buts_contre_1 </font> : écart entre les nombres de buts encaissés du club et de l'adversaire sur le dernier match ;",                      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Activité_def_5</font> : écart entre les activités des défenseurs du club et de l'adversaire sur les 5 derniers matchs ;",                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Activité_def_3</font> : écart entre les activités des défenseurs du club et de l'adversaire sur les 3 derniers matchs ;",                  unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Activité_def_1</font> : écart entre les activités des défenseurs du club et de l'adversaire sur le dernier match ;",                       unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Fautes_5      </font> : écart entre les nombres de fautes commises par le club et l'adversaire sur les 5 derniers matchs ;",               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Fautes_3      </font> : écart entre les nombres de fautes commises par le club et l'adversaire sur les 3 derniers matchs ;",               unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Fautes_1      </font> : écart entre les nombres de fautes commises par le club et l'adversaire sur le dernier match ;",                    unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Tirs_5        </font> : écart entre les nombres de tirs proches du but effectués par le club et l'adversaire sur les 5 derniers matchs ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Tirs_3        </font> : écart entre les nombres de tirs proches du but effectués par le club et l'adversaire sur les 3 derniers matchs ;", unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">Écart_Tirs_1        </font> : écart entre les nombres de tirs proches du but effectués par le club et l'adversaire sur le dernier match ;",      unsafe_allow_html=True)
            st.markdown("- <font color=" + st.session_state.cols_color + ">[Résultat]          </font> : indique la victoire ou la défaite (variable cible).",                                                              unsafe_allow_html=True)
    
    if st.checkbox('Afficher le code pour la création de la table « data »'):
        st.code("""
        # Suppression des lignes ayant des NA dans la table « match_all »
        match_all = match_all.dropna()
        
        # Création de la table « data »
        
        ##########################
        ## ISSUES MATCH_RESULTS ##
        ##########################
        
        data                        = pd.DataFrame(match_all[['Dom_Ext', 'CLB_Nb_jours', 'ADV_Nb_jours']])
        
        data['Écart_Série_vic']     = match_all['CLB_Série_vic']   - match_all['ADV_Série_vic']
        data['Écart_Série_vic_5']   = match_all['CLB_Série_vic_5'] - match_all['ADV_Série_vic_5']
        data['Écart_Série_vic_3']   = match_all['CLB_Série_vic_3'] - match_all['ADV_Série_vic_3']
        data['Écart_Série_vic_1']   = match_all['CLB_Série_vic_1'] - match_all['ADV_Série_vic_1']
        
        data['Écart_Buts_diff']     = match_all['CLB_Tot_buts_diff']   - match_all['ADV_Tot_buts_diff']
        data['Écart_Buts_diff_5']   = match_all['CLB_Tot_buts_diff_5'] - match_all['ADV_Tot_buts_diff_5']
        data['Écart_Buts_diff_3']   = match_all['CLB_Tot_buts_diff_3'] - match_all['ADV_Tot_buts_diff_3']
        data['Écart_Buts_diff_1']   = match_all['CLB_Tot_buts_diff_1'] - match_all['ADV_Tot_buts_diff_1']
        
        data['Écart_Buts_pour']     = match_all['CLB_Tot_buts_pour']   - match_all['ADV_Tot_buts_pour']
        data['Écart_Buts_pour_5']   = match_all['CLB_Tot_buts_pour_5'] - match_all['ADV_Tot_buts_pour_5']
        data['Écart_Buts_pour_3']   = match_all['CLB_Tot_buts_pour_3'] - match_all['ADV_Tot_buts_pour_3']
        data['Écart_Buts_pour_1']   = match_all['CLB_Tot_buts_pour_1'] - match_all['ADV_Tot_buts_pour_1']
        
        data['Écart_Buts_contre']   = match_all['ADV_Tot_buts_contre']   - match_all['CLB_Tot_buts_contre']
        data['Écart_Buts_contre_5'] = match_all['ADV_Tot_buts_contre_5'] - match_all['CLB_Tot_buts_contre_5']
        data['Écart_Buts_contre_3'] = match_all['ADV_Tot_buts_contre_3'] - match_all['CLB_Tot_buts_contre_3']
        data['Écart_Buts_contre_1'] = match_all['ADV_Tot_buts_contre_1'] - match_all['CLB_Tot_buts_contre_1']
        
        ########################
        ## ISSUES MATCH_INFOS ##
        ########################
        
        # Activité des attaquants
        data['Écart_Activité_att_5'] = match_all['CLB_Act_Tot_att_3H_5'] - match_all['ADV_Act_Tot_att_3H_5']
        data['Écart_Activité_att_3'] = match_all['CLB_Act_Tot_att_3H_3'] - match_all['ADV_Act_Tot_att_3H_3']
        data['Écart_Activité_att_1'] = match_all['CLB_Act_Tot_att_3H_1'] - match_all['ADV_Act_Tot_att_3H_1']
        
        # Activité des défenseurs
        data['Écart_Activité_def_5'] = match_all['CLB_Act_Tot_def_3H_5'] - match_all['ADV_Act_Tot_def_3H_5']
        data['Écart_Activité_def_3'] = match_all['CLB_Act_Tot_def_3H_3'] - match_all['ADV_Act_Tot_def_3H_3']
        data['Écart_Activité_def_1'] = match_all['CLB_Act_Tot_def_3H_1'] - match_all['ADV_Act_Tot_def_3H_1']
        
        # Nombre d'arrêts du gardien
        data['Écart_Arrêts_5']       = match_all['CLB_Gar_Tot_3H_5'] - match_all['ADV_Gar_Tot_3H_5']
        data['Écart_Arrêts_3']       = match_all['CLB_Gar_Tot_3H_3'] - match_all['ADV_Gar_Tot_3H_3']
        data['Écart_Arrêts_1']       = match_all['CLB_Gar_Tot_3H_1'] - match_all['ADV_Gar_Tot_3H_1']
        
        # Nombre de fautes des joueurs adverses dans la zone proche de leurs buts
        data['Écart_Fautes_5']       = match_all['CLB_Fte_Tot_3H_5'] - match_all['ADV_Fte_Tot_3H_5']
        data['Écart_Fautes_3']       = match_all['CLB_Fte_Tot_3H_3'] - match_all['ADV_Fte_Tot_3H_3']
        data['Écart_Fautes_1']       = match_all['CLB_Fte_Tot_3H_1'] - match_all['ADV_Fte_Tot_3H_1']
        
        # Nombre de passes intelligentes de la part de l'équipe
        data['Écart_Passes_5']       = match_all['CLB_Pass_Tot_3H_5'] - match_all['ADV_Pass_Tot_3H_5']
        data['Écart_Passes_3']       = match_all['CLB_Pass_Tot_3H_3'] - match_all['ADV_Pass_Tot_3H_3']
        data['Écart_Passes_1']       = match_all['CLB_Pass_Tot_3H_1'] - match_all['ADV_Pass_Tot_3H_1']
        
        # Nombre de tirs dans la zone proche du but adverse
        data['Écart_Tirs_5']         = match_all['CLB_Tirs_Tot_3H_5'] - match_all['ADV_Tirs_Tot_3H_5']
        data['Écart_Tirs_3']         = match_all['CLB_Tirs_Tot_3H_3'] - match_all['ADV_Tirs_Tot_3H_3']
        data['Écart_Tirs_1']         = match_all['CLB_Tirs_Tot_3H_1'] - match_all['ADV_Tirs_Tot_3H_1']
        
        ######################
        ## VARIABLES CIBLES ##
        ######################
        
        data['[Résultat]']           = match_all['[Résultat]']
        data['[CLB_Buts]']           = match_all['[CLB_Buts]']
        data['[ADV_Buts]']           = match_all['[ADV_Buts]']
        """, language='python')
    
    # Affichage d'un extrait de la table « data »
    if st.checkbox('Afficher un extrait de la table « data »'):
        # Chargement des données
        df_data = pd.read_csv(path_data + 'data_samples.csv', index_col=0)
        
        df_data


###############################################################################
## PARTIE 5 : MACHINE LEARNING                                               ##
###############################################################################

if menu_sel == menu_lst[5]:
    st.header(title)
    st.subheader(menu_sel, anchor='Machine-Learning')
    
    #######################################
    ## DONNÉES +  ALGORITHME             ##
    #######################################
    
    # Chargement des données
    df_data = pd.read_csv(path_data + 'data.csv', index_col=0)
    
    if st.checkbox('Afficher la table « data »'):
        df_data
    
    ## Phase de sélection pour l'utilisateur
    
    # Sélection des données par l'utilisateur
    lst_features = list(df_data.columns[:-3])                             # Liste des variables explicatives
    lst_targets  = list(df_data.columns[-3:])                             # Liste des variables cibles
    lst_algos    = ['KNN', 'Logistic Regression', 'Random Forest', 'SVM'] # Liste des algos
    
    st.session_state.sel_features = st.multiselect('Sélection des variables explicatives', lst_features, default=st.session_state.sel_features)
    sel_target                    = st.selectbox('Sélection de la variable cible', lst_targets)
    
    # Sélection de l'algorithme de machine learning par l'utilisateur
    sel_algo = st.selectbox("Sélection de l'algorithme de machine learning", lst_algos)
    
    #######################################
    ## K-NEAREST NEIGHBORS               ##
    #######################################
    
    if sel_algo == 'KNN':
        st.title('K-NEAREST NEIGHBORS')
        
        # Sélection des hyperparamètres par l'utilisateur
        lst_metrics = ['minkowski']
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.sel_KNN_ratio  = st.slider('Proportion des données de test (%)', value=int(st.session_state.sel_KNN_ratio*100), min_value=0, max_value=100) / 100.
            st.session_state.sel_KNN_metric = st.multiselect('Choix de la métrique', lst_metrics, default=st.session_state.sel_KNN_metric)
        with col2:
            st.session_state.sel_KNN_neighbors_min = st.number_input('Nombre de voisins minimaux', value=st.session_state.sel_KNN_neighbors_min, min_value=1, max_value=50)
            if st.session_state.sel_KNN_neighbors_max < st.session_state.sel_KNN_neighbors_min:
                st.session_state.sel_KNN_neighbors_max = st.session_state.sel_KNN_neighbors_min
            st.session_state.sel_KNN_neighbors_max = st.number_input('Nombre de voisins maximaux', value=st.session_state.sel_KNN_neighbors_max, min_value=st.session_state.sel_KNN_neighbors_min, max_value=50)
        
        if len(st.session_state.sel_KNN_metric) > 0:
            if st.button('Lancement KNN'):
                if len(st.session_state.sel_features) == 0:
                    # /!\ Au moins une variable explicative doit être renseignée
                    st.error('Veuillez sélectionner au moins une variable explicative !')
                
                else:
                    # Initialisation de la barre de progression
                    knn_bar = st.progress(0)
                    
                    # Récupération et séparation du JDD
                    X_train, X_test, y_train, y_test = train_test_split(df_data[st.session_state.sel_features], df_data[sel_target], test_size=st.session_state.sel_KNN_ratio, shuffle=False)
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    knn_bar.progress(10)
                    
                    # Normalisation des données
                    scaler = preprocessing.StandardScaler().fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled  = scaler.transform(X_test)
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    knn_bar.progress(20)
                    
                    # Sans GridSearch
                    if 1 == len(st.session_state.sel_KNN_metric) and st.session_state.sel_KNN_neighbors_min == st.session_state.sel_KNN_neighbors_max:
                        knn_params_grid = {'n_neighbors': st.session_state.sel_KNN_neighbors_min,
                                           'metric': st.session_state.sel_KNN_metric[0]}
                        
                        # Affichage des hyperparamètres
                        st.subheader('Hyperparamètres')
                        for key in knn_params_grid:
                            st.write(key + ' : ' + str(knn_params_grid[key]))
                        
                        # Mise à jour de la barre de progression
                        sleep(.1)
                        knn_bar.progress(50)
                        
                        # Entraînement du modèle
                        knn_clf = neighbors.KNeighborsClassifier()
                        knn_clf.set_params(**knn_params_grid)
                        
                        knn_clf.fit(X_train_scaled, y_train)
                    
                    # Avec GridSearch
                    else:
                        knn_params_grid = {'n_neighbors': range(st.session_state.sel_KNN_neighbors_min, st.session_state.sel_KNN_neighbors_max + 1),
                                           'metric': st.session_state.sel_KNN_metric}
                        
                        # Entraînement du modèle + recherche des meilleures hyperparamètres
                        knn_clf = GridSearchCV(estimator=neighbors.KNeighborsClassifier(), param_grid=knn_params_grid)
                        knn_clf.fit(X_train_scaled, y_train)
                        
                        knn_params_grid = knn_clf.best_params_
                        
                        # Affichage des hyperparamètres
                        st.subheader('Hyperparamètres')
                        for key in knn_params_grid:
                            st.write(key + ' : ' + str(knn_params_grid[key]))
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    knn_bar.progress(90)
                    
                    # Prédiction
                    knn_y_pred = knn_clf.predict(X_test_scaled)
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    knn_bar.progress(100)
                    
                    # Évaluation
                    st.subheader('Évaluation')
                    st.write('Score de la prédiction : {} %'.format(round(knn_clf.score(X_test_scaled, y_test) * 100., 2)))
                    
                    # Rapport de classification
                    st.subheader('Rapport de classification')
                    display_classification_report(classification_report(y_test, knn_y_pred))
                    
                    # Matrice de confusion
                    st.subheader('Matrice de confusion')
                    display_matrix(pd.crosstab(y_test, knn_y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']))
                    
                    # Affichage de la courbe ROC
                    st.subheader('Courbe ROC')
                    display_ROC('KNN', y_test, knn_clf.predict_proba(X_test_scaled))
    
    #######################################
    ## LOGISTIC REGRESSION               ##
    #######################################
    
    if sel_algo == 'Logistic Regression':
        st.title('LOGISTIC REGRESSION')
        
        # Sélection des hyperparamètres par l'utilisateur
        lst_values  = [i*10**j for j in range(-3, 2) for i in [1, 5]]
        lst_penalty = ['l1', 'l2', 'elasticnet', 'none']
        lst_solver  = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.sel_LR_ratio   = st.slider('Proportion des données de test (%)', value=int(st.session_state.sel_LR_ratio*100), min_value=0, max_value=100) / 100.
            st.session_state.sel_LR_C       = st.multiselect('C', lst_values, default=st.session_state.sel_LR_C)
        with col2:
            st.session_state.sel_LR_penalty = st.multiselect('penalty', lst_penalty, default=st.session_state.sel_LR_penalty)
            st.session_state.sel_LR_solver  = st.multiselect('solver', lst_solver, default=st.session_state.sel_LR_solver)
        
        if len(st.session_state.sel_LR_C) > 0 and len(st.session_state.sel_LR_penalty) > 0 and len(st.session_state.sel_LR_solver) > 0:
            if st.button('Lancement LR'):
                if len(st.session_state.sel_features) == 0:
                    # /!\ Au moins une variable explicative doit être renseignée
                    st.error('Veuillez sélectionner au moins une variable explicative !')
                
                else:
                    # Initialisation de la barre de progression
                    lr_bar = st.progress(0)
                    
                    # Récupération et séparation du JDD
                    X_train, X_test, y_train, y_test = train_test_split(df_data[st.session_state.sel_features], df_data[sel_target], test_size=st.session_state.sel_LR_ratio, shuffle=False)
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    lr_bar.progress(10)
                    
                    # Normalisation des données
                    scaler = preprocessing.StandardScaler().fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled  = scaler.transform(X_test)
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    lr_bar.progress(20)
                    
                    # Sans GridSearch
                    if 1 == len(st.session_state.sel_LR_C) == len(st.session_state.sel_LR_penalty) == len(st.session_state.sel_LR_solver):
                        lr_params_grid = {'C': st.session_state.sel_LR_C[0],
                                          'penalty': st.session_state.sel_LR_penalty[0],
                                          'solver': st.session_state.sel_LR_solver[0]}
                        
                        # Affichage des hyperparamètres
                        st.subheader('Hyperparamètres')
                        for key in lr_params_grid:
                            st.write(key + ' : ' + str(lr_params_grid[key]))
                        
                        # Mise à jour de la barre de progression
                        sleep(.1)
                        lr_bar.progress(50)
                        
                        # Entraînement du modèle
                        lr_clf = LogisticRegression()
                        lr_clf.set_params(**lr_params_grid)
                        
                        lr_clf.fit(X_train_scaled, y_train)
                    
                    # Avec GridSearch
                    else:
                        lr_params_grid = {'C': st.session_state.sel_LR_C,
                                          'penalty': st.session_state.sel_LR_penalty,
                                          'solver': st.session_state.sel_LR_solver}
                        
                        # Entraînement du modèle + recherche des meilleures hyperparamètres
                        lr_clf = GridSearchCV(estimator=LogisticRegression(), param_grid=lr_params_grid)
                        lr_clf.fit(X_train_scaled, y_train)
                        
                        lr_params_grid = lr_clf.best_params_
                        
                        # Affichage des hyperparamètres
                        st.subheader('Hyperparamètres')
                        for key in lr_params_grid:
                            st.write(key + ' : ' + str(lr_params_grid[key]))
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    lr_bar.progress(90)
                    
                    # Prédiction
                    lr_y_pred = lr_clf.predict(X_test_scaled)
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    lr_bar.progress(100)
                    
                    # Évaluation
                    st.subheader('Évaluation')
                    st.write('Score de la prédiction : {} %'.format(round(lr_clf.score(X_test_scaled, y_test) * 100., 2)))
                    
                    # Rapport de classification
                    st.subheader('Rapport de classification')
                    display_classification_report(classification_report(y_test, lr_y_pred))
                    
                    # Matrice de confusion
                    st.subheader('Matrice de confusion')
                    display_matrix(pd.crosstab(y_test, lr_y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']))
                    
                    # Affichage de la courbe ROC
                    st.subheader('Courbe ROC')
                    display_ROC('Logistic Regression', y_test, lr_clf.predict_proba(X_test_scaled))
    
    #######################################
    ## SUPPORT VECTOR MACHINE            ##
    #######################################
    
    if sel_algo == 'SVM':
        st.title('SUPPORT VECTOR MACHINE')
        
        # Sélection des hyperparamètres par l'utilisateur
        lst_values = [i*10**j for j in range(-3, 2) for i in [1, 5]]
        lst_kernel = ['rbf', 'linear', 'poly']
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.sel_SVM_ratio  = st.slider('Proportion des données de test (%)', value=int(st.session_state.sel_SVM_ratio*100), min_value=0, max_value=100) / 100.
            st.session_state.sel_SVM_C      = st.multiselect('C', lst_values, default=st.session_state.sel_SVM_C)
        with col2:
            st.session_state.sel_SVM_kernel = st.multiselect('kernel', lst_kernel, default=st.session_state.sel_SVM_kernel)
            st.session_state.sel_SVM_gamma  = st.multiselect('gamma', lst_values, default=st.session_state.sel_SVM_gamma)
        
        if len(st.session_state.sel_SVM_C) > 0 and len(st.session_state.sel_SVM_kernel) > 0 and len(st.session_state.sel_SVM_gamma) > 0:
            if st.button('Lancement SVM'):
                if len(st.session_state.sel_features) == 0:
                    # /!\ Au moins une variable explicative doit être renseignée
                    st.error('Veuillez sélectionner au moins une variable explicative !')
                
                else:
                    # Initialisation de la barre de progression
                    svm_bar = st.progress(0)
                    
                    # Récupération et séparation du JDD
                    X_train, X_test, y_train, y_test = train_test_split(df_data[st.session_state.sel_features], df_data[sel_target], test_size=st.session_state.sel_SVM_ratio, shuffle=False)
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    svm_bar.progress(10)
                    
                    # Normalisation des données
                    scaler = preprocessing.StandardScaler().fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled  = scaler.transform(X_test)
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    svm_bar.progress(20)
                    
                    # Sans GridSearch
                    if 1 == len(st.session_state.sel_SVM_C) == len(st.session_state.sel_SVM_kernel) == len(st.session_state.sel_SVM_gamma):
                        svm_params_grid = {'C': st.session_state.sel_SVM_C[0],
                                           'gamma': st.session_state.sel_SVM_gamma[0],
                                           'kernel': st.session_state.sel_SVM_kernel[0]}
                        
                        # Affichage des hyperparamètres
                        st.subheader('Hyperparamètres')
                        for key in svm_params_grid:
                            st.write(key + ' : ' + str(svm_params_grid[key]))
                        
                        # Mise à jour de la barre de progression
                        sleep(.1)
                        svm_bar.progress(50)
                        
                        # Entraînement du modèle
                        svm_clf = svm.SVC(probability=True)
                        svm_clf.set_params(**svm_params_grid)
                        
                        svm_clf.fit(X_train_scaled, y_train)
                    
                    # Avec GridSearch
                    else:
                        svm_params_grid = {'C': st.session_state.sel_SVM_C,
                                           'gamma': st.session_state.sel_SVM_gamma,
                                           'kernel': st.session_state.sel_SVM_kernel}
                        
                        # Entraînement du modèle + recherche des meilleures hyperparamètres
                        svm_clf = GridSearchCV(estimator=svm.SVC(probability=True), param_grid=svm_params_grid)
                        svm_clf.fit(X_train_scaled, y_train)
                        
                        svm_params_grid = svm_clf.best_params_
                        
                        # Affichage des hyperparamètres
                        st.subheader('Hyperparamètres')
                        for key in svm_params_grid:
                            st.write(key + ' : ' + str(svm_params_grid[key]))
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    svm_bar.progress(90)
                    
                    # Prédiction
                    svm_y_pred = svm_clf.predict(X_test_scaled)
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    svm_bar.progress(100)
                    
                    # Évaluation
                    st.subheader('Évaluation')
                    st.write('Score de la prédiction : {} %'.format(round(svm_clf.score(X_test_scaled, y_test) * 100., 2)))
                    
                    # Rapport de classification
                    st.subheader('Rapport de classification')
                    display_classification_report(classification_report(y_test, svm_y_pred))
                    
                    # Matrice de confusion
                    st.subheader('Matrice de confusion')
                    display_matrix(pd.crosstab(y_test, svm_y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']))
                    
                    # Affichage de la courbe ROC
                    st.subheader('Courbe ROC')
                    display_ROC('SVM', y_test, svm_clf.predict_proba(X_test_scaled))
    
    #######################################
    ## RANDOM FOREST                     ##
    #######################################
    
    if sel_algo == 'Random Forest':
        st.title('RANDOM FOREST')
        
        # Sélection des hyperparamètres par l'utilisateur
        lst_max_feat = ['auto', 'log2','sqrt', None]
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.sel_RF_ratio    = st.slider('Proportion des données de test (%)', value=int(st.session_state.sel_RF_ratio*100), min_value=0, max_value=100) / 100.
            st.session_state.sel_RF_max_feat = st.multiselect('Choix de la fonction pour le calcul de "max_features"', lst_max_feat, default=st.session_state.sel_RF_max_feat)
        with col2:
            st.session_state.sel_RF_min_samples_min = st.number_input("Nombre minimum d'échantillons", value=st.session_state.sel_RF_min_samples_min, min_value=1, max_value=50, step=2)
            if st.session_state.sel_RF_min_samples_max < st.session_state.sel_RF_min_samples_min:
                st.session_state.sel_RF_min_samples_max = st.session_state.sel_RF_min_samples_min
            st.session_state.sel_RF_min_samples_max = st.number_input("Nombre maximum d'échantillons", value=st.session_state.sel_RF_min_samples_max, min_value=st.session_state.sel_RF_min_samples_min, max_value=50, step=2)
        
        if len(st.session_state.sel_RF_max_feat) > 0:
            if st.button('Lancement RF'):
                if len(st.session_state.sel_features) == 0:
                    # /!\ Au moins une variable explicative doit être renseignée
                    st.error('Veuillez sélectionner au moins une variable explicative !')
                
                else:
                    # Initialisation de la barre de progression
                    rf_bar = st.progress(0)
                    
                    # Récupération et séparation du JDD
                    X_train, X_test, y_train, y_test = train_test_split(df_data[st.session_state.sel_features], df_data[sel_target], test_size=st.session_state.sel_RF_ratio, shuffle=False)
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    rf_bar.progress(10)
                    
                    # Normalisation des données
                    scaler = preprocessing.StandardScaler().fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled  = scaler.transform(X_test)
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    rf_bar.progress(20)
                    
                    # Sans GridSearch
                    if 1 == len(st.session_state.sel_RF_max_feat) and st.session_state.sel_RF_min_samples_min == st.session_state.sel_RF_min_samples_max:
                        rf_params_grid = {'max_features': st.session_state.sel_RF_max_feat[0],
                                          'min_samples_split': st.session_state.sel_RF_min_samples_min}
                        
                        # Affichage des hyperparamètres
                        st.subheader('Hyperparamètres')
                        for key in rf_params_grid:
                            st.write(key + ' : ' + str(rf_params_grid[key]))
                        
                        # Mise à jour de la barre de progression
                        sleep(.1)
                        rf_bar.progress(50)
                        
                        # Entraînement du modèle
                        rf_clf = RandomForestClassifier()
                        rf_clf.set_params(**rf_params_grid)
                        
                        rf_clf.fit(X_train_scaled, y_train)
                    
                    # Avec GridSearch
                    else:
                        rf_params_grid = {'max_features': st.session_state.sel_RF_max_feat,
                                          'min_samples_split': range(st.session_state.sel_RF_min_samples_min, st.session_state.sel_RF_min_samples_max + 1, 2)}
                        
                        # Entraînement du modèle + recherche des meilleures hyperparamètres
                        rf_clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=rf_params_grid)
                        rf_clf.fit(X_train_scaled, y_train)
                        
                        rf_params_grid = rf_clf.best_params_
                        
                        # Affichage des hyperparamètres
                        st.subheader('Hyperparamètres')
                        for key in rf_params_grid:
                            st.write(key + ' : ' + str(rf_params_grid[key]))
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    rf_bar.progress(90)
                    
                    # Prédiction
                    rf_y_pred = rf_clf.predict(X_test_scaled)
                    
                    # Mise à jour de la barre de progression
                    sleep(.1)
                    rf_bar.progress(100)
                    
                    # Évaluation
                    st.subheader('Évaluation')
                    st.write('Score de la prédiction : {} %'.format(round(rf_clf.score(X_test_scaled, y_test) * 100., 2)))
                    
                    # Rapport de classification
                    st.subheader('Rapport de classification')
                    display_classification_report(classification_report(y_test, rf_y_pred))
                    
                    # Matrice de confusion
                    st.subheader('Matrice de confusion')
                    display_matrix(pd.crosstab(y_test, rf_y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']))
                    
                    # Affichage de la courbe ROC
                    st.subheader('Courbe ROC')
                    display_ROC('Random Forest', y_test, rf_clf.predict_proba(X_test_scaled))


###############################################################################
## PARTIE 6 : CONCLUSION ET PERSPECTIVE                                      ##
###############################################################################

if menu_sel == menu_lst[6]:
    st.header(title)
    st.subheader('Conclusion')
    
    st.markdown("""
                Ce projet nous a permis de travailler sur les grandes étapes d'un projet type en *data science*
                (exploration et nettoyage des données, détermination et création des variables explicatives,
                *machine learning*, présentation des résultats).
                """)
    
    st.subheader('Perspective')
    
    st.markdown("""
                Afin d'améliorer les résultats que nous avons obtenus, il est possible de jouer sur plusieurs points :
                
                - enrichissement du JDD :
                    - récupération d'informations supplémentaires sur les clubs (budget annuel, classement UEFA, ...) ;
                    - récupération de nouvelles statistiques comme *xG* ou *xGA* (*expected Goals* et *expected Goals Against*)
                      afin de prendre en compte la capacité des équipes à sur ou sous performer ;
                    - ajouter les matchs nuls ;
                
                - création de nouvelles variables explicatives :
                    - revoir les variables de tendance (actuellement sur les 1, 3 et 5 derniers matchs) ;
                    - création ou récupération de notes de joueurs ou d'équipe sur les matchs précédents ;
                - avoir une meilleure sélection de variables explicatives à l'entrée du *machine learning* ;
                - chercher dans l'état de l'art des algorithmes de *machine learning* plus adaptés au problème.
                """)
    
    st.markdown("""
                Et pour aller encore plus loin, il serait intéressant de récupérer les côtes de *bookmakers*
                et d'entrainer un modèle afin de tenter de « gagner de l'argent ».
                """)
