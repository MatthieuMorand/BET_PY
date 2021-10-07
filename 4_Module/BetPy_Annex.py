###############################################################################
## Fichier : BetPy_Annex.py                                                  ##
## Date de création : 12/04/21                                               ##
## Date de dernière modification : 19/07/21                                  ##
## Version : 1.0                                                             ##
## Description :                                                             ##
##                                                                           ##
###############################################################################
## Auteur | Date     | Description                                           ##
## ------------------------------------------------------------------------- ##
## MMO    | 12/04/21 | Création du fichier.                                  ##
##        |          | Ajout de la fonction « Collect_results ».             ##
## MMO    | 13/04/21 | Ajout des fonctions « Collect_infos » et              ##
##        |          | « Aggregate_infos ».                                  ##
## MMO    | 15/04/21 | Ajout de la fonction « Exploit_infos ».               ##
## MMO    | 16/04/21 | Correction de la fonction « Exploit_infos ».          ##
## MMO    | 17/04/21 | Correction de la fonction « Collect_results ».        ##
## MMO    | 18/04/21 | Correction de la récupération des variables           ##
##        |          | « [CLB_Buts] » et « [ADV_Buts] » dans la fonction     ##
##        |          | « Collect_results ».                                  ##
## MMO    | ../../21 | Modification de la fonction « Collect_results » pour  ##
##        |          | pour création de nouveaux indicateurs depuis la table ##
##        |          | « match ».                                            ##
## MMO    | 23/06/21 | Modification de la fonction « Collect_infos » pour    ##
##        |          | création de nouveaux indicateurs sur le déroulé de    ##
##        |          | chaque match.                                         ##
## MMO    | 01/07/21 | Fin du VABF sur la fonction « Collect_infos ».        ##
##        |          | Refonte de la fonction « Aggregate_infos ».           ##
## MMO    | 04/07/21 | Fin du VABF sur la fonction « Aggregate_infos ».      ##
## MMO    | 05/07/21 | Refonte de la fonction « Exploit_infos ».             ##
## MMO    | 07/07/21 | VABF et correction de la fonction « Exploit_infos ».  ##
## MMO    | 08/07/21 | Fin du VABF de la fonction « Exploit_infos ».         ##
## MMO    | 19/07/21 | Ajout de la fonction « Build_tree ».                  ##
###############################################################################

## IMPORATION DES MODULES

import numpy  as np
import pandas as pd

import re

## ÉCRITURE DES FONCTIONS

###############################################################################
## Fonction : Collect_results                                                ##
## Description :                                                             ##
##                                                                           ##
## Arguments :                                                               ##
## Complexité :                                                              ##
###############################################################################
def Collect_results(df):
    """
    Cette fonction permet de créé la table « match_results »
    à partir de la table « match ».
    
    Argument :
     - df : la table « match ».
    
    Prérequis : « df » doit être trié sur la colonne « dateutc » (ordre croissant).
    """
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


###############################################################################
## Fonction : Collect_infos                                                  ##
## Description :                                                             ##
##                                                                           ##
## Arguments :                                                               ##
## Complexité :                                                              ##
###############################################################################
def Collect_infos(df):
    """
    Cette fonction permet de créé une ligne, à insérée dans la table « match_infos »,
    qui contient des informations extraites à partir des événements d'un match particulier de la table
    « match_event ».
    
    Argument :
     - df : les données de la table « match_event » correspondant à un match particulier.
    
    Prérequis : « df » doit être trié sur les colonnes « matchperiod » et « eventsec » (ordre croissant).
    """
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
        
        ## Initialisation des informations des joueurs
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


###############################################################################
## Fonction : Aggregate_infos                                                ##
## Description :                                                             ##
##                                                                           ##
## Arguments :                                                               ##
## Complexité :                                                              ##
###############################################################################
def Aggregate_infos(match_id, meta_infos, infos, n_index=0):
    """
    TODO
    """
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
    ## LA POSSESSION          ##
    ############################

    # Calcul du temps total de possession du ballon par un joueur sur l'intégralité du match
    # pattern = '^Possession_.*'
    # tot_all = infos.filter(regex=pattern, axis=1).sum().sum()
    
    # for ptr_res, ptr_team in zip([results1, results2], [team1, team2]):
    #     ## Agrégation des informations pour l'équipe courante
    #     for half in ['1H', '2H']:
    #         # Calcul du temps total de possession de l'équipe courante
    #         pattern = '^Possession_' + ''.join(map(lambda x: '(' + str(x) + ')?', ptr_team['Tous'])) + '_' + half + '$'
    #         tot = infos.filter(regex=pattern, axis=1).sum().sum()

    #         # Calcul du pourcentage de possession pour l'équipe
    #         ptr_res['(Possession_Équipe_' + half + ')'] = round(tot / tot_all * 100., 2)

    #         # Boucle sur les différents postes
    #         for t in ['Attaquants', 'Milieux', 'Défenseurs', 'Gardiens']:
    #             pattern = '^Possession_' + ''.join(map(lambda x: '(' + str(x) + ')?', ptr_team[t])) + '_' + half + '$'
    #             ptr_res['(Possession_' + t + '_' + half + ')'] = round(infos.filter(regex=pattern, axis=1).sum().sum() / tot * 100., 2)
    
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


###############################################################################
## Fonction : Exploit_infos                                                  ##
## Description :                                                             ##
##                                                                           ##
## Arguments :                                                               ##
## Complexité :                                                              ##
###############################################################################
def Exploit_infos(df):
    """
    Cette fonction permet de d'exploiter la table « match_infos » afin de créer
    de nouvelles variables explicatives.
    
    Arguments :
     - df : la table « match_infos ».
    
    Prérequis : « df » doit être trié sur la colonne « Date » (ordre croissant).
    """
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
    
    ## Les nouvelles variables explicatives
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
    
    ## Boucle sur l'ensemble des lignes
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
        
        ## Calcul des nouvelles variables explicatives
        
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


###############################################################################
## Fonction : Build_tree                                                     ##
## Description :                                                             ##
##                                                                           ##
## Arguments :                                                               ##
## Complexité :                                                              ##
###############################################################################
def Build_tree(export):
    """
    TODO: ajouter description...
    """
    
    ## FONCTION SUPPLÉMENTAIRE
    
    def fill_node(text_node, text_parent):
        # Récupération du numéro du noeud parent
        parent = int(re.match(r'(?P<Parent>.*) ->.*', text_parent).group('Parent'))
        
        # Récupération des informations sur le noeud
        pattern = r'(?P<Numéro>\d+) \[label=\"(?P<Question>.*);gini = (?P<gini>.*);samples = (?P<samples>.*);value = \[(?P<val1>\d+), (?P<val2>\d+)\]'
        elts = re.match(pattern, text_node)
        
        # Le noeud est une feuille
        if elts == None:
            pattern = r'(?P<Numéro>\d+) \[label=\"gini = (?P<gini>.*);samples = (?P<samples>.*);value = \[(?P<val1>\d+), (?P<val2>\d+)\]'
            elts = re.match(pattern, text_node)
            
            return {'Type'    : 'Leaf',
                    'Numéro'  : int(elts.group('Numéro')),
                    'Parent'  : parent,
                    'Niveau'  : 0,
                    'gini'    : float(elts.group('gini')),
                    'samples' : int(elts.group('samples')),
                    'value'   : [int(elts.group('val1')), int(elts.group('val2'))]}
        
        # Le noeud N'est PAS une feuille
        else:
            pattern = r'(?P<Variable>.*) <= (?P<Valeur>-?.*)'
            question = re.match(pattern, elts.group('Question'))
            
            return {'Type'    : 'Node',
                    'Numéro'  : int(elts.group('Numéro')),
                    'Parent'  : parent,
                    'Niveau'  : 0, # Valeur par défaut (recalculé par la suite)
                    'Variable': question.group('Variable'),
                    'Valeur'  : float(question.group('Valeur')),
                    'gini'    : float(elts.group('gini')),
                    'samples' : int(elts.group('samples')),
                    'value'   : [int(elts.group('val1')), int(elts.group('val2'))]}
    
    ## CORPS DE LA PROCÉDURE
    
    # Récupération des informations sur le premier noeud
    tree = [fill_node(export[1].strip().replace('\\n', ';'), '-1 -> 0')]
    
    # Boucle sur les autres noeuds
    for i in range(2, len(export) - 1, 2):
        # Récupération des informations sur le noeud courant
        current_node = fill_node(export[i].strip().replace('\\n', ';'), export[i + 1].strip())
        
        # Calcul du niveau du noeud courant
        current_node['Niveau'] = tree[current_node['Parent']]['Niveau'] + 1
        
        # Ajout du noeud courant dans la liste
        tree.append(current_node)
    
    return tree
