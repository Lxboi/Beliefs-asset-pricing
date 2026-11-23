import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class AnalyseurSentimentFinancier:
    """
    Pipeline complet :
    1. Nettoyage lexical et scoring via Loughran & McDonald (2011)
    2. Calcul du sentiment global par actif et de la dispersion (désaccord)
    3. Clustering des articles en classes d'investisseurs (alpha)
    4. Estimation empirique des matrices de transition Q^alpha (Harrison & Kreps)
    """

    def __init__(self, chemin_dictionnaire_lm: Optional[str] = None):
        """
        Charge le dictionnaire officiel Loughran & McDonald (CSV).
        Si aucun chemin n'est fourni, charge un mini-lexique de démonstration.
        """
        if chemin_dictionnaire_lm and Path(chemin_dictionnaire_lm).exists():
            self.lexique = self._charger_dictionnaire_lm(chemin_dictionnaire_lm)
            print(f"Dictionnaire Loughran & McDonald chargé : {len(self.lexique['positive'])} positifs, "
                  f"{len(self.lexique['negative'])} négatifs, {len(self.lexique['uncertainty'])} incertitude.")
        else:
            print("Aucun fichier CSV fourni : chargement du mini-lexique LM de démonstration.")
            self.lexique = self._lexique_demo()

    @staticmethod
    def _charger_dictionnaire_lm(chemin_csv: str) -> Dict[str, Set[str]]:
        """
        Lit le CSV officiel Loughran-McDonald_MasterDictionary.
        Note : dans le CSV officiel, une valeur > 0 indique l'année d'ajout du mot
        (une valeur négative indique un mot retiré lors d'une révision).
        """
        df_lm = pd.read_csv(chemin_csv)
        df_lm["Word"] = df_lm["Word"].astype(str).str.upper().str.strip()

        return {
            "positive": set(df_lm.loc[df_lm["Positive"] > 0, "Word"]),
            "negative": set(df_lm.loc[df_lm["Negative"] > 0, "Word"]),
            "uncertainty": set(df_lm.loc[df_lm["Uncertainty"] > 0, "Word"]),
            "litigious": set(df_lm.loc[df_lm["Litigious"] > 0, "Word"]),
        }

    @staticmethod
    def _lexique_demo() -> Dict[str, Set[str]]:
        """Sous-ensemble fidèle au lexique Loughran-McDonald pour tester le code immédiatement."""
        return {
            "positive": {
                "STRONG", "PROFIT", "PROFITABLE", "GAIN", "GAINS", "GROWTH", "IMPROVE",
                "IMPROVED", "SUCCESS", "POSITIVE", "OUTPERFORM", "SURGE", "BULLISH",
                "REBOUND", "OPPORTUNITY", "DIVIDEND", "INCREASE", "STABLE", "CONFIDENT",
                "RECORD", "EXPAND", "EXPANSION", "RESILIENT", "BENEFIT", "FAVORABLE",
            },
            "negative": {
                "LOSS", "LOSSES", "DECLINE", "DECLINING", "CRASH", "RECESSION", "DEFICIT",
                "BANKRUPTCY", "DEFAULT", "WEAK", "WEAKNESS", "DROP", "PLUNGE", "BEARISH",
                "DOWNTURN", "FAIL", "FAILURE", "RISK", "RISKS", "VOLATILE", "VOLATILITY",
                "DEBT", "PENALTY", "LAWSUIT", "FRAUD", "CONCERN", "NEGATIVE", "SELLOFF",
                "SEVERE", "SLUMP", "WORSE", "DETERIORATE", "COLLAPSE",
            },
            "uncertainty": {
                "UNCERTAIN", "UNCERTAINTY", "VOLATILE", "VOLATILITY", "RISK", "RISKS",
                "UNPREDICTABLE", "MAY", "MIGHT", "COULD", "POSSIBLE", "SPECULATION",
                "SPECULATIVE", "DOUBT", "UNUSUAL", "FLUCTUATE", "VARIABLE", "UNKNOWN",
            },
            "litigious": {
                "LAWSUIT", "LITIGATION", "REGULATORY", "INVESTIGATION", "BAN", "PENALTY",
                "SANCTION", "COURT", "LEGAL", "COMPLIANCE",
            },
        }

    @staticmethod
    def nettoyer_et_tokeniser(texte: str) -> list[str]:
        """Extrait uniquement les mots alphabétiques (en majuscules pour matcher LM)."""
        if not isinstance(texte, str):
            return []
        return re.findall(r"\b[A-Za-z]{2,}\b", texte.upper())

    def scorer_article(self, texte: str) -> Dict[str, float]:
        """
        Calcule les métriques lexicales d'un article :
        Sentiment = (#positifs - #négatifs) / #mots_totaux
        """
        tokens = self.nettoyer_et_tokeniser(texte)
        nb_totaux = len(tokens)

        if nb_totaux == 0:
            return {
                "nb_mots": 0,
                "nb_pos": 0,
                "nb_neg": 0,
                "nb_incertitude": 0,
                "ratio_pos": 0.0,
                "ratio_neg": 0.0,
                "ratio_incertitude": 0.0,
                "sentiment_score": 0.0,
            }

        nb_pos = sum(1 for mot in tokens if mot in self.lexique["positive"])
        nb_neg = sum(1 for mot in tokens if mot in self.lexique["negative"])
        nb_inc = sum(1 for mot in tokens if mot in self.lexique["uncertainty"])

        return {
            "nb_mots": nb_totaux,
            "nb_pos": nb_pos,
            "nb_neg": nb_neg,
            "nb_incertitude": nb_inc,
            "ratio_pos": nb_pos / nb_totaux,
            "ratio_neg": nb_neg / nb_totaux,
            "ratio_incertitude": nb_inc / nb_totaux,
            "sentiment_score": (nb_pos - nb_neg) / nb_totaux,
        }

    def traiter_corpus(self, df_articles: pd.DataFrame, col_texte: str = "texte_brut") -> pd.DataFrame:
        """
        Applique le scoring à l'ensemble d'un DataFrame d'articles.
        Colonnes attendues : ['date', 'actif_cible', 'source', 'texte_brut']
        """
        df = df_articles.copy()
        df["date"] = pd.to_datetime(df["date"])

        scores = df[col_texte].apply(self.scorer_article)
        df_scores = pd.DataFrame(scores.tolist(), index=df.index)

        return pd.concat([df, df_scores], axis=1)

    def calculer_sentiment_global(
        self,
        df_scores: pd.DataFrame,
        frequence: str = "D",
    ) -> pd.DataFrame:
        """
        Agrège le sentiment global du marché par actif et par période ('D' = jour, 'W' = semaine).
        Calcule aussi l'écart-type des scores (proxy direct du désaccord entre articles).
        """
        df_global = (
            df_scores.set_index("date")
            .groupby("actif_cible")
            .resample(frequence)
            .agg(
                nb_articles=("sentiment_score", "count"),
                sentiment_global=("sentiment_score", "mean"),
                desaccord_std=("sentiment_score", "std"),
                incertitude_moyenne=("ratio_incertitude", "mean"),
            )
            .reset_index()
        )
        # Supprime les périodes sans aucun article et remplace les std NaN (1 seul article) par 0
        df_global = df_global[df_global["nb_articles"] > 0].copy()
        df_global["desaccord_std"] = df_global["desaccord_std"].fillna(0.0)
        return df_global

    def clusteriser_investisseurs(
        self,
        df_scores: pd.DataFrame,
        n_clusters: int = 2,
        par_actif: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Regroupe les articles en n_clusters catégories d'individus (ex: 2 classes pour Harrison & Kreps).
        Les labels sont triés par sentiment moyen croissant :
        Classe 0 = la plus pessimiste/prudente, Classe (n-1) = la plus optimiste.
        """
        colonnes_features = ["sentiment_score", "ratio_pos", "ratio_neg", "ratio_incertitude"]
        df = df_scores.copy()

        def _appliquer_kmeans(sous_df: pd.DataFrame) -> pd.DataFrame:
            if len(sous_df) < n_clusters:
                sous_df["classe_alpha"] = 0
                return sous_df

            X = StandardScaler().fit_transform(sous_df[colonnes_features])
            modele = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            sous_df["cluster_brut"] = modele.fit_predict(X)

            # Réordonnancement déterministe du plus pessimiste (0) au plus optimiste (n-1)
            ordre = (
                sous_df.groupby("cluster_brut")["sentiment_score"]
                .mean()
                .sort_values()
                .index
            )
            mapping = {ancien: nouveau for nouveau, ancien in enumerate(ordre)}
            sous_df["classe_alpha"] = sous_df["cluster_brut"].map(mapping)
            return sous_df.drop(columns=["cluster_brut"])

        if par_actif:
         groupes_traites = [
             _appliquer_kmeans(sous_df.copy())
             for _, sous_df in df.groupby("actif_cible", sort=False)
         ]
         df = pd.concat(groupes_traites).sort_index()
        else:
         df = _appliquer_kmeans(df)

        # Comparaison : sentiment par cluster vs sentiment global du marché
        resume_clusters = (
            df.groupby(["actif_cible", "classe_alpha"])
            .agg(
                nb_articles=("sentiment_score", "count"),
                sentiment_moyen_cluster=("sentiment_score", "mean"),
                incertitude_moyenne=("ratio_incertitude", "mean"),
            )
            .reset_index()
        )
        moyenne_globale = (
            df.groupby("actif_cible")["sentiment_score"]
            .mean()
            .rename("sentiment_global_marche")
        )
        resume_clusters = resume_clusters.merge(moyenne_globale, on="actif_cible")
        resume_clusters["ecart_vs_marche"] = (
            resume_clusters["sentiment_moyen_cluster"] - resume_clusters["sentiment_global_marche"]
        )

        return df, resume_clusters

    @staticmethod
    def estimer_matrices_transition(
        df_clustered: pd.DataFrame,
        actif: str,
        col_etat_marche: str = "etat_marche",
        lissage_laplace: float = 1.0,
    ) -> Dict[int, np.ndarray]:
        """
        Estime les matrices de transition 2x2 Q^alpha pour le modèle de Harrison & Kreps :
        q^alpha(i, j) = probabilité attribuée par la classe alpha de passer à l'état j
        sachant que l'état actuel observé du marché est i (0 = défavorable, 1 = favorable).

        Proxy empirique : proportion d'articles optimistes (sentiment > 0) dans la classe alpha
        lorsque le marché se trouve dans l'état i, avec lissage de Laplace pour éviter les 0/1 stricts.
        """
        sous_df = df_clustered[df_clustered["actif_cible"] == actif]
        matrices = {}

        for classe, groupe in sous_df.groupby("classe_alpha"):
            Q = np.zeros((2, 2))
            for etat_actuel in [0, 1]:
                subset = groupe[groupe[col_etat_marche] == etat_actuel]
                n_total = len(subset)
                n_optimistes = int((subset["sentiment_score"] > 0).sum())

                # Probabilité de transition vers l'état 1 (favorable) avec lissage de Laplace
                q_i1 = (n_optimistes + lissage_laplace) / (n_total + 2 * lissage_laplace)
                q_i0 = 1.0 - q_i1

                Q[etat_actuel, 0] = round(q_i0, 4)
                Q[etat_actuel, 1] = round(q_i1, 4)

            matrices[int(classe)] = Q

        return matrices


# ==============================================================================
# DÉMONSTRATION EXÉCUTABLE IMMÉDIATEMENT
# ==============================================================================
if __name__ == "__main__":
    # 1. Jeu de données d'exemple simulant des articles datés (en anglais pour LM)
    donnees_exemple = [
        # --- S&P 500 ---
        ("2025-01-10", "SP500", "Reuters", "Strong corporate profit and stable dividend growth improve outlook.", 1),
        ("2025-01-10", "SP500", "Reddit", "Recession risk and severe market volatility could trigger a crash.", 1),
        ("2025-01-11", "SP500", "Bloomberg", "Companies outperform forecasts with record expansion and resilient earnings.", 1),
        ("2025-01-11", "SP500", "Blog", "Weak consumer demand and rising debt create uncertainty and decline concerns.", 0),
        ("2025-01-12", "SP500", "WSJ", "Favorable earnings rebound supports positive investor sentiment.", 0),
        ("2025-01-12", "SP500", "X", "Selloff continues amid lawsuit penalties and deteriorating margins.", 0),
        # --- Bitcoin ---
        ("2025-01-10", "BTC", "CoinDesk", "Bitcoin surge continues as bullish institutional adoption drives record gains.", 1),
        ("2025-01-10", "BTC", "FT", "Speculative bubble risk and regulatory investigation threaten collapse.", 1),
        ("2025-01-11", "BTC", "X", "Strong opportunity for massive profit and expansion in crypto markets.", 1),
        ("2025-01-11", "BTC", "Reuters", "Extreme volatility, fraud concerns and unpredictable plunge worry analysts.", 0),
        ("2025-01-12", "BTC", "Reddit", "Positive rebound expected despite short term uncertainty and fluctuation.", 0),
        ("2025-01-12", "BTC", "Bloomberg", "Bearish downturn and bankruptcy fears accelerate crypto losses.", 0),
    ]

    df_brut = pd.DataFrame(
        donnees_exemple,
        columns=["date", "actif_cible", "source", "texte_brut", "etat_marche"],
    )

    # 2. Initialisation (remplacez None par "data/Loughran-McDonald_MasterDictionary_1993-2024.csv")
    analyseur = AnalyseurSentimentFinancier(
    chemin_dictionnaire_lm="data/Loughran-McDonald_MasterDictionary_1993-2025.csv"
)

    # 3. Calcul des scores par article
    df_scores = analyseur.traiter_corpus(df_brut)

    # 4. Sentiment global quotidien + mesure du désaccord
    df_marche = analyseur.calculer_sentiment_global(df_scores, frequence="D")
    print("\n--- SENTIMENT GLOBAL DU MARCHÉ (PAR JOUR ET ACTIF) ---")
    print(df_marche.to_string(index=False))

    # 5. Clustering en 2 catégories d'individus (Pessimistes = 0, Optimistes = 1)
    df_clustered, resume = analyseur.clusteriser_investisseurs(df_scores, n_clusters=2, par_actif=True)
    print("\n--- COMPARAISON CLUSTERS VS SENTIMENT GLOBAL ---")
    print(resume.to_string(index=False))

    # 6. Estimation des matrices Q^alpha pour Harrison & Kreps
    for actif in ["SP500", "BTC"]:
        matrices = analyseur.estimer_matrices_transition(df_clustered, actif=actif)
        print(f"\n--- MATRICES DE TRANSITION ESTIMÉES Q^alpha ({actif}) ---")
        for classe, Q in matrices.items():
            print(f"Classe {classe} :\n{Q}")
        