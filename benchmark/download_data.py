#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de téléchargement du dataset SIFT1M

Usage:
    python download_sift1m.py

Le script télécharge et extrait automatiquement les fichiers SIFT1M
dans le dossier data/
"""

import urllib.request
import tarfile
import os
import sys
from pathlib import Path

# Fix pour l'encodage Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configuration
DATA_DIR = Path("data")
SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
ARCHIVE_PATH = DATA_DIR / "sift.tar.gz"

# Fichiers attendus après extraction
EXPECTED_FILES = [
    "sift_base.fvecs",
    "sift_query.fvecs",
    "sift_groundtruth.ivecs",
    "sift_learn.fvecs"
]

def download_with_progress(url, destination):
    """
    Télécharge un fichier avec barre de progression.

    Args:
        url (str): URL du fichier à télécharger
        destination (Path): Chemin de destination
    """
    print(f"📥 Téléchargement depuis : {url}")
    print(f"📁 Destination : {destination}")

    def reporthook(blocknum, blocksize, totalsize):
        """Callback pour afficher la progression."""
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 100 / totalsize
            s = f"\r   Progression: {percent:5.1f}% ({readsofar:,} / {totalsize:,} bytes)"
            sys.stderr.write(s)
            if readsofar >= totalsize:
                sys.stderr.write("\n")
        else:
            sys.stderr.write(f"\r   Téléchargé: {readsofar:,} bytes")

    try:
        urllib.request.urlretrieve(url, destination, reporthook)
        print("✅ Téléchargement terminé !")
        return True
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        return False


def extract_archive(archive_path, extract_to):
    """
    Extrait une archive tar.gz.

    Args:
        archive_path (Path): Chemin de l'archive
        extract_to (Path): Dossier d'extraction
    """
    print(f"\n📦 Extraction de l'archive...")
    print(f"   Archive : {archive_path}")
    print(f"   Destination : {extract_to}")

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # Extraire tous les fichiers
            tar.extractall(path=extract_to)

        print("✅ Extraction terminée !")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction : {e}")
        return False


def move_files_to_root(data_dir):
    """
    Déplace les fichiers depuis data/sift/ vers data/ si nécessaire.

    Args:
        data_dir (Path): Dossier data
    """
    sift_subdir = data_dir / "sift"

    if sift_subdir.exists():
        print(f"\n📂 Déplacement des fichiers vers {data_dir}...")

        for filename in EXPECTED_FILES:
            source = sift_subdir / filename
            destination = data_dir / filename

            if source.exists():
                if destination.exists():
                    print(f"   ⚠️  {filename} existe déjà, ignore")
                else:
                    source.rename(destination)
                    print(f"   ✓ {filename} déplacé")
            else:
                print(f"   ⚠️  {filename} non trouvé dans {sift_subdir}")

        # Nettoyer le dossier sift/ si vide
        try:
            if not any(sift_subdir.iterdir()):
                sift_subdir.rmdir()
                print(f"   ✓ Dossier {sift_subdir} supprimé (vide)")
        except:
            pass


def verify_files(data_dir):
    """
    Vérifie que tous les fichiers nécessaires sont présents.

    Args:
        data_dir (Path): Dossier data

    Returns:
        bool: True si tous les fichiers sont présents
    """
    print(f"\n🔍 Vérification des fichiers dans {data_dir}...")

    all_present = True
    for filename in EXPECTED_FILES[:3]:  # On vérifie seulement les 3 premiers (les essentiels)
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✅ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {filename} manquant")
            all_present = False

    return all_present


def main():
    """Fonction principale."""
    print("=" * 70)
    print("TÉLÉCHARGEMENT DU DATASET SIFT1M")
    print("=" * 70)

    # Créer le dossier data s'il n'existe pas
    DATA_DIR.mkdir(exist_ok=True)
    print(f"\n✓ Dossier {DATA_DIR} prêt")

    # Vérifier si les fichiers existent déjà
    if verify_files(DATA_DIR):
        print("\n✅ Tous les fichiers SIFT1M sont déjà présents !")
        print("   Aucun téléchargement nécessaire.")
        return 0

    # Télécharger l'archive
    print(f"\n📥 Début du téléchargement...")
    print(f"   Taille : ~161 MB (compressé)")
    print(f"   Temps estimé : 2-5 minutes selon votre connexion")

    if not download_with_progress(SIFT_URL, ARCHIVE_PATH):
        print("\n❌ Échec du téléchargement.")
        print("   Vous pouvez télécharger manuellement depuis :")
        print(f"   {SIFT_URL}")
        return 1

    # Extraire l'archive
    if not extract_archive(ARCHIVE_PATH, DATA_DIR):
        print("\n❌ Échec de l'extraction.")
        return 1

    # Déplacer les fichiers vers data/
    move_files_to_root(DATA_DIR)

    # Vérification finale
    if verify_files(DATA_DIR):
        print("\n" + "🎉" * 35)
        print("✅ DATASET SIFT1M INSTALLÉ AVEC SUCCÈS !")
        print("🎉" * 35)

        # Nettoyer l'archive
        if ARCHIVE_PATH.exists():
            ARCHIVE_PATH.unlink()
            print(f"\n✓ Archive {ARCHIVE_PATH.name} supprimée (économie d'espace)")

        print("\n📊 Vous pouvez maintenant exécuter les notebooks dans benchmark/")
        print("   hnsw_benchmark.ipynb, lsh_benchmark.ipynb, ivfpq_benchmark.ipynb, pca_benchmark.ipynb")
        return 0
    else:
        print("\n⚠️  Installation incomplète. Certains fichiers sont manquants.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Téléchargement interrompu par l'utilisateur.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
