#!/usr/bin/env bash
# ==============================================================================
# EvoScan - Hugging Face Spaces Deployment Script
# ==============================================================================
# Questo script contiene i passaggi per configurare ed eseguire il deploy
# dell'applicazione Streamlit EvoScan su Hugging Face Spaces.
#
# PREREQUISITI:
# 1. Crea un account su Hugging Face: https://huggingface.co
# 2. Crea un nuovo Space:
#    - Space Name: EvoScan (o il nome desiderato)
#    - License: MIT
#    - SDK: Streamlit (Python 3.10+)
#    - Hardware: CPU basic (gratuita) o GPU T4 per inferenza accelerata
# 3. Genera un Access Token con permessi di 'Write':
#    https://huggingface.co/settings/tokens
# ==============================================================================

set -e

# --- CONFIGURAZIONE ---
# Sostituisci con il tuo username Hugging Face e il nome dello Space
HF_USERNAME="YOUR_HF_USERNAME"
SPACE_NAME="EvoScan"

echo "🧬 Preparazione al Deploy di EvoScan su Hugging Face Spaces..."

if [ "$HF_USERNAME" = "YOUR_HF_USERNAME" ]; then
    echo "⚠️  ATTENZIONE: Modifica prima la variabile HF_USERNAME con il tuo username Hugging Face."
    echo ""
    echo "Passaggi manuali da eseguire:"
    echo "1. git remote add space https://huggingface.co/spaces/<TUO_USERNAME>/$SPACE_NAME"
    echo "2. git push --force space main"
    exit 1
fi

SPACE_REPO_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

echo "🔗 Configurazione remote 'space': $SPACE_REPO_URL"

# Rimuovi il remote esistente se già presente
if git remote | grep -q "^space$"; then
    git remote remove space
fi

# Aggiungi il remote Hugging Face
git remote add space "$SPACE_REPO_URL"

echo "🚀 Esecuzione push su Hugging Face Space ($SPACE_REPO_URL)..."
echo "ℹ️  Se richiesta la password, inserisci il tuo Hugging Face User Access Token (Write)."

git push --force space main

echo "✅ Deploy completato con successo!"
echo "🌐 Visita la tua app all'indirizzo: https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
