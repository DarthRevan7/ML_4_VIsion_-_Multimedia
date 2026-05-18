import os
import sys
import glob
import re
from datetime import datetime
import pandas as pd

# Assicuriamo che la cartella corrente sia nel path di sistema
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Importiamo il modulo originale (eseguirà solo le inizializzazioni top-level)
import evaluate


def format_param(s):
    """
    Ripristina il punto decimale se la stringa inizia con 0 e non lo contiene.
    Es: '05' -> '0.5', '00025' -> '0.00025'
    """
    if s.startswith('0') and '.' not in s:
        return f"0.{s[1:]}"
    return s


def parse_filename(filename):
    """
    Estrae i tre iperparametri fondamentali usando le espressioni regolari.
    """
    margin_match = re.search(r'(?:margin|M)([0-9.]+)', filename, re.IGNORECASE)
    epochs_match = re.search(r'E([0-9]+)', filename, re.IGNORECASE)
    lr_match = re.search(r'LR([0-9.]+)', filename, re.IGNORECASE)
    
    margin_str = format_param(margin_match.group(1)) if margin_match else "0.5"
    epochs_str = epochs_match.group(1) if epochs_match else "5"
    lr_str = format_param(lr_match.group(1)) if lr_match else "0.001"
    
    return epochs_str, margin_str, lr_str


def main():
    # Creiamo la cartella dei risultati se non esiste
    os.makedirs("results", exist_ok=True)
    
    # 1. Raccolta dei file dei modelli (.pth)
    models_root = glob.glob(os.path.join("models", "*.pth"))
    models_olds = glob.glob(os.path.join("models", "olds", "*.pth"))
    
    if not models_root and not models_olds:
        print("❌ Nessun modello .pth trovato in 'models' o 'models/olds'. Check dei path fallito.")
        return
        
    # STRATEGIA DI TEST RICHIESTA: 
    # Mettiamo in cima alla coda un modello di 'models/olds' e uno di 'models'
    ordered_models = []
    if models_olds:
        ordered_models.append(models_olds[0])
    if models_root:
        ordered_models.append(models_root[0])
        
    # Aggiungiamo tutti gli altri modelli rimanenti evitando duplicati
    for m in models_olds[1:]:
        if m not in ordered_models:
            ordered_models.append(m)
    for m in models_root[1:]:
        if m not in ordered_models:
            ordered_models.append(m)
            
    print(f"📂 Trovati {len(ordered_models)} modelli totali da elaborare.")
    print(f"🔬 I primi due modelli usati per il test provengono da cartelle diverse.")
    
    final_dfs = []
    continue_all = False  # Flag per lo skip dei prompt futuri
    
    # 2. Ciclo di pianificazione dello scheduling
    for idx, model_path in enumerate(ordered_models):
        filename = os.path.basename(model_path)
        print(f"\n" + "="*60)
        print(f"🔄 [{idx+1}/{len(ordered_models)}] Inizio Valutazione Modello: {model_path}")
        print("="*60)
        
        # Estrazione della nomenclatura
        n_epoche, margin, learning_rate = parse_filename(filename)
        print(f"📝 Parametri rilevati -> Epoche: {n_epoche}, Margin: {margin}, LR: {learning_rate}")
        
        # Generazione stringa della prima colonna custom
        custom_col_name = f"Dataset - E{n_epoche} - M{margin} - LR{learning_rate}"
        
        # Definizione path intermedio richiesto: "results/nome_file_estratto_da_modello"
        base_name_no_ext = os.path.splitext(filename)[0]
        intermediate_csv = os.path.join("results", f"final_eval_{base_name_no_ext}.csv")
        
        # 3. MONKEY PATCHING DELLE VARIABILI DI EVALUATE.PY
        evaluate.model_pth = model_path
        evaluate.result_file_path = intermediate_csv
        evaluate.MARGIN = float(margin)
        evaluate.epoche = n_epoche
        
        # Gestione stringhe pulite necessarie a evaluate.py per i grafici CMC interni
        # Estraiamo la porzione di testo grezzo numerico originale per non rompere i path grafici di evaluate.py
        margin_match = re.search(r'(?:margin|M)(([0-9.]+))', filename, re.IGNORECASE)
        lr_match = re.search(r'LR(([0-9.]+))', filename, re.IGNORECASE)
        evaluate.margin_cl = margin_match.group(1) if margin_match else "05"
        evaluate.learning_rate = lr_match.group(1) if lr_match else "00025"
        
        # Esecuzione della valutazione nativa
        try:
            evaluate.run_evaluation()
        except Exception as e:
            print(f"❌ Errore critico durante la valutazione di {filename}: {e}")
            continue
            
        # Post-elaborazione dell'output appena generato per rinominare la prima colonna
        if os.path.exists(intermediate_csv):
            df = pd.read_csv(intermediate_csv)
            if not df.empty and 'Dataset' in df.columns:
                df = df.rename(columns={'Dataset': custom_col_name})
                df.to_csv(intermediate_csv, index=False)
                final_dfs.append(df)
        
        # 4. GESTIONE DEI PROMPT INTERATTIVI (INTERFACCIA UTENTE)
        if idx == 0 and not continue_all:
            ans1 = input("\n❓ Posso continuare fino alla fine? (y/n): ").strip().lower()
            if ans1 == 'y':
                continue_all = True
            else:
                ans2 = input("❓ Posso passare alla prossima valutazione? (y/n): ").strip().lower()
                if ans2 != 'y':
                    print("⏹️ Procedura interrotta dall'utente al primo step.")
                    break
        elif idx > 0 and not continue_all:
            ans2 = input("\n❓ Posso passare alla prossima valutazione? (y/n): ").strip().lower()
            if ans2 != 'y':
                print("⏹️ Procedura interrotta dall'utente.")
                break

    # 5. UNIFICAZIONE DEI RISULTATI NEL FILE FINALE
    if final_dfs:
        print(f"\n📊 Generazione del file unificato finale...")
        
        # Nota di design: dato che ogni dataframe ha la prima colonna con un header *diverso* 
        # (perché include i parametri specifici di quel modello), per fare in modo che "le colonne 
        # dell'ultimo excel siano uguali a quelle dei precedenti" senza generare colonne sfasate,
        # applichiamo un allineamento posizionale resettando temporaneamente i nomi delle colonne.
        
        standardized_dfs = []
        target_columns = ["Configurazione Dataset"] + list(final_dfs[0].columns[1:])
        
        for df_temp in final_dfs:
            df_copy = df_temp.copy()
            df_copy.columns = target_columns
            standardized_dfs.append(df_copy)
            
        final_df = pd.concat(standardized_dfs, axis=0, ignore_index=True)
        
        # Salvataggio con la data di oggi nel formato richiesto
        today_str = datetime.now().strftime("%d_%m_%Y")
        final_csv_path = os.path.join("results", f"final_eval_data_{today_str}.csv")
        final_df.to_csv(final_csv_path, index=False)
        
        print(f"🎉 Processo completato con successo!")
        print(f"💾 File di riepilogo globale salvato in: {final_csv_path}")
    else:
        print("\n❌ Nessun dato raccolto da poter unificare.")


if __name__ == "__main__":
    main()