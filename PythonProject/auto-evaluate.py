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

# Importiamo il modulo originale
import evaluate


def parse_filename_for_label(filename):
    """
    Estrae i token letterali dal nome del file per costruire 
    la stringa esatta richiesta nella terza immagine.
    """
    epochs_match = re.search(r'[eE]([0-9]+)', filename)
    margin_match = re.search(r'(?:margin|M)([0-9.eE+-]+)', filename)
    lr_match = re.search(r'[lL][rR]([0-9.eE+-]*)', filename)
    
    ep = epochs_match.group(1) if epochs_match else ""
    ma = margin_match.group(1) if margin_match else ""
    lr = lr_match.group(1) if lr_match else ""
    
    # Costruiamo la dicitura dinamica adattandoci ai casi reali (es. se LR è vuoto)
    parts = []
    if ep: parts.append(f"E{ep}")
    if ma: parts.append(f"M{ma}")
    parts.append(f"LR{lr}" if lr else "LR")
    
    return " - ".join(["Dataset"] + parts)


def to_float_value(s):
    """
    Converte in float i parametri estratti per non rompere i calcoli interni di evaluate.py
    Es: '05' -> 0.5, '0.4' -> 0.4
    """
    if not s:
        return 0.5
    if s.startswith('0') and '.' not in s and len(s) > 1:
        return float(f"0.{s[1:]}")
    try:
        return float(s)
    except ValueError:
        return 0.5


def main():
    os.makedirs("results", exist_ok=True)
    
    # 1. Raccolta dei file dei modelli (.pth)
    models_root = glob.glob(os.path.join("models", "*.pth"))
    models_olds = glob.glob(os.path.join("models", "olds", "*.pth"))
    
    if not models_root and not models_olds:
        print("❌ Nessun modello .pth trovato. Verifica le cartelle 'models' e 'models/olds'.")
        return
        
    # STRATEGIA DI TEST: Un modello da olds e uno da root in cima alla lista
    ordered_models = []
    if models_olds:
        ordered_models.append(models_olds[0])
    if models_root:
        ordered_models.append(models_root[0])
        
    for m in models_olds[1:]:
        if m not in ordered_models:
            ordered_models.append(m)
    for m in models_root[1:]:
        if m not in ordered_models:
            ordered_models.append(m)
            
    print(f"📂 Trovati {len(ordered_models)} modelli totali da elaborare.")
    
    # Questa lista conterrà tutte le righe (comprese le intestazioni ripetute e le righe vuote)
    final_rows = []
    continue_all = False  
    
    # 2. Ciclo di esecuzione sui modelli
    for idx, model_path in enumerate(ordered_models):
        filename = os.path.basename(model_path)
        print(f"\n" + "="*60)
        print(f"🔄 [{idx+1}/{len(ordered_models)}] Valutazione Modello: {model_path}")
        print("="*60)
        
        # Estrazione della nomenclatura letterale per la tabella
        custom_col_name = parse_filename_for_label(filename)
        print(f"📝 Identificativo generato: {custom_col_name}")
        
        # Estrazione parametri grezzi per il funzionamento interno di evaluate.py
        epochs_match = re.search(r'[eE]([0-9]+)', filename)
        margin_match = re.search(r'(?:margin|M)([0-9.eE+-]+)', filename)
        lr_match = re.search(r'[lL][rR]([0-9.eE+-]*)', filename)
        
        ep_str = epochs_match.group(1) if epochs_match else "5"
        ma_str = margin_match.group(1) if margin_match else "05"
        lr_str = lr_match.group(1) if lr_match else "00025"
        
        # Path intermedio del singolo file csv richiesto
        base_name_no_ext = os.path.splitext(filename)[0]
        intermediate_csv = os.path.join("results", f"final_eval_{base_name_no_ext}.csv")
        
        # 3. MONKEY PATCHING DI EVALUATE.PY
        evaluate.model_pth = model_path
        evaluate.result_file_path = intermediate_csv
        evaluate.epoche = ep_str
        evaluate.margin_cl = ma_str
        evaluate.learning_rate = lr_str
        evaluate.MARGIN = to_float_value(ma_str)
        
        # Esecuzione del processo di valutazione originale
        try:
            evaluate.run_evaluation()
        except Exception as e:
            print(f"❌ Errore durante la valutazione di {filename}: {e}")
            continue
            
        # 4. ACQUISIZIONE E STRUTTURAZIONE DATI (STILE IMMAGINE 3)
        if os.path.exists(intermediate_csv):
            df = pd.read_csv(intermediate_csv)
            if not df.empty:
                # Aggiorniamo prima il file .csv singolo intermedio (Immagine 1)
                if 'Dataset' in df.columns:
                    df = df.rename(columns={'Dataset': custom_col_name})
                    df.to_csv(intermediate_csv, index=False)
                
                # Prepariamo i componenti del blocco per il file finale globale
                other_headers = list(df.columns[1:])
                block_header = [custom_col_name] + other_headers
                
                # Se non è il primo modello in assoluto, inseriamo un rigo vuoto di stacco
                if final_rows:
                    final_rows.append([''] * len(block_header))
                
                # Aggiungiamo la riga d'intestazione del blocco corrente
                final_rows.append(block_header)
                
                # Aggiungiamo le righe dei dati (LogoDet-3K, FlickrLogos-32, ecc.)
                for row in df.values.tolist():
                    final_rows.append(row)
        
        # 5. INTERFACCIA UTENTE (PROMPT INTERATTIVI)
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

    # 6. SALVATAGGIO REPERTORIO FINALE UNIFICATO
    if final_rows:
        print(f"\n📊 Scrittura del file unificato finale...")
        
        # Costruiamo il DataFrame direttamente dalla matrice di righe accumulata
        final_df = pd.DataFrame(final_rows)
        
        today_str = datetime.now().strftime("%d_%m_%Y")
        final_csv_path = os.path.join("results", f"final_eval_data_{today_str}.csv")
        
        # CRUCIALE: Salviamo con header=False e index=False perché le intestazioni di colonna
        # cambiano ad ogni blocco e sono già state iniettate come righe di dati.
        final_df.to_csv(final_csv_path, index=False, header=False)
        
        print(f"🎉 Fatto! Il file con la struttura a blocchi è pronto.")
        print(f"💾 Salvato in: {final_csv_path}")
    else:
        print("\n❌ Nessun dato raccolto.")


if __name__ == "__main__":
    main()