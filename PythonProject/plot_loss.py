import pandas as pd
import matplotlib.pyplot as plt


def generate_loss_plot(csv_path, output_image):
    try:
        # 1. Carica i dati dal file CSV
        df = pd.read_csv(csv_path)

        # 2. Configura l'estetica del grafico
        plt.figure(figsize=(10, 6))
        if 'Train_Loss' in df.columns:
            plt.plot(df['Epoch'], df['Train_Loss'],
                     marker='o',
                     linestyle='-',
                     color='#1f77b4',
                     linewidth=2,
                     markersize=6,
                     label='Train Loss')

        if 'Val_Loss' in df.columns:
            plt.plot(df['Epoch'], df['Val_Loss'],
                     marker='s',
                     linestyle='--',
                     color='#d62728',
                     linewidth=2,
                     markersize=6,
                     label='Val Loss')

        if 'Train_Loss' not in df.columns and 'Loss' in df.columns:
            plt.plot(df['Epoch'], df['Loss'],
                     marker='o',
                     linestyle='-',
                     color='#1f77b4',
                     linewidth=2,
                     markersize=6,
                     label='Loss')

        # 3. Aggiungi titoli e label
        plt.title('Andamento della Loss di Training (Margin 0.4)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoca', fontsize=12)
        plt.ylabel('Loss', fontsize=12)

        # 4. Griglia e Legenda
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=11)

        # 5. Annotazione automatica dell'ultimo valore (opzionale ma utile)
        last_epoch = df['Epoch'].iloc[-1]
        loss_column = 'Train_Loss' if 'Train_Loss' in df.columns else 'Loss'
        last_loss = df[loss_column].iloc[-1]
        plt.annotate(f'Ultima {loss_column}: {last_loss:.4f}',
                 xy=(last_epoch, last_loss),
                 xytext=(last_epoch - 4, last_loss + 0.01),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=10, fontweight='bold', color='red')

        # 6. Salva l'immagine
        plt.tight_layout()
        plt.savefig(output_image, dpi=300)  # Alta risoluzione per la stampa
        plt.show()
        print(f"✅ Grafico salvato con successo in: {output_image}")

    except Exception as e:
        print(f"❌ Si è verificato un errore: {e}")


# Esecuzione
if __name__ == "__main__":
    # Assicurati che il nome del file CSV sia corretto
    generate_loss_plot('training_log_margin04_E15_LR_decay_aug.csv', 'loss_plot_margin04_E15_LR_decay_aug.png')