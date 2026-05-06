import pandas as pd
import matplotlib.pyplot as plt


def generate_loss_plot(csv_path, output_image):
    try:
        # 1. Carica i dati dal file CSV
        df = pd.read_csv(csv_path)

        # 2. Configura l'estetica del grafico
        plt.figure(figsize=(10, 6))
        plt.plot(df['Epoch'], df['Loss'],
                 marker='o',  # Pallino su ogni epoca
                 linestyle='-',  # Linea continua
                 color='#1f77b4',  # Blu professionale
                 linewidth=2,
                 markersize=6,
                 label='Training Loss')

        # 3. Aggiungi titoli e label
        plt.title('Andamento della Training Loss (Margin 0.2)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoca', fontsize=12)
        plt.ylabel('Loss', fontsize=12)

        # 4. Griglia e Legenda
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=11)

        # 5. Annotazione automatica dell'ultimo valore (opzionale ma utile)
        last_epoch = df['Epoch'].iloc[-1]
        last_loss = df['Loss'].iloc[-1]
        plt.annotate(f'Ultima Loss: {last_loss:.4f}',
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
    generate_loss_plot('training_log_margin02.csv', 'loss_plot_margin02.png')