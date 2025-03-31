import cv2
import time
import ollama
import os
import tempfile

def process_frame(frame):
    """Riduce la risoluzione del frame e lo passa a Ollama per l'analisi."""
    # Ridimensiona il frame per ridurre la quantità di dati
    resized_frame = cv2.resize(frame, (1920, 1280))
    
    # Salva il frame in un file temporaneo (di solito in una directory veloce)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image_filename = tmp.name
        cv2.imwrite(image_filename, resized_frame)
    
    if not os.path.exists(image_filename):
        return "Errore: Impossibile salvare il frame.", None

    try:
        start_time = time.time()
        
        # Chiamata al modello Llama3.2 Vision tramite l'API Python di Ollama
        response = ollama.chat(model="llama3.2-vision:latest", messages=[
            {
            "role": "system",
            "content": "You are an expert in photography and art. Analyze the images by describing the composition, lighting, artistic style, and meaning. But you have to give a fast response"
            },
            {
            "role": "user",
            "content": "Describe this image.",
            "images": [image_filename]}
        ])
        
        elapsed_time = time.time() - start_time
    except Exception as e:
        elapsed_time = None
        response = {"message": {"content": f"Errore nell'esecuzione di Ollama: {e}"}}
    
    # Elimina il file temporaneo
    os.remove(image_filename)
    
    return response['message']['content'], elapsed_time

def main():
    """Apre la webcam e analizza un frame ogni 60 secondi."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossibile aprire la webcam.")
        return
    
    print("Avvio della demo. Premere Ctrl+C per terminare.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Errore nell'acquisizione del frame dalla webcam.")
                break
            
            # Messaggio che indica l'acquisizione e salvataggio del frame
            print("Frame salvato")
            
            # Analizza il frame in modo sincrono
            output, elapsed_time = process_frame(frame)
            print("Output IA:", output)
            if elapsed_time is not None:
                print(f"Tempo di elaborazione: {elapsed_time:.2f} secondi")
            
            # Aspetta 60 secondi prima di catturare il prossimo frame
            time.sleep(60)
    
    except KeyboardInterrupt:
        print("\nDemo terminata dall'utente.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
