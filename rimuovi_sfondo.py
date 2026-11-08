import os
import sys
import subprocess
import torch
import numpy as np
import cv2
import argparse
from sam2.build_sam import build_sam2_video_predictor

def main():
    parser = argparse.ArgumentParser(description="Rimozione sfondo video con SAM 2")
    parser.add_argument("--video", type=str, required=True, help="Percorso del video di input (es. vacanza3.mp4)")
    args = parser.parse_args()

    VIDEO = args.video
    if not os.path.exists(VIDEO):
        print(f"Errore: Il file video '{VIDEO}' non esiste.")
        return

    # 1. Download del checkpoint se mancante
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = 'checkpoints/sam2.1_hiera_base_plus.pt'
    if not os.path.exists(checkpoint_path):
        print("Download del checkpoint di SAM 2.1...")
        subprocess.run(["wget", "-q", "-O", checkpoint_path, "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"], check=True)

    # 2. Estrazione fotogrammi con ffmpeg
    os.makedirs('frames', exist_ok=True)
    print("Estrazione dei fotogrammi in corso...")
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', VIDEO, '-q:v', '2', '-start_number', '0', 'frames/%05d.jpg'], check=True)

    # Recupero FPS
    r = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', VIDEO]).decode().strip()
    a, b = r.split('/')
    FPS = float(a) / float(b)
    N = len(os.listdir('frames'))
    print(f"Estratti {N} fotogrammi a {FPS:.2f} FPS.")

    # 3. Inizializzazione SAM 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilizzo dispositivo: {device}")
    if device == "cpu":
        print("ATTENZIONE: Stai usando la CPU. L'elaborazione potrebbe essere estremamente lenta!")

    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
    inference_state = predictor.init_state(video_path="frames")

    # Definisci qui le tue coordinate reali per il tracciamento
    PROMPTS = {
        0: {
            'pos': [[250, 450]], 
            'neg': [[10, 10]]
        }
    }

    predictor.reset_state(inference_state)
    for frame_idx, pt in PROMPTS.items():
        points = np.array(pt['pos'] + pt.get('neg', []), dtype=np.float32)
        labels = np.array([1]*len(pt['pos']) + [0]*len(pt.get('neg', [])), dtype=np.int32)
        predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=frame_idx, obj_id=1, points=points, labels=labels)

    # 4. Tracciamento
    os.makedirs('masks', exist_ok=True)
    print("Avvio tracking in corso...")
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # 5. Salvataggio maschere
    print("Salvataggio delle maschere binarie...")
    h, w = cv2.imread('frames/00000.jpg').shape[:2]
    for frame_idx in range(N):
        mask = np.zeros((h, w), dtype=np.uint8)
        if frame_idx in video_segments:
            for obj_id, obj_mask in video_segments[frame_idx].items():
                mask = np.maximum(mask, (obj_mask[0] * 255).astype(np.uint8))
        cv2.imwrite(f'masks/{frame_idx:05d}.png', mask)

    # 6. Applicazione Trasparenza
    print("Creazione fotogrammi RGBA con trasparenza...")
    FEATHER = 1.0
    os.makedirs('rgba', exist_ok=True)
    for i in range(N):
        img = cv2.imread(f'frames/{i:05d}.jpg')
        m = cv2.imread(f'masks/{i:05d}.png', 0)
        if FEATHER > 0:
            m = cv2.GaussianBlur(m, (0, 0), FEATHER)
        cv2.imwrite(f'rgba/{i:05d}.png', np.dstack([img, m]))

    # 7. Generazione video finale con FFMPEG
    print("Esportazione video trasparenti...")
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
        '-i', 'rgba/%05d.png', '-i', VIDEO,
        '-map', '0:v', '-map', '1:a?', '-c:v', 'libvpx-vp9',
        '-pix_fmt', 'yuva420p', '-b:v', '0', '-crf', '22', 'video_trasparente.webm'
    ], check=True)

    print("Processo completato! Troverai i file 'video_trasparente.webm' nella cartella corrente.")

if __name__ == "__main__":
    main()
