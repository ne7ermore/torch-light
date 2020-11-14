import os
import pickle
import librosa

from encode import *

def preprocess_training_dataset(
    data_dir="wavs",
    output="data/datasets.pkl"):

    embedding = pickle.load(open("data/embedding.pt", "rb"))
    cfg = get_config("pretrained_model/config.yml")

    mel_bias_t = get_mel_bias_transpose(cfg["sampling_rate"])
    b, a = butter_highpass(30, cfg["sampling_rate"])

    datas = []
    dirName, subdirList, _ = next(os.walk(data_dir))

    for subdir in sorted(subdirList):
        _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
        spk_emb = embedding[subdir]

        for fileName in sorted(fileList):
            x, _ = librosa.load(os.path.join(dirName,subdir,fileName), sr=cfg["sampling_rate"], mono=True)
            wav = signal.filtfilt(b, a, x)

            D = pySTFT(wav).T
            D_mel = np.dot(D, mel_bias_t)
            D_db = normalize(amp_to_db(D_mel))

            f0_norm = speaker_f0(wav, cfg["sampling_rate"])
            f0_norm = np.clip(f0_norm, 0, 1)            

            assert len(f0_norm) == len(D_db)

            datas.append([spk_emb, D_db, f0_norm])

    with open(output, "wb") as handle:
        pickle.dump(datas, handle)

if __name__ == "__main__":
    preprocess_training_dataset()