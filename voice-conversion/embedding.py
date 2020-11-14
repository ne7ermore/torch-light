from pathlib import Path
from itertools import groupby
import os
import pickle

import torch
import numpy as np
from tqdm import tqdm

from speaker_embedding import VoiceEncoder, preprocess_wav, plot_histograms, plot_similarity_matrix

def embedding(d_path="data", show=True):
    encoder = VoiceEncoder("pretrained_model/speaker.pt", device="cuda" if torch.cuda.is_available() else "cpu")
    wav_fpaths = list(Path("wavs").glob("**/*.wav"))

    speaker_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in groupby(tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit="wavs"), lambda wav_fpath: wav_fpath.parent.stem)}
    speakers = {spk: encoder.embed_speaker(wavs) for spk, wavs in speaker_wavs.items()}
    
    os.makedirs(d_path, exist_ok=True)
    with open(f"{d_path}/embedding.pt", "wb") as handle:
        pickle.dump(speakers, handle)

    if show:
        import matplotlib.pyplot as plt

        embeds_a = np.array([encoder.embed_utterance(wavs[0]) for wavs in speaker_wavs.values()])
        embeds_b = np.array([encoder.embed_utterance(wavs[1]) for wavs in speaker_wavs.values()])                        
        utt_sim_matrix = np.inner(embeds_a, embeds_b)

        spk_embeds_a = np.array([encoder.embed_speaker(wavs[:len(wavs) // 2]) for wavs in speaker_wavs.values()])
        spk_embeds_b = np.array([encoder.embed_speaker(wavs[len(wavs) // 2:]) for wavs in speaker_wavs.values()])
        spk_sim_matrix = np.inner(spk_embeds_a, spk_embeds_b)

        _, axs = plt.subplots(2, 2, figsize=(8, 10))
        labels_a = ["%s-A" % i for i in speaker_wavs.keys()]
        labels_b = ["%s-B" % i for i in speaker_wavs.keys()]
        mask = np.eye(len(utt_sim_matrix), dtype=np.bool)
        plot_similarity_matrix(utt_sim_matrix, labels_a, labels_b, axs[0, 0],
                            "Cross-similarity between utterances\n(speaker_id-utterance_group)")
        plot_histograms((utt_sim_matrix[mask], utt_sim_matrix[np.logical_not(mask)]), axs[0, 1],
                        ["Same speaker", "Different speakers"], 
                        "Normalized histogram of similarity\nvalues between utterances")
        plot_similarity_matrix(spk_sim_matrix, labels_a, labels_b, axs[1, 0],
                            "Cross-similarity between speakers\n(speaker_id-utterances_group)")
        plot_histograms((spk_sim_matrix[mask], spk_sim_matrix[np.logical_not(mask)]), axs[1, 1],
                        ["Same speaker", "Different speakers"], 
                        "Normalized histogram of similarity\nvalues between speakers")
        plt.show()        

if __name__ == "__main__":
    embedding()