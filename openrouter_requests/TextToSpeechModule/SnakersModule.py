import torch
import numpy as np


def create_tts(language='ru', model_id='v5_ru', speaker='xenia', output_sr: int = 48000):
    device = torch.device('cpu')
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language=language,
        speaker=model_id
    )
    model.to(device)

    def tts(text):
        audio_tensor = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=output_sr
        )
        audio_numpy = audio_tensor.cpu().numpy()
        audio_int16 = (audio_numpy * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        return audio_bytes

    return tts

