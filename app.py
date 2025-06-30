import librosa
import shutil

from service.redution import reduce_audio_noise
from service.cut_sound import cut_sound_per_action
from utils.plot_compare import plot_compare
from utils.convert_to_byte import convert_to_2bytes




if __name__ == "__main__":
    input_path = "/Users/prasitpaisan/Desktop/Raw bottle/raw_bottle2.wav"
    sample_rate = 22050
  
    original_audio, sr = librosa.load(input_path, sr=sample_rate)
    reduced_audio, _sr = reduce_audio_noise(input_path, sample_rate)

    audio_segment = convert_to_2bytes(reduced_audio, _sr)

    cut_sound_per_action(audio_segment, "./results",_sr)
 
    plot_compare(original_audio=original_audio, reduced_audio=reduced_audio, sample_rate=sr)

    # Detete the results after finish prediction
    shutil.rmtree('./results')