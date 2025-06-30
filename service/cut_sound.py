from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os 

def cut_sound_per_action(sound, output_dir, sample_rate, action_duration=400, length_duration = 1500):
    print("Cutting sound per action . . . ")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    nonsilent_ranges = detect_nonsilent(sound, min_silence_len=action_duration, silence_thresh=-35 )
    
    if not nonsilent_ranges:
        print("No sound detection in this file")
        return
    
    for i, (start, end) in enumerate(nonsilent_ranges):
        segment = sound[start:end]
        if len(segment) < length_duration:
            silence_to_add = AudioSegment.silent(duration=length_duration - len(segment))
            segment += silence_to_add

        segment = segment[:length_duration] 

        output_file = f"{output_dir}/value_{i + 1}.wav"
        segment.export(output_file, format="wav")  
    print(f"Finished cutting sound per action , {len(nonsilent_ranges)} actions")     
    return