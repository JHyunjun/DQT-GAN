import os
from pydub import AudioSegment

def main():
    audio_length = 4  # 초
    audio_length_ms = audio_length * 1000
    data_overlap = 50  # 퍼센트
    data_overlap_ps = data_overlap / 100
    sampling_rate = 8192

    os.makedirs("data_folder/wav_data", exist_ok=True)
    os.makedirs("data_folder/mp3_data", exist_ok=True)

    wav_path = "data_folder/wav_data"
    mp3_path = "data_folder/mp3_data"

    base_wav = AudioSegment.from_wav("CN7 PE ENG.wav")
    audio = base_wav.set_frame_rate(sampling_rate)

    num_segments = int(len(audio) / (audio_length_ms * data_overlap_ps))

    for i in range(1, num_segments):
        tmp_fname_wav = f"audio_wav_{i:04}.wav"
        tmp_fname_mp3 = f"audio_mp3_{i:04}.mp3"
        tmp_audio = audio[(i-1)*audio_length_ms*data_overlap_ps : (i+1)*audio_length_ms*data_overlap_ps]
        tmp_audio.export(os.path.join(wav_path, tmp_fname_wav), format="wav")
        tmp_audio.export(os.path.join(mp3_path, tmp_fname_mp3), format="mp3")


if __name__ == "__main__":
    main()
