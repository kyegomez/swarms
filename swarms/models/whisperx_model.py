import os
import subprocess

try:
    import whisperx
    from pydub import AudioSegment
    from pytube import YouTube
except Exception as error:
    print("Error importing pytube. Please install pytube manually.")
    print("pip install pytube")
    print("pip install pydub")
    print("pip install whisperx")
    print(f"Pytube error: {error}")


class WhisperX:
    def __init__(
        self,
        video_url,
        audio_format="mp3",
        device="cuda",
        batch_size=16,
        compute_type="float16",
        hf_api_key=None,
    ):
        """
        # Example usage
        video_url = "url"
        speech_to_text = WhisperX(video_url)
        transcription = speech_to_text.transcribe_youtube_video()
        print(transcription)

        """
        self.video_url = video_url
        self.audio_format = audio_format
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.hf_api_key = hf_api_key

    def install(self):
        subprocess.run(["pip", "install", "whisperx"])
        subprocess.run(["pip", "install", "pytube"])
        subprocess.run(["pip", "install", "pydub"])

    def download_youtube_video(self):
        audio_file = f"video.{self.audio_format}"

        # Download video 📥
        yt = YouTube(self.video_url)
        yt_stream = yt.streams.filter(only_audio=True).first()
        yt_stream.download(filename="video.mp4")

        # Convert video to audio 🎧
        video = AudioSegment.from_file("video.mp4", format="mp4")
        video.export(audio_file, format=self.audio_format)
        os.remove("video.mp4")

        return audio_file

    def transcribe_youtube_video(self):
        audio_file = self.download_youtube_video()

        device = "cuda"
        batch_size = 16
        compute_type = "float16"

        # 1. Transcribe with original Whisper (batched) 🗣️
        model = whisperx.load_model(
            "large-v2", device, compute_type=compute_type
        )
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)

        # 2. Align Whisper output 🔍
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        # 3. Assign speaker labels 🏷️
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self.hf_api_key, device=device
        )
        diarize_model(audio_file)

        try:
            segments = result["segments"]
            transcription = " ".join(
                segment["text"] for segment in segments
            )
            return transcription
        except KeyError:
            print("The key 'segments' is not found in the result.")

    def transcribe(self, audio_file):
        model = whisperx.load_model(
            "large-v2", self.device, self.compute_type
        )
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=self.batch_size)

        # 2. Align Whisper output 🔍
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=self.device
        )

        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        # 3. Assign speaker labels 🏷️
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self.hf_api_key, device=self.device
        )

        diarize_model(audio_file)

        try:
            segments = result["segments"]
            transcription = " ".join(
                segment["text"] for segment in segments
            )
            return transcription
        except KeyError:
            print("The key 'segments' is not found in the result.")
