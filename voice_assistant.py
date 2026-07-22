import threading
import time
import io
import traceback
import queue
import numpy as np

class VoiceAssistant:
    def __init__(self, state):
        self.state = state
        self.running = False
        self.thread = None
        self.is_speaking_tts = False
        
        # Load dependencies safely
        try:
            import sounddevice as sd
            import speech_recognition as sr
            import pyttsx3
            
            self.sd = sd
            self.sr = sr
            self.recognizer = sr.Recognizer()
            
            # Init TTS
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 190) # Slightly faster than default 200
            
            # Try to find a male voice (often named 'David' on Windows)
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if 'David' in voice.name or 'male' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
                    
            self.is_available = True
        except ImportError as e:
            print(f"[VOICE] Missing dependency: {e}. Voice disabled.")
            self.is_available = False

    def start(self):
        if not self.is_available: return
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        print("[VOICE] Thread started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def speak(self, text):
        if not hasattr(self, 'tts_engine'):
            return
            
        def _speak_thread():
            self.is_speaking_tts = True
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"[TTS] Speech error: {e}")
            finally:
                self.is_speaking_tts = False
                
        threading.Thread(target=_speak_thread, daemon=True).start()

    def _listen_loop(self):
        fs = 16000
        audio_q = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            audio_q.put(bytes(indata))

        self.awake_until = 0

        print("[VOICE] Starting continuous audio stream...")
        try:
            with self.sd.RawInputStream(samplerate=fs, blocksize=4000, dtype='int16', channels=1, callback=audio_callback):
                audio_buffer = []
                silence_frames = 0
                is_speaking = False
                # RMS threshold for speech detection (adjust if too sensitive)
                ENERGY_THRESHOLD = 500

                while self.running:
                    # UI sync
                    is_awake = time.time() < getattr(self, 'awake_until', 0)
                    self.state.is_listening = is_awake and is_speaking

                    try:
                        chunk = audio_q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                        
                    if self.is_speaking_tts:
                        # Drop audio while PRISM is talking so he doesn't hear himself
                        audio_buffer = []
                        is_speaking = False
                        silence_frames = 0
                        continue

                    # Calculate energy (volume)
                    audio_data = np.frombuffer(chunk, dtype=np.int16)
                    rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))

                    if rms > ENERGY_THRESHOLD:
                        is_speaking = True
                        silence_frames = 0
                        audio_buffer.append(chunk)
                    elif is_speaking:
                        silence_frames += 1
                        audio_buffer.append(chunk)

                        # If silent for ~1 second (4 chunks at 4000/16000)
                        if silence_frames > 4:
                            full_audio = b"".join(audio_buffer)
                            
                            # Reset for next phrase
                            audio_buffer = []
                            is_speaking = False
                            silence_frames = 0
                            
                            # Only process if it was long enough to be a word
                            if len(full_audio) > fs * 0.5:  # > 0.5 seconds
                                self.state.is_listening = False
                                
                                # Directly feed raw bytes to speech_recognition, avoiding heavy scipy/wavfile I/O
                                # sample_width = 2 for 16-bit audio
                                audio = self.sr.AudioData(full_audio, fs, 2)
                                
                                # Dispatch network request to a background thread so it doesn't block listening
                                threading.Thread(target=self._recognize_audio, args=(audio,), daemon=True).start()
                                

        except Exception as e:
            print(f"[VOICE] Stream Error: {e}")
            traceback.print_exc()

    def _recognize_audio(self, audio):
        try:
            # We set a 5 second timeout on the API request so hanging network doesn't leave thread dangling
            text = self.recognizer.recognize_google(audio).lower()
            if text:
                print(f"[VOICE] Heard: '{text}'")
                self._parse_command(text)
        except self.sr.UnknownValueError:
            pass
        except self.sr.RequestError:
            print("[VOICE] API Request error")
        except Exception as e:
            print(f"[VOICE] Unexpected recognition error: {e}")

    def _parse_command(self, text):
        # Wake word logic
        is_awake = time.time() < getattr(self, 'awake_until', 0)
        
        wake_words = ["prism", "prison", "listen", "prizm", "rhythm"]
        if any(w in text for w in wake_words):
            is_awake = True
            self.awake_until = time.time() + 8.0  # Stay awake for 8 seconds
            
            # Remove the wake word from text so we can parse the rest of the sentence
            for w in wake_words:
                text = text.replace(w, "").strip()
                
            self.state.last_voice_text = "YES, SIR?" if not text else text
            self.state.last_voice_time = time.time()
            
            # If they just said "PRISM", don't process further
            if not text:
                self.speak("Yes, sir?")
                return
        
        if not is_awake:
            return  # Ignore all other speech if not awake
            
        self.state.last_voice_text = text
        self.state.last_voice_time = time.time()
        
        cmd = None
        # Keyword mapping
        if any(w in text for w in ["suit up", "deploy", "shoot up", "set up", "sweet up", "straight up"]):
            cmd = "suit_up"
        elif any(w in text for w in ["track all", "everything"]):
            cmd = "track_0"
        elif any(w in text for w in ["track face and hands", "face and hands"]):
            cmd = "track_1"
        elif any(w in text for w in ["track face", "face only"]):
            cmd = "track_2"
        elif any(w in text for w in ["track hands", "hands only"]):
            cmd = "track_3"
        elif any(w in text for w in ["track arms", "arms only"]):
            cmd = "track_4"
        elif any(w in text for w in ["nano", "iron man", "ironman", "mark"]):
            cmd = "set_mode_0"
        elif any(w in text for w in ["quantum", "prism mode", "blue mode"]):
            cmd = "set_mode_1"
        elif any(w in text for w in ["hacker", "cipher", "neon", "matrix", "green mode"]):
            cmd = "set_mode_2"
        elif any(w in text for w in ["edith", "surgical", "white"]):
            cmd = "set_mode_3"
        elif any(w in text for w in ["thermal", "heat", "predator", "infra"]):
            cmd = "set_mode_4"
        elif any(w in text for w in ["ultron", "void", "purple"]):
            cmd = "set_mode_5"
        elif "mode" in text:
            for num, mode_idx in [("1", 0), ("one", 0),
                                   ("2", 1), ("two", 1),
                                   ("3", 2), ("three", 2),
                                   ("4", 3), ("four", 3),
                                   ("5", 4), ("five", 4),
                                   ("6", 5), ("six", 5)]:
                if f"mode {num}" in text:
                    cmd = f"set_mode_{mode_idx}"
                    break
        elif any(w in text for w in ["hide data", "hide hud", "clear screen"]):
            cmd = "toggle_hud"
        elif any(w in text for w in ["disable skip", "hide skip", "remove skip"]):
            cmd = "skip_off"
        elif any(w in text for w in ["enable skip", "show skip"]):
            cmd = "skip_on"
        
        if cmd:
            print(f"[VOICE] Command triggered: {cmd} from '{text}'")
            if hasattr(self.state, "voice_command_queue"):
                self.state.voice_command_queue.append(cmd)
                
            # Play voice lines for commands
            if cmd == "suit_up":
                self.speak("Deploying nanotech, sir.")
            elif cmd == "track_0":
                self.speak("Tracking all targets.")
            elif cmd == "track_1":
                self.speak("Tracking face and hands.")
            elif cmd == "track_2":
                self.speak("Tracking face only.")
            elif cmd == "track_3":
                self.speak("Tracking hands only.")
            elif cmd == "track_4":
                self.speak("Tracking full body pose.")
            elif cmd.startswith("set_mode_"):
                self.speak("Switching HUD interface.")
            elif cmd == "toggle_hud":
                self.speak("Toggling telemetry.")
            elif cmd in ["skip_on", "skip_off"]:
                self.speak("Updating skip protocol.")
                
            # Go back to sleep after executing a command
            self.awake_until = 0
