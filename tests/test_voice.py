import speech_recognition as sr
import pyttsx3
import traceback

def run_audio_test():
    print("=== P.R.I.S.M. AUDIO & VOICE DIAGNOSTIC ===")
    
    print("\n[1/2] Testing Text-to-Speech Engine (pyttsx3)...")
    try:
        engine = pyttsx3.init()
        print("  -> Engine initialized.")
        engine.say("Audio systems are fully operational.")
        engine.runAndWait()
        print("[OK] Voice synthesis successful. You should have heard a voice.")
    except Exception as e:
        print(f"[FAIL] Text-to-Speech failed:")
        traceback.print_exc()
        
    print("\n[2/2] Testing Microphone & Google Speech Recognition...")
    try:
        import sounddevice as sd
        import queue
        import numpy as np
        
        recognizer = sr.Recognizer()
        
        print("  -> Microphone detected. LISTENING... Please say something out loud! (e.g. 'hello world')")
        
        fs = 16000
        audio_q = queue.Queue()
        def audio_callback(indata, frames, time_info, status):
            audio_q.put(bytes(indata))
            
        audio_buffer = []
        with sd.RawInputStream(samplerate=fs, blocksize=4000, dtype='int16', channels=1, callback=audio_callback):
            # Record for exactly 4 seconds
            import time
            end_t = time.time() + 4.0
            while time.time() < end_t:
                try:
                    chunk = audio_q.get(timeout=0.1)
                    audio_buffer.append(chunk)
                except queue.Empty:
                    pass

        print("\n  -> Processing audio with Google...")
        raw_data = b''.join(audio_buffer)
        audio_data = sr.AudioData(raw_data, fs, 2)
        text = recognizer.recognize_google(audio_data).lower()
        print(f"[OK] Successfully recognized: '{text}'")
        print("\n=== SUCCESS! AUDIO SYSTEMS ARE 100% HEALTHY ===")
    except sr.UnknownValueError:
        print("[FAIL] Google Speech could not understand the audio (Too quiet or mumbled).")
    except sr.RequestError as e:
        print(f"[FAIL] Could not request results from Google Speech Recognition service (No internet?). {e}")
    except Exception as e:
        print(f"[CRITICAL FAIL] Microphone access error:")
        traceback.print_exc()

if __name__ == "__main__":
    run_audio_test()
