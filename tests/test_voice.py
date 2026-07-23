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
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("  -> Microphone detected. Calibrating for background noise (2 seconds)...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("\n  -> LISTENING... Please say something out loud! (e.g. 'hello world')")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            print("\n  -> Processing audio with Google...")
            text = recognizer.recognize_google(audio).lower()
            print(f"[OK] Successfully recognized: '{text}'")
            print("\n=== SUCCESS! AUDIO SYSTEMS ARE 100% HEALTHY ===")
    except sr.WaitTimeoutError:
        print("[FAIL] Microphone timed out! Make sure your mic isn't muted and speak louder.")
    except sr.UnknownValueError:
        print("[FAIL] Google Speech could not understand the audio (Too quiet or mumbled).")
    except sr.RequestError as e:
        print(f"[FAIL] Could not request results from Google Speech Recognition service (No internet?). {e}")
    except Exception as e:
        print(f"[CRITICAL FAIL] Microphone access error:")
        traceback.print_exc()

if __name__ == "__main__":
    run_audio_test()
