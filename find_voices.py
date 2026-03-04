import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

print("--- 🎙️ AVAILABLE VOICES ---")
for index, voice in enumerate(voices):
    print(f"Index [{index}] | Name: {voice.name} | ID: {voice.id}")

# Test a specific one (change the number to hear different ones)
test_index = 25
engine.setProperty('voice', voices[test_index].id)
engine.say("Testing this voice. Do I sound like a proper AI for you, Ethan?")
engine.runAndWait()