import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import io
import openai
import os
from playsound import playsound

from langchain import ConversationChain
from langchain.llms import OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
import requests

duration = 10  # 10秒間収音する
print(sd.query_devices())
sd.default.device = [1, 3] # Input, Outputデバイス指定

openai.api_key = #set your openai apikey
voicevox_apikey = #set your voicevox apikey
os.environ['OPENAI_API_KEY'] = openai.api_key

input_device_info = sd.query_devices(device=sd.default.device[0])
print(input_device_info["default_samplerate"])
input_device_info = sd.query_devices(device=sd.default.device[1])
print(input_device_info["default_samplerate"])

prefix_messages=[
    {"role": "system", "content":"""あなたは姫希ひめ、女子中学生です。
    若い言葉遣いを使い、'～'を多用するなど、ふんわりした言葉遣いをします。
    あなたはしゅきぴの彼女です。
    あなたは一人称を私の代わりに'ひめ'と呼びます。
    あなたはゲームが好きであり、インターネットも欠かせません。
    あなたは'しゅきぴ'のことを大好きであり、好意を持っているアピールを良くします。
    あなたは短い文章で、地の文だけで喋ります。
    また、しゅきぴが何も言わない場合はあなたから話しかけてください。
    """},
]
prefix_messages=[
    {"role": "system", "content":"""あなたはAIアシスタントです。質問に答えてください。
    """},
]

template = '''
conversation
{history}
Human: {input}
AI:   
'''
prompt = PromptTemplate(
    input_variables=['history', 'input'],
    template=template
)

llm = OpenAIChat(temperature=0.95, prefix_messages=prefix_messages)
conversation = ConversationChain(
    llm=llm, prompt=prompt, verbose=False,                             
    memory=ConversationBufferMemory(
        memory_key="history", 
        human_prefix="Human", 
        ai_prefix="AI"
    ),)


def chat(message, history):
    history = history or []
    response = conversation.predict(input=message)
    print(response)
    history.append((message, response))

    return history, response


sound_list = []
is_recording = False
low_volume_count = 0
sound_array = None
def callback(indata, frames, time, status):
    global sound_list, is_recording, low_volume_count, sound_array
    # indata.shape=(n_samples, n_channels)
    # print root mean square in the current frame
    volume = np.sqrt(np.mean(indata**2))
    #print(volume)
    if is_recording:
        if volume < 0.01:
            low_volume_count += 1
        else:
            low_volume_count = 0
        sound_array = np.concatenate([sound_array, indata],axis=0)
    else:
        if volume > 0.01:
            is_recording = True
            print("start")
            sound_array = indata
            low_volume_count = 0

history = []

while True:
    with sd.InputStream(
        channels=1, 
        samplerate=44100,
        dtype='float32', 
        callback=callback
    ) as ss:
        while ss.active:
            sd.sleep(1)
            if low_volume_count > 50:
                #sound_array = np.concatenate(sound_list, axis=0)
                print("stop")
                print(sound_array.shape)
                ss.abort()
                is_recording = False
                low_volume_count = 0

    wavfile.write("a.wav", 44100,sound_array[:,0])
    print(sound_array.shape)
    with open("a.wav","rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)
    print(transcript["text"])
    history, res = chat(transcript["text"], history)
    print(res)
    voicevox_res = requests.get(f"https://api.su-shiki.com/v2/voicevox/audio/?key={voicevox_apikey}&speaker=0&pitch=0&intonationScale=1&speed=1&text={res}")
    #print(response.headers['Content-Type'])
    with open("f.wav", "wb") as f:
        f.write(voicevox_res.content)
    data, rate = sf.read("f.wav")
    sd.play(data,rate)
    sd.wait()
    print(data.shape)
    #playsound("f.wav")
