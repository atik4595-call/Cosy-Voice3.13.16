import os
import sys
import collections
import collections.abc
if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping

# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
# Modified by Atik Hasan for Movie Dubbing Pipeline
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

# মুভি ডাবিংয়ের জন্য শুধুমাত্র দরকারি মোডগুলো রাখা হলো
inference_mode_list = ['ক্রস-লিঙ্গুয়াল ক্লোনিং (ইংরেজি থেকে বাংলা ডাবিং)', '৩ সেকেন্ড দ্রুত ক্লোনিং']
stream_mode_list = [('না', False), ('হ্যাঁ', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def generate_audio(tts_text, mode_checkbox_group, prompt_text, prompt_wav_upload, prompt_wav_record, seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None

    if prompt_wav is None:
        gr.Warning('দয়া করে প্রম্পট অডিও ফাইল আপলোড করুন অথবা রেকর্ড করুন!')
        yield (cosyvoice.sample_rate, default_data)
        return

    if 'ক্রস-লিঙ্গুয়াল' in mode_checkbox_group:
        logging.info('get cross_lingual inference request for movie dubbing')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        if prompt_text == '':
            gr.Warning('৩ সেকেন্ড দ্রুত ক্লোনিং মোডের জন্য প্রম্পট টেক্সট আবশ্যক!')
            yield (cosyvoice.sample_rate, default_data)
            return
        logging.info('get zero_shot inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


def main():
    with gr.Blocks() as demo:
        gr.Markdown("## 🎬 Atik Hasan's Movie Dubbing & Voice Cloning Pipeline")
        gr.Markdown("#### মুভি ডাবিংয়ের জন্য আপনার সোর্স অডিও আপলোড করুন এবং কাঙ্ক্ষিত বাংলা টেক্সট দিয়ে জেনারেট করুন।")

        tts_text = gr.Textbox(label="টার্গেট বাংলা ডায়লগ (যে কথাটি বলিয়ে নিতে চান)", lines=2, value="হ্যালো! আতিক হাসানের মুভি ডাবিং পাইপলাইনে আপনাকে স্বাগতম।")
        
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='ডাবিং মোড নির্বাচন করুন', value=inference_mode_list[0])
            stream = gr.Radio(choices=stream_mode_list, label='স্ট্রিম মোড', value=stream_mode_list[0][1])
            speed = gr.Number(value=1.0, label="স্পিড কন্ট্রোল", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="🎲 সিড রিসেট")
                seed = gr.Number(value=0, label="র‍্যান্ডম সিড")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='সোর্স অডিও স্যাম্পল আপলোড করুন (যেমন: চরিত্রের ইংরেজি ডায়লগ)')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='অথবা মাইক্রোফোনে রেকর্ড করুন')
            
        prompt_text = gr.Textbox(label="প্রম্পট টেক্সট (শুধুমাত্র দ্রুত ক্লোনিং মোডের জন্য, ক্রস-লিঙ্গুয়ালে ফাঁকা রাখতে পারেন)", lines=1, placeholder="সোর্স অডিওতে যা বলা হয়েছে...", value='')

        generate_button = gr.Button("🎬 ডাবিং অডিও জেনারেট করুন", variant="primary", scale=2)

        audio_output = gr.Audio(label="জেনারেট হওয়া বাংলা ডাবিং আউটপুট", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, prompt_text, prompt_wav_upload, prompt_wav_record, seed, stream, speed],
                              outputs=[audio_output])
        
    demo.queue()
    demo.launch(server_name='0.0.0.0', server_port=args.port, share=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B', help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)

    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
