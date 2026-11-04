import os
import sys
import collections
import collections.abc
if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping

# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
# Modified by Atik Hasan for Bangla Localization
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

inference_mode_list = ['প্রি-ট্রেইনড ভয়েস', '৩ সেকেন্ড দ্রুত ক্লোনিং', 'ক্রস-লিঙ্গুয়াল ক্লোনিং', 'ন্যাচারাল ল্যাঙ্গুয়েজ কন্ট্রোল']
instruct_dict = {
    'প্রি-ট্রেইনড ভয়েস': '১. প্রি-ট্রেইনড ভয়েস নির্বাচন করুন\n২. অডিও জেনারেট করুন বাটনে ক্লিক করুন',
    '৩ সেকেন্ড দ্রুত ক্লোনিং': '১. প্রম্পট অডিও ফাইল সিলেক্ট করুন বা রেকর্ড করুন (৩০ সেকেন্ডের কম), উভয় থাকলে ফাইল প্রাধান্য পাবে\n২. প্রম্পট টেক্সট ইনপুট দিন\n৩. অডিও জেনারেট করুন বাটনে ক্লিক করুন',
    'ক্রস-লিঙ্গুয়াল ক্লোনিং': '১. প্রম্পট অডিও ফাইল সিলেক্ট করুন বা রেকর্ড করুন (৩০ সেকেন্ডের কম)\n২. অডিও জেনারেট করুন বাটনে ক্লিক করুন',
    'ন্যাচারাল ল্যাঙ্গুয়েজ কন্ট্রোল': '১. প্রি-ট্রেইনড ভয়েস নির্বাচন করুন\n২. ইন্সট্রাক্ট টেক্সট লিখুন\n৩. অডিও জেনারেট করুন বাটনে ক্লিক করুন'
}
stream_mode_list = [('না', False), ('হ্যাঁ', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
        
    if mode_checkbox_group in ['ন্যাচারাল ল্যাঙ্গুয়েজ কন্ট্রোল']:
        if instruct_text == '':
            gr.Warning('দয়া করে ইন্সট্রাক্ট টেক্সট লিখুন')
            yield (cosyvoice.sample_rate, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('ন্যাচারাল ল্যাঙ্গুয়েজ কন্ট্রোল মোডে প্রম্পট অডিও/টেক্সট উপেক্ষা করা হবে')
            
    if mode_checkbox_group in ['ক্রস-লিঙ্গুয়াল ক্লোনিং']:
        if instruct_text != '':
            gr.Info('ক্রস-লিঙ্গুয়াল মোডে ইন্সট্রাক্ট টেক্সট উপেক্ষা করা হবে')
        if prompt_wav is None:
            gr.Warning('দয়া করে প্রম্পট অডিও প্রদান করুন')
            yield (cosyvoice.sample_rate, default_data)
        gr.Info('দয়া করে নিশ্চিত করুন যে সিন্থেসিস টেক্সট এবং প্রম্পট টেক্সট ভিন্ন ভাষার')
        
    if mode_checkbox_group in ['৩ সেকেন্ড দ্রুত ক্লোনিং', 'ক্রস-লিঙ্গুয়াল ক্লোনিং']:
        if prompt_wav is None:
            gr.Warning('প্রম্পট অডিও খালি রয়েছে!')
            yield (cosyvoice.sample_rate, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('প্রম্পট অডিওর স্যামপ্লি রেড {} যা সর্বনিম্ন {} এর চেয়ে কম'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (cosyvoice.sample_rate, default_data)
            
    if mode_checkbox_group in ['প্রি-ট্রেইনড ভয়েস']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('প্রি-ট্রেইনড ভয়েস মোড ব্যবহার করা হচ্ছে, প্রম্পট টেক্সট/অডিও/ইন্সট্রাক্ট টেক্সট উপেক্ষা করা হবে!')
        if sft_dropdown == '':
            gr.Warning('কোনো প্রি-ট্রেইনড ভয়েস উপলব্ধ নেই!')
            yield (cosyvoice.sample_rate, default_data)
            
    if mode_checkbox_group in ['৩ সেকেন্ড দ্রুত ক্লোনিং']:
        if prompt_text == '':
            gr.Warning('প্রম্পট টেক্সট খালি রয়েছে!')
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('৩ সেকেন্ড দ্রুত ক্লোনিং মোড ব্যবহার করা হচ্ছে, প্রি-ট্রেইনড ভয়েস/ইন্সট্রাক্ট টেক্সট উপেক্ষা করা হবে!')

    if mode_checkbox_group == 'প্রি-ট্রেইনড ভয়েস':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '৩ সেকেন্ড দ্রুত ক্লোনিং':
        logging.info('get zero_shot inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == 'クロス语种复刻' or mode_checkbox_group == 'ক্রস-লিঙ্গুয়াল ক্লোনিং':
        logging.info('get cross_lingual inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 🎙️ Atik Hasan's Voice Cloning Pipeline (CosyVoice)")
        gr.Markdown("#### অনুগ্রহ করে নিচের বক্সে আপনার কাঙ্ক্ষিত টেক্সট লিখুন, 推理 (ইনফারেন্স) মোড সিলেক্ট করুন এবং প্রয়োজনীয় নির্দেশনা অনুসরণ করুন।")

        tts_text = gr.Textbox(label="সংশ্লেষণ করার জন্য টেক্সট লিখুন", lines=1, value="হ্যালো! এটি আতিক হাসান এর কাস্টমাইজড ভয়েস ক্লোনিং পাইপলাইন।")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='ইনফারেন্স মোড নির্বাচন করুন', value=inference_mode_list[0])
            instruction_text = gr.Text(label="নির্দেশনা ধাপসমূহ", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='প্রি-ট্রেইনড ভয়েস সিলেক্ট করুন', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='স্ট্রিম মোড চালু করবেন কি?', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="স্পিড কন্ট্রোল (শুধুমাত্র নন-স্ট্রিম মোডের জন্য)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="র‍্যান্ডম সিড (Seed)")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='প্রম্পট অডিও ফাইল আপলোড করুন (স্যাম্পলিং রেট অন্তত ১৬ কিলোহার্জ)')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='মাইক্রোফোনে প্রম্পট অডিও রেকর্ড করুন')
            
        prompt_text = gr.Textbox(label="প্রম্পট টেক্সট ইনপুট দিন", lines=1, placeholder="প্রম্পট অডিওর সাথে হুবহু মিল থাকতে হবে...", value='')
        instruct_text = gr.Textbox(label="ইন্সট্রাক্ট টেক্সট লিখুন", lines=1, placeholder="এখানে ইন্সট্রাকশন লিখুন...", value='')

        generate_button = gr.Button("অডিও জেনারেট করুন 🎙️", variant="primary")

        audio_output = gr.Audio(label="জেনারেট হওয়া আউটপুট অডিও", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
        
    demo.queue()
    demo.launch(server_name='0.0.0.0', server_port=args.port, share=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
