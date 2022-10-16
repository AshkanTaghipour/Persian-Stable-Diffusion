import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import googletrans
from googletrans import Translator
from huggingface_hub import notebook_login
import matplotlib.pyplot as plt
import streamlit as st

#### The header of the Web app ####
st.title("Streamlit example")
st.header(""" Powered by Irancell Labs """)

#### The input text section on the web app ####
fa_input_text = st.text_area("متن اینجا وارد شود لطفا", "")

#### Loading the pretrained model from hugging face ####
notebook_login()
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

token = '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'
#fa_input_text = 'دختری با موهای طلایی در مزرعه'
fa_input_text = fa_input_text

#@st.cache
# def pretrained_SD(model_id, token):
#   pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token = token)
#   pipe = pipe.to(device)
#   return pipe

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token = token)
pipe = pipe.to(device)

#### persian to english using translate ####
translator = Translator() 
result = translator.translate(fa_input_text, src='fa', dest='en')
#pipe = pretrained_SD(model_id, token)

#### employing the stable diffusion for generating image ####
generated_image = pipe(result.text, guidance_scale=7.5).images[0]

#### Show generated image on the Web app ####
st.write("تصویر تولید شده")
gen_image = st.image(generated_image)