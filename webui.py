import os
import random 
import gradio as gr
import time
from zhconv import convert
from LLM import LLM
from src.cost_time import calculate_time
import pdb

from configs import *
os.environ["GRADIO_TEMP_DIR"]= './temp'
os.environ["WEBUI"] = "true"
def get_title(title = 'Linly 智能对话系统 (Linly-Talker)'):
    description = f"""
    <p style="text-align: center; font-weight: bold;">
        <span style="font-size: 28px;">{title}</span>
        <br>
        <span style="font-size: 18px;" id="paper-info">
            [<a href="https://github.com/YongXie66/Honor-of-Kings_RolePlay" target="_blank">主页</a>]
        </span>
        <br> 
    </p>
    """
    return description


# 设置默认system
default_system = '你是一个很有帮助的助手'
# 设置默认的prompt
prefix_prompt = '''请用少于25个字回答以下问题\n\n'''


# 设定默认参数值，可修改
blink_every = True
size_of_image = 256
preprocess_type = 'crop'
facerender = 'facevid2vid'
enhancer = False
is_still_mode = False
exp_weight = 1
use_ref_video = False
ref_video = None
ref_info = 'pose'
use_idle_mode = False
length_of_audio = 5

@calculate_time
def Asr(audio):
    try:
        question = asr.transcribe(audio)
        question = convert(question, 'zh-cn')
    except Exception as e:
        print("ASR Error: ", e)
        question = '麦克风模式可能音频还未传入，请重新点击一下语音识别即可'
        gr.Warning(question)
    return question

@calculate_time
def TTS_response(text, 
                 inp_ref, prompt_text, prompt_language, text_language, how_to_cut, 
                 question_audio, question, 
                 tts_method = '', save_path = 'answer.wav'):
    if tts_method == 'GPT-SoVITS克隆声音':
        try:
            vits.predict(ref_wav_path = inp_ref,
                            prompt_text = prompt_text,
                            prompt_language = prompt_language,
                            text = text, # 回答
                            text_language = text_language,
                            how_to_cut = how_to_cut,
                            save_path = 'answer.wav')
            print(text, tts_method, save_path)
            return 'answer.wav', None
        except Exception as e:
            gr.Warning("无克隆环境或者无克隆模型权重，无法克隆声音", e)
            return None, None
    return None, None
@calculate_time
def LLM_response(question_audio, question, 
                 inp_ref = None, prompt_text = "", prompt_language = "", text_language = "", how_to_cut = "", 
                 tts_method = ''):
    answer = llm.generate(question, default_system)
    print(answer)
    driven_audio, driven_vtt = TTS_response(answer, 
                 inp_ref, prompt_text, prompt_language, text_language, how_to_cut, question_audio, question, 
                 tts_method)
    return driven_audio, driven_vtt, answer

@calculate_time
def Talker_response(question_audio = None, method = 'SadTalker', text = '',
                    voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 100, pitch = 0, 
                    am = 'fastspeech2', voc = 'pwgan', lang = 'zh', male = False, 
                    inp_ref = None, prompt_text = "", prompt_language = "", text_language = "", how_to_cut = "", 
                    tts_method = 'Edge-TTS',batch_size = 2, character = '女性角色', 
                    progress=gr.Progress(track_tqdm=True)):
    default_voice = None
    if character == '女性角色':
        # 女性角色
        source_image, pic_path = r'inputs/girl.png', r'inputs/girl.png'
        crop_pic_path = "./inputs/first_frame_dir_girl/girl.png"
        first_coeff_path = "./inputs/first_frame_dir_girl/girl.mat"
        crop_info = ((403, 403), (19, 30, 502, 513), [40.05956541381802, 40.17324339233366, 443.7892505041507, 443.9029284826663])
        default_voice = 'zh-CN-XiaoxiaoNeural'
    elif character == '男性角色':
        # 男性角色
        source_image = r'./inputs/boy.png'
        pic_path = "./inputs/boy.png"
        crop_pic_path = "./inputs/first_frame_dir_boy/boy.png"
        first_coeff_path = "./inputs/first_frame_dir_boy/boy.mat"
        crop_info = ((876, 747), (0, 0, 886, 838), [10.382158280494476, 0, 886, 747.7078990925525])
        default_voice = 'zh-CN-YunyangNeural'
    else:
        gr.Warning('未知角色')
        return None
    
    voice = default_voice if not voice else voice
    
    if not voice:
        gr.Warning('请选择声音')
    
    driven_audio, driven_vtt, _ = LLM_response(question_audio, text, 
                                               voice, rate, volume, pitch, 
                                               am, voc, lang, male, 
                                               inp_ref, prompt_text, prompt_language, text_language, how_to_cut, 
                                               tts_method)
    
    if method == 'SadTalker':
        pose_style = random.randint(0, 45)
        video = talker.test(pic_path,
                        crop_pic_path,
                        first_coeff_path,
                        crop_info,
                        source_image,
                        driven_audio,
                        preprocess_type,
                        is_still_mode,
                        enhancer,
                        batch_size,                            
                        size_of_image,
                        pose_style,
                        facerender,
                        exp_weight,
                        use_ref_video,
                        ref_video,
                        ref_info,
                        use_idle_mode,
                        length_of_audio,
                        blink_every,
                        fps=20)
    elif method == 'Wav2Lip':
        video = talker.predict(crop_pic_path, driven_audio, batch_size, enhancer)
    elif method == 'ER-NeRF':
        video = talker.predict(driven_audio)
    else:
        gr.Warning("不支持的方法：" + method)
        return None
    if driven_vtt:
        return video, driven_vtt
    else:
        return video

@calculate_time
def Talker_response_img(question_audio, method, text, 
                        inp_ref , prompt_text, prompt_language, text_language, how_to_cut,
                        tts_method,
                        source_image,
                        preprocess_type, 
                        is_still_mode,
                        enhancer,
                        batch_size,                            
                        size_of_image,
                        pose_style,
                        facerender,
                        exp_weight,
                        blink_every,
                        fps, progress=gr.Progress(track_tqdm=True)
                    ):

    driven_audio, driven_vtt, answer = LLM_response(question_audio, text,  
                                               inp_ref, prompt_text, prompt_language, text_language, how_to_cut,
                                               tts_method = tts_method)
    # pdb.set_trace()
    if method == 'SadTalker':
        video = talker.test2(source_image,
                        driven_audio,
                        preprocess_type,
                        is_still_mode,
                        enhancer,
                        batch_size,                            
                        size_of_image,
                        pose_style,
                        facerender,
                        exp_weight,
                        use_ref_video,
                        ref_video,
                        ref_info,
                        use_idle_mode,
                        length_of_audio,
                        blink_every,
                        fps=fps)
    else:
        return None
    if driven_vtt:
        return video, driven_vtt, answer
    else:
        return video, answer

@calculate_time
def Talker_Say(preprocess_type, 
                        is_still_mode,
                        enhancer,
                        batch_size,                            
                        size_of_image,
                        pose_style,
                        facerender,
                        exp_weight,
                        blink_every,
                        fps,source_image = None, source_video = None, question_audio = None, method = 'SadTalker', text = '', 
                    voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 100, pitch = 0, 
                    am = 'fastspeech2', voc = 'pwgan', lang = 'zh', male = False, 
                    inp_ref = None, prompt_text = "", prompt_language = "", text_language = "", how_to_cut = "", 
                    tts_method = 'Edge-TTS', character = '女性角色',
                    progress=gr.Progress(track_tqdm=True)):
    if source_video:
        source_image = source_video
    default_voice = None
    
    voice = default_voice if not voice else voice
    
    if not voice:
        gr.Warning('请选择声音')
    
    driven_audio, driven_vtt = TTS_response(text, voice, rate, volume, pitch, 
                 am, voc, lang, male, 
                 inp_ref, prompt_text, prompt_language, text_language, how_to_cut, question_audio, text, 
                 tts_method)
    
    if method == 'SadTalker':
        pose_style = random.randint(0, 45)
        video = talker.test2(source_image,
                        driven_audio,
                        preprocess_type,
                        is_still_mode,
                        enhancer,
                        batch_size,                            
                        size_of_image,
                        pose_style,
                        facerender,
                        exp_weight,
                        use_ref_video,
                        ref_video,
                        ref_info,
                        use_idle_mode,
                        length_of_audio,
                        blink_every,
                        fps=fps)
    elif method == 'Wav2Lip':
        video = talker.predict(source_image, driven_audio, batch_size, enhancer)
    elif method == 'ER-NeRF':
        video = talker.predict(driven_audio)
    else:
        gr.Warning("不支持的方法：" + method)
        return None
    if driven_vtt:
        return video, driven_vtt
    else:
        return video


def chat_response(system, message, history):
    # response = llm.generate(message)
    response, history = llm.chat(system, message, history)
    print(history)
    # 流式输出
    for i in range(len(response)):
        time.sleep(0.01)
        yield "", history[:-1] + [(message, response[:i+1])]
    return "", history

def modify_system_session(system: str) -> str:
    if system is None or len(system) == 0:
        system = default_system
    llm.clear_history()
    return system, system, []

def clear_session():
    # clear history
    llm.clear_history()
    return '', []

def clear_text():
    return "", ""

def human_response(history, question_audio, talker_method, 
                   voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 0, pitch = 0, batch_size = 2, 
                  am = 'fastspeech2', voc = 'pwgan', lang = 'zh', male = False, 
                  inp_ref = None, prompt_text = "", prompt_language = "", text_language = "", how_to_cut = "", use_mic_voice = False,
                  tts_method = 'Edge-TTS', character = '女性角色', progress=gr.Progress(track_tqdm=True)):
    response = history[-1][1]
    qusetion = history[-1][0]
    # driven_audio, video_vtt = 'answer.wav', 'answer.vtt'
    if character == '女性角色':
        # 女性角色
        source_image, pic_path = r'./inputs/girl.png', r"./inputs/girl.png"
        crop_pic_path = "./inputs/first_frame_dir_girl/girl.png"
        first_coeff_path = "./inputs/first_frame_dir_girl/girl.mat"
        crop_info = ((403, 403), (19, 30, 502, 513), [40.05956541381802, 40.17324339233366, 443.7892505041507, 443.9029284826663])
        default_voice = 'zh-CN-XiaoxiaoNeural'
    elif character == '男性角色':
        # 男性角色
        source_image = r'./inputs/boy.png'
        pic_path = "./inputs/boy.png"
        crop_pic_path = "./inputs/first_frame_dir_boy/boy.png"
        first_coeff_path = "./inputs/first_frame_dir_boy/boy.mat"
        crop_info = ((876, 747), (0, 0, 886, 838), [10.382158280494476, 0, 886, 747.7078990925525])
        default_voice = 'zh-CN-YunyangNeural'
    voice = default_voice if not voice else voice
    # tts.predict(response, voice, rate, volume, pitch, driven_audio, video_vtt)
    driven_audio, driven_vtt = TTS_response(response, voice, rate, volume, pitch, 
                 am, voc, lang, male, 
                 inp_ref, prompt_text, prompt_language, text_language, how_to_cut, question_audio, qusetion, use_mic_voice,
                 tts_method)
    
    if talker_method == 'SadTalker':
        pose_style = random.randint(0, 45)
        video = talker.test(pic_path,
                        crop_pic_path,
                        first_coeff_path,
                        crop_info,
                        source_image,
                        driven_audio,
                        preprocess_type,
                        is_still_mode,
                        enhancer,
                        batch_size,                            
                        size_of_image,
                        pose_style,
                        facerender,
                        exp_weight,
                        use_ref_video,
                        ref_video,
                        ref_info,
                        use_idle_mode,
                        length_of_audio,
                        blink_every,
                        fps=20)
    elif talker_method == 'Wav2Lip':
        video = talker.predict(crop_pic_path, driven_audio, batch_size, enhancer)
    elif talker_method == 'ER-NeRF':
        video = talker.predict(driven_audio)
    else:
        gr.Warning("不支持的方法：" + talker_method)
        return None
    if driven_vtt:
        return video, driven_vtt
    else:
        return video


GPT_SoVITS_ckpt = "GPT_SoVITS/pretrained_models"
def load_vits_model(gpt_path, sovits_path, progress=gr.Progress(track_tqdm=True)):
    global vits
    print("gpt_sovits模型加载中...", gpt_path, sovits_path)
    all_gpt_path, all_sovits_path = os.path.join(GPT_SoVITS_ckpt, gpt_path), os.path.join(GPT_SoVITS_ckpt, sovits_path)
    vits.load_model(all_gpt_path, all_sovits_path)
    gr.Info("模型加载成功")
    return gpt_path, sovits_path

def list_models(dir, endwith = ".pth"):
    list_folder = os.listdir(dir)
    list_folder = [i for i in list_folder if i.endswith(endwith)]
    return list_folder

def character_change(character):
    if character == '女性角色':
        # 女性角色
        source_image = r'./inputs/girl.png'
    elif character == '男性角色':
        # 男性角色
        source_image = r'./inputs/boy.png'
    elif character == '自定义角色':
        # gr.Warnings("自定义角色暂未更新，请继续关注后续，可通过自由上传图片模式进行自定义角色")
        source_image = None
    return source_image

def webui_setting(talk = True):
    if not talk:
        with gr.Tabs():
            with gr.TabItem('数字人形象设定'):
                source_image = gr.Image(label="Source image", type="filepath")
    else:
        source_image = gr.Image(value='inputs/DaJi.png', label="DaJi image", type="filepath", elem_id="img2img_image", width=256, interactive=False, visible=False)  


    # inp_ref = gr.Textbox(value='./GPT_SoVITS/ref_audio/主人的命令,是绝对的.wav', visible=False)
    inp_ref = gr.Audio(value="GPT_SoVITS/ref_audio/主人的命令,是绝对的.wav", type="filepath", visible=False)
    prompt_text = gr.Textbox(value='主人的命令，是绝对的', visible=False)
    prompt_language = gr.Textbox(value="中文", visible=False)
    text_language = gr.Textbox(value="中文", visible=False)
    how_to_cut = gr.Textbox(value="凑四句一切", visible=False)
    batch_size = gr.Textbox(value=2, visible=False)

    character = gr.Textbox(value='自定义角色', visible=False)
    tts_method = gr.Textbox(value='GPT-SoVITS克隆声音', visible=False)
    asr_method = gr.Textbox(value='Whisper-tiny', visible=False)
    talker_method = gr.Textbox(value='SadTalker', visible=False)
    llm_method = gr.Textbox(value='Qwen', visible=False)
    return  (source_image, 
             inp_ref, prompt_text, prompt_language, text_language, how_to_cut, 
             tts_method, batch_size, character, talker_method, asr_method, llm_method)


def exmaple_setting(asr, text, character, talk , tts, voice, llm):
    # 默认text的Example
    examples =  [
        ['Whisper-base', '应对压力最有效的方法是什么？', '女性角色', 'SadTalker', 'Edge-TTS', 'zh-CN-XiaoxiaoNeural', 'Qwen'],
        ['FunASR', '如何进行时间管理？','男性角色', 'SadTalker', 'Edge-TTS', 'zh-CN-YunyangNeural', 'Qwen'],
        ['Whisper-tiny', '为什么有些人选择使用纸质地图或寻求方向，而不是依赖GPS设备或智能手机应用程序？','女性角色', 'Wav2Lip', 'PaddleTTS', 'None', 'Qwen'],
        ]

    with gr.Row(variant='panel'):
        with gr.Column(variant='panel'):
            gr.Markdown("## Test Examples")
            gr.Examples(
                examples = examples,
                inputs = [asr, text, character, talk , tts, voice, llm],
            )


def app_chatty():
    with gr.Blocks(analytics_enabled=False, title = 'DaJi_RolePlay') as inference:
        gr.HTML(get_title("Chatty_DaJi~小狐仙🌟陪你聊天"))
        with gr.Row():
            with gr.Column():
                # (source_image, voice, rate, volume, pitch, 
                # am, voc, lang, male, 
                # inp_ref, prompt_text, prompt_language, text_language, how_to_cut,  use_mic_voice,
                # tts_method, batch_size, character, talker_method, asr_method, llm_method)= webui_setting()
                source_image = gr.Image(value='inputs/DaJi.png', label="DaJi image", type="filepath", elem_id="img2img_image", interactive=False, visible=True)  

            with gr.Column():
                system_input = gr.Textbox(value=default_system, lines=1, label='System (设定角色)', visible=False)
                chatbot = gr.Chatbot(height=400, show_copy_button=True)
                with gr.Group():
                    question_audio = gr.Audio(sources=['microphone','upload'], type="filepath", label='语音对话', autoplay=False)
                    asr_text = gr.Button('🎤 语音识别（语音对话后点击）')
                
                # 创建一个文本框组件，用于输入 prompt。
                msg = gr.Textbox(label="Prompt/问题")
                asr_text.click(fn=Asr,inputs=[question_audio],outputs=[msg])
                
                with gr.Row():
                    sumbit = gr.Button("🚀 发送", variant = 'primary')
                    clear_history = gr.Button("🧹 清除历史对话")
                    
            # 设置按钮的点击事件。当点击时，调用上面定义的 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
            sumbit.click(chat_response, inputs=[system_input, msg, chatbot], 
                         outputs=[msg, chatbot])
            
            # 点击后清空后端存储的聊天记录
            clear_history.click(fn = clear_session, outputs = [msg, chatbot])
            
        # exmaple_setting(asr_method, msg, character, talker_method, tts_method, voice, llm_method)
    return inference


def app_lively():
    with gr.Blocks(analytics_enabled=False, title = 'DaJi_RolePlay') as inference:
        gr.HTML(get_title("Vivid_DaJi~小狐仙🌟陪你聊天"))
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'):
                # with gr.Tabs(elem_id="sadtalker_source_image"):
                #         with gr.TabItem('Source image'):
                #             with gr.Row():
                #                 source_image_path = "inputs/DaJi.png" 
                #                 source_image = gr.Image(value=source_image_path, label="DaJi image", type="filepath", elem_id="img2img_image", width=256, interactive=False)                                
                (source_image,  
                inp_ref, prompt_text, prompt_language, text_language, how_to_cut, 
                tts_method, batch_size, character, talker_method, asr_method, llm_method)= webui_setting()
                             
                with gr.Tabs():
                    with gr.TabItem('ASR'):
                        # chatbot = gr.Chatbot(height=400, show_copy_button=True)
                        with gr.Group():
                            question_audio = gr.Audio(sources=['microphone','upload'], type="filepath", label = '语音输入')
                            asr_text = gr.Button('🎤 语音识别（语音输入后点击）')

                with gr.Tabs(): 
                    with gr.TabItem('Text'):
                        # gr.Markdown("## Text Examples")
                        examples =  [
                            ['你好呀，你是谁？'],
                            ['我今天心情很好，来和我聊天吧！'],
                            ['你知道如何应对压力吗？'],
                        ]
                        
                        input_text = gr.Textbox(label="Input Text", lines=5)
                        output_text = gr.Textbox(label="Output Text", lines=8)
                        asr_text.click(fn=Asr,inputs=[question_audio],outputs=[input_text])
                        gr.Examples(
                            examples = examples,
                            inputs = [input_text],
                        )
                        
                        with gr.Row():
                            submit = gr.Button('🚀 发送', elem_id="LLM&sadtalker_generate", variant='primary')
                            clear_history = gr.Button("🧹 清除对话")
                        
                        clear_history.click(fn=clear_text, outputs=[input_text, output_text])
            
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('数字人参数设置'):
                        with gr.Accordion("Advanced Settings", open=False):
                            with gr.Row():
                                size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model? 256 is faster")
                                batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=1) 
                                enhancer = gr.Checkbox(label="GFPGAN as Face enhancer(take a long time)", value=False)        
                                pose_style = gr.Number(value=0, visible=False)
                                exp_weight = gr.Number(value=1, visible=False)
                                blink_every = gr.Checkbox(value=True, visible=False)
                                preprocess_type = gr.Textbox(value='full', visible=False)
                                is_still_mode = gr.Checkbox(value=True, visible=False)
                                facerender = gr.Textbox(value='facevid2vid', visible=False)
                                fps = gr.Number(value=20, visible=False)

                with gr.Tabs(elem_id="sadtalker_genearted"):
                    gen_video = gr.Video(label="Generated video", format="mp4", value='inputs/DaJi_initial.mp4')  # avi,mp4

                submit.click(
                fn=Talker_response_img,
                inputs=[question_audio,
                        talker_method, 
                        input_text, 
                        inp_ref, prompt_text, prompt_language, text_language, how_to_cut, 
                        tts_method,
                        source_image, 
                        preprocess_type,
                        is_still_mode,
                        enhancer,
                        batch_size,                            
                        size_of_image,
                        pose_style,
                        facerender,
                        exp_weight,
                        blink_every,
                        fps], 
                outputs=[gen_video,
                         output_text]
                )
        
    return inference


def asr_model_change(model_name, progress=gr.Progress(track_tqdm=True)):
    global asr
    if model_name == "Whisper-tiny":
        try:
            asr = WhisperASR('tiny')
            # asr = WhisperASR('Whisper/tiny.pt')
            gr.Info("Whisper-tiny模型导入成功")
        except Exception as e:
            gr.Warning(f"Whisper-tiny模型下载失败 {e}")
    elif model_name == "Whisper-base":
        try:
            asr = WhisperASR('base')
            # asr = WhisperASR('Whisper/base.pt')
            gr.Info("Whisper-base模型导入成功")
        except Exception as e:
            gr.Warning(f"Whisper-base模型下载失败 {e}")
    elif model_name == 'FunASR':
        try:
            from ASR import FunASR
            asr = FunASR()
            gr.Info("FunASR模型导入成功")
        except Exception as e:
            gr.Warning(f"FunASR模型下载失败 {e}")
    else:
        gr.Warning("未知ASR模型，可提issue和PR 或者 建议更新模型")
    return model_name

def llm_model_change(model_name, progress=gr.Progress(track_tqdm=True)):
    global llm
    gemini_apikey = ""
    openai_apikey = ""
    proxy_url = None
    if model_name == 'Linly':
        try:
            llm = llm_class.init_model('Linly', 'Linly-AI/Chinese-LLaMA-2-7B-hf', prefix_prompt=prefix_prompt)
            gr.Info("Linly模型导入成功")
        except Exception as e:
            gr.Warning(f"Linly模型下载失败 {e}")
    elif model_name == 'Qwen':
        try:
            llm = llm_class.init_model('Qwen', 'Qwen/Qwen-1_8B-Chat', prefix_prompt=prefix_prompt)
            gr.Info("Qwen模型导入成功")
        except Exception as e:
            gr.Warning(f"Qwen模型下载失败 {e}")
    elif model_name == 'Qwen2':
        try:
            llm = llm_class.init_model('Qwen2', 'Qwen/Qwen1.5-0.5B-Chat', prefix_prompt=prefix_prompt)
            gr.Info("Qwen2模型导入成功")
        except Exception as e:
            gr.Warning(f"Qwen2模型下载失败 {e}")
    elif model_name == 'Gemini':
        if gemini_apikey:
            llm = llm_class.init_model('Gemini', 'gemini-pro', gemini_apikey, proxy_url)
            gr.Info("Gemini模型导入成功")
        else:
            gr.Warning("请填写Gemini的api_key")
    elif model_name == 'ChatGLM':
        try:
            llm = llm_class.init_model('ChatGLM', 'THUDM/chatglm3-6b', prefix_prompt=prefix_prompt)
            gr.Info("ChatGLM模型导入成功")
        except Exception as e:
            gr.Warning(f"ChatGLM模型导入失败 {e}")
    elif model_name == 'ChatGPT':
        if openai_apikey:
            llm = llm_class.init_model('ChatGPT', api_key=openai_apikey, proxy_url=proxy_url, prefix_prompt=prefix_prompt)
        else:
            gr.Warning("请填写OpenAI的api_key")
    # elif model_name == 'Llama2Chinese':
    #     try:
    #         llm =llm_class.init_model('Llama2Chinese', 'Llama2-chat-13B-Chinese-50W')
    #         gr.Info("Llama2Chinese模型导入成功")
    #     except Exception as e:
    #         gr.Warning(f"Llama2Chinese模型下载失败 {e}")
    elif model_name == 'GPT4Free':
        try:
            llm = llm_class.init_model('GPT4Free', prefix_prompt=prefix_prompt)
            gr.Info("GPT4Free模型导入成功, 请注意GPT4Free可能不稳定")
        except Exception as e:
            gr.Warning(f"GPT4Free模型下载失败 {e}")
    else:
        gr.Warning("未知LLM模型，可提issue和PR 或者 建议更新模型")
    return model_name
    
def talker_model_change(model_name, progress=gr.Progress(track_tqdm=True)):
    global talker
    if model_name not in ['SadTalker', 'Wav2Lip', 'ER-NeRF']:
        gr.Warning("其他模型还未集成，请等待")
    if model_name == 'SadTalker':
        try:
            from TFG import SadTalker
            talker = SadTalker(lazy_load=True)
            gr.Info("SadTalker模型导入成功")
        except Exception as e:
            gr.Warning("SadTalker模型下载失败", e)
    elif model_name == 'Wav2Lip':
        try:
            from TFG import Wav2Lip
            talker = Wav2Lip("checkpoints/wav2lip_gan.pth")
            gr.Info("Wav2Lip模型导入成功")
        except Exception as e:
            gr.Warning("Wav2Lip模型下载失败", e)
    elif model_name == 'ER-NeRF':
        try:
            from TFG import ERNeRF
            talker = ERNeRF()
            talker.init_model('checkpoints/Obama_ave.pth', 'checkpoints/Obama.json')
            gr.Info("ER-NeRF模型导入成功")
        except Exception as e:
            gr.Warning("ER-NeRF模型下载失败", e)
    else:
        gr.Warning("未知TFG模型，可提issue和PR 或者 建议更新模型")
    return model_name

def tts_model_change(model_name, progress=gr.Progress(track_tqdm=True)):
    global tts
    if model_name == 'GPT-SoVITS克隆声音':
        try:
            gpt_path = "GPT_SoVITS/pretrained_models/DaJi-e15.ckpt"
            sovits_path = "GPT_SoVITS/pretrained_models/DaJi_e12_s240.pth"
            vits.load_model(gpt_path, sovits_path)
            # gr.Info("模型加载成功")
        except Exception as e:
            gr.Warning(f"GPT-SoVITS模型加载失败 {e}")
    else:
        gr.Warning("未知TTS模型")
    return model_name

def success_print(text):
    print(f"\033[1;31;42m{text}\033[0m")

def error_print(text):
    print(f"\033[1;37;41m{text}\033[0m")

if __name__ == "__main__":
    llm_class = LLM(mode='offline')
    try:
        llm = llm_class.init_model('Qwen', 'Qwen/Qwen-1_8B-Chat', prefix_prompt=prefix_prompt)
        success_print("Success!!! LLM模块加载成功，默认使用Qwen模型")
    except Exception as e:
        error_print(f"Qwen Error: {e}")
        error_print("如果使用Qwen，请先下载Qwen模型和安装环境")
    
    try:
        from VITS import *
        vits = GPT_SoVITS()
        gpt_path = "DaJi-e15.ckpt"
        sovits_path = "DaJi_e12_s240.pth"
        load_vits_model(gpt_path, sovits_path)
        success_print("Success!!! GPT-SoVITS模块加载成功，语音克隆默认使用GPT-SoVITS模型")
    except Exception as e:
        error_print(f"GPT-SoVITS Error: {e}")
        error_print("如果使用VITS，请先下载GPT-SoVITS模型和安装环境")
    
    try:
        from TFG import SadTalker
        talker = SadTalker(lazy_load=True)
        success_print("Success!!! SadTalker模块加载成功，默认使用SadTalker模型")
    except Exception as e:
        error_print(f"SadTalker Error: {e}")
        error_print("如果使用SadTalker，请先下载SadTalker模型")
    
    try:
        from ASR import WhisperASR
        asr = WhisperASR('base')
        success_print("Success!!! WhisperASR模块加载成功，默认使用Whisper-base模型")
    except Exception as e:
        error_print(f"ASR Error: {e}")
        error_print("如果使用FunASR，请先下载WhisperASR模型和安装环境")

    gr.close_all()
    demo_chatty = app_chatty()
    demo_lively = app_lively()
    demo = gr.TabbedInterface(interface_list = [ 
                                                demo_chatty,
                                                demo_lively,
                                                ], 
                              tab_names = [
                                            " Chatty_DaJi", 
                                            " Lively_DaJi", 
                                           ],
                              title = "DaJi-RolePlay WebUI")
    demo.queue()
    demo.launch(server_name=ip, # 本地端口localhost:127.0.0.1 全局端口转发:"0.0.0.0"
                server_port=port,
                # 似乎在Gradio4.0以上版本可以不使用证书也可以进行麦克风对话
                # ssl_certfile=ssl_certfile,
                # ssl_keyfile=ssl_keyfile,
                # ssl_verify=False,
                debug=True,
                ) 
