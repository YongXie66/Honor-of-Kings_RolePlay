# Honor of Kings - RolePlay

![img](datasets/妲己Images/时之奇旅.jpg)

## 背景

王者荣耀是一款MOBA类国产游戏，王者荣耀中英雄众多，每个英雄都有自己鲜明的故事背景、性格特征、技能招式等，因此王者荣耀的有关数据包含了丰富的文本和语音，适合创建角色扮演聊天机器人。

| 分路       | 代表英雄 | 经典台词                               | 故事背景                                                     | 被动技能                                                     |
| ---------- | -------- | -------------------------------------- | ------------------------------------------------------------ | :----------------------------------------------------------- |
| **发育路** | 黄忠     | 正义或许会迟到，但绝不会忘记砸到你头顶 | 黄忠，字汉升，东汉末年的猛将之一。早年追随刘表，刘表死后，成为长沙太守韩玄的部将 ... ... | 黄忠普攻时可以提高6-12点攻击和0.75%-1.5%暴击率，持续1.5秒，此效果最多可叠加5层。黄忠处于炮台形态时享受双倍增益效果，黄忠在进入或退出炮台形态时，刷新身上留存的增益效果持续时间并将增益系数替换成当前形态的倍数。 |
| **中路**   | 妲己     | 请尽情吩咐妲己，主人\n来和妲己玩耍吧   | 纣王身边妲己实际上是狐狸精。她蛊惑纣王干下了种种祸害百姓、残害忠良的倒行逆施，最终断送了商朝的天下 ... ... | 妲己技能命中敌人会减少目标30~72点法术防御，持续3秒，最多叠加3层 |
| **打野**   | 赵云     | 心怀不惧，才能翱翔于天际               | 三国时的蜀汉名将。常山真定人，字子龙。东汉末年大乱，群雄并起之时，起兵归于公孙瓒。公孙瓒败亡，投奔刘备，时刘备未成气候 ... ... | 赵云每损失3%最大生命就会获得1%减伤                           |
| **对抗路** | 老夫子   | 老夫年少时，也曾像他们一样，征战四方   | 历史原型为儒家学派创始人孔子，他广收门徒，周游列国，号称三千弟子，传播其学说 ... ... | 老夫子普通攻击命中会增加1点训诫值，最多叠加5点，叠满后会获得强化自身，持续5秒；强化时老夫子会增加60点移动速度和25%攻击速度，同时普通攻击将会附带60点真实伤害，每次攻击能够减少1秒圣人训诫和举一反三的冷却时间 |
| **游走**   | 庄周     | 蝴蝶是我，我就是蝴蝶                   | 庄周，一般人称庄子。是战国时的思想家，跟老子一道，为道家的代表人物之一，后世老庄并称。其学说崇尚自然，推崇自由 ... ... | 在自然梦境中，庄周解除并免疫所有控制效果，获得15%减伤，并增加15%移速，持续2秒。庄周每6秒进入一次自然梦境 |

## 🔊介绍

本项目基于**书生浦语🌟InternLM2**模型，通过构造生成训练数据，采用**Xtuner微调**的方式，打造了一个**王者荣耀**领域的**角色扮演**聊天机器人--**峡谷小狐仙**，同时结合🌟**ASR**技术实现**语音输入**、🌟**TTS**技术实现**声音克隆**和**语音输出**、🌟**数字人**技术实现了**视频输出**功能。**峡谷小狐仙**将王者荣耀手游中特定游戏角色妲己的形象带入书生浦语语言大模型，在实现①知识输出的同时，也实现②角色扮演的效果：

1. **知识输出**：使**峡谷小狐仙**对话表现得像《王者荣耀》游戏专家一样，为使用者提供游戏相关的知识查询
   - **峡谷小狐仙**通晓关于《王者荣耀》中100多位英雄的知识，包括英雄被动技能、英雄主动技能、英雄的角色背景以及英雄故事，相关游戏人物的历史故事
2. **角色扮演**：使**峡谷小狐仙**表现得像《王者荣耀》游戏里的英雄角色妲己一样
   - 人物设定符合王者荣耀游戏中妲己的角色背景和英雄故事
   - 采用符合游戏人物妲己的性格特点、语气、行为方式和表达方式来回复问题
   - 目前实现了英雄妲己的角色扮演，以后会支持更多的英雄角色，也可以根据使用的需求设定创建属于自己的英雄，语音音色和添加特定的对话方式

### 功能亮点

- ASR语音识别技术🎤：支持用户的语音输入
- RAG 检索增强生成📚：用户进行王者荣耀领域相关的提问，模型根据知识储备做出回答
- SFT 大模型微调🧠：以峡谷小狐仙的语气口吻回答问题
- TTS 文字转语音＋语音克隆📢：模型模拟妲己的声音，并将LLM的回答以音频形式输出
- 数字人👁：虚拟小狐仙在线陪伴

## 📺demo

效果示例：

| 文字/语音提问 |                          数字人回答                          |
| :-----------: | :----------------------------------------------------------: |
|    你知道如何应对压力吗     | <video src="[demo.mp4](https://github.com/YongXie66/Honor-of-Kings_RolePlay/assets/88486439/c27ebda4-8a96-45a3-841b-fc3de57602d6)"></video> |

## 行动

### 数据收集

王者荣耀数据的收集，来源于**兄弟项目**[Honor_of_Kings_Multi-modal_Dataset](https://github.com/chg0901/Honor_of_Kings_Multi-modal_Dataset/)，欢迎大家前去star~

```bash
|-- Honor-of-Kings_RolePlay/
    |-- README.md
    |-- 王者荣耀英雄名单.txt
    |-- RAG_Data/
        |-- 上官婉儿.txt
        |-- 不知火舞.txt
        |-- 东皇太一.txt
        ... ...
    |-- 妲己Images/
        |-- 仙境爱丽丝.jpg
        |-- 女仆咖啡.jpg
        |-- 时之奇旅.jpg
        ... ...
    |-- 妲己Texts/
        |-- 妲己介绍.txt
        |-- 妲己介绍.xlsx
        |-- 妲己台词.txt
        |-- 妲己性格特点.txt
        |-- 妲己英雄故事.txt
        |-- 妲己角色背景.txt
    |-- 妲己Voices/
        |-- 109_妲己__魅力之狐.txt
        |-- 109_妲己__魅力之狐/
            |-- 109_妲己_妲己,一直爱主人,因为被设定成这样..wav
            |-- 109_妲己_妲己,陪你玩.wav
            ... ...
```

### 微调数据生成
自我认知数据集使用大模型生成，由 【自我介绍 + 背景关系 + 打招呼用词 + 主要功能介绍】组成，其中自我介绍和背景关系部分参考的是妲己的 `性格特点`，`角色背景`， 以及 `英雄故事`。`打招呼用词` 以及 `主要功能介绍` 则是我们团队自己定义的，回答风格则是参考`妲己台词`。

数据集例子：

> `【尾巴，不只能用来挠痒痒哦。我是基于王者峡谷英雄小妲己构建的小助手，名为小狐仙小助手，姜子牙是我的师傅，师傅把我送到纣王身边，我可以帮助你获取信息，解答关于小狐仙和其他英雄的背景故事，以及提供建议。】`，
>
> `【我与纣王有深厚联系，曾是姜子牙的机关术作品，在王者峡谷里，召唤师们喜欢叫我小妲己，而在在这里，我是狐仙召唤师的小狐仙~，由松龄后裔团队开发。我能提供游戏知识，解答问题，帮助您深入了解英雄故事】`。

持续更新中...

### InternLM2微调

持续更新中...

### 自动语音识别(ASR)

ASR技术用于将用户的语音输入转换为文本。本项目支持用户通过麦克风在线录入音频，或者上传本地已有的音频文件。我们采用了开源的 [whisper](https://github.com/openai/whisper) 模型，该模型在多个语音识别任务上表现优异，能够高效、准确地将语音转化为文本。这些转换后的文本将作为输入，传递给LLM。

### 文本转语音(TTS) + 语音克隆

TTS技术可以将文本转化为自然的语音输出。在本项目中，我们集成了强大的少样本语音转换与语音克隆方法 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)，利用其 `few-shot TTS`功能，通过收集王者荣耀英雄角色妲己的台词语音来微调 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 模型，实现了对妲己音色的克隆。能够将 LLM 输出的文本回答转换为语音，并以妲己的声音进行播报。

### 数字人

数字人技术使得虚拟角色可以以更真实的方式与用户互动。本项目集成了 [SadTalker](https://github.com/OpenTalker/SadTalker) 技术，一种从音频中学习生成3D运动系数，使用全新的3D面部渲染器来生成头部运动，并生成高质量的视频的方法。通过输入上一步 `TTS 输出的音频文件`以及`妲己的海报`，可以生成`动态说话视频`。这使得虚拟小狐仙不仅可以用文本、声音与用户交流，还可以以虚拟人的模式进行互动。

### RAG
本项目采用的是基于 Langchain 的 `Metadata` RAG 方案。Metadata 结构是【英雄名 + 被动/一技能/二技能/三技能/英雄故事/历史】 ，例如：【上官婉儿二技能】，对应的 Document 就是 `技能名称` 和 `技能介绍`。RAG 由以下用户提问关键词触发：【"被动", "一技能", "二技能", "三技能", "英雄故事", "历史"】。

## 使用指南

Clone the repo

```bash
git clone https://github.com/YongXie66/Honor-of-Kings_RolePlay.git
```

install the environment

```bash
conda create -n hok-roleplay python=3.10
conda activate hok-roleplay

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

conda install -q ffmpeg
```

download models

```bash
# LLM 下载
>>>from openxlab.model import download
>>>download(model_repo=LLM_path,output='./InternLM2/InternLM2_7b')
# or
apt install git
apt install git-lfs
git clone https://code.openxlab.org.cn/shenfeilang/Honor-of-Kings_RolePlay.git InternLM2/InternLM2_7b/

# gpt_sovits, sadtalker 相关模型下载
git clone https://code.openxlab.org.cn/YongXie66/DaJi_RolePlay.git ./DaJi_RolePlay

# 模型位置移动
mv ./DaJi_RolePlay/GPT_SoVITS/pretrained_models/* ./GPT_SoVITS/pretrained_models/
mv ./DaJi_RolePlay/checkpoints/* ./checkpoints
mv ./DaJi_RolePlay/gfpgan/* ./gfpgan/
```

Web UI 启动 !

```bash
python webui.py
```

## web UI

目前 Web UI 中提供了**Chatty_DaJi** 和 **Lively_DaJi** 两种对话模式

- **Chatty_DaJi：InternLM2-Chat-7b 微调后的基础小狐仙对话模型 + ASR** 

![image-20240613235252894](assets/image-20240613235252894.png)

- **Lively_DaJi：InternLM2-Chat-7b 微调 + ASR + TTS + voice clone + 数字人** 

![image-20240614000226211](assets/image-20240614000226211.png)



## 项目成员

|      | 成员                                        | 贡献（更新中...） |
| ---- | ------------------------------------------- | ----------------- |
| 主创 | [谢勇](https://github.com/YongXie66/)       |                   |
| 主创 | [程宏](https://github.com/chg0901)          |                   |
| 共创 | [Wong Tack Hwa](https://github.com/tackhwa) |                   |
| 共创 | [沈飞](https://github.com/shenfeilang)      |                   |

## 致谢

感谢上海人工智能实验室推出的 **[书生·浦语大模型实战营](https://openxlab.org.cn/models/InternLM/subject)** 学习活动！

感谢上海人工智能实验室对本项目的技术指导和算力支持！

感谢各个开源项目，包括但不限于：

- [InternLM](https://github.com/InternLM/InternLM)
- [xtuner](https://github.com/InternLM/xtuner)
- [whisper](https://github.com/openai/whisper)
- [LMDeploy](https://github.com/InternLM/LMDeploy)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [SadTalker](https://github.com/OpenTalker/SadTalker)

