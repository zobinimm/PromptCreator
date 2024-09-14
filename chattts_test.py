import ChatTTS
import torch
from scipy.io import wavfile

class TorchSeedContext:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = torch.random.get_rng_state()
        torch.manual_seed(self.seed)

    def __exit__(self, type, value, traceback):
        torch.random.set_rng_state(self.state)

def replace_arabic_with_chinese(text: str) -> str:
    arabic_to_chinese = {
        '0': '零',
        '1': '一',
        '2': '二',
        '3': '三',
        '4': '四',
        '5': '五',
        '6': '六',
        '7': '七',
        '8': '八',
        '9': '九'
    }
    chinese_text = ''.join(arabic_to_chinese[char] if char in arabic_to_chinese else char for char in text)
    return chinese_text

chat = ChatTTS.Chat()
chat.load(compile=True) # Set to True for better performance
# if not chat.has_loaded():
#     print("Model loading failed") , source="custom", custom_path="E:/Self/001-SelfGit/PromptCreator"

# texts = ["PUT YOUR 1st TEXT HERE", "PUT YOUR 2nd TEXT HERE"]
#
# wavs = chat.infer(texts)
#
# for i in range(len(wavs)):
#     """
#     In some versions of torchaudio, the first line works but in other versions, so does the second line.
#     """
#     try:
#         wavfile.write(f"basic_output{i}.wav", 24000, wavs[i])
#         # torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
#     except:
#         wavfile.write(f"basic_output{i}.wav", 24000, wavs[i])
#         # torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]), 24000)

text = "例如，在“采用指南”中，我们编写了易于遵循的操作指南，逐步引导您和团队了解将 Dynamics 365 推广到组织中的最佳方法。 本指南重点关注创建全面的采用和变更管理 (ACM) 计划的需求。 要了解为什么变更管理对于 Dynamics 365 数字化转型取得成功至关重要，这段“推动数字化转型变更”的视频将为您解答疑问。"


with TorchSeedContext(1997):
    rand_spk = chat.sample_random_speaker()

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=rand_spk,
    temperature=0.3,
    top_P=0.7,
    top_K=20,
    manual_seed=1997,
)

wavs = chat.infer(replace_arabic_with_chinese(text), skip_refine_text=True, params_infer_code=params_infer_code,)

for i in range(len(wavs)):
    wavfile.write(f"basic_output{i}.wav", 24000, wavs[i])


