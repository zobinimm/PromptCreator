import spacy

# region paragraphs
# # 加载spaCy的英文模型
# nlp = spacy.load("en_core_web_sm")
#
# # 示例文本（小说的片段）
# text = """
# The scene of the death was quite beautiful, now, years after the fact. A small creek cut through a little hollow in the prairie; the bowl-shaped depression had probably been a crater centuries ago before rain and wind had blunted its edges and nature filled it with field grasses and singing cicadas. In the exact center, in a vaguely star-shaped swath of emerald green moss interrupting the golden tallgrass, stood a stone marker bearing the carved sunburst of Omnu, the victim’s name and the dates bracketing her pitifully short life.
# She knelt before the tiny monument, apparently studying it but in truth merely listening as he approached. The crunch of his boots, the rattle of spurs had given him away long before he spoke, to her annoyance; his heavy tread obscured the other sounds for which she listened.
# “You can feel free to tell Father Reyfield I said so,” the man went on, coming to a stop at the lip of the little crater. His shadow loomed beside her, an elongated figure in a ten-gallon hat, hands tucked into his belt in the stationary swagger of a man who kept order in his little town by the sheer force of his personality. “The old poof and I don’t see eye-to-eye on much anyway. Here’s little June Witwill, just plain the best girl in the province, near enough. Sang in the choir, donated all her pocket money to the local mission… Always spoke respectfully of Emperor and country, and up to her eyes in everything ever went on at the church. She once got caught up in a stagecoach robbery when she was twelve, and talked one of the bandits into turning himself in. He went on to become a monk in Omnu’s temple, used to send June letters all the time.” The shadow of his hat oscillated as he shook his head slowly. “Just…best kid I ever knew, is all. And here she is, walkin’ out to catch crawdads in the stream, and just…burns. Just went up like a goddamn firework. Town’s almost a mile away and we heard her scream like it was happening right there. What the hell kind of thing is that, except an act of the gods? And why the hell would they wanna pick on one of the sweetest things they ever created? Yeah, I ain’t been to church since. They’re just plain bastards, is all, and I’ve got enough of those comin’ through my town as it is.”
# """
#
# # 使用spaCy处理文本
# doc = nlp(text)
#
# # 分段（基于空行）
# paragraphs = text.strip().split('\n\n')
# print("Paragraphs:")
# for i, paragraph in enumerate(paragraphs):
#     print(f"Paragraph {i + 1}:")
#     print(paragraph)
#     print()
#
# # 分句
# print("Sentences:")
# for sentence in doc.sents:
#     print(sentence.text)
# endregion

# text = "Tian Xingjian lay motionless in the crater, the soil on his body blending him in with the surrounding environment."

# region Matcher
# from spacy.matcher import Matcher
#
# # 加载 spaCy 模型
# nlp = spacy.load("en_core_web_trf")
#
# # 输入文本
# text = "The tall man with a beard ran quickly through the dense forest. He was searching for his lost dog while trying to avoid the heavy rain."
# text = "A boy lay motionless in the crater, the soil on his body blending him in with the surrounding environment."
# # 处理文本
# doc = nlp(text)
#
#
# for ent in doc.ents:
#     print(f"Entity: {ent.text}, Label: {ent.label_}")
#
# # 创建模式匹配器
# matcher = Matcher(nlp.vocab)
#
# # 定义匹配模式：一个形容词后接一个名词
# pattern = [{"POS": "DET", "OP": "?"}, {"POS": "ADJ", "OP": "*"}, {"POS": "NOUN"}]
# matcher.add("PERSON_DESC", [pattern])
#
# # 查找匹配项
# matches = matcher(doc)
#
# # 提取匹配结果
# features = [doc[start:end].text for match_id, start, end in matches]
#
# # 提取人物动作
# actions = [token.lemma_ for token in doc if token.pos_ == "VERB"]
#
# # 提取背景信息
# background = []
# for chunk in doc.noun_chunks:
#     if any(tok.dep_ in ["amod", "nmod"] for tok in chunk):
#         background.append(chunk.text)
#
# # 输出结果
# print("人物特征:")
# print(", ".join(features))
#
# print("人物动作:")
# print(", ".join(actions))
#
# print("人物背景信息:")
# print(", ".join(background))
# endregion



import spacy
from spacy.matcher import Matcher

# 加载模型
nlp = spacy.load("en_core_web_trf")

# 创建 Matcher 对象
matcher = Matcher(nlp.vocab)

# 定义模式：用于捕获人物姓名（如 Alice）
person_pattern = [
    {"POS": "PROPN"},  # 专有名词
    {"POS": "PROPN", "OP": "*"},  # 可选的第二个专有名词（用于包含姓名和姓）
]

# 定义模式：用于捕获场景（如 sunny park）
scene_pattern = [
    {"POS": "ADJ", "OP": "*"},  # 可能的形容词（描述场景的特征）
    {"POS": "NOUN"},  # 名词（场景的名称）
]

# 添加模式到 Matcher
matcher.add("PERSON", [person_pattern])
matcher.add("SCENE", [scene_pattern])

# 示例文本
text = "The tall man with a beard ran quickly through the dense forest. He was searching for his lost dog while trying to avoid the heavy rain."

# 处理文本
doc = nlp(text)

# 查找匹配
matches = matcher(doc)

# 提取并显示匹配的实体及其描述
for match_id, start, end in matches:
    span = doc[start:end]
    match_id_str = nlp.vocab.strings[match_id]
    # 获取实体的描述（在句子中的上下文）
    description = doc[start-3:end+3].text  # 获取实体前后各3个词的上下文
    #print(f"{span.text},{match_id_str},")
    #print(f"{span.text},")

# 输出例子


import spacy
from spacy.matcher import Matcher

# 加载spaCy的transformer模型
nlp = spacy.load("en_core_web_trf")

# 初始化Matcher
matcher = Matcher(nlp.vocab)

# 定义模式以匹配更多描述词
patterns = [
    [{"POS": "ADJ"}],  # 单个形容词
    [{"POS": "ADJ"}, {"POS": "NOUN"}],  # 形容词 + 名词
    [{"POS": "NOUN"}],  # 单个名词
    [{"POS": "ADJ", "OP": "*", "TAG": "JJ"}, {"POS": "NOUN", "OP": "*"}]  # 形容词及名词短语
]

# 添加每个模式到matcher
for i, pattern in enumerate(patterns):
    matcher.add(f"DESCRIPTION_PATTERN_{i}", [pattern])


# 函数：提取描述词
def extract_description(text):
    doc = nlp(text)
    matches = matcher(doc)
    descriptions = set()

    # 查看所有匹配的文本和它们的类型
    for match_id, start, end in matches:
        span = doc[start:end]
        print(f"Matched Span: {span.text} (Start: {start}, End: {end})")  # Debug输出
        descriptions.add(span.text.lower())  # 转为小写以避免因大小写不同而重复

    # 将描述词列表按长度从长到短排序以便优先保留较长的描述词
    sorted_descriptions = sorted(descriptions, key=len, reverse=True)

    # 去除包含关系的重复项
    unique_descriptions = []
    for desc in sorted_descriptions:
        if not any(desc in existing for existing in unique_descriptions):
            unique_descriptions.append(desc)

    return unique_descriptions


# 示例句子
text = "A boy lay motionless in the crater, the soil on his body blending him in with the surrounding environment."

# 提取描述词
descriptions = extract_description(text)
print("Extracted Descriptions:", descriptions)

# 生成AI绘画提示词
prompt = ",".join(descriptions)
print("Generated Prompt for AI Drawing:", prompt)







