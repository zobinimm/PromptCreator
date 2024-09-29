# Prompt Creator

## Dependencies and Installation in Windows

Find a suitable directory and use `git clone` to get the "Prompt Creator" project code onto your local machine.

```bash
git clone https://github.com/zobinimm/PromptCreator.git
```

Execute the following commands in the project directory to complete the model application build:

```bash
pip install -r requirements.txt
or
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# en_core_web_trf
python -m spacy download en_core_web_trf
or
pip install /path/to/en_core_web_trf-3.7.3-py3-none-any.whl 

# zh_core_web_trf
python -m spacy download zh_core_web_trf
or
pip install /path/to/zh_core_web_trf-3.7.2-py3-none-any.whl
```

## ChatTTS
GPU before
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```
Copy torio from ChatTTS's package into the package
```bash
pip install git+https://github.com/2noise/ChatTTS
```
## pyJianYingDraft
```bash
cd libs\pyJianYingDraft
pip install .
```
## Run
```bash
python app.py
```