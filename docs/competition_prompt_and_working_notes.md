competetion: https://www.kaggle.com/competitions/multilingual-speech-recognition

Multilingual Speech Recognition
Transcribe the audio given in multiple languages


Overview
The NPPE-2: Multilingual Speech Recognition competition challenges participants to build a system capable of accurately transcribing speech from audio recordings in multiple languages.

The primary goal of this competition is to develop practical skills required for real-world speech recognition tasks, including:

Multilingual modeling: Learning how to build models that generalize across multiple languages with diverse phonetic and linguistic characteristics.
Data preprocessing as a core skill: Developing the ability to clean, normalize, and prepare speech datasets, which is essential for achieving strong performance in ASR systems.
Model selection for speech tasks: Identifying and fine-tuning suitable architectures (e.g., Whisper, wav2vec2, or other sequence-to-sequence models) that can effectively handle multilingual speech under limited computational resources.
This challenge emphasizes practical deep learning expertise, requiring participants to:

Design a robust data preprocessing pipeline for audio and transcripts.
Select and fine-tune appropriate ASR models for multilingual transcription.
Work within strict compute constraints using Kaggle Notebooks only.
Develop an end-to-end system that converts raw audio into accurate verbatim transcripts.
Participants are encouraged to experiment with data augmentation, language balancing strategies, and efficient fine-tuning techniques to improve transcription accuracy across languages.

Evaluation
Metric: Word Error Rate (WER)
Your submissions will be evaluated using the Word Error Rate (WER), which is the standard metric for assessing the accuracy of automatic speech recognition systems. It measures the number of errors (substitutions, deletions, and insertions) between your predicted transcript and the ground truth.

Goal: To minimize the difference between the predicted and actual verbatim transcript.
Range: 0.0 to ∞ (a score of 0.0 means a perfect prediction).
Lower is better.
Scoring Formula:
WER = (Substitutions + Deletions + Insertions) / Number of Words in Reference
Substitutions: A word in the reference is replaced by another word.
Deletions: A word in the reference is missing from your prediction.
Insertions: A word is present in your prediction but not in the reference.

Dataset Description(D:\sagan\dlp\nppe 2\data\multilingual-speech-recognition.zip):
train.csv - the training set
test.csv - the test set
competition_data - contains audio files
submission.csv should have "audio",	"text" columns

I want to optimize for least wer.
I only have 6 hours left for competetion. 
I will upload the submission through kaggle.
My kaggle gpu hours are used so, I  will be using colab's gpu
I have download the zipped data in drive and locally also.
I have multiple csv files and notebooks here locally.

kaggle path: 

'''
RUNNING_IN_KAGGLE = False # Set to True when you move this to Kaggle

if RUNNING_IN_KAGGLE:
    # 1. KAGLGE PATHS (Standard structure)
    INPUT_PATH = "/kaggle/input/multilingual-speech-recognition"
    TRAIN_CSV = os.path.join(INPUT_PATH, "train.csv")
    TEST_CSV = os.path.join(INPUT_PATH, "test.csv")

    # Paths to the actual audio folders
    TRAIN_DIR = os.path.join(INPUT_PATH, "competition_data", "train")
    TEST_DIR = os.path.join(INPUT_PATH, "competition_data", "test")
    SUBMISSION_PATH = "/kaggle/working/submission.csv"

else:
    # 2. COLAB PATHS (Drive structure)
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')

    # Unzip if the folder doesn't exist yet
    if not os.path.exists('/content/competition_data'):
        # 🚨 Make sure this filename matches your Drive exactly!
        !unzip -q /content/drive/MyDrive/multilingual-speech-recognition.zip -d /content/competition_data
        print("✅ Dataset unzipped!")

    # Path where we unzipped
    BASE_EXTRACT_DIR = "/content/competition_data"

    # CSVs are usually in the root of the extract or the root of the zip
    TRAIN_CSV = os.path.join(BASE_EXTRACT_DIR, "train.csv")
    TEST_CSV = os.path.join(BASE_EXTRACT_DIR, "test.csv")

    # Audio folders (Adjusting for the common 'competition_data/competition_data' unzip quirk)
    TRAIN_DIR = os.path.join(BASE_EXTRACT_DIR, "competition_data", "train")
    TEST_DIR = os.path.join(BASE_EXTRACT_DIR, "competition_data", "test")
    SUBMISSION_PATH = "/content/submission.csv"
'''

i want wer of 0.1, someone told me 
"
This is what I did , that got me 0.19 , I hope its helpful.
You dont even need finetuning for it, It took me just 5 - 7mins to run the entire nb. I tried finetuning but it was worse for me.
Just try this
- Use LID to detect the lang. 
- Use two models (1. Whisper large, 2. indic conformer 600m , both of this can be imported from the transformer library , just make a req. for indic in their hf repo)
- Make some helper func to Route the eng. lang to whisper and the Indian lang to indic conformer. (Note that the dataset has mainly Tamil, Hindi, Eng as the major lang. It has some other lang too but they are small in no.)
- Convert eng to small case before submitting

Bonus- Some audio which were detected by LID as urdu (ur) but their true transcription is in hindi, but if you apply the above routing make sure the urdu also goes hindi same for ml should go in ta(tamil)
All the best.
"
Another 
make a notebook for me to get wer of 0.1. and run it in colab using gpu. Then, I will upload it in kaggle.
Or you can just give me a ready notebook to submit in kaggle, I can submit directly without running in google colab.
I have saved 
"
sagandeep/whisper-medium-hindi-lora-v2
sagandeep/whisper-medium-checkpoint-675
sagandeep/whisper-medium-checkpoint-500
" these model checkpoints, after finetuning on training data , saved in hugging face.
make sure paths are right.
if you want extra info, you can ask me.
