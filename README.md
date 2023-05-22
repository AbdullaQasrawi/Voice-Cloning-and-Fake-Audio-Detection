# Voice-Cloning-and-Fake-Audio-Detection


## Background

We are a technology company working in the Cyber Security industry. Our goal is to build algorithms that can synthesize spoken audio by converting a speaker’s voice to another speaker’s voice and detect if any spoken audio is authentic or fake.

## Data Description

We have two datasets available for this project:

1. TIMIT Dataset: The TIMIT corpus of read speech is designed to provide speech data for acoustic-phonetic studies and for the development and evaluation of automatic speech recognition systems. It contains 6300 sentences spoken by 630 speakers from 8 major dialect regions of the United States. [Dataset Link](https://github.com/philipperemy/timit)

2. CommonVoice Dataset: Common Voice is a corpus of speech data read by users on the Common Voice website. It contains speech samples from various sources and is primarily used for training and testing automatic speech recognition (ASR) systems. [Dataset Link](https://commonvoice.mozilla.org/en/datasets)

## Goal

The goal of this project is to build a machine learning system that can:
1. Clone a speaker's voice to another speaker's voice using the TIMIT dataset.
2. Detect if a spoken audio is synthetically generated or natural using the CommonVoice dataset.

## Voice Cloning (VC) System

The voice cloning system utilizes the TIMIT dataset for training and synthesizing spoken audio. It consists of the following components:
- Encoder: Pretrained encoder model for extracting speaker embeddings.
- Synthesizer: Model for synthesizing spectrograms from text and speaker embeddings.
- Vocoder: Pretrained vocoder model for generating the waveform from spectrograms.

To clone audio, follow these steps:
1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Clone the Real-Time-Voice-Cloning repository from [GitHub](https://github.com/misbah4064/Real-Time-Voice-Cloning.git).
3. Load the pretrained encoder and vocoder models.
4. Synthesize new audio by providing a source speaker's spoken audio and target speaker's voice.

## Fake Audio Detection (FAD) System

The fake audio detection system utilizes the CommonVoice dataset for training and testing. It involves the following steps:
1. Collect real and fake audio samples.
2. Extract audio features such as MFCCs, chroma, mel, contrast, and tonnetz.
3. Scale the features using StandardScaler.
4. Train a binary classification model using the scaled features.
5. Save the trained model for future use.

## Usage

Provide instructions on how to use the voice cloning and fake audio detection systems, including code examples and any specific steps or considerations.

## License

Specify the license under which the code and other project materials are released.

## Acknowledgements

Mention any acknowledgements or credits for any external libraries, code snippets, or datasets used in the project.

## Contributing

Specify guidelines for contributing to the project, including how others can submit bug reports, feature requests, and pull requests.

## Contact

Provide contact information (email, website, etc.) for users to get in touch with the project maintainers or contributors.

