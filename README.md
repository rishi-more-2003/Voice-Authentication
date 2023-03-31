# Voice Based Authentication
* Voice Authentication Application created using a Siamese Network Approach.
* Dataset Used: [Vox-Celeb1 Indian](https://www.kaggle.com/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity)
* Trained two variants of the model:
  * 1.4M Parameters on 100K samples.
  * 3M Parameters on 10K samples.
  * 900K Parameters on 1M samples

* Preprocessed the audio dataset using Librosa with Fast Fourier Transform to extract vocal features.
* Further used torch modules to preprocess audio in real time.
* Improved the model accuracy to about 90%.
* Created a novel siamese model architecture for extraction of audio features to identify speaker and verify speaker while keeping low overhead and computational delay.
