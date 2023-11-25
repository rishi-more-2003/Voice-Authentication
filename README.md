# Voice Based Authentication
* Voice Authentication Application created using a Siamese Network Approach.
* Dataset Used: [Vox-Celeb1 Indian](https://www.kaggle.com/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity)
* Trained three variants of the model:
  * 1.4M Parameters on 100K samples.
  * 3M Parameters on 10K samples.
  * 900K Parameters on 1M samples

* Preprocessed the audio dataset using Librosa with Fast Fourier Transform to extract vocal features.
* Further used torch to preprocess audio in real time.
* Created a novel siamese model architecture for extraction of audio features to identify speaker and verify speaker while keeping low overhead and computational delay.
* Improved the model accuracy to around 90%.
* :memo: Currently working on detailing the architectural optimizations achieved by globally caching hann windows and embedding the mel spectrograms into 450*80 matrices through a research paper. 
