# V1-Visual-Cortex-Visualization

This repository generates images based upon V1 visual cortex neural activity.
This is brain-based image reconstruction with generalization to arbitrary input.

---

[Dataset](https://crcns.org/data-sets/vc/pvc-1) | <- Link to the dataset.

---

[Prompt](https://grok.com/share/bGVnYWN5_d17bf28b-bfa4-49fe-a9a1-c21408b8f558) | <- Link to grok response:

I am working on a project that will make a prediction of what stimulus image is associated with what neural response. This will then be leveraged to create a generative model that will create images based upon neural input. 

This is the dataset:

https://crcns.org/data-sets/vc/pvc-1

Here is the video of generative imagery:
https://www.youtube.com/watch?v=88I7gLR5v_A&list=PL9rU625vkl4XmGq7i-zZbVuVw3g5ezl6o


## Objective

- **Prediction**: Train a model to predict the neural response based on pairing of natural movie segments and associated neural responses recorded from the primary visual cortex (V1) of macaque monkeys. In other words, given an image, what is the predicted neural response?

- **Generation**: Use the neural responses to generate new images using the association of the neural responses recorded from the primary visual cortex (V1) of macaque monkeys and the stimulus images as training data. In other words, reverse the previous prediction: given a neural response, generate the image that ellictited the neural response. What is the monkey seeing?

---

## References:
- [Generative Imagery](https://www.youtube.com/watch?v=88I7gLR5v_A&list=PL9rU625vkl4XmGq7i-zZbVuVw3g5ezl6o)

- [Single and multi-unit recordings from primary visual cortex](https://crcns.org/data-sets/vc/pvc-1)

---

## Data Flow Diagram

```
     (Train Phase)
   ┌──────────────┐
   │  Image Input │
   └──────┬───────┘
          ▼
  ┌─────────────────┐
  │  Image Encoder  │  (ResNet / ViT)
  └─────────────────┘
          ▼
    z_image_latent ───────┐
                          ▼
              ┌────────────────────┐
              │  Waveform Decoder  │ (MLP or 1D CNN)
              └────────────────────┘
                          ▼
                 Synthetic Waveform
                          ▼
              ┌────────────────────┐
              │  Waveform Encoder  │
              └────────────────────┘
                          ▼
    z_waveform_latent ◄────── latent alignment loss ──────► z_image_latent
                          ▼
              ┌──────────────────┐
              │  Image Decoder   │
              └──────────────────┘
                          ▼
               Reconstructed Image
```

```
Simulation Path (How to See)
Image ─▶ Image Encoder ─▶ z_image_latent ─▶ Waveform Decoder ─▶ Synthetic Waveform
```

```
Reconstruction Path (How to Visualize Sight, Imagination, and Dreams)
Waveform ─▶ Waveform Encoder ─▶ z_waveform_latent ─▶ Image Decoder ─▶ Reconstructed Image
```

## Model Project Architecture
```
project/
├── data/
│   ├── dataset.py             # Image + waveform loader
├── models/
│   ├── image_encoder.py       # CNN (ResNet) image → z
│   ├── waveform_decoder.py    # MLP z → waveform
│   ├── waveform_encoder.py    # MLP waveform → z
│   ├── image_decoder.py       # CNN decoder z → image
│   ├── __init__.py            # Shared architecture utils
├── train.py                   # Trains everything (2 phases)
├── eval.py                    # Runs SSIM, PSNR, MSE
├── config.yaml                # Configurable hyperparams
├── utils.py                   # Logger, metrics, visualizer
└── README.md                  # Usage + dependencies
```

## Full-Stack Project Architecture
```
[Simulation API: WebSocket Server]
┌─────────────────────────────┐
│ Accept a random image       │
│ └── image_encoder → z       │
│     └── waveform_decoder    │
│         └── Send to Relay   │
└─────────────────────────────┘

[Relay API: WebSocket Server]
┌───────────────────────────┐
│ Receive waveform_latent   │
│ └── waveform_encoder → z  │
│     └── image_decoder     │
│         └── Buffer image  │
│             └── Respond   │
└───────────────────────────┘

[Frontend: React]
┌───────────────────────────┐
│ Thought-to-Image button   │
│ └── Poll Relay API WS     │
│     └── Receive image     │
│         └── Display       │
└───────────────────────────┘

```

## Full-Stack Development Time:
```
| Task                                    | Time Estimate   |
| --------------------------------------- | --------------- |
| ✅ Webcam capture & preprocessing        | 0.5 hour        |
| ✅ Integrate image encoder model         | 0.5 hour        |
| ✅ Generate waveform latent              | 0.5 hour        |
| ✅ Send waveform to relay via WebSocket  | 0.5 hour        |
| ✅ Relay receives, decodes image         | 1.5 hours       |
| ✅ Frontend polling WebSocket + image UI | 1.5 hours       |
| ✅ Testing + debugging                   | 1.5 hours       |
| **Total**                               | **\~7.5 hours** |
```

## Simulation -> Relay Message Format
```
{
  "type": "waveform_latent",
  "session_id": "xyz123",
  "payload": [0.023, 0.55, ..., -0.011]  // z_waveform_latent vector
}
```

## Relay -> Simulation Message Format
```
{
  "type": "reconstructed_image",
  "session_id": "xyz123",
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSk..."
}
```

