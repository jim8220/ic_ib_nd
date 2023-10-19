# Inter-channel information-based noise detector

![fig1](https://github.com/jim8220/ic_ib_nd/assets/68427972/cc192122-db30-4c8b-a06e-4fe142e92381)

# Dataset preparation
Run mixer.py
# Noise detection
## on-site
Modify machine_type, bg, rseed of baseline.yaml and run noise_detection.py
## other site
Modify machine_type, bg_from, bg_to, rseed of infer2others.yaml and run noise_detection_infer2others.py

### draw
Modify machine_type, bg, rseed of baseline.yaml and run draw.py

If you have any questions, please email to lasscap@kaist.ac.kr
