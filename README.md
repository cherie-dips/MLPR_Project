# MLPR_Project

### __AI-Assisted pH-Based Colorimetric Monitoring of Chronic Wound Healing Using Pyranine-Infused Silk Fibroin Hydrogel Dressings__

Chronic wounds, like diabetic foot ulcers (DFUs), are a significant global health burden, requiring continuous, non-invasive monitoring to prevent complications like infection and amputation. Conventional wound care involves manual dressing removal and exudate analysis, which is painful, time-consuming, raise amputation risks, impairing recovery. To address this challenge, we develop a machine learning (ML) model to support and complement the smart hydrogel wound dressing being created at the Centre for Equitable and Personalized Healthcare (CEPH), Plaksha University.

This solution integrates pyranine (HPTS)-infused silk fibroin (SF) hydrogels into dressings that visually respond to wound pH changes through colorimetric shifts, enabling real-time, point-of-care monitoring. We constructed a dataset of ~2000 stereomicroscopic images of these hydrogels under lab-simulated wound conditions across four pH levels: pH5, pH6, pH7, and pH8. Each image was preprocessed to extract 33 features—24 from HSV color histograms and 9 from statistical moments across RGB channels.

A Random Forest Classifier (RFC) combined with ResNet-18, trained on the extracted feature set, achieved 80% accuracy in classifying images into their respective pH classes based on the hydrogel image and time point. This pH level is subsequently correlated to wound healing status as the end output.

This enables wound status assessment without specialized equipment, promoting accurate self-monitoring, reducing hospital visits, and supporting timely, cost-effective care at home or point-of-care settings.

Keywords: Chronic Wounds, Diabetic Foot Ulcers (DFUs), Smart Hydrogel Dressings, Pyranine (HPTS), Silk Fibroin (SF) Hydrogels, Machine Learning, Image Analysis, pH Monitoring, Wound Healing, Random Forest Classifier, Image Segmentation, Non-invasive Monitoring, Point-of-Care Healthcare, Bandage Lifetime, Hydrogel Degradation.
