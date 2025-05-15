# MLPR_Project

### __Machine learning assisted Chronic Wound status assessment using smart dressings__

Chronic wounds, like diabetic foot ulcers (DFUs), are a significant global health burden, requiring continuous, non-invasive monitoring to prevent complications like infection and amputation. Conventional wound care involves manual dressing removal and exudate analysis, which is painful, time-consuming, raise amputation risks, impairing recovery. To address this challenge, we develop a machine learning (ML) model to support and complement the smart hydrogel wound dressing being created at the Centre for Equitable and Personalized Healthcare (CEPH), Plaksha University.

This solution integrates pyranine (HPTS)-infused silk fibroin (SF) hydrogels into dressings that visually respond to wound pH changes through colorimetric shifts, enabling real-time, point-of-care monitoring. We constructed a dataset of ~1500 stereomicroscopic images of these hydrogels under lab-simulated wound conditions across four pH levels: pH5, pH6, pH7, and pH8. Each image was resized and normalized and then converted to tensors for ResNet.

Then the images were passed through a ResNet-18 backbone for deep feature extraction, followed by a Random Forest Classifier (RFC) for pH classification. The model achieved 80% accuracy in assigning images to their respective pH classes, which were then correlated to wound healing status. Additionally, hydrogel degradation was monitored through imaging to estimate bandage lifetime, enabling timely replacement for accurate diagnosis.

This enables wound status assessment without specialized equipment, promoting accurate self-monitoring, reducing hospital visits, and supporting timely, cost-effective care at home or point-of-care settings.

![WOUND HEALING-3](https://github.com/user-attachments/assets/de49864a-026c-4d51-aa76-3a3fdcbb0d9a)
