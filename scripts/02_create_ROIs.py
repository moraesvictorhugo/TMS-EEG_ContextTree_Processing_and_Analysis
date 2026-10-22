# This script should import Epochs cleanned and export new epochs of ROIs based in defined ROIs of config
# The ROI analisys will not substitute the channel analisys

# THis analisys must not be hardcoded. Must use the ROIs setup in config. Using at first this configuration:

# This ROI analisys must calculate the average signal of groups of electrodes. I want to create an epoch object with this ROIs, if possible:

'''
Frontal: esquerda com F3 e F7; direita com F4 e F8.

Sensório-motora: esquerda com FC1, FC5, C3, CP1 e CP5; direita com FC2, FC6, C4, CP2 e CP6. Essa seria a principal comparação para estimulação de M1 esquerdo: região ipsilateral versus contralateral. É uma ROI sensório-motora ampla, não uma medida exclusiva de M1.

Parietal: esquerda com P3 e P7; direita com P4 e P8. Nessa divisão, CP1/CP2 e CP5/CP6 ficam exclusivamente na ROI sensório-motora, evitando sobreposição.

Occipital: esquerda com O1 e PO3; direita com O2 e PO4, caso PO3/PO4 estejam disponíveis.

Temporal: T7 à esquerda e T8 à direita, como análise secundária. São medidas de eletrodo único, portanto menos beneficiadas pela redução de ruído obtida com a média espacial.

Mantenha os eletrodos da linha média separados — por exemplo, Fz, Cz e Pz — em vez de incluí-los nos dois hemisférios. Fp1/Fp2 e AF3/AF4, quando presentes, podem formar um par de ROIs frontopolares separado, se isso responder a uma hipótese específica.
'''