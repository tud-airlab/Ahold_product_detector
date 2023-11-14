from pathlib import Path
from typing import Union

import numpy as np
import torch
from torchvision.transforms import v2
import PIL.Image as Image


class ImageLoader:
    def __init__(self, image_size, mean_pixel_values=None, std_pixel_values=None):
        if std_pixel_values is None:
            std_pixel_values = [1, 1, 1]
        if mean_pixel_values is None:
            mean_pixel_values = [0, 0, 0]
        self.image_size = image_size
        self.mean_pixel_values = mean_pixel_values
        self.std_pixel_values = std_pixel_values

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __call__(self, image: Union[Path, np.ndarray, Image.Image]):
        transforms = [
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.mean_pixel_values, std=self.std_pixel_values),
        ]
        if isinstance(image, Path):
            transforms = [v2.ToImage()] + transforms
            image = Image.open(image).convert('RGB')
        if isinstance(image, Image.Image):
            transforms = [v2.ToImage()] + transforms
        transform = v2.Compose(transforms)
        return transform(image)


##############################
# Dataset dependent values
##############################
IMAGE_SIZE = 80
MEAN_PIXEL_VALUES = [0.4416, 0.4286, 0.4058]
STD_PIXEL_VALUES = [0.2010, 0.1880, 0.2029]
IMAGE_LOADER = ImageLoader(image_size=IMAGE_SIZE,
                           mean_pixel_values=MEAN_PIXEL_VALUES,
                           std_pixel_values=STD_PIXEL_VALUES)

SEEN_CLASSES = ['10_AH_Sesam_Woksaus - 8718907306744',
                '13_AH_Terriyaki_Woksaus - 8718907306751',
                '14_AH_Volle_Melk - 8718907056298',
                '15_AH_appelmoes - 8710400011668',
                '16_Bonduelle_Beluga_Linzen_2_pack - 3083681149630',
                '17_Bonduelle_Crispy_Mais - 3083681025484',
                '18_Bonduelle_Kikkererwten_2_pack - 3083681068146',
                '19_Bonduelle_Rode_Bietenblokjes_2_pack - 3083681126471',
                '20_Bonduelle_Rode_Kidneybonen_2_pack - 3083681068122',
                '21_Campina_Halfvolle_Melk - 8712800147008',
                '22_Campina_Volle_Melk - 8712800147770',
                '23_Conimex_Kip_Kerrie_Madras - 8714100795699',
                '24_Conimex_Nasi_Speciaal - 8714100795774',
                '26_HAK_Rode_Bieten - 8720600609893',
                '27_HAK_appelmoes - 8720600606731',
                '29_Kokh_Thai_Rode_Curry - 8717662264368',
                '30_Servero_Appelmoes - 87343267',
                '31_Zaanse_Hoeve_Halfvolle_Melk - 8710400514107',
                '32_Zaanse_Hoeve_Halfvolle_Yoghurt - 8718906872844',
                '33_Zaanse_Hoeve_Magere_Yoghurt - 8718907039987',
                '34_Zaanse_Hoeve_Volle_Melk - 8710400416395',
                '35_Zaanse_Hoeve_Volle_Yoghurt - 8718907039963',
                '36_Croky_Paprika - 5414359921711',
                '38_AH_Biologisch_Tomatensoep - 8718265082151',
                '39_AH_Romige_Tomatensoep - 8718906536265',
                '3_AH_Chocolate_Chip_Cookies - 8718907400701',
                '40_Palmolive_Naturals_Shampoo - 8718951065826',
                '41_Listerine_Mondwater_Sensitive - 3574661734712',
                '43_Sun_All-in-one - 8720181388774',
                '46_Spa_Intense_Mineraalwater - 5410013114697',
                '48_Mutti_Polpa - 80042556',
                '49_Mutti_Passata - 80042563',
                '50_Mutti_Polpa_2_pack - 8005110170324',
                '51_DeCecco_Penne_Rigate_Integrale - 8001250310415',
                '52_LaMolisana_Penne_rigate - 8004690052044',
                '53_AH_bio_Sinaasappelsap - 8718906124066',
                '55_Cocacola_1-5L - 9999',
                '6_AH_Hollandse_Bruine_Bonen - 8710400035862',
                '7_AH_Kwarktaart_Aardbei - 8710400687122',
                '8_AH_Magere_Melk - 8718907056311',
                '9_AH_Rodekool - 8710400145981']
SEEN_COLOR = (107, 191, 61)

UNSEEN_CLASSES = ['1_AH_Brownies - 8718907400718',
                  '25_HAK_Bruine_Bonen - 8720600612848',
                  '2_AH_Cheesecake - 8718907457583',
                  '37_AH_Mini_Friet - 8718906948631',
                  '42_Lenor_Wasverzachter_Orchidee - 8006540896778',
                  '4_AH_Fijngesneden_Tomaten - 8718906697560',
                  '54_AH_bio_appel_sap - 8718906124073',
                  '0_AH_Bio_Houdbare_Halfvolle_Melk - 8710400280507',
                  '11_AH_Sweet_Basil_Woksaus - 8718907306775',
                  '12_AH_Sweet_and_Sour_Woksaus - 8718907306737',
                  '28_Kokh_Thai_Groene_Curry - 8717662264382',
                  '44_RedBull_Cactusvrucht - 90453656',
                  '45_RedBull_Energy_Drink - 90453533',
                  '47_Spa_Reine - 5410013136149',
                  '5_AH_Halfvolle_Melk - 8718907056274']
UNSEEN_COLOR = (97, 165, 232)

DEFAULT_COLOR = (133, 143, 153)
