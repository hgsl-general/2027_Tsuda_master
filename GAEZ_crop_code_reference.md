# GAEZ v5 crop code reference

This document maps crop abbreviations used in the downloaded GAEZ v5 yield layers.

## Important distinction

- `RES02-YLD` is the climate potential yield dataset. In this project these files are stored under `ClimatePotential/`.
- `RES05-YLX` is the attainable yield dataset. In this project these files are stored under `AttainableY/`.
- The two datasets use different crop code systems. For example, wheat is `WHEA` in `RES02-YLD`, but `WHE` in `RES05-YLX`.
- Do not reuse a crop code from one dataset in the other dataset without checking this reference.

## Quick reference for commonly used crops

| crop | RES02 climate_code | RES05 attainable_code |
| --- | --- | --- |
| Wheat | WHEA | WHE |
| Maize | MAIZ | MZE |
| Soybean | SOYB | SOY |
| Wetland rice | RICW | RCW |
| Dryland rice | RICD | RCD |
| Barley | BARL | BRL |
| Sorghum | SORG | SRG |
| White potato | WPOT | WPO |
| Sweet potato | SPOT | SPO |
| Cassava | CASV | CSV |
| Sugar cane | SUGC | SUC |
| Sugar beet | SUGB | SUB |
| Sunflower | SUNF | SFL |
| Rapeseed | RAPE | RSD |
| Cotton | COTT | COT |
| Groundnut | GRND | GRD |
| Alfalfa | ALFA | ALF |
| Pasture grasses | GRAS | PST |

## RES02-YLD climate potential yield codes

- Source: `C:\masterresearch\Comparative_advantage\GAEZ\_readme_res02.xlsx` sheet `CODES_CROP`.
- CSV version: `C:\masterresearch\Comparative_advantage\GAEZ\GAEZ_RES02_climate_potential_crop_codes.csv`.
- Rows: 112.

| climate_code | crop_name | crop_group | uploaded_to_gcs | published_on_platform | npy_file | npy_exists |
| --- | --- | --- | --- | --- | --- | --- |
| ALFA | Alfalfa | Fodder crops | YES | YES | ClimatePotential/climate_potential_yield_ALFA.npy | yes |
| AVOC | Avocado | Fruits | YES | YES | ClimatePotential/climate_potential_yield_AVOC.npy | yes |
| AVOCST | Avocado subtropical ecotype | Fruits | YES | NO | ClimatePotential/climate_potential_yield_AVOCST.npy | yes |
| AVOCTH | Avocado tropical highland | Fruits | YES | NO | ClimatePotential/climate_potential_yield_AVOCTH.npy | yes |
| AVOCTL | Avocado tropical lowland | Fruits | YES | NO | ClimatePotential/climate_potential_yield_AVOCTL.npy | yes |
| BANA | Banana | Fruits | YES | YES | ClimatePotential/climate_potential_yield_BANA.npy | yes |
| BARL | Barley (best sub-type) | Cereals | YES | YES | ClimatePotential/climate_potential_yield_BARL.npy | yes |
| BCKW | Buckwheat | Cereals | YES | YES | ClimatePotential/climate_potential_yield_BCKW.npy | yes |
| BEAN | Phaseolous bean | Pulses | YES | YES | ClimatePotential/climate_potential_yield_BEAN.npy | yes |
| BHSG | Biomass sorghum, highland types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_BHSG.npy | yes |
| BLSG | Biomass sorghum, lowland types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_BLSG.npy | yes |
| BRCH | Brachiaria | Fodder crops | YES | YES | ClimatePotential/climate_potential_yield_BRCH.npy | yes |
| BSRG | Biomass sorghum (best sub-type) | Cereals | YES | YES | ClimatePotential/climate_potential_yield_BSRG.npy | yes |
| BTSG | Biomass sorghum, temperate/sub-tropical types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_BTSG.npy | yes |
| CABB | Cabbage | Vegetables | YES | YES | ClimatePotential/climate_potential_yield_CABB.npy | yes |
| CAME | Camelina (best sub-type) | Oil crops | YES | YES | ClimatePotential/climate_potential_yield_CAME.npy | yes |
| CAMSP | Camelina, spring types | Oil crops | YES | NO | ClimatePotential/climate_potential_yield_CAMSP.npy | yes |
| CAMW | Camelina, autumn and winter types | Oil crops | YES | NO | ClimatePotential/climate_potential_yield_CAMW.npy | yes |
| CARI | Carinata (best sub-type) | Oil crops | YES | YES | ClimatePotential/climate_potential_yield_CARI.npy | yes |
| CARIS | Carinata, spring types | Oil crops | YES | NO | ClimatePotential/climate_potential_yield_CARIS.npy | yes |
| CARIW | Carinata, winter types | Oil crops | YES | NO | ClimatePotential/climate_potential_yield_CARIW.npy | yes |
| CARR | Carrot | Vegetables | YES | YES | ClimatePotential/climate_potential_yield_CARR.npy | yes |
| CASH | Cashew | Fruits | YES | YES | ClimatePotential/climate_potential_yield_CASH.npy | yes |
| CAST | Castor bean | Oil crops | YES | YES | ClimatePotential/climate_potential_yield_CAST.npy | yes |
| CASV | Cassava | Roots and tubers | YES | YES | ClimatePotential/climate_potential_yield_CASV.npy | yes |
| CHCK | Chickpea | Pulses | YES | YES | ClimatePotential/climate_potential_yield_CHCK.npy | yes |
| CITR | Citrus | Fruits | YES | YES | ClimatePotential/climate_potential_yield_CITR.npy | yes |
| COC1 | Coconut, tall | Fruits | YES | NO | ClimatePotential/climate_potential_yield_COC1.npy | yes |
| COC2 | Coconut, hybrid | Fruits | YES | NO | ClimatePotential/climate_potential_yield_COC2.npy | yes |
| COC3 | Coconut, dwarf | Fruits | YES | NO | ClimatePotential/climate_potential_yield_COC3.npy | yes |
| COCC | Cacao, comum | Narcotics and stimulants | YES | NO | ClimatePotential/climate_potential_yield_COCC.npy | yes |
| COCH | Cacao, hybrid | Narcotics and stimulants | YES | NO | ClimatePotential/climate_potential_yield_COCH.npy | yes |
| COCN | Coconut (best sub-type) | Fruits | YES | YES | ClimatePotential/climate_potential_yield_COCN.npy | yes |
| COCO | Cacao (the better of comum and hybrid) | Narcotics and stimulants | YES | YES | ClimatePotential/climate_potential_yield_COCO.npy | yes |
| COFA | Coffee, arabica | Narcotics and stimulants | YES | NO | ClimatePotential/climate_potential_yield_COFA.npy | yes |
| COFF | Coffee (the better of arabica and robusta) | Narcotics and stimulants | YES | YES | ClimatePotential/climate_potential_yield_COFF.npy | yes |
| COFR | Coffee, robusta | Narcotics and stimulants | YES | NO | ClimatePotential/climate_potential_yield_COFR.npy | yes |
| COTT | Cotton | Industrial crops | YES | YES | ClimatePotential/climate_potential_yield_COTT.npy | yes |
| COWP | Cowpea | Pulses | YES | YES | ClimatePotential/climate_potential_yield_COWP.npy | yes |
| DPEA | Dry peas | Pulses | YES | YES | ClimatePotential/climate_potential_yield_DPEA.npy | yes |
| ECAN | Energy cane (the better of V2 and V3) | Bioenergy feedstocks | YES | YES | ClimatePotential/climate_potential_yield_ECAN.npy | yes |
| ECANV2 | Energy cane, V2 | Bioenergy feedstocks | YES | NO | ClimatePotential/climate_potential_yield_ECANV2.npy | yes |
| ECANV3 | Energy cane, V3 | Bioenergy feedstocks | YES | NO | ClimatePotential/climate_potential_yield_ECANV3.npy | yes |
| FIMLT | Finger millet | Cereals | YES | YES | ClimatePotential/climate_potential_yield_FIMLT.npy | yes |
| FLAX | Flax fibre | Industrial crops | YES | YES | ClimatePotential/climate_potential_yield_FLAX.npy | yes |
| FMLT | Foxtail millet | Cereals | YES | YES | ClimatePotential/climate_potential_yield_FMLT.npy | yes |
| FONIO | Fonio | Cereals | YES | YES | ClimatePotential/climate_potential_yield_FONIO.npy | yes |
| GRAM | Gram | Pulses | YES | YES | ClimatePotential/climate_potential_yield_GRAM.npy | yes |
| GRAS | Pasture grasses | Fodder crops | YES | YES | ClimatePotential/climate_potential_yield_GRAS.npy | yes |
| GRLG | Pasture legumes | Fodder crops | YES | YES | ClimatePotential/climate_potential_yield_GRLG.npy | yes |
| GRND | Groundnut | Oil crops | YES | YES | ClimatePotential/climate_potential_yield_GRND.npy | yes |
| GYAM | Yam, greater | Roots and tubers | YES | NO | ClimatePotential/climate_potential_yield_GYAM.npy | yes |
| HBRL | Barley, hybernating types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_HBRL.npy | yes |
| HMZE | Maize, tropical highland types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_HMZE.npy | yes |
| HSRG | Sorghum, tropical highland types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_HSRG.npy | yes |
| HWHE | Wheat, hybernating types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_HWHE.npy | yes |
| JATR | Jatropha | Oil crops | YES | YES | ClimatePotential/climate_potential_yield_JATR.npy | yes |
| LMZE | Maize, tropical lowland types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_LMZE.npy | yes |
| LSRG | Sorghum, tropical lowland types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_LSRG.npy | yes |
| MAIZ | Maize (best sub-type) | Cereals | YES | YES | ClimatePotential/climate_potential_yield_MAIZ.npy | yes |
| MANG | Mango | Fruits | YES | YES | ClimatePotential/climate_potential_yield_MANG.npy | yes |
| MCAU | Macauba palm | Oil crops | YES | YES | ClimatePotential/climate_potential_yield_MCAU.npy | yes |
| MISC | Miscanthus | Bioenergy feedstocks | YES | YES | ClimatePotential/climate_potential_yield_MISC.npy | yes |
| MLLT | Millet (best millet type) | Cereals | YES | YES | ClimatePotential/climate_potential_yield_MLLT.npy | yes |
| MZSI | Maize, silage | Cereals | YES | YES | ClimatePotential/climate_potential_yield_MZSI.npy | yes |
| NAPR | Napier grass | Fodder crops | YES | YES | ClimatePotential/climate_potential_yield_NAPR.npy | yes |
| OATS | Oat | Cereals | YES | YES | ClimatePotential/climate_potential_yield_OATS.npy | yes |
| OILP | Oil palm | Oil crops | YES | YES | ClimatePotential/climate_potential_yield_OILP.npy | yes |
| OKRA | Okra | Vegetables | YES | YES | ClimatePotential/climate_potential_yield_OKRA.npy | yes |
| OLIV | Olive | Oil crops | YES | YES | ClimatePotential/climate_potential_yield_OLIV.npy | yes |
| ONIO | Onion | Vegetables | YES | YES | ClimatePotential/climate_potential_yield_ONIO.npy | yes |
| PIGP | Pigeon pea | Pulses | YES | YES | ClimatePotential/climate_potential_yield_PIGP.npy | yes |
| PMLT | Pearl millet | Cereals | YES | YES | ClimatePotential/climate_potential_yield_PMLT.npy | yes |
| PRUB | Para-rubber | Industrial crops | YES | YES | ClimatePotential/climate_potential_yield_PRUB.npy | yes |
| RAPE | Rapeseed (best sub-type) | Oil crops | YES | YES | ClimatePotential/climate_potential_yield_RAPE.npy | yes |
| RCGR | Reed canary grass | Bioenergy feedstocks | YES | YES | ClimatePotential/climate_potential_yield_RCGR.npy | yes |
| RICD | Rice, dryland | Cereals | YES | YES | ClimatePotential/climate_potential_yield_RICD.npy | yes |
| RICW | Rice, wetland | Cereals | YES | YES | ClimatePotential/climate_potential_yield_RICW.npy | yes |
| RYES | Rye (best sub-type) | Cereals | YES | YES | ClimatePotential/climate_potential_yield_RYES.npy | yes |
| SBRL | Barley, spring types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_SBRL.npy | yes |
| SESA | Sesame | Oil crops | YES | YES | ClimatePotential/climate_potential_yield_SESA.npy | yes |
| SOLA | Solaris energy tobacco | Bioenergy feedstocks | YES | YES | ClimatePotential/climate_potential_yield_SOLA.npy | yes |
| SORG | Sorghum (best sub-type) | Cereals | YES | YES | ClimatePotential/climate_potential_yield_SORG.npy | yes |
| SOYB | Soybean | Oil crops | YES | YES | ClimatePotential/climate_potential_yield_SOYB.npy | yes |
| SPOT | Sweet potato | Roots and tubers | YES | YES | ClimatePotential/climate_potential_yield_SPOT.npy | yes |
| SRAP | Rapeseed, spring types | Oil crops | YES | NO | ClimatePotential/climate_potential_yield_SRAP.npy | yes |
| SRYE | Rye, spring types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_SRYE.npy | yes |
| SUGB | Sugar beet | Sugar crops | YES | YES | ClimatePotential/climate_potential_yield_SUGB.npy | yes |
| SUGC | Sugar cane | Sugar crops | YES | YES | ClimatePotential/climate_potential_yield_SUGC.npy | yes |
| SUNF | Sunflower | Oil crops | YES | YES | ClimatePotential/climate_potential_yield_SUNF.npy | yes |
| SWGR | Switchgrass | Bioenergy feedstocks | YES | YES | ClimatePotential/climate_potential_yield_SWGR.npy | yes |
| SWHE | Wheat, spring types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_SWHE.npy | yes |
| TANN | Tannia (Xanthosoma sagittifolium) | Roots and tubers | YES | YES | ClimatePotential/climate_potential_yield_TANN.npy | yes |
| TARODL | Taro, dryland (Colocasia esculenta) | Roots and tubers | YES | YES | ClimatePotential/climate_potential_yield_TARODL.npy | yes |
| TAROWL | Taro, wetland (Colocasia esculenta) | Roots and tubers | YES | YES | ClimatePotential/climate_potential_yield_TAROWL.npy | yes |
| TBRL | Barley, tropical/sub-tropical types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_TBRL.npy | yes |
| TEAS | Tea (best of China, Assam and hybrid types) | Narcotics and stimulants | YES | YES | ClimatePotential/climate_potential_yield_TEAS.npy | yes |
| TEFF | Tef | Cereals | YES | YES | ClimatePotential/climate_potential_yield_TEFF.npy | yes |
| TMZE | Maize, temperate/sub-tropical types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_TMZE.npy | yes |
| TOBA | Tobacco | Narcotics and stimulants | YES | YES | ClimatePotential/climate_potential_yield_TOBA.npy | yes |
| TOMA | Tomato | Vegetables | YES | YES | ClimatePotential/climate_potential_yield_TOMA.npy | yes |
| TRIT | Triticale | Cereals | YES | YES | ClimatePotential/climate_potential_yield_TRIT.npy | yes |
| TSRG | Sorghum, temperate/sub-tropical types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_TSRG.npy | yes |
| TWHE | Wheat, tropical/sub-tropical types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_TWHE.npy | yes |
| WHEA | Wheat (best sub-type) | Cereals | YES | YES | ClimatePotential/climate_potential_yield_WHEA.npy | yes |
| WMEL | Watermelon | Fruits | YES | YES | ClimatePotential/climate_potential_yield_WMEL.npy | yes |
| WPOT | White potato | Roots and tubers | YES | YES | ClimatePotential/climate_potential_yield_WPOT.npy | yes |
| WRAP | Rapeseed, hybernating types | Oil crops | YES | NO | ClimatePotential/climate_potential_yield_WRAP.npy | yes |
| WRYE | Rye, hybernating types | Cereals | YES | NO | ClimatePotential/climate_potential_yield_WRYE.npy | yes |
| WYAM | Yam, white | Roots and tubers | YES | NO | ClimatePotential/climate_potential_yield_WYAM.npy | yes |
| YAMS | Yam (best of white, greater and yellow yam types) | Roots and tubers | YES | YES | ClimatePotential/climate_potential_yield_YAMS.npy | yes |
| YYAM | Yam, yellow | Roots and tubers | YES | NO | ClimatePotential/climate_potential_yield_YYAM.npy | yes |

## RES05-YLX attainable yield codes

- Source: `C:\masterresearch\Comparative_advantage\GAEZ\res05-ylx.json`.
- CSV version: `C:\masterresearch\Comparative_advantage\GAEZ\GAEZ_RES05_attainable_yield_crop_codes.csv`.
- Rows: 72.

| attainable_code | crop_name | npy_file | npy_exists |
| --- | --- | --- | --- |
| ALF | Alfalfa | AttainableY/attainable_yield_ALF.npy | yes |
| AVOC | Avocado | AttainableY/attainable_yield_AVOC.npy | yes |
| BAN | Banana | AttainableY/attainable_yield_BAN.npy | yes |
| BCH | Brachiaria | AttainableY/attainable_yield_BCH.npy | yes |
| BCK | Buckwheat | AttainableY/attainable_yield_BCK.npy | yes |
| BRL | Barley | AttainableY/attainable_yield_BRL.npy | yes |
| BSG | Biomass sorghum | AttainableY/attainable_yield_BSG.npy | yes |
| CAB | Cabbage | AttainableY/attainable_yield_CAB.npy | yes |
| CAM | Camelina | AttainableY/attainable_yield_CAM.npy | yes |
| CAR | Carrot | AttainableY/attainable_yield_CAR.npy | yes |
| CHK | Chickpea | AttainableY/attainable_yield_CHK.npy | yes |
| CIT | Citrus | AttainableY/attainable_yield_CIT.npy | yes |
| COC | Cocoa | AttainableY/attainable_yield_COC.npy | yes |
| COF | Coffee | AttainableY/attainable_yield_COF.npy | yes |
| CON | Coconut | AttainableY/attainable_yield_CON.npy | yes |
| COT | Cotton | AttainableY/attainable_yield_COT.npy | yes |
| COW | Cowpea | AttainableY/attainable_yield_COW.npy | yes |
| CRT | Carinata | AttainableY/attainable_yield_CRT.npy | yes |
| CSH | Cashew | AttainableY/attainable_yield_CSH.npy | yes |
| CST | Castor bean | AttainableY/attainable_yield_CST.npy | yes |
| CSV | Cassava | AttainableY/attainable_yield_CSV.npy | yes |
| ECN | Energy cane | AttainableY/attainable_yield_ECN.npy | yes |
| FIML | Finger millet | AttainableY/attainable_yield_FIML.npy | yes |
| FLX | Flax | AttainableY/attainable_yield_FLX.npy | yes |
| FML | Foxtail millet | AttainableY/attainable_yield_FML.npy | yes |
| FONIO | Fonio | AttainableY/attainable_yield_FONIO.npy | yes |
| GRD | Groundnut | AttainableY/attainable_yield_GRD.npy | yes |
| GRM | Gram | AttainableY/attainable_yield_GRM.npy | yes |
| JTR | Jatropha | AttainableY/attainable_yield_JTR.npy | yes |
| MCA | Macauba palm | AttainableY/attainable_yield_MCA.npy | yes |
| MIS | Miscanthus | AttainableY/attainable_yield_MIS.npy | yes |
| MNG | Mango | AttainableY/attainable_yield_MNG.npy | yes |
| MZE | Maize | AttainableY/attainable_yield_MZE.npy | yes |
| MZS | Silage maize | AttainableY/attainable_yield_MZS.npy | yes |
| NAP | Napier grass | AttainableY/attainable_yield_NAP.npy | yes |
| OAT | Oat | AttainableY/attainable_yield_OAT.npy | yes |
| OKRA | Okra | AttainableY/attainable_yield_OKRA.npy | yes |
| OLP | Oil palm | AttainableY/attainable_yield_OLP.npy | yes |
| OLV | Olive | AttainableY/attainable_yield_OLV.npy | yes |
| ONI | Onion | AttainableY/attainable_yield_ONI.npy | yes |
| PEA | Dry pea | AttainableY/attainable_yield_PEA.npy | yes |
| PHB | Phaseolus bean | AttainableY/attainable_yield_PHB.npy | yes |
| PIG | Pigeonpea | AttainableY/attainable_yield_PIG.npy | yes |
| PML | Pearl millet | AttainableY/attainable_yield_PML.npy | yes |
| PST | Pasture grasses | AttainableY/attainable_yield_PST.npy | yes |
| RCD | Dryland rice | AttainableY/attainable_yield_RCD.npy | yes |
| RCG | Reed canary grass | AttainableY/attainable_yield_RCG.npy | yes |
| RCW | Wetland rice | AttainableY/attainable_yield_RCW.npy | yes |
| RSD | Rapeseed | AttainableY/attainable_yield_RSD.npy | yes |
| RUB | Rubber | AttainableY/attainable_yield_RUB.npy | yes |
| RYE | Rye | AttainableY/attainable_yield_RYE.npy | yes |
| SES | Sesame | AttainableY/attainable_yield_SES.npy | yes |
| SFL | Sunflower | AttainableY/attainable_yield_SFL.npy | yes |
| SOL | Solaris energy tobacco  | AttainableY/attainable_yield_SOL.npy | yes |
| SOY | Soybean | AttainableY/attainable_yield_SOY.npy | yes |
| SPO | Sweet potato | AttainableY/attainable_yield_SPO.npy | yes |
| SRG | Sorghum | AttainableY/attainable_yield_SRG.npy | yes |
| SUB | Sugar beet | AttainableY/attainable_yield_SUB.npy | yes |
| SUC | Sugar cane | AttainableY/attainable_yield_SUC.npy | yes |
| SWG | Switchgrass | AttainableY/attainable_yield_SWG.npy | yes |
| TANN | Tannia | AttainableY/attainable_yield_TANN.npy | yes |
| TAROD | Taro, dryaland | AttainableY/attainable_yield_TAROD.npy | yes |
| TAROW | Taro, wetland | AttainableY/attainable_yield_TAROW.npy | yes |
| TEA | Tea | AttainableY/attainable_yield_TEA.npy | yes |
| TEF | Tef | AttainableY/attainable_yield_TEF.npy | yes |
| TOB | Tobacco | AttainableY/attainable_yield_TOB.npy | yes |
| TOM | Tomato | AttainableY/attainable_yield_TOM.npy | yes |
| TRI | Triticale | AttainableY/attainable_yield_TRI.npy | yes |
| WHE | Wheat | AttainableY/attainable_yield_WHE.npy | yes |
| WMEL | Watermelon | AttainableY/attainable_yield_WMEL.npy | yes |
| WPO | White potato | AttainableY/attainable_yield_WPO.npy | yes |
| YAM | Yam | AttainableY/attainable_yield_YAM.npy | yes |

## Notes

- `published_on_platform = NO` in the RES02 table generally indicates crop subtypes or internal variants; they were still available in Google Cloud and were downloaded as NPY files in this project.
- RES02 includes more subtype-specific codes than RES05. For cross-dataset comparisons, prefer best-subtype or aggregate crop codes where possible.
- All downloaded arrays are expected to be `2160 x 4320` and stored as `float32` NPY files.
