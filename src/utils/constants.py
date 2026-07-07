
from typing import List
import matplotlib.pyplot as plt
import platform
import torch
import gc

# ============================================================================
# KOREAN TO ENGLISH VARIABLE NAME MAPPING
# ============================================================================
KOREAN_TO_ENGLISH_MAP = {
    '수온 (℃)': 'Water Temperature (°C)',
    '수소이온농도': 'pH',
    '전기전도도 (μS/cm)': 'Conductivity (μS/cm)',
    '용존산소 (mg/L)': 'DO (mg/L)',
    '탁도 (NTU)': 'Turbidity (NTU)',
    '총유기탄소 (mg/L)': 'TOC (mg/L)',
    '클로로필-a (mg/㎥)': 'Chl-a (mg/m³)',
    '염화메틸렌 (μg/L)': 'Chloromethane (μg/L)',
    '1.1.1-트리클로로에테인 (μg/L)': '1,1,1-Trichloroethane (μg/L)',
    '사염화탄소 (μg/L)': 'Carbon tetrachloride (μg/L)',
    '트리클로로에틸렌 (μg/L)': 'Trichloroethylene (μg/L)',
    '테트라클로로에틸렌 (μg/L)': 'Tetrachloroethylene (μg/L)',
    '벤젠 (μg/L)': 'Benzene (μg/L)',
    '톨루엔 (μg/L)': 'Toluene (μg/L)',
    '에틸벤젠 (μg/L)': 'Ethylbenzene (μg/L)',
    'm,p-자일렌 (μg/L)': 'm,p-Xylene (μg/L)',
    'o-자일렌 (μg/L)': 'o-Xylene (μg/L)',
    '[ECD]염화메틸렌 (μg/L)': 'Chloromethane (μg/L)',
    '[ECD]1.1.1-트리클로로에테인 (μg/L)': '1,1,1-Trichloroethane (μg/L)',
    '[ECD]사염화탄소 (μg/L)': 'Carbon tetrachloride (μg/L)',
    '[ECD]트리클로로에틸렌 (μg/L)': 'Trichloroethylene (μg/L)',
    '[ECD]테트라클로로에틸렌 (μg/L)': 'Tetrachloroethylene (μg/L)',
    '총질소 (mg/L)': 'TN (mg/L)',
    '총인 (mg/L)': 'TP (mg/L)',
    '클로로필-a (mg/㎥)_diff1': 'Chl-a diff1 (1-day)',
    '클로로필-a (mg/㎥)_ma7': 'Chl-a (7-day moving avg)',
}

SITE_NAME_MAP = {
    '공주': 'Gongju',
    '대청호': 'Daecheong',
    '갑천': 'Gapcheon',
    '부여': 'Buyeo',
    '용담호': 'Yongdam'
}

def translate_site_names(site_names: List[str]) -> List[str]:
    """Translate Korean site names to English"""
    return [SITE_NAME_MAP.get(s, s) for s in site_names]

def translate_feature_names(feature_names: List[str]) -> List[str]:
    """Translate Korean feature names to English for visualization"""
    translated = []
    for name in feature_names:
        # Check if exact match exists
        if name in KOREAN_TO_ENGLISH_MAP:
            translated_name = KOREAN_TO_ENGLISH_MAP[name]
        else:
            # Try partial match for engineered features
            translated_name = name
            for kr, en in KOREAN_TO_ENGLISH_MAP.items():
                if kr in name:
                    # Replace Korean part with English
                    translated_name = name.replace(kr, en)
                    break
            if translated_name == name:
                translated_name = name  # Keep original if no match
        
        # Remove [ECD] prefix if present
        if '[ECD]' in translated_name:
            translated_name = translated_name.replace('[ECD]', '').strip()
        
        # Fix spacing issues: remove space between unit prefix and unit
        # e.g., "μg/L" should not have space before "g"
        translated_name = translated_name.replace('μ g/', 'μg/').replace('μ g/', 'μg/')
        translated_name = translated_name.replace('° C', '°C')
        
        # Fix "ma7" to "7-day moving avg" if present
        if 'ma7' in translated_name.lower():
            translated_name = translated_name.replace('ma7', '7-day moving avg')
            translated_name = translated_name.replace('(7-day avg)', '(7-day moving avg)')
        
        translated.append(translated_name)
    return translated

def setup_korean_font():
    """Setup matplotlib font for Korean support based on OS"""
    try:
        if platform.system() == 'Darwin':  # macOS
            plt.rcParams['font.family'] = 'AppleGothic'
        elif platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        else:  # Linux
            plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
    except:
        pass  # Use default font if Korean font not available

def cleanup_memory():
    """Memory cleanup routine for Apple Silicon"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def get_device():
    """Get the appropriate device (MPS, CUDA, or CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
